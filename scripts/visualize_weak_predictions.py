import argparse
import math
import os
import os.path as osp
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
SRC = osp.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dprt.datasets import init as init_dataset
from dprt.datasets.weak_labels import projected_boxes, weak_label_config
from dprt.models import load_model
from dprt.models.confidence import confidence_model_args, load_confidence_modules
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 192, 255),
    (255, 0, 192),
    (128, 255, 0),
    (0, 128, 255),
    (255, 0, 0),
]


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install opencv-python to draw prediction overlays.") from exc
    return cv2


def class_names_from_config(config: Dict) -> List[str]:
    categories = config.get("data", {}).get("categories", {})
    return [
        name
        for name, idx in sorted(
            ((name, idx) for name, idx in categories.items() if isinstance(idx, int) and idx >= 0),
            key=lambda item: item[1],
        )
    ]


def to_device(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in data.items()}


def batchify(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in data.items()
    }


def load_projection(path: str) -> np.ndarray:
    projection = np.asarray(np.load(path), dtype=np.float64)
    if projection.shape == (3, 4):
        return projection
    if projection.shape == (4, 4):
        return projection[:3]
    if projection.shape == (3, 3):
        return np.concatenate([projection, np.zeros((3, 1), dtype=projection.dtype)], axis=1)
    flat = projection.reshape(-1)
    if flat.size == 12:
        return flat.reshape(3, 4)
    if flat.size == 16:
        return flat.reshape(4, 4)[:3]
    raise ValueError(f"Unsupported projection shape {projection.shape}: {path}")


def resolve_indices(dataset, args) -> List[int]:
    if args.indices:
        return [int(index) for index in args.indices]

    if args.sequence:
        matches = []
        sequence_name = str(args.sequence)
        for index, paths in enumerate(dataset.dataset_paths):
            sample_dir = osp.dirname(paths["camera_mono"])
            current_sequence = osp.basename(osp.dirname(sample_dir))
            if current_sequence == sequence_name:
                matches.append(index)
        if not matches:
            raise ValueError(f"Sequence not found in split '{args.split}': {args.sequence}")
        return matches[: args.max_samples]

    if args.sample:
        normalized = osp.normpath(args.sample)
        matches = []
        for index, paths in enumerate(dataset.dataset_paths):
            sample_dir = osp.normpath(osp.dirname(paths["camera_mono"]))
            if normalized == sample_dir or normalized in sample_dir:
                matches.append(index)
        if not matches:
            raise ValueError(f"Sample not found in split '{args.split}': {args.sample}")
        return matches[: args.max_samples]

    return list(range(min(args.max_samples, len(dataset))))


def predictions_to_boxes(
    output: Dict[str, torch.Tensor],
    projection: np.ndarray,
    image_shape: Tuple[int, int],
    config: Dict,
    score_threshold: float,
):
    class_scores = output["class"][0].detach().cpu()
    centers = output["center"][0].detach().cpu().numpy()
    sizes = output["size"][0].detach().cpu().numpy()

    pred_score, pred_label = torch.max(class_scores, dim=-1)
    keep = (pred_label > 0) & (pred_score >= score_threshold)
    if not keep.any():
        return np.zeros((0, 4), dtype=np.float32), [], []

    keep_np = keep.numpy()
    class_ids = pred_label[keep].numpy().astype(np.int64) - 1
    scores = pred_score[keep].numpy()
    centers = centers[keep_np]
    size_lh = sizes[keep_np][:, (0, 2)]

    weak_cfg = weak_label_config(config.get("data", {}).get("weak_label", {}))
    depth_ratios = np.asarray(
        [weak_cfg["class_depth_ratios"].get(int(class_id), 1.0) for class_id in class_ids],
        dtype=np.float32,
    )
    bbox_scales = np.asarray(
        [weak_cfg["class_bbox_scales"].get(int(class_id), 1.0) for class_id in class_ids],
        dtype=np.float32,
    )
    yaw = np.zeros((centers.shape[0],), dtype=np.float32)

    boxes = projected_boxes(
        centers=centers,
        length_heights=size_lh,
        yaw=yaw,
        depth_ratios=depth_ratios,
        bbox_scales=bbox_scales,
        projection=projection,
        image_shape=image_shape,
        bbox_mode=str(weak_cfg["bbox_mode"]),
    )
    valid = np.isfinite(boxes).all(axis=1)
    return boxes[valid], class_ids[valid], scores[valid]


def draw_predictions(image, boxes, class_ids, scores, class_names):
    cv2 = require_cv2()
    canvas = image.copy()
    for box, class_id, score in zip(boxes, class_ids, scores):
        color = COLORS[int(class_id) % len(COLORS)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = class_names[int(class_id)] if 0 <= int(class_id) < len(class_names) else str(class_id)
        text = f"{name} {float(score):.2f}"
        cv2.putText(canvas, text, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return canvas


def main():
    parser = argparse.ArgumentParser("Visualize weak DPRT predictions on mono camera images.")
    parser.add_argument('--src', type=str, default='/mnt/disk1/zhangzhibin/dataset/kradar/',
                        help="Path to the processed dataset folder.")
    parser.add_argument('--cfg', type=str, default='/home/yangqilin/code/dpft_v4/config/la-tom.json',
                        help="Path to the configuration file.")
    parser.add_argument('--dst', type=str, default="/mnt/disk1/yangqilin/dpft/visualize_weak/",
                        help="Path to save the training log.")
    parser.add_argument('--checkpoint', type=str,default="/mnt/disk1/yangqilin/dpft/log/20260508-173506-590/checkpoints/20260508-173506-590_checkpoint_0195.pt",
                        help="Path to a model checkpoint to resume training from.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--max-samples", type=int, default=math.inf, help="Number of samples to visualize.")
    parser.add_argument("--indices", nargs="*", help="Optional dataset indices to visualize.")
    parser.add_argument("--sequence", default=None, help="Optional scene/sequence id, e.g. 12 for test/12/xxxx.")
    parser.add_argument("--sample", help="Optional sample directory or unique path fragment.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Prediction score threshold.")
    args = parser.parse_args()

    cv2 = require_cv2()
    os.makedirs(args.dst, exist_ok=True)

    config = load_config(args.cfg)
    set_seed(config["computing"]["seed"])
    device = torch.device(config["computing"]["device"])
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else config.get("evaluate", {}).get("weak_metrics", {}).get("weak", {}).get("score_threshold", 0.3)
    )

    dataset = init_dataset(dataset=config["dataset"], src=args.src, split=args.split, config=config)
    model, epoch, timestamp = load_model(args.checkpoint, config)
    model.to(device)
    model.eval()
    confidence = load_confidence_modules(config, device)
    if confidence is not None:
        confidence.eval()

    class_names = class_names_from_config(config)
    indices = resolve_indices(dataset, args)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Epoch: {epoch}, run: {timestamp}")
    print(f"Split: {args.split}, samples: {len(indices)}, score_threshold: {score_threshold}")

    for index in indices:
        sample_paths = dataset.dataset_paths[index]
        image_path = sample_paths["camera_mono"]
        projection_path = sample_paths["label_to_camera_mono"]

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Cannot read image: {image_path}")
            continue

        data, _ = dataset[index]
        batch = to_device(batchify(data), device)
        with torch.no_grad():
            output = model(batch, *confidence_model_args(confidence, batch))

        projection = load_projection(projection_path)
        boxes, class_ids, scores = predictions_to_boxes(
            output=output,
            projection=projection,
            image_shape=image.shape[:2],
            config=config,
            score_threshold=score_threshold,
        )
        overlay = draw_predictions(image, boxes, class_ids, scores, class_names)

        sample_dir = osp.basename(osp.dirname(image_path))
        sequence = osp.basename(osp.dirname(osp.dirname(image_path)))
        out_dir = osp.join(args.dst, args.split, sequence)
        os.makedirs(out_dir, exist_ok=True)
        out_name = f"{sample_dir}_{index:06d}.jpg"
        out_path = osp.join(out_dir, out_name)
        cv2.imwrite(out_path, overlay)
        print(f"[{index}] predictions={len(boxes)} -> {out_path}")


if __name__ == "__main__":
    main()
