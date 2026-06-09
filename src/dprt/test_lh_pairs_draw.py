from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import torch

from torch.utils.data import DataLoader, Subset
from torchvision.ops import nms

from dprt.datasets import init as init_dataset
from dprt.datasets.loader import listed_collating
from dprt.evaluation.metric import mAP2D
from dprt.models import build as build_model
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


PRED_COLOR = (0, 0, 255)
GT_COLOR = (0, 200, 0)
TEXT_BG = (0, 0, 0)


def _load_detector(checkpoint: str, config: Dict, device: torch.device) -> torch.nn.Module:
    model = build_model(config['model']['name'], config)
    obj = torch.load(checkpoint, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and 'model_state_dict' in obj:
        model.load_state_dict(obj['model_state_dict'], strict=False)
    elif isinstance(obj, dict):
        model.load_state_dict(obj, strict=False)
    else:
        raise TypeError(f'Unsupported checkpoint type: {type(obj)}')

    model.to(device)
    model.eval()
    return model


def _to_device(data, device: torch.device):
    if torch.is_tensor(data):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_device(v, device) for v in data]
    return data


def _class_names(categories: Dict[str, int]) -> List[str]:
    pairs = sorted(
        ((name, int(idx)) for name, idx in categories.items() if isinstance(idx, int) and int(idx) >= 0),
        key=lambda item: item[1],
    )
    return [name for name, _ in pairs]


def _postprocess(
    output: Dict[str, torch.Tensor],
    score_threshold: float,
    nms_iou: float,
    max_detections: int,
) -> Dict[str, torch.Tensor]:
    class_scores = output['class'][0]
    boxes_xyxy = output['boxes_xyxy'][0]
    boxes_cxcywh = output['boxes'][0]
    num_classes = class_scores.shape[-1]

    scores, labels = class_scores[:, 1:].max(dim=-1)
    labels = labels + 1
    keep = scores >= score_threshold
    if keep.sum() == 0:
        empty_boxes = boxes_xyxy.new_zeros((0, 4))
        empty_scores = class_scores.new_zeros((0, num_classes))
        return {
            'boxes_xyxy': empty_boxes.unsqueeze(0),
            'boxes': empty_boxes.unsqueeze(0),
            'class': empty_scores.unsqueeze(0),
            'scores': boxes_xyxy.new_zeros((0,)),
            'labels': labels.new_zeros((0,)),
        }

    boxes_xyxy = boxes_xyxy[keep]
    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    labels = labels[keep]

    kept_indices = []
    for class_idx in labels.unique(sorted=True):
        class_mask = labels == class_idx
        local = torch.nonzero(class_mask, as_tuple=False).flatten()
        if nms_iou > 0:
            selected = nms(boxes_xyxy[local], scores[local], nms_iou)
            local = local[selected]
        kept_indices.append(local)

    if kept_indices:
        keep_idx = torch.cat(kept_indices)
        order = scores[keep_idx].argsort(descending=True)
        keep_idx = keep_idx[order[:max_detections]]
    else:
        keep_idx = torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)

    boxes_xyxy = boxes_xyxy[keep_idx]
    boxes_cxcywh = boxes_cxcywh[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    filtered_class = class_scores.new_zeros((labels.shape[0], num_classes))
    if labels.numel():
        filtered_class[torch.arange(labels.shape[0], device=labels.device), labels] = scores

    return {
        'boxes_xyxy': boxes_xyxy.unsqueeze(0),
        'boxes': boxes_cxcywh.unsqueeze(0),
        'class': filtered_class.unsqueeze(0),
        'scores': scores,
        'labels': labels,
    }


def _tensor_image_to_bgr(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu().clamp(0, 1).numpy()
    image = (image * 255).astype('uint8')
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _draw_text(image, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - base - 2)
    cv2.rectangle(image, (x, y0), (x + tw + 4, y0 + th + base + 4), TEXT_BG, -1)
    cv2.putText(image, text, (x + 2, y0 + th + 1), font, scale, color, thickness, cv2.LINE_AA)


def _draw_boxes(
    image,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor | None,
    names: List[str],
    color: Tuple[int, int, int],
    prefix: str,
) -> None:
    h, w = image.shape[:2]
    boxes = boxes_xyxy.detach().cpu().clone()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    boxes = boxes.round().to(torch.int64)
    labels = labels.detach().cpu().to(torch.int64)
    scores_cpu = scores.detach().cpu() if scores is not None else None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        class_idx = int(labels[i])
        name = names[class_idx] if 0 <= class_idx < len(names) else f'class_{class_idx}'
        if scores_cpu is None:
            text = f'{prefix}:{name}'
        else:
            text = f'{prefix}:{name} {float(scores_cpu[i]):.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        _draw_text(image, text, x1, y1, color)


def _metric_table(results: Dict[str, torch.Tensor], class_names: List[str]) -> str:
    lines = []
    lines.append(f"{'Class':<18}{'AP':>8}{'GT':>8}{'Pred':>8}{'TP':>8}{'FP':>8}{'FN':>8}{'Prec':>8}{'Rec':>8}")
    for name in class_names[1:]:
        if f'GT_{name}' not in results:
            continue
        ap = float(results.get(name, torch.tensor(0.0)).detach().cpu())
        gt = int(float(results[f'GT_{name}'].detach().cpu()))
        pred = int(float(results.get(f'Pred_{name}', torch.tensor(0.0)).detach().cpu()))
        tp = int(float(results.get(f'TP_{name}', torch.tensor(0.0)).detach().cpu()))
        fp = int(float(results.get(f'FP_{name}', torch.tensor(0.0)).detach().cpu()))
        fn = int(float(results.get(f'FN_{name}', torch.tensor(0.0)).detach().cpu()))
        prec = float(results.get(f'Precision_{name}', torch.tensor(0.0)).detach().cpu())
        rec = float(results.get(f'Recall_{name}', torch.tensor(0.0)).detach().cpu())
        lines.append(f'{name:<18}{ap:>8.4f}{gt:>8}{pred:>8}{tp:>8}{fp:>8}{fn:>8}{prec:>8.4f}{rec:>8.4f}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser('LH RGB/IR 2D detector test and visualization')
    parser.add_argument('--src', default='/mnt/disk1/zhangzhibin/dataset/LH_pairs_dataset')
    parser.add_argument('--cfg', default='/mnt/disk1/zhangzhibin/dpft_v4/config/la-tom.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dst', default='/mnt/disk1/zhangzhibin/test/2D-vis')
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--draw-size', choices=['original', 'resized'], default='original',
                        help='Draw boxes on original visible image or resized model input.')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--score-threshold', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.5)
    parser.add_argument('--max-detections', type=int, default=100)
    parser.add_argument('--draw-gt', action='store_true', default=True)
    parser.add_argument('--no-draw-gt', action='store_false', dest='draw_gt')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    config = load_config(args.cfg)
    set_seed(config['computing']['seed'])
    if args.device is not None:
        config['computing']['device'] = args.device
    config['train']['batch_size'] = args.batch_size
    config['computing']['workers'] = 0

    device = torch.device(config['computing'].get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    raw_dataset = init_dataset(dataset=config['dataset'], src=args.src, split=args.split, config=config)
    dataset = raw_dataset
    if args.num_samples > 0:
        dataset = Subset(dataset, range(min(args.num_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=listed_collating)

    names = _class_names(config['data']['categories'])
    model = _load_detector(args.checkpoint, config, device)
    metric = mAP2D(
        threshold=config.get('evaluate', {}).get('metrics', {}).get('mAP', {}).get('threshold', 0.5),
        score_threshold=args.score_threshold,
        class_names=names,
    )

    out_dir = Path(args.dst)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Testing split={args.split}, samples={len(dataset)}, checkpoint={args.checkpoint}')
    print(f'Draw size: {args.draw_size}')
    print(f'Visualizations will be saved to: {out_dir}')

    metric_inputs = []
    metric_targets = []
    saved = 0
    with torch.no_grad():
        for batch_idx, (batch, targets) in enumerate(loader):
            batch = _to_device(batch, device)
            targets = _to_device(list(targets), device)
            output = model(batch)

            for sample_idx, target in enumerate(targets):
                single_output = {k: v[sample_idx:sample_idx + 1] for k, v in output.items() if torch.is_tensor(v)}
                processed = _postprocess(
                    single_output,
                    score_threshold=args.score_threshold,
                    nms_iou=args.nms_iou,
                    max_detections=args.max_detections,
                )
                metric_inputs.append({k: v.squeeze(0).detach() for k, v in processed.items() if k in {'boxes', 'boxes_xyxy', 'class'}})
                metric_targets.append({k: v.detach() for k, v in target.items()})

                image_id = int(target.get('image_id', torch.tensor(saved)).detach().cpu())
                if args.draw_size == 'original' and hasattr(raw_dataset, 'samples') and image_id < len(raw_dataset.samples):
                    image_path = raw_dataset.samples[image_id]['rgb_image']
                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        image = _tensor_image_to_bgr(batch['camera_mono'][sample_idx])
                else:
                    image = _tensor_image_to_bgr(batch['camera_mono'][sample_idx])

                if args.draw_gt and target['boxes'].numel():
                    _draw_boxes(
                        image,
                        target['boxes'],
                        target['labels'],
                        None,
                        names,
                        GT_COLOR,
                        'GT',
                    )
                pred_labels = processed['labels']
                pred_scores = processed['scores']
                if pred_labels.numel():
                    _draw_boxes(
                        image,
                        processed['boxes_xyxy'][0],
                        pred_labels,
                        pred_scores,
                        names,
                        PRED_COLOR,
                        'P',
                    )

                save_path = out_dir / f'{batch_idx:05d}_{sample_idx:02d}_id{image_id:06d}.jpg'
                cv2.imwrite(str(save_path), image)
                saved += 1

    results = metric.compute_dataset(metric_inputs, metric_targets)
    print('\n2D detection result after score filtering / NMS')
    print(f"mAP@{float(results['IoU_threshold']):.2f}: {float(results['mAP']):.4f}")
    print(_metric_table(results, names))
    print(f'\nSaved {saved} visualization images to {out_dir}')


if __name__ == '__main__':
    main()
