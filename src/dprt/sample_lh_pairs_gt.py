from __future__ import annotations

import argparse
import os
import os.path as osp
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import torch

from dprt.datasets import init as init_dataset
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed

BOX_COLORS = {
    'rgb': (0, 200, 0),
    'ir': (0, 140, 255),
    'micro': (255, 120, 0),
}
TEXT_BG = (0, 0, 0)
PANEL_BG = (32, 32, 32)


def _to_bgr(image: torch.Tensor):
    image = image.detach().cpu().clamp(0, 1).numpy()
    image = (image * 255).astype('uint8')
    image = image.transpose(1, 2, 0)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _draw_text(image, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - base - 4)
    cv2.rectangle(image, (x, y0), (x + tw + 8, y0 + th + base + 8), TEXT_BG, -1)
    cv2.putText(image, text, (x + 4, y0 + th + 2), font, scale, color, thickness, cv2.LINE_AA)


def _draw_boxes(image, boxes: torch.Tensor, labels: torch.Tensor, class_names: List[str], color, title: str):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        class_idx = int(labels[i])
        name = class_names[class_idx] if 0 <= class_idx < len(class_names) else str(class_idx)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        _draw_text(canvas, name, x1, y1, color)
    _draw_text(canvas, title, 8, 26, color)
    return canvas


def _pad_to_height(image, height: int):
    h, w = image.shape[:2]
    if h == height:
        return image
    out = image.copy()
    if h > height:
        return cv2.resize(out, (int(round(w * height / h)), height), interpolation=cv2.INTER_LINEAR)
    pad = height - h
    top = pad // 2
    bottom = pad - top
    return cv2.copyMakeBorder(out, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=PANEL_BG)


def _concat_panels(images: List) -> any:
    height = max(image.shape[0] for image in images)
    normalized = [_pad_to_height(image, height) for image in images]
    return cv2.hconcat(normalized)


def _class_names(categories: Dict[str, int]) -> List[str]:
    max_idx = max(int(v) for v in categories.values())
    names = ['Background'] * (max_idx + 1)
    for name, idx in categories.items():
        idx = int(idx)
        if idx >= 0:
            names[idx] = name
    return names


def _sample_indices(length: int, count: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    count = min(count, length)
    return sorted(rng.sample(range(length), count))


def _panel_title(split: str, index: int, rel_path: str, labels: torch.Tensor) -> str:
    return f'{split} idx={index} boxes={int(labels.numel())} {rel_path}'


def main():
    parser = argparse.ArgumentParser('Randomly visualize LH pairs GT boxes')
    parser.add_argument('--src', default='/mnt/disk1/zhangzhibin/dataset/LH_pairs_dataset_auto_v2')
    parser.add_argument('--cfg', default='/mnt/disk1/zhangzhibin/dpft_v4/config/la-tom-qwen14b.json')
    parser.add_argument('--dst', default='/mnt/disk1/zhangzhibin/test/lh_pairs_gt_check')
    parser.add_argument('--per-split', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    config = load_config(args.cfg)
    set_seed(args.seed)
    config['computing']['workers'] = 0
    config['model']['input_enable']['camera_mono'] = True
    config['model']['input_enable']['ir_image'] = True
    config['model']['input_enable']['micro_light'] = True

    out_root = Path(args.dst)
    out_root.mkdir(parents=True, exist_ok=True)
    class_names = _class_names(config['data']['categories'])

    for split_offset, split in enumerate(args.splits):
        dataset = init_dataset(dataset=config['dataset'], src=args.src, split=split, config=config)
        indices = _sample_indices(len(dataset), args.per_split, args.seed + split_offset)
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f'{split}: total={len(dataset)}, sampled={len(indices)}, save_dir={split_dir}')

        for order, dataset_index in enumerate(indices):
            sample = dataset.samples[dataset_index]
            target = dataset._load_target(sample)

            rgb_image = dataset._load_image(sample['rgb_image']) if sample.get('rgb_image') is not None else None
            ir_image = dataset._load_image(sample['ir_image']) if sample.get('ir_image') is not None else None
            micro_image = dataset._load_image(sample['micro_light']) if sample.get('micro_light') is not None else None

            labels = target['labels']
            rgb_panel = None
            ir_panel = None
            micro_panel = None

            if rgb_image is not None:
                rgb_panel = _draw_boxes(_to_bgr(rgb_image), target['boxes'], labels, class_names, BOX_COLORS['rgb'], 'RGB GT')
            if ir_image is not None:
                ir_boxes = target['ir_boxes'][target['ir_valid']]
                ir_labels = labels[target['ir_valid']]
                ir_panel = _draw_boxes(_to_bgr(ir_image), ir_boxes, ir_labels, class_names, BOX_COLORS['ir'], 'IR projected GT')
            if micro_image is not None:
                micro_boxes = target['micro_boxes'][target['micro_valid']]
                micro_labels = labels[target['micro_valid']]
                micro_panel = _draw_boxes(_to_bgr(micro_image), micro_boxes, micro_labels, class_names, BOX_COLORS['micro'], 'MICRO projected GT')

            panels = [panel for panel in [rgb_panel, ir_panel, micro_panel] if panel is not None]
            if not panels:
                continue
            merged = _concat_panels(panels)
            rel_path = str(Path(sample['json']).parent.relative_to(Path(args.src)))
            footer = _panel_title(split, dataset_index, rel_path, labels)
            cv2.rectangle(merged, (0, 0), (merged.shape[1] - 1, 34), PANEL_BG, -1)
            _draw_text(merged, footer, 8, 30, (255, 255, 255))

            stem = f'{order:04d}_idx{dataset_index:06d}'
            cv2.imwrite(str(split_dir / f'{stem}.jpg'), merged)

    print(f'Done. Visualizations saved to {out_root}')


if __name__ == '__main__':
    main()
