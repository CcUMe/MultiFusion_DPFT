from __future__ import annotations

import os
import os.path as osp

from typing import Any, Dict, List

import torch

from dprt.utils.data import decollate_batch


class LHPairsExporter:
    def __init__(self,
                 conf_thrs: List[float] = None,
                 categories: Dict[str, int] = None,
                 **kwargs):
        del kwargs
        self.conf_thrs = conf_thrs if conf_thrs is not None else [0.0, 0.3, 0.5, 0.7, 0.9]
        self.categories = categories

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, value: Dict[str, int] | None):
        if value is None:
            self._categories = {0: 'Background'}
            return
        self._categories = {int(idx): name for name, idx in value.items()}

    def __call__(self, *args, **kwargs) -> None:
        self.export(*args, **kwargs)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LHPairsExporter':
        evaluate_config = config.get('evaluate', {})
        exporter_config = evaluate_config.get('exporter', {})
        return cls(
            conf_thrs=exporter_config.get('conf_thrs'),
            categories=config.get('data', {}).get('categories'),
        )

    @staticmethod
    def write(lines: List[str], dst: str) -> None:
        os.makedirs(osp.dirname(dst), exist_ok=True)
        with open(dst, 'w', encoding='utf-8') as f:
            if lines:
                f.writelines(s + "\n" for s in lines)

    @staticmethod
    def append(lines: List[str], dst: str) -> None:
        os.makedirs(osp.dirname(dst), exist_ok=True)
        with open(dst, 'a', encoding='utf-8') as f:
            if lines:
                f.writelines(s + "\n" for s in lines)

    def _sample_step(self, target: Dict[str, torch.Tensor], fallback_step: int) -> int:
        image_id = target.get('image_id')
        if image_id is None or image_id.numel() == 0:
            return fallback_step
        return int(image_id.reshape(-1)[0].item())

    def _serialize_predictions(self,
                               output: Dict[str, torch.Tensor],
                               conf_thr: float) -> List[str]:
        probs = output['class']
        pred_scores, pred_labels = probs[:, 1:].max(dim=-1)
        pred_labels = pred_labels + 1
        pred_boxes = output.get('boxes_xyxy')
        if pred_boxes is None:
            pred_boxes = output['boxes']
        keep = pred_scores >= conf_thr
        lines = []
        for score, label, box in zip(pred_scores[keep], pred_labels[keep], pred_boxes[keep]):
            class_name = self.categories.get(int(label.item()), f'class_{int(label.item())}')
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            lines.append(f'{class_name} {float(score.item()):.6f} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}')
        return lines

    def _serialize_targets(self, target: Dict[str, torch.Tensor]) -> List[str]:
        labels = target.get('labels')
        boxes = target.get('boxes')
        if labels is None or boxes is None:
            return []
        lines = []
        for label, box in zip(labels, boxes):
            class_name = self.categories.get(int(label.item()), f'class_{int(label.item())}')
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            lines.append(f'{class_name} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}')
        return lines

    def export(self,
               outputs: Dict[str, torch.Tensor],
               targets: List[Dict[str, torch.Tensor]],
               step: int,
               dst: str) -> None:
        output_batch = decollate_batch(outputs, detach=True, pad=False)
        for conf_thr in self.conf_thrs:
            folder = osp.join(dst, 'exports', 'lh_pairs', str(conf_thr))
            for index, (output, target) in enumerate(zip(output_batch, targets)):
                sample_step = self._sample_step(target, step + index)
                stem = str(sample_step).zfill(6)
                self.write(self._serialize_predictions(output, conf_thr), osp.join(folder, 'preds', f'{stem}.txt'))
                self.write(self._serialize_targets(target), osp.join(folder, 'gts', f'{stem}.txt'))
                self.append([stem], osp.join(folder, 'val.txt'))


def build_lh_pairs(*args, **kwargs):
    return LHPairsExporter.from_config(*args, **kwargs)
