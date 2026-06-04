from __future__ import annotations  # noqa: F407

from typing import Any, Dict, List, Optional

import torch

from torch import nn
from torchvision.ops import box_iou
from dprt.utils.iou import iou3d, giou3d
from dprt.utils.bbox import get_box_corners
from dprt.utils.data import decollate_batch
from dprt.utils.misc import interp


class mAP3D(nn.modules.loss._Loss):
    def __init__(self,
                 threshold: float = 0.5,
                 nelem: int = 101,
                 class_names: Optional[List[str]] = None,
                 **kwargs):
        """Mean average percision for 3D bounding boxes.

        Arguments:
            threshold: IoU threshold for bounding box matching.
            nelem: Number of elements to be used for the
                discretization of the precision recall curve.
        """
        super().__init__()

        self.threshold = threshold
        self.nelem = nelem
        self.class_names = class_names if class_names is not None else []

    def _class_name(self, class_idx: int) -> str:
        class_name_idx = class_idx - 1
        if 0 <= class_name_idx < len(self.class_names):
            return self.class_names[class_name_idx]
        return f'class_{class_idx}'

    def _average_precision(self, tp: torch.Tensor, fp: torch.Tensor, npos: int) -> torch.Tensor:
        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)

        prec = torch.zeros_like(tp)
        div_mask = (fp + tp != 0)
        prec[div_mask] = tp[div_mask] / (fp[div_mask] + tp[div_mask])

        if npos == 0:
            rec = torch.ones_like(tp)
        else:
            rec = tp / float(npos)

        rec_interp = torch.linspace(0, 1, self.nelem, dtype=rec.dtype, device=rec.device)
        prec = interp(rec_interp, rec, prec, right=0)
        return torch.sum(prec * 1 / (self.nelem - 1))

    def compute_dataset(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute AP once over all validation samples."""
        if not inputs:
            return {'mAP': torch.ones((), dtype=torch.float)}

        device = inputs[0]['class'].device
        num_classes = targets[0]['gt_class'].shape[-1]
        aps = torch.zeros((num_classes, ), dtype=torch.float, device=device)
        present_classes = []
        gt_counts = torch.zeros((num_classes, ), dtype=torch.long, device=device)
        pred_counts = torch.zeros((num_classes, ), dtype=torch.long, device=device)
        tp_counts = torch.zeros((num_classes, ), dtype=torch.long, device=device)
        fp_counts = torch.zeros((num_classes, ), dtype=torch.long, device=device)

        per_sample = []
        for input, target in zip(inputs, targets):
            label = torch.argmax(input['class'], dim=-1)
            gt_label = torch.argmax(target['gt_class'], dim=-1)
            if gt_label.numel():
                gt_counts += torch.bincount(gt_label, minlength=num_classes)
            if label.numel():
                pred_counts += torch.bincount(label, minlength=num_classes)
            angle = torch.atan2(input['angle'][..., 0], input['angle'][..., 1])
            gt_angle = torch.atan2(target['gt_angle'][..., 0], target['gt_angle'][..., 1])

            per_sample.append({
                'scores': input['class'],
                'label': label,
                'gt_label': gt_label,
                'corners': get_box_corners(
                    input['center'].unsqueeze(0),
                    input['size'].unsqueeze(0),
                    angle.unsqueeze(0)
                ).squeeze(0),
                'gt_corners': get_box_corners(
                    target['gt_center'].unsqueeze(0),
                    target['gt_size'].unsqueeze(0),
                    gt_angle.unsqueeze(0)
                ).squeeze(0),
            })

        for class_idx in range(1, num_classes):
            detections = []
            gt_by_sample = []
            npos = 0

            for sample_idx, sample in enumerate(per_sample):
                pred_mask = (sample['label'] == class_idx)
                gt_mask = (sample['gt_label'] == class_idx)

                gt_corners = sample['gt_corners'][gt_mask]
                gt_by_sample.append({
                    'corners': gt_corners,
                    'matched': torch.zeros(
                        gt_corners.shape[0],
                        dtype=torch.bool,
                        device=device
                    )
                })
                npos += gt_corners.shape[0]

                pred_indices = torch.nonzero(pred_mask, as_tuple=False).flatten()
                for pred_idx in pred_indices:
                    detections.append((
                        sample['scores'][pred_idx, class_idx],
                        sample_idx,
                        sample['corners'][pred_idx]
                    ))

            if npos == 0:
                continue

            present_classes.append(class_idx)

            if not detections:
                continue

            detections.sort(key=lambda item: float(item[0]), reverse=True)
            tp = torch.zeros((len(detections), ), dtype=torch.float, device=device)
            fp = torch.ones((len(detections), ), dtype=torch.float, device=device)

            for det_idx, (_, sample_idx, pred_corners) in enumerate(detections):
                gt_entry = gt_by_sample[sample_idx]
                gt_corners = gt_entry['corners']
                unmatched = ~gt_entry['matched']

                if gt_corners.numel() == 0 or not unmatched.any():
                    continue

                ious = iou3d(
                    pred_corners.reshape(1, 1, 8, 3),
                    gt_corners.reshape(1, -1, 8, 3)
                ).reshape(-1)
                ious[~unmatched] = -1
                best_iou, best_idx = torch.max(ious, dim=0)

                if best_iou > self.threshold:
                    tp[det_idx] = 1
                    fp[det_idx] = 0
                    gt_entry['matched'][best_idx] = True

            aps[class_idx] = self._average_precision(tp, fp, npos)
            tp_counts[class_idx] = tp.sum().to(torch.long)
            fp_counts[class_idx] = fp.sum().to(torch.long)

        results: Dict[str, torch.Tensor] = {}
        for class_idx in present_classes:
            class_name = self._class_name(class_idx)
            gt_count = gt_counts[class_idx].to(torch.float)
            pred_count = pred_counts[class_idx].to(torch.float)
            tp_count = tp_counts[class_idx].to(torch.float)
            fp_count = fp_counts[class_idx].to(torch.float)
            fn_count = torch.clamp(gt_count - tp_count, min=0)
            precision = tp_count / torch.clamp(tp_count + fp_count, min=1)
            recall = tp_count / torch.clamp(gt_count, min=1)

            results[class_name] = aps[class_idx]
            results[f"GT_{class_name}"] = gt_count
            results[f"Pred_{class_name}"] = pred_count
            results[f"TP_{class_name}"] = tp_count
            results[f"FP_{class_name}"] = fp_count
            results[f"FN_{class_name}"] = fn_count
            results[f"Precision_{class_name}"] = precision
            results[f"Recall_{class_name}"] = recall

        for class_idx in range(1, num_classes):
            if gt_counts[class_idx] == 0 and pred_counts[class_idx] > 0:
                class_name = self._class_name(class_idx)
                results[f"Pred_{class_name}"] = pred_counts[class_idx].to(torch.float)
                results[f"FP_only_{class_name}"] = pred_counts[class_idx].to(torch.float)

        if not present_classes:
            results['mAP'] = torch.ones((), dtype=torch.float, device=device)
        else:
            results['mAP'] = torch.mean(aps[torch.as_tensor(present_classes, device=device)])

        results['IoU_threshold'] = torch.tensor(self.threshold, dtype=torch.float, device=device)
        results['AP_points'] = torch.tensor(self.nelem, dtype=torch.float, device=device)
        results['num_eval_classes'] = torch.tensor(len(present_classes), dtype=torch.float, device=device)

        return results

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns the mean average precision values.

        Arguments:
            inputs: This is a dict that contains at least these entries:
                "class": Bounding box class probabilities of shape (B, N, C)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).

            targets: This is a dict of targets that contains at least these entries:
                "gt_class": Bounding box class probabilities of shape (B, M, C)
                "gt_center": Bounding box center coordinates of shape (B, M, 3).
                "gt_size": Bounding box size values of shape (B, M, 3).
                "gt_angle": Bounding box orientation values of shape (B, M, 2).

        Returns:
            Dictionary containing the overall mAP under key 'mAP' and per-class
            AP values under keys 'class_1', 'class_2', ...
        """
        # Get device
        device = targets['gt_class'].device

        # Determine the number of classes
        num_classes = targets['gt_class'].shape[-1]

        label = torch.argmax(inputs['class'], dim=-1)
        gt_label = torch.argmax(targets['gt_class'], dim=-1)

        # Reconstruc angle from sin and cos part
        angle = torch.atan2(inputs['angle'][..., 0], inputs['angle'][..., 1])
        gt_angle = torch.atan2(targets['gt_angle'][..., 0], targets['gt_angle'][..., 1])

        # Initialize average precision values
        aps = torch.zeros((num_classes, ), dtype=torch.float, device=device)

        for l in range(num_classes):
            # Get class label mask with shape (B, N) and (B, M)
            mask = (label == l)
            gt_mask = (gt_label == l)

            # Get 3d boundng box corners with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners = get_box_corners(inputs['center'], inputs['size'], angle)
            gt_corners = get_box_corners(targets['gt_center'], targets['gt_size'], gt_angle)

            # Get box corners mask with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners_mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)
            gt_corners_mask = gt_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)

            # Get intersection over union with shape (B, N, M)
            iou = iou3d(torch.mul(corners, corners_mask), torch.mul(gt_corners, gt_corners_mask))

            # Flatten iou and masks along batch dimension (B, N, M) -> (B * N, M)
            iou = iou.flatten(0, 1)
            mask = mask.flatten(0, 1)
            gt_mask = gt_mask.flatten(0, 1)

            # Get number of ground truth elements
            npos = torch.sum(gt_mask).type(torch.float)

            # Sort iou and masks by confidence score
            sort_idx = torch.argsort(inputs['class'][..., l], descending=True).flatten(0, 1)
            iou = iou[sort_idx, :]
            mask = mask[sort_idx]

            # Get mask for all ious that are lower than the required threshold
            thr_mask = (iou > self.threshold)

            # Get final iou mask with shape (B * N, B * M)
            iou_mask = torch.logical_and(*torch.meshgrid(mask, gt_mask, indexing='ij'))

            # Get true positive candidates mask
            tp_c_mask = torch.logical_and(iou_mask, thr_mask)

            # Initialize true positives and false positives
            tp = torch.zeros(iou.shape[0], dtype=torch.float, device=device)
            fp = torch.ones(iou.shape[0], dtype=torch.float, device=device)

            # Get true positives
            tp_value, tp_idx = torch.max(tp_c_mask, dim=0)
            tp[tp_idx[tp_value]] = 1
            fp[tp_idx[tp_value]] = 0

            # Adjust for true negatives
            fp[~mask] = 0

            # Accumulate values
            tp = torch.cumsum(tp, dim=0)
            fp = torch.cumsum(fp, dim=0)

            # Calculate precision (avoid div by zero)
            prec = torch.zeros_like(tp)
            div_mask = (fp + tp != 0)
            prec[div_mask] = tp[div_mask] / (fp[div_mask] + tp[div_mask])

            # Calculate recall (avoid div by zero)
            if npos == 0:
                rec = torch.ones_like(tp)
            else:
                rec = tp / npos

            # Interpolate precision and recall
            rec_interp = torch.linspace(0, 1, self.nelem, dtype=rec.dtype, device=device)
            prec = interp(rec_interp, rec, prec, right=0)
            rec = rec_interp

            # Calculate average precision
            aps[l] = torch.sum(prec * 1 / (self.nelem - 1))

        # Select contributing (present) classes only and keep index 0 as ignore/background.
        selection = torch.sort(torch.unique(torch.concatenate([label, gt_label], dim=1)))[0][1:]

        results: Dict[str, torch.Tensor] = {}

        # Report per-class AP values for contributing classes only
        for class_idx in selection:
            results[self._class_name(class_idx)] = aps[class_idx]

        # Avoid empty selection
        if not selection.numel() or not selection.any():
            results['mAP'] = torch.ones((), dtype=torch.float, device=device)
            return results

        # Calculate mAP and ignore first class
        results['mAP'] = torch.mean(aps[selection])

        return results


class mGIoU3D(nn.modules.loss._Loss):
    def __init__(self):
        """Generalized intersection over union.
        """
        super().__init__()

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns the generalized intersection over union value.

        Arguments:
            inputs: This is a dict that contains at least these entries:
                "class": Bounding box class probabilities of shape (B, N, C)
                "center": Bounding box center coordinates of shape (B, N, 3).
                "size": Bounding box size values of shape (B, N, 3).
                "angle": Bounding box orientation values of shape (B, N, 2).

            targets: This is a dict of targets that contains at least these entries:
                "gt_class": Bounding box class probabilities of shape (B, M, C)
                "gt_center": Bounding box center coordinates of shape (B, M, 3).
                "gt_size": Bounding box size values of shape (B, M, 3).
                "gt_angle": Bounding box orientation values of shape (B, M, 2).

        Returns:
            giou: Generalized intersection over union value.
        """
        # Get device
        device = targets['gt_class'].device

        # Get input shapes
        num_classes = targets['gt_class'].shape[-1]

        label = torch.argmax(inputs['class'], dim=-1)
        gt_label = torch.argmax(targets['gt_class'], dim=-1)

        # Reconstruc angle from sin and cos part
        angle = torch.atan2(inputs['angle'][..., 0], inputs['angle'][..., 1])
        gt_angle = torch.atan2(targets['gt_angle'][..., 0], targets['gt_angle'][..., 1])

        # Initialize giou values
        gious = -torch.ones((num_classes, ), dtype=torch.float, device=device)

        for l in range(num_classes):
            # Get class label mask with shape (B, N) and (B, M)
            mask = (label == l)
            gt_mask = (gt_label == l)

            # Get 3d boundng box corners with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners = get_box_corners(inputs['center'], inputs['size'], angle)
            gt_corners = get_box_corners(targets['gt_center'], targets['gt_size'], gt_angle)

            # Get box corners mask with shape (B, N, 8, 3) and (B, M, 8, 3)
            corners_mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)
            gt_corners_mask = gt_mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 8, 3)

            # Get intersection over union with shape (B, N, M)
            giou = giou3d(torch.mul(corners, corners_mask), torch.mul(gt_corners, gt_corners_mask))

            # Flatten iou and masks along batch dimension (B, N, M) -> (B * N, M)
            giou = giou.flatten(0, 1)
            mask = mask.flatten(0, 1)
            gt_mask = gt_mask.flatten(0, 1)

            # Sort iou and masks by confidence score
            sort_idx = torch.argsort(inputs['class'][..., l], descending=True).flatten(0, 1)
            giou = giou[sort_idx, :]
            mask = mask[sort_idx]

            # Get final iou mask with shape (B * N, B * M)
            giou_mask = torch.logical_and(*torch.meshgrid(mask, gt_mask, indexing='ij'))

            # Set unmatched values to -1
            giou[~giou_mask] = -1

            # Get most confident match
            match_giou, _ = torch.max(giou, dim=0)

            # Add class GIoU
            if gt_mask.sum() == 0:
                gious[l] = 1.0

            if match_giou.numel() > 0 and giou_mask.any():
                gious[l] = torch.mean(match_giou)

        # Select contributing (present) classes only
        selection = torch.sort(torch.unique(torch.concatenate([label, gt_label], dim=1)))[0][1:]

        # Avoid empty selection
        if not selection.numel() or not selection.any():
            return torch.ones((), dtype=torch.float, device=device)

        # Calculate GIoU and ignore first class
        giou = torch.mean(gious[selection])

        return giou


class WeakBEVMetric(nn.modules.loss._Loss):
    def __init__(self,
                 distance_threshold: float = 1.0,
                 score_threshold: float = 0.3,
                 ap_score_threshold: float = 0.0,
                 nelem: int = 101,
                 class_names: Optional[List[str]] = None,
                 **kwargs):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.score_threshold = score_threshold
        self.ap_score_threshold = ap_score_threshold
        self.nelem = nelem
        self.class_names = class_names if class_names is not None else []

    def _class_name(self, class_idx: int) -> str:
        class_name_idx = class_idx - 1
        if 0 <= class_name_idx < len(self.class_names):
            return self.class_names[class_name_idx]
        return f'class_{class_idx}'

    def _average_precision(self,
                           tp: torch.Tensor,
                           fp: torch.Tensor,
                           npos: torch.Tensor) -> torch.Tensor:
        if tp.numel() == 0:
            return torch.zeros((), dtype=torch.float, device=tp.device)

        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(npos, min=1.0)
        recall_points = torch.linspace(
            0, 1, self.nelem, dtype=recall.dtype, device=recall.device
        )
        sampled_precision = []
        for recall_point in recall_points:
            keep = recall >= recall_point
            sampled_precision.append(
                precision[keep].max() if keep.any()
                else torch.zeros((), dtype=torch.float, device=tp.device)
            )
        return torch.stack(sampled_precision).mean()

    def compute_dataset(
        self,
        inputs: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if not inputs:
            return {'Recall': torch.zeros((), dtype=torch.float)}

        device = inputs[0]['class'].device
        num_classes = targets[0]['gt_class'].shape[-1]
        gt_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        pred_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        tp_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        center_errors = []
        per_sample = []

        for input, target in zip(inputs, targets):
            pred_score, pred_label = torch.max(input['class'], dim=-1)
            gt_label = torch.argmax(target['gt_class'], dim=-1)
            pred_center = input['center'][:, :2]
            gt_center = target['gt_center'][:, :2]
            foreground = (pred_label > 0) & (pred_score >= self.score_threshold)
            per_sample.append({
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_center': pred_center,
                'gt_label': gt_label,
                'gt_center': gt_center,
            })

            if gt_label.numel():
                gt_counts += torch.bincount(gt_label, minlength=num_classes).to(device=device, dtype=torch.float)
            if foreground.any():
                pred_counts += torch.bincount(
                    pred_label[foreground],
                    minlength=num_classes
                ).to(device=device, dtype=torch.float)

            for class_idx in range(1, num_classes):
                pred_idx = torch.nonzero((pred_label == class_idx) & foreground, as_tuple=False).flatten()
                gt_idx = torch.nonzero(gt_label == class_idx, as_tuple=False).flatten()
                if pred_idx.numel() == 0 or gt_idx.numel() == 0:
                    continue

                scores = input['class'][pred_idx, class_idx]
                pred_idx = pred_idx[torch.argsort(scores, descending=True)]
                matched = torch.zeros((gt_idx.numel(),), dtype=torch.bool, device=device)

                for p_idx in pred_idx:
                    distances = torch.linalg.norm(gt_center[gt_idx] - pred_center[p_idx], dim=-1)
                    distances[matched] = float('inf')
                    best_distance, best_local_idx = torch.min(distances, dim=0)
                    if best_distance <= self.distance_threshold:
                        matched[best_local_idx] = True
                        tp_counts[class_idx] += 1
                        center_errors.append(best_distance)

        aps = torch.zeros((num_classes,), dtype=torch.float, device=device)
        for class_idx in range(1, num_classes):
            npos = gt_counts[class_idx]
            if npos == 0:
                continue

            detections = []
            gt_by_sample = []
            for sample_idx, sample in enumerate(per_sample):
                gt_idx = torch.nonzero(sample['gt_label'] == class_idx, as_tuple=False).flatten()
                gt_by_sample.append({
                    'center': sample['gt_center'][gt_idx],
                    'matched': torch.zeros((gt_idx.numel(),), dtype=torch.bool, device=device),
                })

                pred_idx = torch.nonzero(
                    (sample['pred_label'] == class_idx)
                    & (sample['pred_score'] >= self.ap_score_threshold),
                    as_tuple=False
                ).flatten()
                for p_idx in pred_idx:
                    detections.append((
                        sample['pred_score'][p_idx],
                        sample_idx,
                        sample['pred_center'][p_idx],
                    ))

            if not detections:
                continue

            detections.sort(key=lambda item: float(item[0]), reverse=True)
            tp = torch.zeros((len(detections),), dtype=torch.float, device=device)
            fp = torch.ones((len(detections),), dtype=torch.float, device=device)
            for det_idx, (_, sample_idx, pred_center) in enumerate(detections):
                gt_entry = gt_by_sample[sample_idx]
                if gt_entry['center'].numel() == 0 or gt_entry['matched'].all():
                    continue

                distances = torch.linalg.norm(gt_entry['center'] - pred_center, dim=-1)
                distances[gt_entry['matched']] = float('inf')
                best_distance, best_local_idx = torch.min(distances, dim=0)
                if best_distance <= self.distance_threshold:
                    gt_entry['matched'][best_local_idx] = True
                    tp[det_idx] = 1
                    fp[det_idx] = 0

            aps[class_idx] = self._average_precision(tp, fp, npos)

        results: Dict[str, torch.Tensor] = {}
        total_tp = tp_counts[1:].sum()
        total_gt = gt_counts[1:].sum()
        total_pred = pred_counts[1:].sum()
        present_classes = torch.nonzero(gt_counts[1:] > 0, as_tuple=False).flatten() + 1
        results['center_mAP'] = (
            aps[present_classes].mean()
            if present_classes.numel() else torch.zeros((), dtype=torch.float, device=device)
        )
        results['Recall'] = total_tp / torch.clamp(total_gt, min=1.0)
        results['Precision'] = total_tp / torch.clamp(total_pred, min=1.0)
        results['GT'] = total_gt
        results['Pred'] = total_pred
        results['TP'] = total_tp
        results['FP'] = torch.clamp(total_pred - total_tp, min=0.0)
        results['FN'] = torch.clamp(total_gt - total_tp, min=0.0)
        results['Center_MAE'] = (
            torch.stack(center_errors).mean()
            if center_errors else torch.zeros((), dtype=torch.float, device=device)
        )
        results['Distance_threshold'] = torch.tensor(self.distance_threshold, dtype=torch.float, device=device)
        results['Score_threshold'] = torch.tensor(self.score_threshold, dtype=torch.float, device=device)
        results['AP_score_threshold'] = torch.tensor(self.ap_score_threshold, dtype=torch.float, device=device)
        results['AP_points'] = torch.tensor(self.nelem, dtype=torch.float, device=device)
        results['num_eval_classes'] = torch.sum(gt_counts[1:] > 0).to(dtype=torch.float)

        for class_idx in range(1, num_classes):
            if gt_counts[class_idx] == 0 and pred_counts[class_idx] == 0:
                continue
            name = self._class_name(class_idx)
            results[f'AP_{name}'] = aps[class_idx]
            results[f'GT_{name}'] = gt_counts[class_idx]
            results[f'Pred_{name}'] = pred_counts[class_idx]
            results[f'TP_{name}'] = tp_counts[class_idx]
            results[f'FP_{name}'] = torch.clamp(pred_counts[class_idx] - tp_counts[class_idx], min=0.0)
            results[f'FN_{name}'] = torch.clamp(gt_counts[class_idx] - tp_counts[class_idx], min=0.0)
            results[f'Recall_{name}'] = tp_counts[class_idx] / torch.clamp(gt_counts[class_idx], min=1.0)
            results[f'Precision_{name}'] = tp_counts[class_idx] / torch.clamp(pred_counts[class_idx], min=1.0)

        return results



class mAP2D(nn.modules.loss._Loss):
    def __init__(self, threshold: float = 0.5, nelem: int = 101, score_threshold: float = 0.05,
                 class_names: Optional[List[str]] = None, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.nelem = nelem
        self.score_threshold = score_threshold
        self.class_names = class_names if class_names is not None else []

    def _class_name(self, class_idx: int) -> str:
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f'class_{class_idx}'

    def _average_precision(self, tp: torch.Tensor, fp: torch.Tensor, npos: int) -> torch.Tensor:
        tp = torch.cumsum(tp, dim=0)
        fp = torch.cumsum(fp, dim=0)
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / float(max(npos, 1))
        rec_interp = torch.linspace(0, 1, self.nelem, dtype=recall.dtype, device=recall.device)
        prec_interp = interp(rec_interp, recall, precision, right=0)
        return prec_interp.mean()

    def compute_dataset(self, inputs: List[Dict[str, torch.Tensor]],
                        targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not inputs:
            return {'mAP': torch.ones((), dtype=torch.float)}
        score_key = 'class' if 'class' in inputs[0] else 'class_logits'
        device = inputs[0][score_key].device
        num_classes = inputs[0][score_key].shape[-1]
        gt_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        pred_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        tp_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        fp_counts = torch.zeros((num_classes,), dtype=torch.float, device=device)
        aps = torch.zeros((num_classes,), dtype=torch.float, device=device)
        per_sample = []
        for input, target in zip(inputs, targets):
            if 'class' in input:
                probs = input['class']
            else:
                probs = torch.softmax(input['class_logits'], dim=-1)
            pred_scores, pred_labels = probs[:, 1:].max(dim=-1)
            pred_labels = pred_labels + 1
            pred_boxes = input.get('boxes_xyxy')
            if pred_boxes is None:
                pred_boxes = cxcywh_to_xyxy_metric(input['boxes'])
            keep = pred_scores >= self.score_threshold
            labels = target['labels']
            boxes = target['boxes']
            if labels.numel():
                gt_counts += torch.bincount(labels, minlength=num_classes).to(dtype=torch.float)
            if keep.any():
                pred_counts += torch.bincount(pred_labels[keep], minlength=num_classes).to(dtype=torch.float)
            per_sample.append({
                'pred_scores': pred_scores[keep],
                'pred_labels': pred_labels[keep],
                'pred_boxes': pred_boxes[keep],
                'gt_labels': labels,
                'gt_boxes': boxes,
            })
        present_classes = []
        for class_idx in range(1, num_classes):
            npos = int(gt_counts[class_idx].item())
            if npos == 0:
                continue
            present_classes.append(class_idx)
            detections = []
            gt_by_sample = []
            for sample_idx, sample in enumerate(per_sample):
                gt_mask = sample['gt_labels'] == class_idx
                gt_boxes = sample['gt_boxes'][gt_mask]
                gt_by_sample.append({
                    'boxes': gt_boxes,
                    'matched': torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=device),
                })
                pred_mask = sample['pred_labels'] == class_idx
                for pred_idx in torch.nonzero(pred_mask, as_tuple=False).flatten():
                    detections.append((sample['pred_scores'][pred_idx], sample_idx, sample['pred_boxes'][pred_idx]))
            if not detections:
                continue
            detections.sort(key=lambda item: float(item[0]), reverse=True)
            tp = torch.zeros((len(detections),), dtype=torch.float, device=device)
            fp = torch.ones((len(detections),), dtype=torch.float, device=device)
            for det_idx, (_, sample_idx, pred_box) in enumerate(detections):
                gt_entry = gt_by_sample[sample_idx]
                gt_boxes = gt_entry['boxes']
                if gt_boxes.numel() == 0 or gt_entry['matched'].all():
                    continue
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes).reshape(-1)
                ious[gt_entry['matched']] = -1
                best_iou, best_idx = torch.max(ious, dim=0)
                if best_iou >= self.threshold:
                    gt_entry['matched'][best_idx] = True
                    tp[det_idx] = 1
                    fp[det_idx] = 0
            aps[class_idx] = self._average_precision(tp, fp, npos)
            tp_counts[class_idx] = tp.sum()
            fp_counts[class_idx] = fp.sum()
        results: Dict[str, torch.Tensor] = {}
        if present_classes:
            idx = torch.as_tensor(present_classes, dtype=torch.long, device=device)
            results['mAP'] = aps[idx].mean()
        else:
            results['mAP'] = torch.zeros((), dtype=torch.float, device=device)
        results['IoU_threshold'] = torch.tensor(self.threshold, dtype=torch.float, device=device)
        results['AP_points'] = torch.tensor(self.nelem, dtype=torch.float, device=device)
        results['num_eval_classes'] = torch.tensor(len(present_classes), dtype=torch.float, device=device)
        for class_idx in range(1, num_classes):
            if gt_counts[class_idx] == 0 and pred_counts[class_idx] == 0:
                continue
            name = self._class_name(class_idx)
            results[name] = aps[class_idx]
            results[f'GT_{name}'] = gt_counts[class_idx]
            results[f'Pred_{name}'] = pred_counts[class_idx]
            results[f'TP_{name}'] = tp_counts[class_idx]
            fp_value = fp_counts[class_idx]
            if gt_counts[class_idx] == 0 and pred_counts[class_idx] > 0:
                fp_value = pred_counts[class_idx]
            results[f'FP_{name}'] = fp_value
            results[f'FN_{name}'] = torch.clamp(gt_counts[class_idx] - tp_counts[class_idx], min=0.0)
            results[f'Precision_{name}'] = tp_counts[class_idx] / torch.clamp(tp_counts[class_idx] + fp_value, min=1.0)
            results[f'Recall_{name}'] = tp_counts[class_idx] / torch.clamp(gt_counts[class_idx], min=1.0)
        return results


def cxcywh_to_xyxy_metric(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1).clamp(0.0, 1.0)


class Metric(nn.modules.loss._Loss):
    def __init__(self,
                 metrics: Dict[str, nn.modules.loss._Loss] = None,
                 reduction: str = 'mean',
                 **kwargs):
        """Metric module.

        Arguments:
            metrics: Dictionary of metric functions. Mapping a
                metric name to a metric function.
            reduction: Reduction mode for the per batch metric values.
                One of either none, sum or mean.
        """
        # Initialize base class
        super().__init__(**kwargs)

        # Check input arguments
        if reduction not in {'none', 'mean', 'sum'}:
            raise ValueError(
                    f"Invalid Value for arg 'reduction': '{self.reduction}"
                    f"\n Supported reduction modes: 'none', 'mean', 'sum'"
                )

        # Initialize instance attributes
        self.metrics = metrics if metrics is not None else {}
        self.reduction = reduction
        self.reset()

        # Get reduction function
        if self.reduction != 'none':
            self.reduction_fn = getattr(torch, self.reduction)

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    categories: Dict[str, int] = None,
                    label_mode: str = 'strict_3d') -> Metric:  # noqa: F821
        metrics = None
        reduction = config.get('reduction', 'mean')
        class_names = None

        if categories:
            class_names = [
                name for name, idx in sorted(
                    ((name, idx) for name, idx in categories.items() if isinstance(idx, int) and idx >= 0),
                    key=lambda item: item[1]
                )
            ]

        if label_mode in {'weak', 'weak_2d', 'weak_2d_bev', 'weak_center_lh'} and config.get('use_weak_metric', True):
            metrics = {
                k: _get_metric(v, class_names=class_names)
                for k, v in config.get('weak_metrics', {
                    'weak': {
                        'name': 'WeakBEVMetric',
                        'distance_threshold': config.get('distance_threshold', 1.0)
                    }
                }).items()
            }
        elif 'metrics' in config:
            metrics = {k: _get_metric(v, class_names=class_names) for k, v in config['metrics'].items()}

        return cls(
            metrics=metrics,
            reduction=reduction
        )

    @staticmethod
    def _detach_dict(data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach()
            for key, value in data.items()
        }

    def reset(self) -> None:
        self._inputs: List[Dict[str, torch.Tensor]] = []
        self._targets: List[Dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def update(self,
               inputs: Dict[str, torch.Tensor],
               targets: List[Dict[str, torch.Tensor]]) -> None:
        inputs: List[Dict[str, torch.Tensor]] = decollate_batch(inputs, detach=True, pad=False)
        for input, target in zip(inputs, targets):
            self._inputs.append(self._detach_dict(input))
            self._targets.append(self._detach_dict(target))

    @torch.no_grad()
    def compute(self) -> Dict[str, torch.Tensor]:
        if not self.metrics:
            return torch.ones(1)

        metric_accum = {}
        batch_metrics = {}

        for name, metric in self.metrics.items():
            if hasattr(metric, 'compute_dataset'):
                result = metric.compute_dataset(self._inputs, self._targets)
                if isinstance(result, dict):
                    for key, value in result.items():
                        batch_metrics[f"{name}_{key}" if key != 'mAP' else name] = value
                else:
                    batch_metrics[name] = result
                continue

            for input, target in zip(self._inputs, self._targets):
                result = metric(
                    {k: v.unsqueeze(0) for k, v in input.items()},
                    {k: v.unsqueeze(0) for k, v in target.items()}
                )
                values = result if isinstance(result, dict) else {name: result}
                for key, value in values.items():
                    metric_key = f"{name}_{key}" if isinstance(result, dict) and key != name else key
                    if isinstance(result, dict) and key == 'mAP':
                        metric_key = name
                    if metric_key not in metric_accum:
                        metric_accum[metric_key] = [value, 1]
                    else:
                        metric_accum[metric_key][0] = metric_accum[metric_key][0] + value
                        metric_accum[metric_key][1] += 1

        for key, (cumsum, count) in metric_accum.items():
            if self.reduction == 'mean':
                batch_metrics[key] = cumsum / count
            elif self.reduction == 'sum':
                batch_metrics[key] = cumsum
            else:
                batch_metrics[key] = cumsum

        return batch_metrics

    @torch.no_grad()
    def forward(self,
                inputs: Dict[str, torch.Tensor],
                targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Returns the loss given a prediction and ground truth.

        Arguments:
            inputs: Dictionary of model predictions with shape (B, N, C).
            targets: List of dictionaries with ground truth values
                with shape (B, M, C).

        Returns:
            metrics: Dictionary of metric values.
        """
        # Initialize metrics accumulators: metric_name -> (cumsum, count)
        metric_accum = {}

        # Decollate inputs
        inputs: List[Dict[str, torch.Tensor]] = decollate_batch(inputs, detach=False, pad=False)

        # Get loss for each item in the batch
        for input, target in zip(inputs, targets):
            # Insert dummy batch dimension
            input = {k: v.unsqueeze(0) for k, v in input.items()}
            target = {k: v.unsqueeze(0) for k, v in target.items()}

            # Get metric values
            metrics = {}
            for name, metric in self.metrics.items():
                result = metric(input, target)

                # Flatten dict metrics so they can be logged directly
                if isinstance(result, dict):
                    for key, value in result.items():
                        metrics[f"{name}_{key}" if key != 'mAP' else name] = value
                else:
                    metrics[name] = result

            # Accumulate metrics only for keys that appear in this batch
            for key, value in metrics.items():
                if key not in metric_accum:
                    metric_accum[key] = [value, 1]
                else:
                    metric_accum[key][0] = metric_accum[key][0] + value
                    metric_accum[key][1] += 1

        # Catch no metric configuration
        if not self.metrics:
            return torch.ones(1)

        # Calculate average for each metric based on count
        batch_metrics = {}
        for key, (cumsum, count) in metric_accum.items():
            if self.reduction == 'mean':
                batch_metrics[key] = cumsum / count
            elif self.reduction == 'sum':
                batch_metrics[key] = cumsum
            else:  # 'none'
                batch_metrics[key] = cumsum

        return batch_metrics


def _get_metric(config: Any, class_names: List[str] = None) -> nn.modules.loss._Loss:
    """Returns a pytorch or custom loss function given its name.

    Attributes:
        name: Name of the loss function (class).

    Returns:
        Instance of a loss function.
    """
    if isinstance(config, str):
        name = config
        params = {}
    else:
        name = config.get('name')
        params = {k: v for k, v in config.items() if k != 'name'}

    if name in {'mAP3D', 'mAP2D', 'WeakBEVMetric'} and class_names is not None:
        params.setdefault('class_names', class_names)

    try:
        return getattr(nn, name)(**params)
    except AttributeError:
        return globals()[name](**params)
    except Exception as e:
        raise e


def build_metric(*args, **kwargs):
    return Metric.from_config(*args, **kwargs)
