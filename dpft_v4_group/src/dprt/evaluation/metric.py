from __future__ import annotations  # noqa: F407

from typing import Any, Dict, List, Optional

import torch

from torch import nn
from torch.utils.data import default_collate

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

        # Select contributing (present) classes only
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

        # Get reduction function
        if self.reduction != 'none':
            self.reduction_fn = getattr(torch, self.reduction)

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    categories: Dict[str, int] = None) -> Metric:  # noqa: F821
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

        if 'metrics' in config:
            metrics = {k: _get_metric(v, class_names=class_names) for k, v in config['metrics'].items()}

        return cls(
            metrics=metrics,
            reduction=reduction
        )

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

    if name == 'mAP3D' and class_names is not None:
        params.setdefault('class_names', class_names)

    try:
        return getattr(nn, name)(**params)
    except AttributeError:
        return globals()[name](**params)
    except Exception as e:
        raise e


def build_metric(*args, **kwargs):
    return Metric.from_config(*args, **kwargs)
