from __future__ import annotations  # noqa: F407

from typing import Any, Dict, List, Optional, Tuple

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
                 distance_ranges: Optional[List[Tuple[float, float]]] = None):
        """Mean average precision for 3D bounding boxes.

        Arguments:
            threshold: IoU threshold for bounding box matching.
            nelem: Number of elements to be used for the
                discretization of the precision recall curve.
            distance_ranges: List of distance ranges for range-wise mAP calculation.
                Example: [(0, 30), (30, 50), (50, 100)]
                If None, only compute overall mAP.
        """
        super().__init__()
        self.threshold = threshold
        self.nelem = nelem
        self.distance_ranges = distance_ranges

    def forward(self,
                inputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns the mean average precision value.

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
            If distance_ranges is None:
                Dictionary with single key 'mAP': Overall mean average precision value
            If distance_ranges is provided:
                Dictionary containing:
                    'mAP': Overall mAP
                    'mAP_0-30m': mAP for range 0-30m (example)
                    ... (for each distance range)
        """
        # Get device
        device = targets['gt_class'].device

        # Determine the number of classes
        num_classes = targets['gt_class'].shape[-1]

        label = torch.argmax(inputs['class'], dim=-1)
        gt_label = torch.argmax(targets['gt_class'], dim=-1)

        # Reconstruct angle from sin and cos part
        angle = torch.atan2(inputs['angle'][..., 0], inputs['angle'][..., 1])
        gt_angle = torch.atan2(targets['gt_angle'][..., 0], targets['gt_angle'][..., 1])

        # Calculate distances from origin (using xy-plane distance for radar)
        gt_distances = torch.norm(targets['gt_center'][..., :2], dim=-1)  # Shape: (B, M)

        # Get unique class labels (excluding background class 0)
        selection = torch.sort(torch.unique(torch.concatenate([label, gt_label], dim=1)))[0][1:]

        # Initialize results dictionary
        results = {}

        # Compute overall mAP (no distance filtering)
        aps_overall = self._compute_ap_for_mask(
            inputs, targets, label, gt_label, angle, gt_angle,
            None, device, num_classes
        )

        # Calculate overall mAP
        if not selection.numel() or not selection.any():
            results['mAP'] = torch.ones((), dtype=torch.float, device=device)
        else:
            results['mAP'] = torch.mean(aps_overall[selection])

        # Compute range-wise mAP if distance_ranges is provided
        if self.distance_ranges is not None:
            for dist_min, dist_max in self.distance_ranges:
                # Create distance mask for ground truth boxes
                dist_mask = (gt_distances >= dist_min) & (gt_distances < dist_max)  # Shape: (B, M)

                # Check if there are any GT boxes in this range
                if not dist_mask.any():
                    # No ground truth in this range, set mAP to 1.0 (no penalty for no GT)
                    results[f'mAP_{dist_min}-{dist_max}m'] = torch.ones((), dtype=torch.float, device=device)
                    continue

                # Compute AP for this distance range
                aps_range = self._compute_ap_for_mask(
                    inputs, targets, label, gt_label, angle, gt_angle,
                    dist_mask, device, num_classes
                )

                # Calculate mAP for this range
                if not selection.numel() or not selection.any():
                    range_mAP = torch.ones((), dtype=torch.float, device=device)
                else:
                    range_mAP = torch.mean(aps_range[selection])

                results[f'mAP_{dist_min}-{dist_max}m'] = range_mAP

        return results

    def _compute_ap_for_mask(self,
                             inputs: Dict[str, torch.Tensor],
                             targets: Dict[str, torch.Tensor],
                             label: torch.Tensor,
                             gt_label: torch.Tensor,
                             angle: torch.Tensor,
                             gt_angle: torch.Tensor,
                             distance_mask: Optional[torch.Tensor],
                             device: torch.device,
                             num_classes: int) -> torch.Tensor:
        """Compute AP for each class given an optional distance mask.

        Arguments:
            inputs: Model predictions dictionary.
            targets: Ground truth dictionary.
            label: Predicted class labels, shape (B, N).
            gt_label: Ground truth class labels, shape (B, M).
            angle: Predicted angles, shape (B, N).
            gt_angle: Ground truth angles, shape (B, M).
            distance_mask: Boolean mask of shape (B, M) for filtering GT boxes by distance.
                          If None, use all GT boxes.
            device: Torch device.
            num_classes: Number of classes.

        Returns:
            aps: Average precision values for each class, shape (num_classes,).
        """
        # Initialize average precision values
        aps = torch.zeros((num_classes,), dtype=torch.float, device=device)

        for l in range(num_classes):
            # Get class label mask with shape (B, N) and (B, M)
            mask = (label == l)
            gt_mask = (gt_label == l)

            # Apply distance mask if provided
            if distance_mask is not None:
                gt_mask = gt_mask & distance_mask

            # Get 3d bounding box corners with shape (B, N, 8, 3) and (B, M, 8, 3)
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

            # Get mask for all ious that are greater than the required threshold
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

        return aps


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

        # Reconstruct angle from sin and cos part
        angle = torch.atan2(inputs['angle'][..., 0], inputs['angle'][..., 1])
        gt_angle = torch.atan2(targets['gt_angle'][..., 0], targets['gt_angle'][..., 1])

        # Initialize giou values
        gious = -torch.ones((num_classes,), dtype=torch.float, device=device)

        for l in range(num_classes):
            # Get class label mask with shape (B, N) and (B, M)
            mask = (label == l)
            gt_mask = (gt_label == l)

            # Get 3d bounding box corners with shape (B, N, 8, 3) and (B, M, 8, 3)
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
                f"Invalid Value for arg 'reduction': '{reduction}'"
                f"\n Supported reduction modes: 'none', 'mean', 'sum'"
            )

        # Initialize instance attributes
        self.metrics = metrics if metrics is not None else {}
        self.reduction = reduction

        # Get reduction function
        if self.reduction != 'none':
            self.reduction_fn = getattr(torch, self.reduction)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Metric':  # noqa: F821
        metrics = None
        reduction = config.get('reduction', 'mean')

        if 'metrics' in config:
            metrics = {k: _get_metric(v) for k, v in config['metrics'].items()}

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
        # Initialize losses
        batch_metrics = []

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
                # Handle both scalar and dict returns from metric
                if isinstance(result, dict):
                    # Flatten dict metrics with prefix
                    for k, v in result.items():
                        metrics[f"{name}_{k}" if k != 'mAP' else name] = v
                else:
                    metrics[name] = result

            # Add metrics to the batch
            batch_metrics.append(metrics)

        # Catch no metric configuration
        if not self.metrics:
            return torch.ones(1)

        # Collate metrics (revert decollating)
        batch_metrics: Dict[str, torch.Tensor] = default_collate(batch_metrics)

        # Reduce batch metrics
        if self.reduction != 'none':
            batch_metrics = {k: self.reduction_fn(v) for k, v in batch_metrics.items()}

        return batch_metrics


def _get_metric(config: Dict[str, Any]) -> nn.modules.loss._Loss:
    """Returns a pytorch or custom loss function given its configuration.

    Attributes:
        config: Configuration dictionary with at least 'name' key.

    Returns:
        Instance of a metric function.
    """
    if isinstance(config, str):
        # Legacy support for string names
        name = config
        try:
            return getattr(nn, name)()
        except AttributeError:
            return globals()[name]()

    # Extract name and other parameters
    name = config.get('name')
    params = {k: v for k, v in config.items() if k != 'name'}

    try:
        return getattr(nn, name)(**params)
    except AttributeError:
        return globals()[name](**params)
    except Exception as e:
        raise e


def build_metric(*args, **kwargs):
    return Metric.from_config(*args, **kwargs)
