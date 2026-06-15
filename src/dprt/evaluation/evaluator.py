from __future__ import annotations  # noqa: F407

from typing import Any, Callable, Dict, Iterable, List

import os.path as osp

import torch

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dprt.models import load_model as load_model
from dprt.models.confidence import confidence_model_args, load_confidence_modules
from dprt.evaluation.exporters import build as build_exporter
from dprt.evaluation.metric import build_metric


class ConfidenceWrappedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, confidence: torch.nn.Module = None):
        super().__init__()
        self.model = model
        self.confidence = confidence

    def forward(self, data: Dict[str, torch.Tensor]):
        return self.model(data, *confidence_model_args(self.confidence, data))


class CentralizedEvaluator():
    def __init__(self,
                 metric: torch.nn.modules.loss._Loss = None,
                 exporter: Callable = None,
                 device: str = None,
                 logging: str = None,
                 config: Dict[str, Any] = None,
                 measure_inference_time: bool = False,
                 measure_complexity: bool = False):
        """
        Arguments:
            logging: Logging frequency. One of either None,
                step or epoch.
        """
        # Initialize instance arrtibutes
        self.config = config
        self.eval_fn = metric
        self.export_fn = exporter
        self.device = device
        self.logging = logging
        self.measure_inference_time = measure_inference_time
        self.measure_complexity = measure_complexity

    @classmethod
    def from_config(cls, config: Dict[str, Any], *args, **kwargs):
        label_mode = config.get('data', {}).get('label_mode', 'strict_3d')
        metric = build_metric(
            config['evaluate'],
            categories=config.get('data', {}).get('categories'),
            label_mode=label_mode
        )
        evaluate_config = config['evaluate']
        exporter_config = evaluate_config.get('exporter')
        if label_mode in {'weak', 'weak_2d', 'weak_2d_bev', 'weak_center_lh'} and evaluate_config.get('disable_exporter_in_weak_mode', True):
            exporter = None
        elif exporter_config is None:
            exporter = None
        else:
            exporter = build_exporter(exporter_config['name'], config)
        device = torch.device(config['computing']['device'])
        logging = config['evaluate'].get('logging', config['train'].get('logging'))
        measure_inference_time = config['evaluate'].get('measure_inference_time', False)
        measure_complexity = config['evaluate'].get('measure_complexity', False)

        return cls(
            metric=metric,
            exporter=exporter,
            device=device,
            logging=logging,
            config=config,
            measure_inference_time=measure_inference_time,
            measure_complexity=measure_complexity
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.evaluate(*args, **kwargs)

    @staticmethod
    def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in data.items()}

    def _filter_tensorboard_scalars(self,
                                    scalars: Dict[str, Any],
                                    split: str) -> Dict[str, Any]:
        del split
        exclude_keys = {'weak_GT', 'mAP_GT'}
        exclude_prefixes = ('weak_GT_', 'mAP_GT_')
        return {
            key: value
            for key, value in scalars.items()
            if key not in exclude_keys and not key.startswith(exclude_prefixes)
        }

    def log_scalars(self, writer, scalars: Dict[str, Any], epoch: int, prefix: str = None) -> None:
        if writer is None:
            return

        scalars = self._filter_tensorboard_scalars(scalars, prefix or '')
        if not scalars:
            return

        # Get prefix
        prefix = f"{prefix}/" if prefix is not None else ""

        # Add scalar values
        for name, scalar in scalars.items():
            writer.add_scalar(prefix + name, scalar, epoch)

    @staticmethod
    def _scalar(value: Any) -> float:
        if hasattr(value, 'detach'):
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _accumulate(accumulator: Dict[str, List[Any]], values: Dict[str, Any]) -> None:
        for key, value in values.items():
            if key not in accumulator:
                accumulator[key] = [value, 1]
            else:
                accumulator[key][0] = accumulator[key][0] + value
                accumulator[key][1] += 1

    @staticmethod
    def _average(accumulator: Dict[str, List[Any]]) -> Dict[str, Any]:
        return {
            key: total / count
            for key, (total, count) in accumulator.items()
        }

    def _expected_map_keys(self) -> List[str]:
        return [
            f"mAP_{name}"
            for name in self._eval_class_names()
        ]

    def _eval_class_names(self) -> List[str]:
        categories = (self.config or {}).get('data', {}).get('categories', {})
        return [
            name
            for name, idx in sorted(
                ((name, idx) for name, idx in categories.items() if isinstance(idx, int) and idx >= 0),
                key=lambda item: item[1]
            )
        ]

    def _format_detection_table(self, metrics: Dict[str, Any]) -> str:
        if 'mAP' not in metrics:
            return ""

        lines = [
            "Detection metrics (dataset-level mAP):",
            f"  IoU={self._scalar(metrics.get('mAP_IoU_threshold', 0.0)):.2f}, "
            f"AP points={self._scalar(metrics.get('mAP_AP_points', 0.0)):.0f}, "
            f"classes with GT={self._scalar(metrics.get('mAP_num_eval_classes', 0.0)):.0f}",
            "  Class                 AP      GT    Pred      TP      FP      FN    Prec     Rec",
        ]

        for class_name in self._eval_class_names():
            ap_key = f"mAP_{class_name}"
            pred_key = f"mAP_Pred_{class_name}"
            if ap_key not in metrics and pred_key not in metrics:
                continue

            ap = f"{self._scalar(metrics[ap_key]):.4f}" if ap_key in metrics else "N/A"
            gt = int(self._scalar(metrics.get(f"mAP_GT_{class_name}", 0.0)))
            pred = int(self._scalar(metrics.get(pred_key, 0.0)))
            tp = int(self._scalar(metrics.get(f"mAP_TP_{class_name}", 0.0)))
            fp = int(self._scalar(metrics.get(f"mAP_FP_{class_name}", metrics.get(f"mAP_FP_only_{class_name}", 0.0))))
            fn = int(self._scalar(metrics.get(f"mAP_FN_{class_name}", 0.0)))
            precision = (
                f"{self._scalar(metrics[f'mAP_Precision_{class_name}']):.4f}"
                if f"mAP_Precision_{class_name}" in metrics else "N/A"
            )
            recall = (
                f"{self._scalar(metrics[f'mAP_Recall_{class_name}']):.4f}"
                if f"mAP_Recall_{class_name}" in metrics else "N/A"
            )

            lines.append(
                f"  {class_name:<18} {ap:>7} {gt:>7} {pred:>7} "
                f"{tp:>7} {fp:>7} {fn:>7} {precision:>7} {recall:>7}"
            )

        return "\n".join(lines)

    def _format_weak_detection_table(self, metrics: Dict[str, Any]) -> str:
        if 'weak_Recall' not in metrics and 'weak_Precision' not in metrics:
            return ""

        lines = [
            "Weak detection metrics (center-distance matching):",
            f"  distance={self._scalar(metrics.get('weak_Distance_threshold', 0.0)):.2f}m, "
            f"score={self._scalar(metrics.get('weak_Score_threshold', 0.0)):.2f}, "
            f"mAP={self._scalar(metrics.get('weak_center_mAP', 0.0)):.4f}, "
            f"classes with GT={self._scalar(metrics.get('weak_num_eval_classes', 0.0)):.0f}",
            "  Class                 AP      GT    Pred      TP      FP      FN    Prec     Rec",
        ]

        for class_name in self._eval_class_names():
            ap_key = f"weak_AP_{class_name}"
            gt_key = f"weak_GT_{class_name}"
            pred_key = f"weak_Pred_{class_name}"
            tp_key = f"weak_TP_{class_name}"
            fp_key = f"weak_FP_{class_name}"
            fn_key = f"weak_FN_{class_name}"
            prec_key = f"weak_Precision_{class_name}"
            rec_key = f"weak_Recall_{class_name}"

            if gt_key not in metrics and pred_key not in metrics:
                continue

            ap = f"{self._scalar(metrics[ap_key]):.4f}" if ap_key in metrics else "N/A"
            gt = int(self._scalar(metrics.get(gt_key, 0.0)))
            pred = int(self._scalar(metrics.get(pred_key, 0.0)))
            tp = int(self._scalar(metrics.get(tp_key, 0.0)))
            fp = int(self._scalar(metrics.get(fp_key, max(pred - tp, 0))))
            fn = int(self._scalar(metrics.get(fn_key, max(gt - tp, 0))))
            precision = f"{self._scalar(metrics[prec_key]):.4f}" if prec_key in metrics else "N/A"
            recall = f"{self._scalar(metrics[rec_key]):.4f}" if rec_key in metrics else "N/A"

            lines.append(
                f"  {class_name:<18} {ap:>7} {gt:>7} {pred:>7} "
                f"{tp:>7} {fp:>7} {fn:>7} {precision:>7} {recall:>7}"
            )

        return "\n".join(lines)

    def _print_metrics(self, title: str, metrics: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

        if 'mAP' in metrics:
            print(f"{'mAP':<25s}: {self._scalar(metrics['mAP']):.4f}")

        expected_map_keys = self._expected_map_keys()
        diagnostic_prefixes = (
            'mAP_GT_', 'mAP_Pred_', 'mAP_TP_', 'mAP_FP_', 'mAP_FN_',
            'mAP_Precision_', 'mAP_Recall_', 'mAP_FP_only_'
        )
        weak_diagnostic_prefixes = (
            'weak_AP_', 'weak_GT_', 'weak_Pred_', 'weak_TP_', 'weak_FP_', 'weak_FN_',
            'weak_Precision_', 'weak_Recall_'
        )
        config_keys = {'mAP_IoU_threshold', 'mAP_AP_points', 'mAP_num_eval_classes'}
        extra_map_keys = sorted(
            k for k in metrics
            if k.startswith('mAP_')
            and k not in expected_map_keys
            and k not in config_keys
            and not k.startswith(diagnostic_prefixes)
        )
        for key in expected_map_keys + extra_map_keys:
            if key in metrics:
                print(f"{key:<25s}: {self._scalar(metrics[key]):.4f}")
            else:
                print(f"{key:<25s}: N/A")

        for key in sorted(
            k for k in metrics
            if k != 'mAP'
            and not k.startswith('mAP_')
            and not k.startswith(weak_diagnostic_prefixes)
        ):
            try:
                print(f"{key:<25s}: {self._scalar(metrics[key]):.4f}")
            except Exception:
                print(f"{key:<25s}: {metrics[key]}")

        table = self._format_detection_table(metrics)
        if table:
            print("\n" + table)
        weak_table = self._format_weak_detection_table(metrics)
        if weak_table:
            print("\n" + weak_table)

        print("=" * 70 + "\n")

    @torch.no_grad()
    def evaluate_complexity(self, epoch: int, model: torch.nn.Module,
                            data_loader: Iterable, writer=None,
                            confidence: torch.nn.Module = None):
        # Set model to evaluation mode
        model.eval()
        if confidence is not None:
            confidence.eval()

        # Get inference test input
        data, _ = next(iter(data_loader))

        # Load test data (to device)
        data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)

        # Determine model complexity
        profiled_model = ConfidenceWrappedModel(model, confidence).to(self.device)
        with get_accelerator().device(self.device):
            flops, macs, params = get_model_profile(
                model=profiled_model, args=(data,),
                print_profile=False, warm_up=10, as_string=False
            )

        # Log model complexity
        self.log_scalars(
            writer, {'FLOPS': flops, 'MACS': macs, 'Parameters': params},
            epoch, 'test'
        )

    @torch.no_grad()
    def evaluate_inference_time(self, epoch: int, model: torch.nn.Module,
                                data_loader: Iterable, writer=None,
                                confidence: torch.nn.Module = None):
        # Set model to evaluation mode
        model.eval()
        if confidence is not None:
            confidence.eval()

        # Get inference test input
        data, _ = next(iter(data_loader))

        # Load test data (to device)
        data: Dict[str, torch.Tensor] = self._dict_to(data, self.device)

        # Initialize loggers
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = torch.zeros((repetitions, 1))

        # GPU warm-up
        for _ in range(10):
            model(data, *confidence_model_args(confidence, data))

        # Measure performance
        for rep in range(repetitions):
            starter.record()
            model(data, *confidence_model_args(confidence, data))
            ender.record()
            # Wait for GPU sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

        # Calculate mean and std inference time in milliseconds
        mean_syn = torch.sum(timings) / repetitions
        std_syn = torch.std(timings)

        # Log inference time measures
        self.log_scalars(
            writer, {'Inference_time_mean_ms': mean_syn, 'Inference_time_std_ms': std_syn},
            epoch, 'test'
        )

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch: int, model: torch.nn.Module,
                           data_loader: Iterable, writer=None, dst: str = None,
                           confidence: torch.nn.Module = None):
        # Set model to evaluation mode
        model.eval()
        if confidence is not None:
            confidence.eval()


        # 鈹€鈹€ 鏂板锛氶噸缃鏃跺巻鍙诧紝閬垮厤娣峰叆 evaluate_inference_time 鐨勯鐑暟鎹?鈹€鈹€
        if hasattr(model, 'reset_timing'):
            model.reset_timing()
        if hasattr(self.eval_fn, 'reset'):
            self.eval_fn.reset()

        with tqdm(total=len(data_loader)) as pbar:
            for i, (data, labels) in enumerate(data_loader):
                # Load data and labels (to device)
                labels: List[Dict[str, torch.Tensor]] = \
                    [self._dict_to(label, self.device) for label in labels]
                data: Dict[str, torch.Tensor] = \
                    self._dict_to(data, self.device)

                # Make prediction
                output = model(data, *confidence_model_args(confidence, data))

                # Evaluate model output
                if hasattr(self.eval_fn, 'update'):
                    self.eval_fn.update(output, labels)
                    metrics = {}
                else:
                    raise RuntimeError("Evaluation requires a dataset-level metric with update()/compute().")

                # Log evaluation step
                if self.logging == 'step':
                    self.log_scalars(writer, metrics, i + epoch * len(data_loader), 'test')

                # Export predictions
                if self.export_fn is not None:
                    self.export_fn(output, labels, i * len(labels), dst)

                # Report training progress
                pbar.update()

        scalars = self.eval_fn.compute()

        if self.logging == 'epoch':
            self.log_scalars(writer, scalars, epoch, 'test')

        if hasattr(model, 'print_timing_stats'):
            model.print_timing_stats(warmup=3)

        self._print_metrics("Test metrics", scalars)

    def evaluate(self, checkpoint: str, data_loader: Iterable, dst: str = None):
        model, epoch, timestamp, _ = load_model(checkpoint, self.config)
        model.to(self.device)
        model.eval()
        confidence = load_confidence_modules(self.config, self.device)

        # Check if destination is provided
        if self.logging is not None:
            dst = osp.join(dst, timestamp)

        # Initialize tensorboard writer (logging)
        writer = None
        if self.logging is not None:
            writer = SummaryWriter(log_dir=dst)

        # Evaluate model performance
        self.evaluate_one_epoch(epoch, model, data_loader, writer, dst, confidence)

        if self.measure_inference_time:
            self.evaluate_inference_time(epoch, model, data_loader, writer, confidence)

        if self.measure_complexity:
            self.evaluate_complexity(epoch, model, data_loader, writer, confidence)

        # Flush and close writer
        if self.logging is not None:
            writer.flush()
            writer.close()


def build_evaluator(*args, **kwargs):
    return CentralizedEvaluator.from_config(*args, **kwargs)
