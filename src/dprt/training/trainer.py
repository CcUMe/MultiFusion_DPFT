from __future__ import annotations  # noqa: F407

import datetime
import os
import os.path as osp

from typing import Any, Dict, Iterable, List

import torch

from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter

from dprt.evaluation.metric import build_metric
from dprt.models.confidence import confidence_model_args, load_confidence_modules
from dprt.training.optimizer import build_optimizer
from dprt.training.loss import build_loss
from dprt.training.scheduler import build_scheduler


class CentralizedTrainer():
    def __init__(self,
                 epochs: int = 1,
                 optimizer: torch.optim.Optimizer = None,
                 loss: torch.nn.modules.loss._Loss = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                 metric: torch.nn.modules.loss._Loss = None,
                 device: str = None,
                 logging: str = None,
                 evaluating: int = 1,
                 config: Dict[str, Any] = None):
        """
        Arguments:
            logging: Logging frequency. One of either None,
                step or epoch.
            evaluating: Evaluation frequency. A value of
                -1 means no evaluation, 0 means an evaluation
                after every step and evey value > 0 descibes the
                number of epoch after which an evaluation is executed.
        """
        # Initialize instance arrtibutes
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss
        self.scheduler = scheduler
        self.eval_fn = metric
        self.device = device
        self.logging = logging
        self.evaluating = evaluating
        self.config = config or {}
        self._printed_val_label_counts = False

    @classmethod
    def from_config(cls,
                    config: Dict[str, Any],
                    *args,
                    **kwargs) -> CentralizedTrainer:  # noqa: F821
        # Get trainer atributes
        epochs = config['train']['epochs']
        optimizer = build_optimizer(
            config['train']['optimizer'].pop('name'),
            **config['train']['optimizer']
        )
        loss = build_loss(
            config
        )


        scheduler = build_scheduler(
            config['train']['scheduler'].pop('name'),
            **config['train']['scheduler']
        )
        metric = build_metric(
            config['evaluate'],
            categories = config.get('data', {}).get('categories'),
            label_mode = config.get('data', {}).get('label_mode', 'strict_3d')
        )
        device = torch.device(config['computing']['device'])
        logging = config['train'].get('logging')

        return cls(
            epochs=epochs,
            optimizer=optimizer,
            loss=loss,
            scheduler=scheduler,
            metric=metric,
            device=device,
            logging=logging,
            config=config
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.train(*args, **kwargs)

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

    def log_scalars(self,
                    writer,
                    scalars: Dict[str, Any],
                    epoch: int,
                    prefix: str = None) -> None:
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
    def _dict_to(data: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in data.items()}

    @staticmethod
    def _parameter_counts(module: torch.nn.Module | None) -> Dict[str, int]:
        if module is None:
            return {'total': 0, 'trainable': 0, 'frozen': 0}
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }

    @classmethod
    def _print_parameter_counts(cls,
                                model: torch.nn.Module,
                                confidence: torch.nn.Module | None = None) -> None:
        model_counts = cls._parameter_counts(model)
        confidence_counts = cls._parameter_counts(confidence)
        combined_counts = {
            key: model_counts[key] + confidence_counts[key]
            for key in model_counts
        }

        def fmt(count: int) -> str:
            return f"{count:>12,}  ({count / 1e6:.3f} M)"

        print("=" * 64)
        print(f"{'Module':<16} {'Total':>22} {'Trainable':>22} {'Frozen':>22}")
        print("-" * 64)

        print(
            f"{'DPRT':<16} {fmt(model_counts['total']):>22} "
            f"{fmt(model_counts['trainable']):>22} {fmt(model_counts['frozen']):>22}"
        )
        print(
            f"{'Confidence':<16} {fmt(confidence_counts['total']):>22} "
            f"{fmt(confidence_counts['trainable']):>22} {fmt(confidence_counts['frozen']):>22}"
        )

        print("-" * 64)
        print(
            f"{'Combined':<16} {fmt(combined_counts['total']):>22} "
            f"{fmt(combined_counts['trainable']):>22} {fmt(combined_counts['frozen']):>22}"
        )

    @staticmethod
    def _scalar(value: Any) -> float:
        if hasattr(value, 'detach'):
            return float(value.detach().item())
        return float(value)

    def _expected_map_keys(self) -> List[str]:
        return [
            f"mAP_{name}"
            for name in self._eval_class_names()
        ]

    def _eval_class_names(self) -> List[str]:
        categories = self.config.get('data', {}).get('categories', {})
        min_class_idx = 1 if self.config.get('task') == '2d_detection' else 0
        return [
            name
            for name, idx in sorted(
                ((name, idx) for name, idx in categories.items()
                 if isinstance(idx, int) and idx >= min_class_idx),
                key=lambda item: item[1]
            )
        ]

    def _format_epoch_metrics(self, prefix: str, epoch: int, metrics: Dict[str, Any]) -> str:
        expected_map_keys = self._expected_map_keys()
        diagnostic_prefixes = (
            'mAP_GT_', 'mAP_Pred_', 'mAP_TP_', 'mAP_FP_', 'mAP_FN_',
            'mAP_Precision_', 'mAP_Recall_', 'mAP_FP_only_'
        )
        config_keys = {'mAP_IoU_threshold', 'mAP_AP_points', 'mAP_num_eval_classes'}
        map_keys = ['mAP'] + expected_map_keys + sorted(
            k for k in metrics if k.startswith('mAP_') and k not in expected_map_keys
            and k not in config_keys
            and not k.startswith(diagnostic_prefixes)
        )
        config_order = ['mAP_IoU_threshold', 'mAP_AP_points', 'mAP_num_eval_classes']
        weak_keys = [
            'weak_center_mAP',
            'weak_Recall',
            'weak_Precision',
            'weak_GT',
            'weak_Pred',
            'weak_TP',
            'weak_FP',
            'weak_FN',
            'weak_Center_MAE',
            'weak_Distance_threshold',
            'weak_Score_threshold',
            'weak_AP_score_threshold',
            'weak_AP_points',
            'weak_num_eval_classes',
        ]
        ordered_keys = ['loss'] + [k for k in map_keys if k in metrics] + [
            k for k in config_order if k in metrics
        ] + [k for k in weak_keys if k in metrics]
        values = [
            f"{key}={self._scalar(metrics[key]):.4f}"
            for key in ordered_keys
            if key in metrics
        ]
        if any(key in metrics for key in map_keys):
            values.extend(
                f"{key}=N/A"
                for key in expected_map_keys
                if key not in metrics
        )
        return f"{prefix} Epoch {epoch + 1}: " + ", ".join(values)

    def _format_detection_table(self, metrics: Dict[str, Any]) -> str:
        if 'mAP' not in metrics:
            return ""

        lines = [
            "Validation detection metrics (dataset-level mAP):",
            f"  IoU={self._scalar(metrics.get('mAP_IoU_threshold', 0.0)):.2f}, "
            f"AP points={self._scalar(metrics.get('mAP_AP_points', 0.0)):.0f}, "
            f"classes with GT={self._scalar(metrics.get('mAP_num_eval_classes', 0.0)):.0f}",
            "  Class                 AP      GT    Pred      TP      FP      FN    Prec     Rec",
        ]

        for class_name in self._eval_class_names():
            ap_key = f"mAP_{class_name}"
            gt_key = f"mAP_GT_{class_name}"
            pred_key = f"mAP_Pred_{class_name}"
            tp_key = f"mAP_TP_{class_name}"
            fp_key = f"mAP_FP_{class_name}"
            fn_key = f"mAP_FN_{class_name}"
            prec_key = f"mAP_Precision_{class_name}"
            rec_key = f"mAP_Recall_{class_name}"

            if ap_key not in metrics and pred_key not in metrics:
                continue

            ap = f"{self._scalar(metrics[ap_key]):.4f}" if ap_key in metrics else "N/A"
            gt = int(self._scalar(metrics.get(gt_key, 0.0)))
            pred = int(self._scalar(metrics.get(pred_key, 0.0)))
            tp = int(self._scalar(metrics.get(tp_key, 0.0)))
            fp = int(self._scalar(metrics.get(fp_key, metrics.get(f'mAP_FP_only_{class_name}', 0.0))))
            fn = int(self._scalar(metrics.get(fn_key, 0.0)))
            precision = f"{self._scalar(metrics[prec_key]):.4f}" if prec_key in metrics else "N/A"
            recall = f"{self._scalar(metrics[rec_key]):.4f}" if rec_key in metrics else "N/A"

            lines.append(
                f"  {class_name:<18} {ap:>7} {gt:>7} {pred:>7} "
                f"{tp:>7} {fp:>7} {fn:>7} {precision:>7} {recall:>7}"
            )

        return "\n".join(lines)

    def _format_weak_detection_table(self, metrics: Dict[str, Any]) -> str:
        if 'weak_Recall' not in metrics and 'weak_Precision' not in metrics:
            return ""

        lines = [
            "Validation weak detection metrics (center-distance matching):",
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

    def train_one_epoch(self, epoch: int, model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    writer: SummaryWriter = None,
                    confidence: torch.nn.Module = None
                    ) -> Dict[str, torch.Tensor]:
        # Make sure gradient tracking is on
        model.train()
        if confidence is not None:
            confidence.eval()

        # Initialize epoch logs
        loss_scalars = {}

        pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc=f"Train Epoch {epoch + 1}/{self.epochs}",
            leave=False
        )
        for i, (data, labels) in enumerate(pbar):
            # Log learning rate
            if self.logging == 'step':
                self.log_scalars(
                    writer,
                    {'learning_rate': optimizer.param_groups[0]['lr']},
                    i + epoch * len(data_loader),
                    'train'
                )

            # Load data and labels (to device)
            labels: List[Dict[str, torch.Tensor]] = \
                [self._dict_to(label, self.device) for label in labels]
            data: Dict[str, torch.Tensor] = \
                self._dict_to(data, self.device)

            # Zero gradients
            optimizer.zero_grad()

            with torch.no_grad():
                model_confidence_args = confidence_model_args(confidence, data)

            # Make prediction
            output = model(data, *model_confidence_args)

            # Compute the loss and its gradients
            loss, losses = self.loss_fn(output, labels)

            # Adjust weights
            if loss > 0:
                loss.backward()
                optimizer.step()

            pbar.set_postfix({
                'loss': float(loss.detach().item()) if hasattr(loss, 'detach') else float(loss)
            })

            # Add prefix to loss values (logging)
            losses = {f'loss_{k}': v for k, v in losses.items()}
            losses['loss'] = loss

            # Log training step
            if self.logging == 'step':
                self.log_scalars(writer, losses, i + epoch * len(data_loader), 'train')

            # Add values to epoch log
            for k, v in losses.items():
                loss_scalars[k] = loss_scalars.get(k, 0) + v

        epoch_metrics = {
            k: v / max(i + 1, 1)
            for k, v in loss_scalars.items()
        }

        if self.logging == 'epoch':
            # Write epoch logs
            self.log_scalars(writer, epoch_metrics, epoch, 'train')
            self.log_scalars(writer, {'learning_rate': optimizer.param_groups[0]['lr']}, epoch, 'train')

        return epoch_metrics

    @torch.no_grad()
    def validate_one_epoch(self, epoch: int, model: torch.nn.Module, data_loader: Iterable,
                        writer: SummaryWriter = None,
                        confidence: torch.nn.Module = None
                        ) -> Dict[str, float]:
        # Make sure the model is in evaluation mode
        model.eval()
        if confidence is not None:
            confidence.eval()
        self.loss_fn.eval()
        if hasattr(self.eval_fn, 'reset'):
            self.eval_fn.reset()

        # Initialize epoch logs
        epoch_loss = 0.0

        pbar = tqdm(
            data_loader,
            total=len(data_loader),
            desc=f"Val Epoch {epoch + 1}/{self.epochs}",
            leave=False
        )
        for i, (data, labels) in enumerate(pbar):
            # Load data and labels (to device)
            labels: List[Dict[str, torch.Tensor]] = \
                [self._dict_to(label, self.device) for label in labels]
            data: Dict[str, torch.Tensor] = \
                self._dict_to(data, self.device)

            model_confidence_args = confidence_model_args(confidence, data)

            # Make prediction
            output = model(data, *model_confidence_args)

            # Compute the loss and its gradients
            loss, losses = self.loss_fn(output, labels)
            epoch_loss += self._scalar(loss)

            # Evaluate model output
            if hasattr(self.eval_fn, 'update'):
                self.eval_fn.update(output, labels)
            else:
                raise RuntimeError("Validation requires a dataset-level metric with update()/compute().")

            # Add prefix to loss values (logging)
            losses = {f'loss_{k}': v for k, v in losses.items()}
            losses['loss'] = loss

            pbar.set_postfix({
                'loss': float(loss.detach().item()) if hasattr(loss, 'detach') else float(loss)
            })

            # Log training step
            if self.logging == 'step':
                self.log_scalars(writer, losses, i + epoch * len(data_loader), 'val')

        epoch_metrics = self.eval_fn.compute()
        epoch_metrics['loss'] = torch.tensor(
            epoch_loss / max(len(data_loader), 1),
            dtype=torch.float,
            device=self.device
        )
        if self.logging == 'epoch':
            self.log_scalars(
                writer,
                epoch_metrics,
                epoch,
                'val'
            )

        return epoch_metrics

    def train(self, model: torch.nn.Module, data_loader: Iterable, val_loader: Iterable = None,
              start_epoch: int = 0, timestamp: str = None, dst: str = None) -> None:
        # Load model and loss function (to device)
        model.to(self.device)
        self.loss_fn.to(self.device)

        # Get current timestamp
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]

        # Create checkpoint directory
        os.makedirs(osp.join(dst, timestamp, 'checkpoints'), exist_ok=True)

        # Check if destination is provided
        if self.logging is not None:
            assert dst is not None

        # Initialize tensorboard writer (logging)
        writer = None
        if self.logging is not None:
            writer = SummaryWriter(log_dir=osp.join(dst, timestamp))

        confidence = load_confidence_modules(self.config, self.device)
        self._print_parameter_counts(model, confidence)

        # Parameterize optimizer
        optimizer = self.optimizer(model.parameters())

        # Pass optimizer to learning rate scheduler
        scheduler = self.scheduler(optimizer)

        # Initialize progressbar iterator
        tbar = trange(start_epoch, self.epochs, initial=start_epoch, total=self.epochs)

        for epoch in tbar:
            # Execute model training
            train_result = self.train_one_epoch(epoch, model, data_loader, optimizer, writer, confidence)
            tqdm.write(self._format_epoch_metrics('Train', epoch, train_result))

            if val_loader is not None:
                result = self.validate_one_epoch(epoch, model, val_loader, writer, confidence)
                tqdm.write(self._format_epoch_metrics('Val', epoch, result))
                detection_table = self._format_detection_table(result)
                if detection_table:
                    tqdm.write(detection_table)
                weak_detection_table = self._format_weak_detection_table(result)
                if weak_detection_table:
                    tqdm.write(weak_detection_table)
            else:
                result = train_result

            # Update learning rate
            scheduler.step()

            # Update progressbar
            postfix = {}
            if 'loss' in result:
                postfix['loss'] = self._scalar(result['loss'])
            if 'mAP' in result:
                postfix['mAP'] = self._scalar(result['mAP'])
            if 'weak_Recall' in result:
                postfix['weak_Recall'] = self._scalar(result['weak_Recall'])
            tbar.set_postfix(postfix, refresh=True)

            # Save checkpoint
            path = osp.join(dst, timestamp, 'checkpoints',
                            f"{timestamp}_checkpoint_{str(epoch).zfill(4)}.pt")
            # torch.save(model, path)
            torch.save(model.state_dict(), path)

        # Flush and close writer
        if self.logging is not None:
            writer.flush()
            writer.close()


def build_trainer(*args, **kwargs):
    return CentralizedTrainer.from_config(*args, **kwargs)
