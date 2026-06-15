# from __future__ import annotations  # noqa: F407
#
# from collections import OrderedDict
# from typing import Any, Callable, Dict, List, Tuple
#
# import torch
# from torch import nn
#
# from dprt.models.backbones import build_backbone
# from dprt.models.necks import build_neck
# from dprt.models.embeddings import build_embedding
# from dprt.models.queries import build_querent
# from dprt.models.fusers import build_fuser
# from dprt.models.heads import build_head
#
#
# def _build_module(build_fn: Callable, module_name: str,
#                   config: Dict[str, Any], computing: Dict[str, Any],
#                   *args, **kwargs) -> nn.Module:
#     """Retruns an module instance given its configuration.
#
#     Arguments:
#         build_fn: Build function for the particular module.
#         module_name: Name of the module.
#         config: Configuration of the module instance.
#
#     Returns:
#         Module instance or Identity module if module_name is not in config.
#     """
#     # Get module configuration
#     module: Dict[str, Any] = config.get(module_name)
#
#     if module is not None:
#         return build_fn(
#             module['name'], dict(computing | module), *args, **kwargs
#         )
#
#     return None
#
#
# def _build_modules(build_fn: Callable, module_name: str,
#                    config: Dict[str, Any], computing: Dict[str, Any],
#                    *args, **kwargs) -> Dict[str, nn.Module]:
#     """Retruns a dict of module instances given their configuration.
#
#     Arguments:
#         build_fn: Build function for the modules.
#         module_name: Name of the module (parent module).
#         config: Configuration of the module instances.
#
#     Returns:
#         Dict of module instances or None if module_name is not in config.
#     """
#     # Get modules configuration
#     modules: Dict[str, Any] = config.get(module_name)
#
#     # Build modules
#     if modules is not None:
#         return {
#             k: _build_module(build_fn, k, modules, computing, *args, **kwargs)
#             for k in modules.keys()
#         }
#
#     return None
#
#
# class DPRT(nn.Module):
#     def __init__(self,
#                  inputs: List[str],
#                  skiplinks: Dict[str, bool] = None,
#                  backbones: Dict[str, nn.Module] = None,
#                  necks: Dict[str, nn.Module] = None,
#                  embeddings: Dict[str, nn.Module] = None,
#                  querent: nn.Module = None,
#                  fuser: nn.Module = None,
#                  head: nn.Module = None,
#                  **kwargs):
#         """Dual Perspective Radar Transformer
#
#         Arguments:
#             inputs: List of input data names representing multiple
#                 perspectives (modalities).
#             skiplinks: Dict of boolean values specifying whether to pass
#                 the raw data of the corresponding input to the fusion module.
#             backbones: Dict of backbone modules used for the feature extraction
#                 of the corresponding input data.
#             necks: Dict of neck modules used for the feature alignment of
#                 the extracked features and input data (if skiplink).
#             embeddings: Dict of embedding modules used for positional encoding
#                 of the feature maps of the corresponding inputs.
#             querent: Query reference point generation module.
#             fuser: Fusion module used to fuse the different views (modalities).
#             head: Head model used to generate the model prediction.
#         """
#         # Initialize base class
#         super().__init__()
#
#         # Initialize instance attributes
#         self.inputs = inputs
#         self.skiplinks = skiplinks if skiplinks is not None else {}
#         self.backbones = backbones if backbones is not None else {}
#         self.necks = necks if necks is not None else {}
#         self.embeddings = embeddings if embeddings is not None else {}
#
#         # Initialize unspecified submodules
#         self.skiplinks = {input: self.skiplinks.get(input, False) for input in inputs}
#         self.backbones = self._init_unspecified(self.backbones)
#         self.necks = self._init_unspecified(self.necks)
#         self.embeddings = self._init_unspecified(self.embeddings)
#         self.querent = self._module_or_identity(querent)
#         self.fuser = self._module_or_identity(fuser)
#         self.head = self._module_or_identity(head)
#         self.language_model = language_model
#
#         self.lidar_bn = nn.ModuleDict()
#
#     @classmethod
#     def from_config(cls, config: Dict[str, Any]) -> DPRT:  # noqa: F821
#         # Get general subconfigs
#         computing: Dict[str, Any] = config['computing']
#         model: Dict[str, Any] = config['model']
#
#         # Build fuser-head combination
#         head = _build_module(build_head, 'head', model, computing)
#         fuser = _build_module(build_fuser, 'fuser', model, computing, head=head)
#
#         return cls(
#             inputs=model.get('inputs'),
#             skiplinks=model.get('skiplinks'),
#             backbones=_build_modules(build_backbone, 'backbones', model, computing),
#             necks=_build_modules(build_neck, 'necks', model, computing),
#             embeddings=_build_modules(build_embedding, 'embeddings', model, computing),
#             querent=_build_module(build_querent, 'querent', model, computing),
#             fuser=fuser,
#             head=head
#         )
#
#     def _init_unspecified(self, submodule: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
#         """Returns a dict of module instances
#
#         Arguments:
#             submodule: Dictionary of submodules.
#
#         Returns:
#             Dictionary of modules with one module for each input.
#             If no module is provided the Identity module is used.
#         """
#         return nn.ModuleDict(
#             {input: self._module_or_identity(submodule.get(input)) for input in self.inputs}
#         )
#
#     @staticmethod
#     def _module_or_identity(module: nn.Module = None) -> nn.Module:
#         """Returns the given module or the Identity module.
#
#         Arguments:
#             module: Module to return.
#
#         Returns:
#             module: The given module or the Identity module,
#                 if the given module is None.
#         """
#         if module is not None:
#             return module
#         return nn.Identity()
#
#     @staticmethod
#     def _add_raw_data(features: OrderedDict[str, torch.Tensor],
#                       raw_data: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
#         """Returns the feature dict extend by the raw data.
#
#         Arguments:
#             features: Dictionary of feature tensors.
#             raw_data: Raw data tensor to insert into the feature dict.
#
#         Returns:
#             features: Dictionary of feature tensors with the raw data
#                 inserted in the front.
#         """
#         features['0'] = raw_data
#         features.move_to_end('0', last=False)
#         return features
#
#     @staticmethod
#     def _get_projetions(inputs: List[str],
#                         batch: Dict[str, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
#         """Returns a tuple of projection matrices for each input.
#
#         Arguments:
#             inputs: List of input names.
#             batch: Batch of input data containing the
#                 transformation and projetion matirces.
#
#         Returns:
#             A list of tuples containing a transformation
#             and projection matirx for every input.
#         """
#         return [
#             (batch[f'label_to_{input}_t'], batch[f'label_to_{input}_p'])
#             for input in inputs
#         ]
#
#     def forward(self, batch: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
#         """Returns the DPRT prediction based on the given input.
#
#         Arguments:
#             batch: Dictionary of batched model input tensors. For every input the
#                 dict must at least contain:
#                 "input_name": Input data with shape (B, H, W, C).
#                 "label_to_input_name_t": A transformation matrix with shape (B, 4, 4).
#                 "label_to_input_name_p": A projection matrix with shape (B, 4, 4).
#                 "input_name_shape": Original shape of the input data with shape (H, W, C).
#
#         Returns:
#             out: Dictionary of batched model predictions. The content of the dict is
#                 defined by the head module.
#         """
#         # Get input data shapes in channel last format (B, H, W, C)
#         shapes = {input: batch[f"{input}_shape"] for input in self.inputs}
#
#         # Feature extraction
#         features = {input: self.backbones[input](batch[input]) for input in self.inputs}
#
#         # Add input features (skip link)
#         features = {
#             input: self._add_raw_data(features[input], batch[input])
#             for input in self.inputs if self.skiplinks[input]
#         }
#
#         # Feature alignment
#         features = {input: self.necks[input](features[input]) for input in self.inputs}
#
#         # 单独给 LiDAR BEV 做 per-level BN
#         if 'lidar_bev' in features:
#             for lvl, feat in features['lidar_bev'].items():
#                 C = feat.shape[-1]  # 当前级别通道数
#                 # # 如果还没给这一 level 创建 BN，就创建并注册
#                 # if lvl not in self.lidar_bn:
#                 # 检查是否需要创建或重建 BN 层,如果不加后面那个or的条件，就会报错，非常奇怪
#                 if lvl not in self.lidar_bn or self.lidar_bn[lvl].num_features != C:
#                     self.lidar_bn[lvl] = nn.BatchNorm2d(C).to(feat.device)
#                 # 应用对应级别的 BN
#                 # 1) 先把 (B, H, W, C) → (B, C, H, W)
#                 x = feat.movedim(-1, 1)
#                 # 2) 做 BatchNorm2d
#                 x = self.lidar_bn[lvl](x)
#                 # 3) 再把 (B, C, H, W) → (B, H, W, C)
#                 features['lidar_bev'][lvl] = x.movedim(1, -1)
#                 # features['lidar_bev'][lvl] = self.lidar_bn[lvl](feat)
#
#         # Positional embedding
#         features = {input: self.embeddings[input](features[input]) for input in self.inputs}
#
#         # Get (global) reference points to query
#         out = self.querent(batch)
#
#         # Sensor fusion
#         out = self.fuser(
#             batch=[features[input] for input in self.inputs],
#             shape=[shapes[input][:, :2] for input in self.inputs],
#             projection=self._get_projetions(self.inputs, batch),
#             out=out
#         )
#
#         return out
#
#
# def build_dprt(*args, **kwargs):
#     return DPRT.from_config(*args, **kwargs)


from __future__ import annotations  # noqa: F407

import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from dprt.models.backbones import build_backbone
from dprt.models.necks import build_neck
from dprt.models.embeddings import build_embedding
from dprt.models.queries import build_querent
from dprt.models.fusers import build_fuser
from dprt.models.heads import build_head
from dprt.models.language_models import build_language_model
from dprt.utils.config import get_active_inputs


def _build_module(build_fn: Callable, module_name: str,
                  config: Dict[str, Any], computing: Dict[str, Any],
                  *args, **kwargs) -> nn.Module:
    """Returns a module instance given its configuration."""
    module: Dict[str, Any] = config.get(module_name)
    if module is not None:
        return build_fn(
            module['name'], dict(computing | module), *args, **kwargs
        )
    return None


def _build_modules(build_fn: Callable, module_name: str,
                   config: Dict[str, Any], computing: Dict[str, Any],
                   *args, **kwargs) -> Dict[str, nn.Module]:
    """Returns a dict of module instances given their configuration."""
    modules: Dict[str, Any] = config.get(module_name)
    if modules is not None:
        return {
            k: _build_module(build_fn, k, modules, computing, *args, **kwargs)
            for k in modules.keys()
        }
    return None


class DPRT(nn.Module):
    def __init__(self,
                 inputs: List[str],
                 available_inputs: List[str] = None,
                 fuser_inputs: List[str] = None,
                 skiplinks: Dict[str, bool] = None,
                 backbones: Dict[str, nn.Module] = None,
                 necks: Dict[str, nn.Module] = None,
                 embeddings: Dict[str, nn.Module] = None,
                 querent: nn.Module = None,
                 fuser: nn.Module = None,
                 head: nn.Module = None,
                 language_model: nn.Module = None,
                 **kwargs):
        """Dual Perspective Radar Transformer"""
        super().__init__()

        self.inputs = inputs
        self.available_inputs = available_inputs if available_inputs is not None else inputs
        self.fuser_inputs = fuser_inputs if fuser_inputs is not None else inputs
        self.skiplinks = skiplinks if skiplinks is not None else {}
        self.backbones = backbones if backbones is not None else {}
        self.necks = necks if necks is not None else {}
        self.embeddings = embeddings if embeddings is not None else {}

        self.skiplinks = {input: self.skiplinks.get(input, False) for input in self.available_inputs}
        self.backbones = self._init_unspecified(self.backbones, self.available_inputs)
        self.necks = self._init_unspecified(self.necks, self.available_inputs)
        self.embeddings = self._init_unspecified(self.embeddings, self.available_inputs)
        self.querent = self._module_or_identity(querent)
        self.fuser = self._module_or_identity(fuser)
        self.head = self._module_or_identity(head)
        self.language_model = language_model

        self.lidar_bn = nn.ModuleDict()

        # ── Timing state ──────────────────────────────────────────────
        self.last_timing: Dict[str, float] = {}
        self.timing_history: List[Dict[str, float]] = []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> DPRT:  # noqa: F821
        computing: Dict[str, Any] = config['computing']
        model: Dict[str, Any] = config['model']
        inputs = get_active_inputs(model)
        configured_fuser_inputs = model.get('fuser', {}).get('inputs')
        if configured_fuser_inputs is None:
            fuser_inputs = inputs
        else:
            fuser_inputs = [input_name for input_name in configured_fuser_inputs if input_name in inputs]
            if not fuser_inputs:
                raise ValueError(
                    'No active fuser inputs remain after applying model.input_enable. '
                    f'Configured fuser inputs: {configured_fuser_inputs}, active inputs: {inputs}.'
                )
        head = _build_module(build_head, 'head', model, computing)
        fuser = _build_module(
            build_fuser, 'fuser', model, computing,
            head=head, inputs=fuser_inputs
        )
        language_config = model.get('language_model')
        language_model = None
        if language_config and language_config.get('enabled', False):
            language_model = build_language_model(
                language_config.get('name', 'qwen_14b'), language_config
            )
        return cls(
            inputs=inputs,
            available_inputs=model.get('inputs'),
            fuser_inputs=fuser_inputs,
            skiplinks=model.get('skiplinks'),
            backbones=_build_modules(build_backbone, 'backbones', model, computing),
            necks=_build_modules(build_neck, 'necks', model, computing),
            embeddings=_build_modules(build_embedding, 'embeddings', model, computing),
            querent=_build_module(build_querent, 'querent', model, computing),
            fuser=fuser,
            head=head,
            language_model=language_model
        )

    def _init_unspecified(self,
                          submodule: Dict[str, nn.Module],
                          input_names: List[str] = None) -> Dict[str, nn.Module]:
        input_names = input_names if input_names is not None else self.inputs
        return nn.ModuleDict(
            {input: self._module_or_identity(submodule.get(input)) for input in input_names}
        )

    @staticmethod
    def _module_or_identity(module: nn.Module = None) -> nn.Module:
        return module if module is not None else nn.Identity()

    @staticmethod
    def _add_raw_data(features: OrderedDict,
                      raw_data: torch.Tensor) -> OrderedDict:
        features['0'] = raw_data
        features.move_to_end('0', last=False)
        return features

    @staticmethod
    def _get_projetions(inputs: List[str],
                        batch: Dict[str, torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        projected_inputs = set(inputs) - {"camera_mono"}
        if "rgb_original_size" in batch and set(inputs).issubset({"camera_mono", "ir_image", "micro_light"}):
            projections = []
            for input_name in inputs:
                if input_name == "camera_mono":
                    projections.append({
                        "input_name": input_name,
                        "homography": None,
                        "rgb_original_size": batch["rgb_original_size"],
                        "original_size": batch["rgb_original_size"],
                    })
                    continue

                homography_key = f"homography_rgb_to_{input_name}"
                original_size_key = f"{input_name}_original_size"
                if homography_key not in batch or original_size_key not in batch:
                    break
                projections.append({
                    "input_name": input_name,
                    "homography": batch[homography_key],
                    "rgb_original_size": batch["rgb_original_size"],
                    "original_size": batch[original_size_key],
                })
            else:
                return projections
        return [
            (batch[f'label_to_{input}_t'], batch[f'label_to_{input}_p'])
            for input in inputs
        ]

    # ── Timing helpers ────────────────────────────────────────────────

    @staticmethod
    def _sync_time() -> float:
        """Wall-clock time (seconds) after syncing CUDA.

        torch.cuda.synchronize() ensures all launched CUDA kernels have
        finished before we record the timestamp, giving accurate per-module
        GPU latency. Without it, timings would reflect CPU dispatch time
        only and be meaningless for GPU-bound modules.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _print_timing(timing: Dict[str, float]) -> None:
        """Prints a formatted per-module timing table for a single forward pass."""
        if not timing:
            return
        total = sum(timing.values())
        print(f"\n{'─' * 50}")
        print(f"  {'Module':<16} {'Time (ms)':>10} {'%':>6}  Bar")
        print(f"{'─' * 50}")
        for name, ms in timing.items():
            bar = '█' * int(ms / total * 20)
            print(f"  {name:<16} {ms:>10.2f} {ms / total * 100:>5.1f}%  {bar}")
        print(f"{'─' * 50}")
        print(f"  {'Total':<16} {total:>10.2f} {'100.0%':>6}")
        print(f"{'─' * 50}\n")

    def print_timing_stats(self, warmup: int = 3) -> None:
        """Prints mean / std / min / max latency statistics across all recorded frames.

        Args:
            warmup: Number of initial frames to discard before computing
                    statistics. The first few forward passes are slower due
                    to CUDA kernel compilation and GPU warm-up, and would
                    skew the mean upward if included.
        """
        history = self.timing_history
        if not history:
            print("No timing data recorded. Run forward() with profile=True first.")
            return

        # Discard warm-up frames
        history = history[warmup:]
        if not history:
            print(f"Not enough frames (need > {warmup}). Reduce warmup or record more frames.")
            return

        modules = list(history[0].keys())
        col = 18

        print(f"\n{'═' * 66}")
        print(f"  Timing Statistics  (frames recorded: {len(self.timing_history)},  "
              f"warmup discarded: {warmup})")
        print(f"{'═' * 66}")
        print(f"  {'Module':<{col}} {'Mean(ms)':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"{'─' * 66}")

        total_mean = 0.0
        for name in modules:
            vals = np.array([h[name] for h in history if name in h])
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            std = float(vals.std())
            mn = float(vals.min())
            mx = float(vals.max())
            total_mean += mean

            print(f"  {name:<{col}} {mean:>10.2f} {std:>8.2f} {mn:>8.2f} {mx:>8.2f}")

        print(f"{'─' * 66}")
        print(f"  {'Total (mean)':<{col}} {total_mean:>10.2f}")
        print(f"{'═' * 66}\n")

    def reset_timing(self) -> None:
        """Clears all recorded timing history."""
        self.last_timing = {}
        self.timing_history = []

    # ── Forward pass ──────────────────────────────────────────────────

    def forward(self,
                batch: Dict[str, torch.Tensor],
                pred_confidence_camera: torch.Tensor = None,
                pred_confidence_radar_bev: torch.Tensor = None,
                pred_confidence_radar_front: torch.Tensor = None,
                pred_confidence_lidar: torch.Tensor = None,
                profile: bool = False) -> OrderedDict:
        """Returns the DPRT prediction based on the given input.

        Args:
            batch: Dictionary of batched model input tensors. For every input
                   the dict must at least contain:
                     "input_name"              – Input data (B, H, W, C).
                     "label_to_input_name_t"   – Transformation matrix (B, 4, 4).
                     "label_to_input_name_p"   – Projection matrix (B, 4, 4).
                     "input_name_shape"         – Original data shape (H, W, C).
            profile: If True, record per-module GPU latency for this forward
                     pass and print a single-pass timing table.
                     Set to False (default) during training to avoid the
                     cuda.synchronize() overhead.

        Returns:
            out: Ordered dict of model predictions (class, center, size, angle).

        Timing access after evaluation loop:
            model.last_timing        – Dict[str, float] for the latest call (ms).
            model.timing_history     – List[Dict] of all recorded calls.
            model.print_timing_stats() – Aggregated mean/std/min/max table.
        """
        timing: Dict[str, float] = {}
        shapes = {input: batch[f"{input}_shape"] for input in self.inputs}

        # ── 1. Backbone ───────────────────────────────────────────────
        t0 = self._sync_time() if profile else None
        features = {input: self.backbones[input](batch[input]) for input in self.inputs}
        if profile:
            timing['backbone'] = (self._sync_time() - t0) * 1000

        # 置信度加权
        if pred_confidence_camera is not None:
            pred_confidence_camera = 1 + 0.5 * (pred_confidence_camera - 0.5)

        if pred_confidence_radar_bev is not None:
            pred_confidence_radar_bev = 1 + 0.5 * (pred_confidence_radar_bev - 0.5)

        if pred_confidence_radar_front is not None:
            pred_confidence_radar_front = 1 + 0.5 * (pred_confidence_radar_front - 0.5)

        if pred_confidence_lidar is not None:
            pred_confidence_lidar = 1 + 0.5 * (pred_confidence_lidar - 0.5)

        weighted_features = {}
        for input_name in self.inputs:
            if input_name == 'camera_mono' and pred_confidence_camera is not None:
                confidence = pred_confidence_camera  # [B, 1]
            elif input_name == 'radar_bev' and pred_confidence_radar_bev is not None:
                confidence = pred_confidence_radar_bev  # [B, 1]
            elif input_name == 'radar_front' and pred_confidence_radar_front is not None:
                confidence = pred_confidence_radar_front  # [B, 1]
            elif input_name == 'lidar_bev' and pred_confidence_lidar is not None:
                confidence = pred_confidence_lidar  # [B, 1]
            else:
                confidence = torch.ones(batch[input_name].shape[0], 1,
                                        device=batch[input_name].device)  # [B, 1]
            input_features = features[input_name]  # OrderedDict: {scale_name: feature_map}

            weighted_scales = OrderedDict()
            for scale_name, feature_map in input_features.items():
                confidence_expanded = confidence.view(confidence.shape[0], 1, 1, 1)
                weighted_scales[scale_name] = feature_map * confidence_expanded

            weighted_features[input_name] = weighted_scales

        features = weighted_features

        # ── 2. Skip-link ──────────────────────────────────────────────
        t0 = self._sync_time() if profile else None
        features = {
            input: self._add_raw_data(features[input], batch[input]) if self.skiplinks[input] else features[input]
            for input in self.inputs
        }
        if profile:
            timing['skiplink'] = (self._sync_time() - t0) * 1000

        # ── 3. Neck ───────────────────────────────────────────────────
        t0 = self._sync_time() if profile else None
        features = {input: self.necks[input](features[input]) for input in self.inputs}
        if profile:
            timing['neck'] = (self._sync_time() - t0) * 1000

        # ── 4. LiDAR BEV BatchNorm ────────────────────────────────────
        t0 = self._sync_time() if profile else None
        if 'lidar_bev' in features:
            for lvl, feat in features['lidar_bev'].items():
                C = feat.shape[-1]
                if lvl not in self.lidar_bn or self.lidar_bn[lvl].num_features != C:
                    self.lidar_bn[lvl] = nn.BatchNorm2d(C).to(feat.device)
                x = feat.movedim(-1, 1)
                x = self.lidar_bn[lvl](x)
                features['lidar_bev'][lvl] = x.movedim(1, -1)
        if profile:
            timing['lidar_bn'] = (self._sync_time() - t0) * 1000

        # ── 5. Positional embedding ───────────────────────────────────
        t0 = self._sync_time() if profile else None
        features = {input: self.embeddings[input](features[input]) for input in self.inputs}
        if profile:
            timing['embedding'] = (self._sync_time() - t0) * 1000

        # ── 6. Querent ────────────────────────────────────────────────
        t0 = self._sync_time() if profile else None
        out = self.querent(batch)
        if profile:
            timing['querent'] = (self._sync_time() - t0) * 1000

        # ── 7. Fuser ──────────────────────────────────────────────────
        t0 = self._sync_time() if profile else None
        out = self.fuser(
            batch=[features[input] for input in self.fuser_inputs],
            shape=[shapes[input][:, :2] for input in self.fuser_inputs],
            projection=self._get_projetions(self.fuser_inputs, batch),
            out=out
        )
        if profile:
            raw_fuser_ms = (self._sync_time() - t0) * 1000
            timing['fuser'] = max(raw_fuser_ms, 0.0)

        # ── 8. Language model ─────────────────────────────────────────
        t0 = self._sync_time() if profile else None
        if self.language_model is not None:
            lm_features = {
                'fused_queries': getattr(self.fuser, 'last_query', None),
                'multi_scale_features': features,
            }
            out = self.language_model(out, batch=batch, features=lm_features)
        if profile:
            timing['language_model'] = (self._sync_time() - t0) * 1000

        # ── Store & display ───────────────────────────────────────────
        if profile:
            self.last_timing = timing
            if not hasattr(self, 'timing_history'):
                self.timing_history = []
            self.timing_history.append(timing)
            # self._print_timing(timing)

        return out


def build_dprt(*args, **kwargs):
    return DPRT.from_config(*args, **kwargs)
