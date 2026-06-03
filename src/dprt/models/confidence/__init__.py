from __future__ import annotations

from collections import OrderedDict
import sys
from typing import Any, Dict, Mapping

import torch
from torch import nn

from dprt.models.confidence import confidence_camera as _confidence_camera_module
from dprt.models.confidence import confidence_lidar as _confidence_lidar_module
from dprt.models.confidence import confidence_radar as _confidence_radar_module
from dprt.models.confidence import feature_selector as _feature_selector_module
from dprt.models.confidence.confidence_camera import ModalityLevelDynamics as CameraConfidence
from dprt.models.confidence.confidence_lidar import ModalityLevelDynamics as LidarConfidence
from dprt.models.confidence.confidence_radar import ModalityLevelDynamics as RadarConfidence
from dprt.models.confidence.feature_selector import MultiScaleFeatureSelector

CONFIDENCE_MODELS = {
    "camera": CameraConfidence,
    "camera_mono": CameraConfidence,
    "radar": RadarConfidence,
    "radar_bev": RadarConfidence,
    "radar_front": RadarConfidence,
    "lidar": LidarConfidence,
    "lidar_bev": LidarConfidence,
}


CONFIDENCE_INPUT_ORDER = (
    "camera_mono",
    "radar_bev",
    "radar_front",
    "lidar_bev",
)


def build_confidence(name: str, **kwargs):
    if name not in CONFIDENCE_MODELS:
        raise KeyError(f"Unknown confidence model: {name}")
    return CONFIDENCE_MODELS[name](**kwargs)


class ConfidenceEnsemble(nn.Module):
    """Registered confidence modules used by DPRT as external modality weights."""

    def __init__(self,
                 models: Mapping[str, nn.Module],
                 selectors: Mapping[str, nn.Module] | None = None):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.selectors = nn.ModuleDict(selectors or {})

    def freeze(self) -> "ConfidenceEnsemble":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def predict(self, data: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor | None]:
        confidences = OrderedDict()
        for input_name in CONFIDENCE_INPUT_ORDER:
            if input_name not in self.models:
                confidences[input_name] = None
                continue
            selector = self.selectors[input_name] if input_name in self.selectors else None
            confidences[input_name] = self.models[input_name](data[input_name], selector)
        return confidences

    def model_args(self, data: Dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, ...]:
        confidences = self.predict(data)
        return tuple(confidences[input_name] for input_name in CONFIDENCE_INPUT_ORDER)


def _confidence_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("confidence") or config.get("model", {}).get("confidence") or {}


def confidence_is_enabled(config: Dict[str, Any]) -> bool:
    return bool(_confidence_config(config).get("enabled", False))


def _missing_checkpoint(path: str | None) -> bool:
    return path in (None, "", "...", "...pt")


def _register_legacy_confidence_imports() -> None:
    """Allow older full-module checkpoints to unpickle after the package move."""
    legacy_modules = {
        "dprt.confidence_camera": _confidence_camera_module,
        "dprt.confidence_lidar": _confidence_lidar_module,
        "dprt.confidence_radar": _confidence_radar_module,
        "dprt.feature_selector": _feature_selector_module,
    }
    for old_name, module in legacy_modules.items():
        sys.modules.setdefault(old_name, module)


def _load_torch_module(path: str, device: torch.device | str) -> nn.Module | Dict[str, torch.Tensor]:
    _register_legacy_confidence_imports()
    return torch.load(path, map_location=device)


def _as_state_dict(loaded: Any) -> Dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    if isinstance(loaded, dict) and "state_dict" in loaded:
        return loaded["state_dict"]
    return loaded


def _load_confidence_model(input_name: str,
                           spec: Dict[str, Any],
                           device: torch.device | str) -> nn.Module:
    checkpoint = spec.get("checkpoint") or spec.get("model")
    if _missing_checkpoint(checkpoint):
        raise ValueError(f"Missing confidence checkpoint for '{input_name}'.")

    loaded = _load_torch_module(checkpoint, device)
    if isinstance(loaded, nn.Module):
        return loaded

    loaded = _as_state_dict(loaded)
    model_type = spec.get("type", input_name)
    model = build_confidence(model_type, **spec.get("kwargs", {}))
    model.load_state_dict(loaded, strict=spec.get("strict", True))
    return model


def _load_selector(input_name: str,
                   spec: Dict[str, Any],
                   device: torch.device | str) -> nn.Module | None:
    selector_spec = spec.get("selector")
    if selector_spec in (None, False):
        return None

    if isinstance(selector_spec, str):
        checkpoint = selector_spec
        selector_kwargs = {}
        strict = True
    else:
        checkpoint = selector_spec.get("checkpoint")
        selector_kwargs = selector_spec.get("kwargs", {})
        strict = selector_spec.get("strict", True)

    if _missing_checkpoint(checkpoint):
        raise ValueError(f"Missing confidence selector checkpoint for '{input_name}'.")

    loaded = _load_torch_module(checkpoint, device)
    if isinstance(loaded, nn.Module):
        return loaded

    loaded = _as_state_dict(loaded)
    if "feature_channels" not in selector_kwargs:
        raise ValueError(
            f"Selector checkpoint for '{input_name}' is a state_dict. "
            "Add selector.kwargs.feature_channels to the config."
        )

    selector = MultiScaleFeatureSelector(**selector_kwargs)
    selector.load_state_dict(loaded, strict=strict)
    return selector


def load_confidence_modules(config: Dict[str, Any],
                            device: torch.device | str) -> ConfidenceEnsemble | None:
    conf_config = _confidence_config(config)
    if not conf_config.get("enabled", False):
        return None

    models = OrderedDict()
    selectors = OrderedDict()
    for input_name in CONFIDENCE_INPUT_ORDER:
        spec = conf_config.get(input_name)
        if spec is None:
            continue
        models[input_name] = _load_confidence_model(input_name, spec, device)
        selector = _load_selector(input_name, spec, device)
        if selector is not None:
            selectors[input_name] = selector

    if not models:
        raise ValueError("confidence.enabled is true, but no confidence modules are configured.")

    return ConfidenceEnsemble(models=models, selectors=selectors).to(device).freeze()


def confidence_model_args(confidence: ConfidenceEnsemble | None,
                          data: Dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, ...]:
    if confidence is None:
        return ()
    return confidence.model_args(data)
