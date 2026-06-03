from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Dict, Optional

import torch
from torch import nn


class Qwen3VL8BAdapter(nn.Module):
    """Reserved Qwen3-VL-8B stage that keeps DPRT predictions unchanged.

    The adapter is intentionally non-generative by default. It gives the model
    graph a stable language-model extension point while preserving the detector
    output contract for training and evaluation.
    """

    def __init__(self,
                 model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
                 enabled: bool = False,
                 mode: str = "passthrough",
                 load_model: bool = False,
                 reasoning_enabled: bool = False,
                 attach_metadata: bool = False,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        super().__init__()

        if mode not in {"passthrough", "metadata"}:
            raise ValueError(f"Unsupported Qwen3-VL adapter mode: {mode}")

        self.model_id = model_id
        self.enabled = enabled
        self.mode = mode
        self.load_model = load_model
        self.reasoning_enabled = reasoning_enabled
        self.attach_metadata = attach_metadata
        self.cache_dir = cache_dir
        self.extra_config = dict(kwargs)
        self.last_context: Dict[str, Any] = {}

        self.processor = None
        self.language_model = None
        if self.enabled and self.load_model:
            self._load_backbone()

    def extra_repr(self) -> str:
        return (
            f"model_id={self.model_id!r}, enabled={self.enabled}, "
            f"mode={self.mode!r}, cache_dir={self.cache_dir!r}, "
            f"reasoning_enabled={self.reasoning_enabled}"
        )

    def _load_backbone(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Qwen3VL8BAdapter requires transformers when load_model=True. "
                "Install transformers or keep load_model=False for the reserved "
                "passthrough stage."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        self.language_model = AutoModel.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            torch_dtype=self.extra_config.get("torch_dtype", "auto")
        )
        self.language_model.eval()
        for param in self.language_model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _summarize_output(out: Any) -> Dict[str, Any]:
        if not isinstance(out, MutableMapping):
            return {"type": type(out).__name__}

        summary = {}
        for key, value in out.items():
            if torch.is_tensor(value):
                summary[key] = {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device)
                }
            else:
                summary[key] = type(value).__name__
        return summary

    def forward(self,
                out: MutableMapping[str, Any],
                batch: Optional[Dict[str, Any]] = None,
                features: Optional[Dict[str, Any]] = None) -> MutableMapping[str, Any]:
        self.last_context = {
            "model_id": self.model_id,
            "enabled": self.enabled,
            "mode": self.mode,
            "cache_dir": self.cache_dir,
            "reasoning_enabled": self.reasoning_enabled,
            "output_summary": self._summarize_output(out),
            "batch_keys": sorted(batch.keys()) if isinstance(batch, dict) else [],
            "feature_keys": sorted(features.keys()) if isinstance(features, dict) else []
        }

        if self.attach_metadata and isinstance(out, MutableMapping):
            out["_qwen3_vl"] = {
                "model_id": self.model_id,
                "mode": self.mode,
                "cache_dir": self.cache_dir,
                "reasoning_enabled": self.reasoning_enabled
            }

        return out


def build_qwen3_vl_adapter(config: Dict[str, Any]) -> Qwen3VL8BAdapter:
    return Qwen3VL8BAdapter(
        model_id=config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"),
        enabled=config.get("enabled", False),
        mode=config.get("mode", "passthrough"),
        load_model=config.get("load_model", False),
        reasoning_enabled=config.get("reasoning_enabled", False),
        attach_metadata=config.get("attach_metadata", False),
        cache_dir=config.get("cache_dir"),
        **{
            key: value for key, value in config.items()
            if key not in {
                "model_id",
                "enabled",
                "mode",
                "load_model",
                "reasoning_enabled",
                "attach_metadata",
                "cache_dir"
            }
        }
    )
