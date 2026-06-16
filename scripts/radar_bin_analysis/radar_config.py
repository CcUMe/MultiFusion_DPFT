"""Load configurable hyper-parameters for low-altitude radar processing."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "/home/yangqilin/code/dpft_v4/config/low_altitude_radar_config.jsonc"

DEFAULT_CONFIG: dict[str, Any] = {
    "pitch_selection": {
        "strategy": "dominant_count",
        "fixed_pitch_deg": 0.0,
        "quantization_step_deg": 0.5,
        "selection_tolerance_deg": 0.11,
        "prefer_nearest_zero_on_tie": True,
    },
    "ra_output": {
        "mapping_csv_name": "image_to_antframe_time_aligned.csv",
        "packet_csv_name": "match_radar_camera_anchor.csv",
        "ra_dir_name": "mmwave_ra_npy",
    },
    "visualization": {
        "preview_dir_name": "low_altitude_preview",
        "radar_subdir_name": "radar_fullframe",
        "compare_subdir_name": "compare",
        "side_by_side_gap_px": 24,
        "side_by_side_margin_px": 16,
        "side_by_side_label_height_px": 72,
        "layer_selection": "selected_pitch",
        "fixed_layer_index": 0,
        "fixed_pitch_deg": 0.0,
        "layer_fallback": "middle",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _strip_jsonc_comments(text: str) -> str:
    return re.sub(r"//.*", "", text)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = (config_path or DEFAULT_CONFIG_PATH).resolve()
    with path.open('r', encoding='utf-8') as f:
        raw = f.read()
    user_cfg = json.loads(_strip_jsonc_comments(raw))
    return _deep_merge(DEFAULT_CONFIG, user_cfg)
