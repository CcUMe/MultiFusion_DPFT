"""Helpers for parsing low-altitude mmWave radar scans from raw bin packets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import match_radar_camera_anchor as anchor_match


PKT = anchor_match.PKT
PKT_WORDS = anchor_match.PKT_WORDS
N_RANGE_FULL = 668
N_RANGE_OUT = 666


@dataclass(frozen=True)
class ScanRange:
    scan_index: int
    pkt_start: int
    pkt_end: int
    complete: bool
    start_flag: int
    end_flag: int


def quantize_pitch_deg(values: np.ndarray, step_deg: float = 0.5) -> np.ndarray:
    """Quantize noisy pitch values to the configured nominal raster."""
    step = float(step_deg) if float(step_deg) > 0 else 0.5
    return np.round(values / step) * step


def build_scan_ranges(bin_path: Path) -> list[ScanRange]:
    """Derive complete/incomplete scan ranges using start/end flags."""
    n_pkts = bin_path.stat().st_size // PKT
    mm = np.memmap(str(bin_path), dtype="<u4", mode="r", shape=(n_pkts, PKT_WORDS))
    start = mm[:, 6].astype(np.uint32)
    end = mm[:, 7].astype(np.uint32)
    del mm

    scans: list[ScanRange] = []
    current_start = 0
    current_start_flag = 0
    scan_index = 0

    for i in range(n_pkts):
        s = int(start[i])
        e = int(end[i])

        if s != 0 and i > current_start:
            scans.append(
                ScanRange(
                    scan_index=scan_index,
                    pkt_start=current_start,
                    pkt_end=i - 1,
                    complete=False,
                    start_flag=current_start_flag,
                    end_flag=0,
                )
            )
            scan_index += 1
            current_start = i
            current_start_flag = s
        elif s != 0 and i == current_start:
            current_start_flag = s

        if e != 0:
            scans.append(
                ScanRange(
                    scan_index=scan_index,
                    pkt_start=current_start,
                    pkt_end=i,
                    complete=True,
                    start_flag=current_start_flag,
                    end_flag=e,
                )
            )
            scan_index += 1
            current_start = i + 1
            current_start_flag = 0

    if current_start < n_pkts:
        scans.append(
            ScanRange(
                scan_index=scan_index,
                pkt_start=current_start,
                pkt_end=n_pkts - 1,
                complete=False,
                start_flag=current_start_flag,
                end_flag=0,
            )
        )

    return scans


def build_packet_to_scan_index(scan_ranges: list[ScanRange], n_pkts: int) -> np.ndarray:
    scan_idx = np.full(n_pkts, -1, dtype=np.int32)
    for scan in scan_ranges:
        scan_idx[scan.pkt_start : scan.pkt_end + 1] = scan.scan_index
    return scan_idx


def read_packet_block(bin_path: Path, pkt_start: int, pkt_end: int) -> np.ndarray:
    n_pkts = pkt_end - pkt_start + 1
    with bin_path.open("rb") as f:
        f.seek(pkt_start * PKT)
        buf = f.read(n_pkts * PKT)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n_pkts, PKT)


def decode_scan_block(arr: np.ndarray) -> dict[str, np.ndarray]:
    ant_az = arr[:, 80:84].copy().view("<f4").ravel().astype(np.float64)
    ant_el = arr[:, 84:88].copy().view("<f4").ravel().astype(np.float64)
    range_km = arr[:, 32:36].copy().view("<f4").ravel().astype(np.float64)
    scan_dir = arr[:, 96:100].copy().view("<u4").ravel().astype(np.int64)
    body = arr[:, 256:5632]
    sum_lin = np.frombuffer(body[:, 8 : 8 + N_RANGE_FULL * 4].tobytes(), dtype="<f4").reshape(len(arr), N_RANGE_FULL)
    diff_lin = np.frombuffer(body[:, 2696 : 2696 + N_RANGE_FULL * 4].tobytes(), dtype="<f4").reshape(len(arr), N_RANGE_FULL)
    return {
        "ant_az": ant_az,
        "ant_el": ant_el,
        "range_km": range_km,
        "scan_dir": scan_dir,
        "sum_lin": sum_lin,
        "diff_lin": diff_lin,
    }


def power_to_db(x: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(x, 1e-3))).astype(np.float32)


def choose_pitch_layer(ant_el: np.ndarray, pitch_cfg: dict[str, Any] | None = None) -> tuple[float, np.ndarray]:
    """Pick one representative pitch layer instead of stacking all layers."""
    cfg = pitch_cfg or {}
    strategy = str(cfg.get("strategy", "dominant_count"))
    step_deg = float(cfg.get("quantization_step_deg", 0.5) or 0.5)
    tolerance_deg = float(cfg.get("selection_tolerance_deg", 0.11) or 0.11)
    fixed_pitch_deg = float(cfg.get("fixed_pitch_deg", 0.0) or 0.0)
    prefer_nearest_zero = bool(cfg.get("prefer_nearest_zero_on_tie", True))

    el_q = quantize_pitch_deg(ant_el, step_deg=step_deg)
    finite = np.isfinite(el_q)
    if not finite.any():
        return 0.0, np.zeros_like(el_q, dtype=bool)

    uniq, counts = np.unique(el_q[finite], return_counts=True)

    if strategy == "fixed_value":
        chosen = float(uniq[np.argmin(np.abs(uniq - fixed_pitch_deg))])
    elif strategy == "nearest_zero":
        chosen = float(uniq[np.argmin(np.abs(uniq))])
    else:
        if prefer_nearest_zero:
            order = sorted(range(len(uniq)), key=lambda i: (-counts[i], abs(float(uniq[i])), float(uniq[i])))
        else:
            order = sorted(range(len(uniq)), key=lambda i: (-counts[i], float(uniq[i])))
        chosen = float(uniq[order[0]])

    mask = finite & (np.abs(el_q - chosen) <= tolerance_deg)
    return chosen, mask


def list_pitch_layers(ant_el: np.ndarray, pitch_cfg: dict[str, Any] | None = None) -> np.ndarray:
    """List all quantized pitch layers present in one scan."""
    cfg = pitch_cfg or {}
    step_deg = float(cfg.get("quantization_step_deg", 0.5) or 0.5)
    el_q = quantize_pitch_deg(ant_el, step_deg=step_deg)
    finite = el_q[np.isfinite(el_q)]
    if finite.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.unique(finite).astype(np.float64)


def build_ra_slice(sum_db_666: np.ndarray, ant_az: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build one RA slice from packets belonging to a single pitch layer."""
    az_values = np.unique(np.round(ant_az[np.isfinite(ant_az)], 3))
    az_values.sort()
    if len(az_values) == 0:
        return np.zeros((sum_db_666.shape[1], 0), dtype=np.float32), az_values

    columns = np.zeros((sum_db_666.shape[1], len(az_values)), dtype=np.float32)
    hit = np.zeros(len(az_values), dtype=bool)

    amp_cols = np.power(10.0, sum_db_666 / 20.0)
    idx = np.searchsorted(az_values, np.round(ant_az, 3))
    idx = np.clip(idx, 0, len(az_values) - 1)

    for src_i, az_i in enumerate(idx):
        columns[:, az_i] = np.maximum(columns[:, az_i], amp_cols[src_i].astype(np.float32))
        hit[az_i] = True

    if not hit.all():
        valid_cols = np.where(hit)[0]
        for az_i in np.where(~hit)[0]:
            nearest = valid_cols[np.argmin(np.abs(valid_cols - az_i))]
            columns[:, az_i] = columns[:, nearest]

    return columns, az_values


def build_ra_volume(
    sum_db_666: np.ndarray,
    ant_az: np.ndarray,
    ant_el: np.ndarray,
    pitch_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 3D RAE tensor with shape (range, azimuth, elevation_layer)."""
    cfg = pitch_cfg or {}
    step_deg = float(cfg.get("quantization_step_deg", 0.5) or 0.5)
    tolerance_deg = float(cfg.get("selection_tolerance_deg", 0.11) or 0.11)

    pitch_values = list_pitch_layers(ant_el, cfg)
    if pitch_values.size == 0:
        return np.zeros((sum_db_666.shape[1], 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float64), pitch_values

    az_values = np.unique(np.round(ant_az[np.isfinite(ant_az)], 3))
    az_values.sort()
    if az_values.size == 0:
        return np.zeros((sum_db_666.shape[1], 0, pitch_values.size), dtype=np.float32), az_values, pitch_values

    ant_el_q = quantize_pitch_deg(ant_el, step_deg=step_deg)
    volume = np.zeros((sum_db_666.shape[1], az_values.size, pitch_values.size), dtype=np.float32)

    for pitch_idx, pitch_deg in enumerate(pitch_values):
        sel = np.isfinite(ant_el_q) & (np.abs(ant_el_q - pitch_deg) <= tolerance_deg)
        if not sel.any():
            continue
        ra_slice, layer_az_values = build_ra_slice(sum_db_666[sel], ant_az[sel])
        if layer_az_values.size == 0:
            continue
        layer_idx = np.searchsorted(az_values, np.round(layer_az_values, 3))
        layer_idx = np.clip(layer_idx, 0, az_values.size - 1)
        volume[:, layer_idx, pitch_idx] = ra_slice

        filled = np.zeros(az_values.size, dtype=bool)
        filled[layer_idx] = True
        if not filled.all():
            valid_cols = np.where(filled)[0]
            for az_i in np.where(~filled)[0]:
                nearest = valid_cols[np.argmin(np.abs(valid_cols - az_i))]
                volume[:, az_i, pitch_idx] = volume[:, nearest, pitch_idx]

    return volume, az_values, pitch_values


def scan_meta_from_selected(
    range_km: np.ndarray,
    az_values: np.ndarray,
    chosen_el: float,
    selected_pitch_idx: int,
    pitch_values: np.ndarray,
    scan_dir: np.ndarray,
) -> dict[str, float | int | str]:
    valid_range = range_km[np.isfinite(range_km)]
    range_max_m = float(np.median(valid_range) * 1000.0) if len(valid_range) else float(N_RANGE_OUT)
    valid_dir = [int(v) for v in scan_dir.tolist() if int(v) != 0]
    predominant_dir = int(np.median(valid_dir)) if valid_dir else 0
    return {
        "range_max_m": range_max_m,
        "az_min_deg": float(az_values[0]) if len(az_values) else 0.0,
        "az_max_deg": float(az_values[-1]) if len(az_values) else 0.0,
        "selected_pitch_deg": float(chosen_el),
        "selected_pitch_idx": int(selected_pitch_idx),
        "pitch_layer_count": int(pitch_values.size),
        "pitch_layers_deg": ",".join(f"{float(v):.2f}" for v in pitch_values.tolist()),
        "scan_dir": predominant_dir,
    }
