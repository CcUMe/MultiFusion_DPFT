#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
毫米波雷达 bin 数据流式解析脚本

功能：
1. 流式读取 bin，不需要一次性把整个 bin 读入内存
2. 自动扫描 0xABABABAB 0xABABABAB 帧头
3. 按协议解析单帧：
   - 数据帧头
   - 和路 668 点
   - 差路 668 点
   - 地形点
   - 高压线/孤立物/密集区
4. 按“天线一次完整扫描”边处理边保存
5. 每遇到 antenna_frame_end != 0，立即保存一个 scan_xxxxxx 文件夹

输出结构：
out_dir/
  scans/
    scans_summary.json
    scan_000000/
      scan_summary.json
      frames_summary.json
      read_points.csv
      terrain_points.csv
      isolated_objects.csv
      powerlines.csv
      dense_areas.csv
      sum_channel.npy
      diff_channel.npy
      raw_frames.bin    # 可选 --save-raw

用法：
  python parse_mmwave_stream_scans.py input.bin --out out_dir
  python parse_mmwave_stream_scans.py input.bin --out out_dir --dump-echo
  python parse_mmwave_stream_scans.py input.bin --out out_dir --keep-incomplete
  python parse_mmwave_stream_scans.py input.bin --out out_dir --save-raw
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# =========================
# 协议常量
# =========================

FRAME_WORDS = 2156
FRAME_BYTES = FRAME_WORDS * 4

HEADER_WORDS = 64
ECHO_WORDS = 1344
TERRAIN_WORDS = 36
TARGET_WORDS = 712

VALID_POINTS = 668

SYNC = {
    "frame_head": 0xABABABAB,
    "frame_tail": 0xBCBCBCBC,

    "sum_head": 0x11221122,
    "sum_tail": 0x22332233,

    "diff_head": 0x33443344,
    "diff_tail": 0x44554455,

    "terrain_head": 0xCDCDCDCD,
    "terrain_tail": 0xDEDEDEDE,

    "target_head": 0xACACACAC,
    "target_tail": 0xBDBDBDBD,
}


# =========================
# 基础工具
# =========================

class DecodeError(Exception):
    pass


@dataclass
class ParseConfig:
    endian: str = "<"
    strict: bool = True
    verbose: bool = False


class Reader:
    def __init__(self, data: bytes, endian: str = "<"):
        self.data = data
        self.endian = endian
        self.off = 0

    def tell(self) -> int:
        return self.off

    def _unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        if self.off + size > len(self.data):
            raise DecodeError(
                f"数据不足: offset={self.off}, need={size}, left={len(self.data) - self.off}"
            )
        v = struct.unpack_from(fmt, self.data, self.off)
        self.off += size
        return v[0] if len(v) == 1 else v

    def u32(self) -> int:
        return int(self._unpack(self.endian + "I"))

    def u64(self) -> int:
        return int(self._unpack(self.endian + "Q"))

    def f32(self) -> float:
        return float(self._unpack(self.endian + "f"))

    def u32s(self, n: int) -> List[int]:
        return [self.u32() for _ in range(n)]

    def f32s(self, n: int) -> List[float]:
        return [self.f32() for _ in range(n)]


def hex32(v: int) -> str:
    return f"0x{v & 0xFFFFFFFF:08X}"


def check_word(
    got: int,
    expected: int,
    name: str,
    warnings: List[str],
    strict: bool,
) -> None:
    if got != expected:
        msg = f"{name} 同步字错误: got={hex32(got)}, expected={hex32(expected)}"
        if strict:
            raise DecodeError(msg)
        warnings.append(msg)


def endian_name(endian: str) -> str:
    return "little" if endian == "<" else "big"


def pack_u32(v: int, endian: str) -> bytes:
    return struct.pack(endian + "I", v)


# =========================
# 字节序判断与流式帧扫描
# =========================

def score_frame_for_endian(frame: bytes, endian: str) -> int:
    if len(frame) < FRAME_BYTES:
        return -9999

    score = 0

    def u32_at(word_idx: int) -> int:
        return struct.unpack_from(endian + "I", frame, word_idx * 4)[0]

    expected_words = {
        0: SYNC["frame_head"],
        1: SYNC["frame_head"],

        64: SYNC["sum_head"],
        65: SYNC["sum_head"],
        734: SYNC["sum_tail"],
        735: SYNC["sum_tail"],

        736: SYNC["diff_head"],
        737: SYNC["diff_head"],
        1406: SYNC["diff_tail"],
        1407: SYNC["diff_tail"],

        1408: SYNC["terrain_head"],
        1409: SYNC["terrain_head"],
        1442: SYNC["terrain_tail"],
        1443: SYNC["terrain_tail"],

        1444: SYNC["target_head"],
        1445: SYNC["target_head"],
        2154: SYNC["target_tail"],
        2155: SYNC["target_tail"],
    }

    for idx, exp in expected_words.items():
        try:
            score += 10 if u32_at(idx) == exp else -10
        except Exception:
            score -= 10

    try:
        valid_points = u32_at(9)
        terrain_count = u32_at(1410)
        line_segment_count = u32_at(1451)
        isolated_count = u32_at(1452)
        dense_count = u32_at(1453)

        if valid_points == VALID_POINTS:
            score += 5
        if 0 <= terrain_count <= 5:
            score += 3
        if 0 <= line_segment_count <= 60:
            score += 3
        if 0 <= isolated_count <= 40:
            score += 3
        if 0 <= dense_count <= 6:
            score += 3
    except Exception:
        pass

    return score


def detect_endian(frame: bytes) -> str:
    le_score = score_frame_for_endian(frame, "<")
    be_score = score_frame_for_endian(frame, ">")
    return "<" if le_score >= be_score else ">"


def iter_frames_from_file(
    bin_file: Path,
    endian_mode: str = "auto",
    max_frames: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> Iterable[Tuple[int, str, bytes]]:
    """
    流式扫描文件，找到完整协议帧后 yield。

    返回：
      file_offset, endian, frame_bytes
    """
    pattern = pack_u32(SYNC["frame_head"], "<") * 2

    buf = bytearray()
    base_offset = 0
    yielded = 0

    with bin_file.open("rb") as fp:
        while True:
            chunk = fp.read(chunk_size)
            if not chunk and not buf:
                break

            if chunk:
                buf.extend(chunk)

            while True:
                idx = bytes(buf).find(pattern)
                if idx < 0:
                    # 保留一小段尾部，防止帧头跨 chunk
                    keep = min(len(buf), len(pattern) - 1)
                    if len(buf) > keep:
                        base_offset += len(buf) - keep
                        del buf[:len(buf) - keep]
                    break

                if idx > 0:
                    base_offset += idx
                    del buf[:idx]

                if len(buf) < FRAME_BYTES:
                    break

                frame = bytes(buf[:FRAME_BYTES])

                if endian_mode == "auto":
                    endian = detect_endian(frame)
                elif endian_mode == "little":
                    endian = "<"
                elif endian_mode == "big":
                    endian = ">"
                else:
                    raise ValueError(f"未知 endian_mode: {endian_mode}")

                score = score_frame_for_endian(frame, endian)

                if score < 80:
                    # 误命中帧头，跳过 1 字节继续找
                    base_offset += 1
                    del buf[:1]
                    continue

                file_offset = base_offset
                yield file_offset, endian, frame

                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    return

                base_offset += FRAME_BYTES
                del buf[:FRAME_BYTES]

            if not chunk:
                break


# =========================
# 单帧解析
# =========================

def parse_header(r: Reader, cfg: ParseConfig, warnings: List[str]) -> Dict[str, Any]:
    check_word(r.u32(), SYNC["frame_head"], "数据帧头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["frame_head"], "数据帧头[1]", warnings, cfg.strict)

    h = {
        "low_power_frame_no": r.u32(),
        "network_frame_no": r.u32(),
        "data_length_words": r.u32(),

        "work_mode": r.u32(),
        "antenna_frame_start": r.u32(),
        "antenna_frame_end": r.u32(),
        "range_km": r.u32(),
        "valid_points": r.u32(),

        "timestamp_raw_u64": r.u64(),

        "aircraft_lon_deg": r.f32(),
        "aircraft_lat_deg": r.f32(),
        "aircraft_true_heading_deg": r.f32(),
        "aircraft_gps_height_m": r.f32(),
        "aircraft_ground_speed_mps": r.f32(),
        "aircraft_east_speed_mps": r.f32(),
        "aircraft_north_speed_mps": r.f32(),
        "aircraft_up_speed_mps": r.f32(),

        "stable_azimuth_deg": r.f32(),
        "stable_pitch_deg": r.f32(),
        "antenna_scan_speed_degps": r.f32(),
        "antenna_scan_range_deg": r.f32(),
        "antenna_scan_dir": r.u32(),

        "azimuth_install_error_deg": r.f32(),
        "pitch_install_error_deg": r.f32(),
        "roll_install_error_deg": r.f32(),

        "reserved_29_62": r.u32s(34),
    }

    tail_1 = r.u32()
    tail_2 = r.u32()

    h["header_tail_1"] = tail_1
    h["header_tail_2"] = tail_2

    if tail_1 != SYNC["frame_tail"]:
        warnings.append(f"数据帧头尾[63] 非 0xBCBCBCBC: {hex32(tail_1)}")
    check_word(tail_2, SYNC["frame_tail"], "数据帧头尾[64]", warnings, cfg.strict)

    if h["valid_points"] != VALID_POINTS:
        warnings.append(f"有效点数不是 668: {h['valid_points']}")

    return h


def parse_echo(r: Reader, cfg: ParseConfig, warnings: List[str]) -> Dict[str, Any]:
    check_word(r.u32(), SYNC["sum_head"], "和路同步头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["sum_head"], "和路同步头[1]", warnings, cfg.strict)

    sum_channel = r.f32s(VALID_POINTS)

    check_word(r.u32(), SYNC["sum_tail"], "和路同步尾[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["sum_tail"], "和路同步尾[1]", warnings, cfg.strict)

    check_word(r.u32(), SYNC["diff_head"], "差路同步头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["diff_head"], "差路同步头[1]", warnings, cfg.strict)

    diff_channel = r.f32s(VALID_POINTS)

    check_word(r.u32(), SYNC["diff_tail"], "差路同步尾[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["diff_tail"], "差路同步尾[1]", warnings, cfg.strict)

    def stats(xs: List[float]) -> Dict[str, Optional[float]]:
        finite = [x for x in xs if math.isfinite(x)]
        if not finite:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(finite),
            "max": max(finite),
            "mean": sum(finite) / len(finite),
        }

    return {
        "sum_channel": sum_channel,
        "diff_channel": diff_channel,
        "sum_stats": stats(sum_channel),
        "diff_stats": stats(diff_channel),
    }


def parse_terrain(r: Reader, cfg: ParseConfig, warnings: List[str]) -> Dict[str, Any]:
    check_word(r.u32(), SYNC["terrain_head"], "地形点同步头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["terrain_head"], "地形点同步头[1]", warnings, cfg.strict)

    count = r.u32()
    reserved = r.u32()

    points_all = []
    for i in range(5):
        points_all.append({
            "index": i,
            "antenna_azimuth_deg": r.f32(),
            "antenna_pitch_deg": r.f32(),
            "pitch_error_deg": r.f32(),
            "range_gate_m": r.f32(),
            "power": r.f32(),
            "target_type": r.u32(),
        })

    check_word(r.u32(), SYNC["terrain_tail"], "地形点同步尾[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["terrain_tail"], "地形点同步尾[1]", warnings, cfg.strict)

    if count > 5:
        warnings.append(f"地形点个数超范围: {count}")

    return {
        "count": count,
        "reserved": reserved,
        "points_all": points_all,
        "points": points_all[:min(count, 5)],
    }


def parse_line_segment_word(word: int) -> Dict[str, int]:
    return {
        "raw": word,
        "segment_id": word & 0xFFFF,
        "head_tail_flag": (word >> 16) & 0xFF,
        "segment_count_in_line": (word >> 24) & 0xFF,
    }


def parse_targets(r: Reader, cfg: ParseConfig, warnings: List[str]) -> Dict[str, Any]:
    check_word(r.u32(), SYNC["target_head"], "目标信息同步头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["target_head"], "目标信息同步头[1]", warnings, cfg.strict)

    target = {
        "ref_aircraft_lon_deg": r.f32(),
        "ref_aircraft_lat_deg": r.f32(),
        "ref_aircraft_true_heading_deg": r.f32(),
        "ref_aircraft_gps_height_m": r.f32(),

        "powerline_count_internal": r.u32(),
        "line_segment_count": r.u32(),
        "isolated_object_count": r.u32(),
        "dense_area_count": r.u32(),

        "reserved_11_16": r.u32s(6),
    }

    raw_segment_words = r.u32s(60)
    line_segment_infos = [
        {"slot": i, **parse_line_segment_word(w)}
        for i, w in enumerate(raw_segment_words)
    ]

    powerline_positions_all = []
    for i in range(60):
        powerline_positions_all.append({
            "slot": i,
            "start_x_north_m": r.f32(),
            "start_y_west_m": r.f32(),
            "start_z_up_m": r.f32(),
            "end_x_north_m": r.f32(),
            "end_y_west_m": r.f32(),
            "end_z_up_m": r.f32(),
        })

    isolated_objects_all = []
    for i in range(40):
        isolated_objects_all.append({
            "slot": i,
            "x_north_m": r.f32(),
            "y_west_m": r.f32(),
            "z_up_m": r.f32(),
        })

    dense_vertex_counts = r.u32s(6)
    dense_x_flat = r.f32s(48)
    dense_y_flat = r.f32s(48)
    dense_z_flat = r.f32s(48)

    dense_areas_all = []
    for area_idx in range(6):
        n = dense_vertex_counts[area_idx]
        vertices = []
        for v_idx in range(8):
            k = area_idx * 8 + v_idx
            vertices.append({
                "vertex_index": v_idx,
                "x_north_m": dense_x_flat[k],
                "y_west_m": dense_y_flat[k],
                "z_up_m": dense_z_flat[k],
            })
        dense_areas_all.append({
            "area_index": area_idx,
            "vertex_count": n,
            "vertices_all": vertices,
            "vertices": vertices[:min(n, 8)],
        })

    target["reserved_707_710"] = r.u32s(4)

    tail_1 = r.u32()
    tail_2 = r.u32()

    check_word(tail_1, SYNC["target_tail"], "目标信息同步尾[0]", warnings, cfg.strict)
    check_word(tail_2, SYNC["target_tail"], "目标信息同步尾[1]", warnings, cfg.strict)

    seg_n = min(target["line_segment_count"], 60)
    iso_n = min(target["isolated_object_count"], 40)
    dense_n = min(target["dense_area_count"], 6)

    if target["line_segment_count"] > 60:
        warnings.append(f"高压线段条数超范围: {target['line_segment_count']}")
    if target["isolated_object_count"] > 40:
        warnings.append(f"孤立物个数超范围: {target['isolated_object_count']}")
    if target["dense_area_count"] > 6:
        warnings.append(f"密集区个数超范围: {target['dense_area_count']}")

    target.update({
        "line_segment_words_raw": raw_segment_words,
        "line_segment_infos_all": line_segment_infos,
        "line_segment_infos": line_segment_infos[:seg_n],

        "powerline_positions_all": powerline_positions_all,
        "powerline_positions": powerline_positions_all[:seg_n],

        "isolated_objects_all": isolated_objects_all,
        "isolated_objects": isolated_objects_all[:iso_n],

        "dense_vertex_counts": dense_vertex_counts,
        "dense_x_flat": dense_x_flat,
        "dense_y_flat": dense_y_flat,
        "dense_z_flat": dense_z_flat,
        "dense_areas_all": dense_areas_all,
        "dense_areas": dense_areas_all[:dense_n],

        "target_tail_1": tail_1,
        "target_tail_2": tail_2,
    })

    return target


def parse_frame(
    frame: bytes,
    offset: int,
    endian: str,
    strict: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if len(frame) != FRAME_BYTES:
        raise DecodeError(f"单帧长度错误: {len(frame)} != {FRAME_BYTES}")

    cfg = ParseConfig(endian=endian, strict=strict, verbose=verbose)
    warnings: List[str] = []
    r = Reader(frame, endian=endian)

    header = parse_header(r, cfg, warnings)
    echo = parse_echo(r, cfg, warnings)
    terrain = parse_terrain(r, cfg, warnings)
    targets = parse_targets(r, cfg, warnings)

    if r.tell() != FRAME_BYTES:
        warnings.append(f"解析后偏移异常: {r.tell()} != {FRAME_BYTES}")

    return {
        "offset": offset,
        "endian": endian_name(endian),
        "warnings": warnings,
        "header": header,
        "echo": echo,
        "terrain": terrain,
        "targets": targets,
    }


# =========================
# READ 转换与表格输出
# =========================

def xyz_nwu_to_read(
    x_north_m: float,
    y_west_m: float,
    z_up_m: float,
) -> Dict[str, Optional[float]]:
    """
    协议坐标：
      X: 北向
      Y: 西向
      Z: 天向

    Azimuth 约定：
      0° 指北
      正角指西
      负角指东
    """
    if x_north_m is None or y_west_m is None or z_up_m is None:
        return {
            "range_m": None,
            "elevation_deg": None,
            "azimuth_deg": None,
            "doppler_mps": None,
        }

    r_xy = math.hypot(x_north_m, y_west_m)
    r = math.sqrt(x_north_m ** 2 + y_west_m ** 2 + z_up_m ** 2)

    elevation = None if r == 0 else math.degrees(math.atan2(z_up_m, r_xy))
    azimuth = math.degrees(math.atan2(y_west_m, x_north_m))

    return {
        "range_m": r,
        "elevation_deg": elevation,
        "azimuth_deg": azimuth,
        "doppler_mps": None,
    }


def add_read_common(
    frame_idx: int,
    f: Dict[str, Any],
    source: str,
    source_index: Any,
    read: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    h = f["header"]

    row = {
        "frame_index": frame_idx,
        "file_offset": f["offset"],
        "network_frame_no": h.get("network_frame_no"),
        "low_power_frame_no": h.get("low_power_frame_no"),
        "source": source,
        "source_index": source_index,
        "range_m": read.get("range_m"),
        "elevation_deg": read.get("elevation_deg"),
        "azimuth_deg": read.get("azimuth_deg"),
        "doppler_mps": read.get("doppler_mps"),
    }

    if extra:
        row.update(extra)

    return row


def flatten_rows_for_read_points(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for local_idx, f in enumerate(frames):
        frame_idx = f.get("_frame_index", local_idx)

        for p in f["terrain"]["points"]:
            read = {
                "range_m": p.get("range_gate_m"),
                "elevation_deg": p.get("antenna_pitch_deg"),
                "azimuth_deg": p.get("antenna_azimuth_deg"),
                "doppler_mps": None,
            }
            rows.append(add_read_common(
                frame_idx,
                f,
                "terrain_point",
                p.get("index"),
                read,
                {
                    "power": p.get("power"),
                    "target_type": p.get("target_type"),
                    "note": "direct_from_terrain_block",
                },
            ))

        for obj in f["targets"]["isolated_objects"]:
            read = xyz_nwu_to_read(
                obj.get("x_north_m"),
                obj.get("y_west_m"),
                obj.get("z_up_m"),
            )
            rows.append(add_read_common(
                frame_idx,
                f,
                "isolated_object",
                obj.get("slot"),
                read,
                {
                    "x_north_m": obj.get("x_north_m"),
                    "y_west_m": obj.get("y_west_m"),
                    "z_up_m": obj.get("z_up_m"),
                    "note": "converted_from_xyz_nwu",
                },
            ))

        for seg in f["targets"]["powerline_positions"]:
            slot = seg.get("slot")
            points = [
                (
                    "powerline_start",
                    seg.get("start_x_north_m"),
                    seg.get("start_y_west_m"),
                    seg.get("start_z_up_m"),
                ),
                (
                    "powerline_end",
                    seg.get("end_x_north_m"),
                    seg.get("end_y_west_m"),
                    seg.get("end_z_up_m"),
                ),
                (
                    "powerline_midpoint",
                    0.5 * (seg.get("start_x_north_m") + seg.get("end_x_north_m")),
                    0.5 * (seg.get("start_y_west_m") + seg.get("end_y_west_m")),
                    0.5 * (seg.get("start_z_up_m") + seg.get("end_z_up_m")),
                ),
            ]

            for source, x, y, z in points:
                read = xyz_nwu_to_read(x, y, z)
                rows.append(add_read_common(
                    frame_idx,
                    f,
                    source,
                    slot,
                    read,
                    {
                        "x_north_m": x,
                        "y_west_m": y,
                        "z_up_m": z,
                        "note": "converted_from_powerline_xyz_nwu",
                    },
                ))

        for area in f["targets"]["dense_areas"]:
            for v in area["vertices"]:
                read = xyz_nwu_to_read(
                    v.get("x_north_m"),
                    v.get("y_west_m"),
                    v.get("z_up_m"),
                )
                rows.append(add_read_common(
                    frame_idx,
                    f,
                    "dense_area_vertex",
                    f"{area.get('area_index')}:{v.get('vertex_index')}",
                    read,
                    {
                        "area_index": area.get("area_index"),
                        "vertex_index": v.get("vertex_index"),
                        "area_vertex_count": area.get("vertex_count"),
                        "x_north_m": v.get("x_north_m"),
                        "y_west_m": v.get("y_west_m"),
                        "z_up_m": v.get("z_up_m"),
                        "note": "converted_from_dense_area_xyz_nwu",
                    },
                ))

    return rows


def flatten_rows_for_terrain(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for local_idx, f in enumerate(frames):
        h = f["header"]
        frame_idx = f.get("_frame_index", local_idx)
        for p in f["terrain"]["points"]:
            rows.append({
                "frame_index": frame_idx,
                "file_offset": f["offset"],
                "network_frame_no": h.get("network_frame_no"),
                "low_power_frame_no": h.get("low_power_frame_no"),
                **p,
            })
    return rows


def flatten_rows_for_isolated(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for local_idx, f in enumerate(frames):
        h = f["header"]
        frame_idx = f.get("_frame_index", local_idx)
        for obj in f["targets"]["isolated_objects"]:
            rows.append({
                "frame_index": frame_idx,
                "file_offset": f["offset"],
                "network_frame_no": h.get("network_frame_no"),
                "low_power_frame_no": h.get("low_power_frame_no"),
                **obj,
            })
    return rows


def flatten_rows_for_powerlines(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for local_idx, f in enumerate(frames):
        h = f["header"]
        frame_idx = f.get("_frame_index", local_idx)
        infos = f["targets"]["line_segment_infos"]
        poss = f["targets"]["powerline_positions"]

        for i, pos in enumerate(poss):
            info = infos[i] if i < len(infos) else {}
            rows.append({
                "frame_index": frame_idx,
                "file_offset": f["offset"],
                "network_frame_no": h.get("network_frame_no"),
                "low_power_frame_no": h.get("low_power_frame_no"),
                **info,
                **pos,
            })

    return rows


def flatten_rows_for_dense_areas(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for local_idx, f in enumerate(frames):
        h = f["header"]
        frame_idx = f.get("_frame_index", local_idx)

        for area in f["targets"]["dense_areas"]:
            for v in area["vertices"]:
                rows.append({
                    "frame_index": frame_idx,
                    "file_offset": f["offset"],
                    "network_frame_no": h.get("network_frame_no"),
                    "low_power_frame_no": h.get("low_power_frame_no"),
                    "area_index": area["area_index"],
                    "area_vertex_count": area["vertex_count"],
                    **v,
                })

    return rows


# =========================
# 文件输出
# =========================

def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = sorted({k for row in rows for k in row.keys()})

    with path.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def strip_large_arrays(f: Dict[str, Any]) -> Dict[str, Any]:
    """
    写 JSON 时去掉 668 点大数组，避免 JSON 太大。
    """
    obj = {
        "offset": f.get("offset"),
        "endian": f.get("endian"),
        "_frame_index": f.get("_frame_index"),
        "warnings": f.get("warnings"),
        "header": f.get("header"),
        "terrain": f.get("terrain"),
        "targets": f.get("targets"),
        "echo": {
            "sum_stats": f.get("echo", {}).get("sum_stats"),
            "diff_stats": f.get("echo", {}).get("diff_stats"),
        },
    }
    return obj


def frame_start_flag(f: Dict[str, Any]) -> int:
    return int(f["header"].get("antenna_frame_start") or 0)


def frame_end_flag(f: Dict[str, Any]) -> int:
    return int(f["header"].get("antenna_frame_end") or 0)


def frame_scan_dir(f: Dict[str, Any]) -> int:
    return int(f["header"].get("antenna_scan_dir") or 0)


def scan_summary(
    scan_index: int,
    frames: List[Dict[str, Any]],
    complete: bool,
    close_reason: str,
) -> Dict[str, Any]:
    first = frames[0]
    last = frames[-1]

    return {
        "scan_index": scan_index,
        "complete": complete,
        "close_reason": close_reason,
        "frame_count": len(frames),

        "start_frame_index": first.get("_frame_index"),
        "end_frame_index": last.get("_frame_index"),

        "start_file_offset": first.get("offset"),
        "end_file_offset": last.get("offset"),

        "start_network_frame_no": first["header"].get("network_frame_no"),
        "end_network_frame_no": last["header"].get("network_frame_no"),

        "start_low_power_frame_no": first["header"].get("low_power_frame_no"),
        "end_low_power_frame_no": last["header"].get("low_power_frame_no"),

        "start_flag": frame_start_flag(first),
        "end_flag": frame_end_flag(last),

        "antenna_scan_dir_first": frame_scan_dir(first),
        "antenna_scan_dir_last": frame_scan_dir(last),

        "warnings_count": sum(len(f.get("warnings", [])) for f in frames),
        "terrain_points_total": sum(int(f["terrain"].get("count") or 0) for f in frames),
        "isolated_objects_total": sum(int(f["targets"].get("isolated_object_count") or 0) for f in frames),
        "line_segments_total": sum(int(f["targets"].get("line_segment_count") or 0) for f in frames),
        "dense_areas_total": sum(int(f["targets"].get("dense_area_count") or 0) for f in frames),

        "network_frame_numbers": [f["header"].get("network_frame_no") for f in frames],
        "low_power_frame_numbers": [f["header"].get("low_power_frame_no") for f in frames],
        "file_offsets": [f.get("offset") for f in frames],
    }


def dump_scan_echo_npy(scan_dir: Path, frames: List[Dict[str, Any]]) -> None:
    try:
        import numpy as np
    except ImportError:
        print("[WARN] 未安装 numpy，跳过 echo npy 输出。可执行 pip install numpy", file=sys.stderr)
        return

    sum_arr = np.asarray(
        [f["echo"]["sum_channel"] for f in frames],
        dtype=np.float32,
    )
    diff_arr = np.asarray(
        [f["echo"]["diff_channel"] for f in frames],
        dtype=np.float32,
    )

    np.save(scan_dir / "sum_channel.npy", sum_arr)
    np.save(scan_dir / "diff_channel.npy", diff_arr)


def save_one_scan(
    out_dir: Path,
    scan_index: int,
    frames: List[Dict[str, Any]],
    complete: bool,
    close_reason: str,
    dump_echo: bool,
    save_raw: bool,
) -> Dict[str, Any]:
    """
    核心函数：
    每次完整扫描结束后，马上调用这个函数保存到 scan_xxxxxx 文件夹。
    """
    scans_root = out_dir / "scans"
    scans_root.mkdir(parents=True, exist_ok=True)

    scan_dir = scans_root / f"scan_{scan_index:06d}"
    scan_dir.mkdir(parents=True, exist_ok=True)

    summary = scan_summary(
        scan_index=scan_index,
        frames=frames,
        complete=complete,
        close_reason=close_reason,
    )

    write_json(scan_dir / "scan_summary.json", summary)
    write_json(
        scan_dir / "frames_summary.json",
        [strip_large_arrays(f) for f in frames],
    )

    write_csv(scan_dir / "read_points.csv", flatten_rows_for_read_points(frames))
    write_csv(scan_dir / "terrain_points.csv", flatten_rows_for_terrain(frames))
    write_csv(scan_dir / "isolated_objects.csv", flatten_rows_for_isolated(frames))
    write_csv(scan_dir / "powerlines.csv", flatten_rows_for_powerlines(frames))
    write_csv(scan_dir / "dense_areas.csv", flatten_rows_for_dense_areas(frames))

    if dump_echo:
        dump_scan_echo_npy(scan_dir, frames)

    if save_raw:
        with (scan_dir / "raw_frames.bin").open("wb") as fp:
            for f in frames:
                raw = f.get("_raw_frame")
                if raw:
                    fp.write(raw)

    return summary


def print_frame_summary(frame_index: int, f: Dict[str, Any]) -> None:
    h = f["header"]
    t = f["terrain"]
    g = f["targets"]
    e = f["echo"]

    print(
        f"[Frame {frame_index:06d}] "
        f"offset={f['offset']} "
        f"endian={f['endian']} "
        f"net_frame={h.get('network_frame_no')} "
        f"start={h.get('antenna_frame_start')} "
        f"end={h.get('antenna_frame_end')} "
        f"dir={h.get('antenna_scan_dir')} "
        f"terrain={t.get('count')} "
        f"line_seg={g.get('line_segment_count')} "
        f"isolated={g.get('isolated_object_count')} "
        f"dense={g.get('dense_area_count')} "
        f"sum_mean={e['sum_stats']['mean']} "
        f"diff_mean={e['diff_stats']['mean']} "
        f"warnings={len(f['warnings'])}"
    )


# =========================
# 主程序：边处理边保存
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(description="毫米波雷达 bin 流式解析，并按完整天线扫描保存")
    ap.add_argument("bin_file", type=Path, help="输入 bin 文件")
    ap.add_argument("--out", type=Path, default=Path("mmwave_stream_out"), help="输出目录")
    ap.add_argument("--endian", choices=["auto", "little", "big"], default="auto", help="字节序")
    ap.add_argument("--max-frames", type=int, default=None, help="最多处理多少帧")
    ap.add_argument("--strict", action="store_true", help="严格模式")
    ap.add_argument("--non-strict", action="store_true", help="非严格模式")
    ap.add_argument("--dump-echo", action="store_true", help="每次扫描输出 sum/diff npy")
    ap.add_argument("--save-raw", action="store_true", help="每次扫描保存 raw_frames.bin")
    ap.add_argument("--keep-incomplete", action="store_true", help="保存文件开头/结尾不完整扫描")
    ap.add_argument("--verbose", action="store_true", help="打印 warning")
    args = ap.parse_args()

    if not args.bin_file.exists():
        print(f"[ERROR] 文件不存在: {args.bin_file}", file=sys.stderr)
        return 2

    strict = True
    if args.non_strict:
        strict = False
    if args.strict:
        strict = True

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 输入文件: {args.bin_file}")
    print(f"[INFO] 输出目录: {args.out.resolve()}")
    print(f"[INFO] 单帧长度: {FRAME_BYTES} bytes")
    print("[INFO] 开始流式解析，遇到完整扫描结束标志后立即保存。")

    current_scan_frames: List[Dict[str, Any]] = []
    scan_summaries: List[Dict[str, Any]] = []

    scan_index = 0
    frame_index = 0
    error_count = 0

    for off, endian, raw_frame in iter_frames_from_file(
        args.bin_file,
        endian_mode=args.endian,
        max_frames=args.max_frames,
    ):
        try:
            f = parse_frame(
                raw_frame,
                offset=off,
                endian=endian,
                strict=strict,
                verbose=args.verbose,
            )

            f["_frame_index"] = frame_index

            if args.save_raw:
                f["_raw_frame"] = raw_frame

            print_frame_summary(frame_index, f)

            if args.verbose and f["warnings"]:
                for w in f["warnings"]:
                    print(f"  [WARN] {w}")

            frame_index += 1

            s = frame_start_flag(f)
            e = frame_end_flag(f)

            # 遇到新扫描开始，但上一轮还没结束
            if s != 0 and current_scan_frames:
                if args.keep_incomplete:
                    summary = save_one_scan(
                        out_dir=args.out,
                        scan_index=scan_index,
                        frames=current_scan_frames,
                        complete=False,
                        close_reason="new_start_before_previous_end",
                        dump_echo=args.dump_echo,
                        save_raw=args.save_raw,
                    )
                    scan_summaries.append(summary)

                    print(
                        f"[SAVE] scan_{scan_index:06d} "
                        f"incomplete, frames={summary['frame_count']}, "
                        f"reason=new_start_before_previous_end"
                    )

                    scan_index += 1

                current_scan_frames = []

            current_scan_frames.append(f)

            # 遇到完整扫描结束，立刻保存
            if e != 0:
                summary = save_one_scan(
                    out_dir=args.out,
                    scan_index=scan_index,
                    frames=current_scan_frames,
                    complete=True,
                    close_reason="end_flag",
                    dump_echo=args.dump_echo,
                    save_raw=args.save_raw,
                )
                scan_summaries.append(summary)

                print(
                    f"[SAVE] scan_{scan_index:06d} "
                    f"complete, frames={summary['frame_count']}, "
                    f"net_frame={summary['start_network_frame_no']}~{summary['end_network_frame_no']}"
                )

                scan_index += 1
                current_scan_frames = []

                # 主索引也边处理边更新，防止中途停止丢失索引
                write_json(args.out / "scans" / "scans_summary.json", scan_summaries)

        except Exception as exc:
            error_count += 1
            print(
                f"[ERROR] 解析失败 offset={off}, endian={endian_name(endian)}: {exc}",
                file=sys.stderr,
            )
            continue

    # 文件结束后，如果还有未闭合的扫描
    if current_scan_frames and args.keep_incomplete:
        summary = save_one_scan(
            out_dir=args.out,
            scan_index=scan_index,
            frames=current_scan_frames,
            complete=False,
            close_reason="eof_without_end_flag",
            dump_echo=args.dump_echo,
            save_raw=args.save_raw,
        )
        scan_summaries.append(summary)

        print(
            f"[SAVE] scan_{scan_index:06d} "
            f"incomplete, frames={summary['frame_count']}, "
            f"reason=eof_without_end_flag"
        )

        scan_index += 1
        current_scan_frames = []

    scans_root = args.out / "scans"
    scans_root.mkdir(parents=True, exist_ok=True)

    write_json(scans_root / "scans_summary.json", scan_summaries)

    run_summary = {
        "input_file": str(args.bin_file),
        "frame_bytes": FRAME_BYTES,
        "processed_frame_count": frame_index,
        "saved_scan_count": len(scan_summaries),
        "error_count": error_count,
        "keep_incomplete": args.keep_incomplete,
        "dump_echo": args.dump_echo,
        "save_raw": args.save_raw,
    }
    write_json(args.out / "run_summary.json", run_summary)

    print("[INFO] 处理完成")
    print(f"[INFO] 总帧数: {frame_index}")
    print(f"[INFO] 保存扫描次数: {len(scan_summaries)}")
    print(f"[INFO] 错误数: {error_count}")
    print(f"[INFO] 扫描目录: {scans_root.resolve()}")
    print(f"[INFO] 扫描索引: {(scans_root / 'scans_summary.json').resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())