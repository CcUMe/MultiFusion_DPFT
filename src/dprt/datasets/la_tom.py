#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
毫米波雷达 bin 数据协议解析/测试脚本

支持：
1. 从 .bin 文件中自动扫描 0xABABABAB 0xABABABAB 帧头
2. 自动尝试小端/大端字节序
3. 按协议解析：
   - 64 words 数据帧头
   - 1344 words 回波数据体：和路 668 点 + 差路 668 点
   - 36 words 地形点数据体
   - 712 words 电力线/孤立物/密集区目标信息
4. 输出 summary.json
5. 可选导出 echo .npy、terrain.csv、isolated_objects.csv、powerlines.csv、dense_areas.csv

用法示例：
    python parse_mmwave_bin.py input.bin
    python parse_mmwave_bin.py input.bin --endian auto --out out_dir --dump-echo
    python parse_mmwave_bin.py input.bin --max-frames 10 --verbose

说明：
- 协议单帧长度：2156 * 4 = 8624 bytes
- 如果你的 bin 是 UDP payload 连续拼接，脚本会自动扫描有效帧头。
- 如果你的 bin 带 pcap/链路层/UDP/IP 头，需要先提取 payload 后再运行本脚本。
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
# 低层读取工具
# =========================

class DecodeError(Exception):
    pass


@dataclass
class ParseConfig:
    endian: str = "<"       # "<" little, ">" big
    strict: bool = True     # True: 同步字不匹配直接报错；False: 记录 warning 后尽量继续
    verbose: bool = False


class Reader:
    def __init__(self, data: bytes, endian: str = "<"):
        self.data = data
        self.endian = endian
        self.off = 0

    def tell(self) -> int:
        return self.off

    def seek(self, off: int) -> None:
        if off < 0 or off > len(self.data):
            raise DecodeError(f"seek 越界: {off}")
        self.off = off

    def _unpack(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        if self.off + size > len(self.data):
            raise DecodeError(f"数据不足: offset={self.off}, need={size}, left={len(self.data)-self.off}")
        v = struct.unpack_from(fmt, self.data, self.off)
        self.off += size
        return v[0] if len(v) == 1 else v

    def u32(self) -> int:
        return int(self._unpack(self.endian + "I"))

    def i32(self) -> int:
        return int(self._unpack(self.endian + "i"))

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
    strict: bool = True,
) -> None:
    if got != expected:
        msg = f"{name} 同步字错误: got={hex32(got)}, expected={hex32(expected)}"
        if strict:
            raise DecodeError(msg)
        warnings.append(msg)


def is_reasonable_float(v: float, min_v: float, max_v: float) -> bool:
    return math.isfinite(v) and min_v <= v <= max_v


# =========================
# 字节序与帧扫描
# =========================

def endian_name(endian: str) -> str:
    return "little" if endian == "<" else "big"


def pack_u32(value: int, endian: str) -> bytes:
    return struct.pack(endian + "I", value)


def find_candidate_offsets(data: bytes, endian: str) -> List[int]:
    """扫描 0xABABABAB 0xABABABAB 帧头。注意该同步字字节对称，大小端字节相同。"""
    pattern = pack_u32(SYNC["frame_head"], endian) * 2
    offsets = []
    start = 0
    while True:
        idx = data.find(pattern, start)
        if idx < 0:
            break
        offsets.append(idx)
        start = idx + 1
    return offsets


def score_frame_for_endian(frame: bytes, endian: str) -> int:
    """粗略评分，用于 auto 字节序判断。"""
    if len(frame) < FRAME_BYTES:
        return -9999

    score = 0

    def u32_at(word_idx: int) -> int:
        return struct.unpack_from(endian + "I", frame, word_idx * 4)[0]

    # 固定同步字位置
    expected_words = {
        0: SYNC["frame_head"],
        1: SYNC["frame_head"],
        63: SYNC["frame_tail"],

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
            if u32_at(idx) == exp:
                score += 10
            else:
                score -= 10
        except Exception:
            score -= 10

    # 一些字段合理性
    try:
        data_len_words = u32_at(4)
        valid_points = u32_at(9)
        terrain_count = u32_at(1410)
        line_segment_count = u32_at(1451)
        isolated_count = u32_at(1452)
        dense_count = u32_at(1453)

        if data_len_words in (FRAME_WORDS, FRAME_WORDS - HEADER_WORDS, ECHO_WORDS + TERRAIN_WORDS + TARGET_WORDS):
            score += 3
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
    s_le = score_frame_for_endian(frame, "<")
    s_be = score_frame_for_endian(frame, ">")
    return "<" if s_le >= s_be else ">"


def iter_frames(
    data: bytes,
    endian_mode: str = "auto",
    max_frames: Optional[int] = None,
) -> Iterable[Tuple[int, str, bytes]]:
    """
    返回: (offset, endian, frame_bytes)
    endian_mode: auto/little/big
    """
    # 由于 ABABABAB 字节对称，扫描 little 或 big 都一样；这里固定用 little pattern 即可
    offsets = find_candidate_offsets(data, "<")
    count = 0

    for off in offsets:
        if off + FRAME_BYTES > len(data):
            continue

        frame = data[off: off + FRAME_BYTES]

        if endian_mode == "auto":
            endian = detect_endian(frame)
        elif endian_mode == "little":
            endian = "<"
        elif endian_mode == "big":
            endian = ">"
        else:
            raise ValueError(f"未知 endian_mode: {endian_mode}")

        # 避免把数据体里偶然出现的 ABABABAB 当成帧，做一次基本评分
        if score_frame_for_endian(frame, endian) < 80:
            continue

        yield off, endian, frame
        count += 1
        if max_frames is not None and count >= max_frames:
            break


# =========================
# 解析各数据块
# =========================

def parse_header(r: Reader, cfg: ParseConfig, warnings: List[str]) -> Dict[str, Any]:
    check_word(r.u32(), SYNC["frame_head"], "数据帧头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["frame_head"], "数据帧头[1]", warnings, cfg.strict)

    h: Dict[str, Any] = {
        "low_power_frame_no": r.u32(),
        "network_frame_no": r.u32(),
        "data_length_words": r.u32(),

        "work_mode": r.u32(),
        "antenna_frame_start": r.u32(),
        "antenna_frame_end": r.u32(),
        "range_km": r.u32(),
        "valid_points": r.u32(),

        # 协议写成 Uint64：年、月、日、时-分秒-毫秒组合。
        # 由于具体编码没有进一步说明，这里保留原始 64bit。
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

    # 表格中第63字没有明确写数值，第64字写 0xBCBCBCBC。
    # 实测如果第63字也为 0xBCBCBCBC 会通过；否则只作为 warning。
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
        point = {
            "index": i,
            "antenna_azimuth_deg": r.f32(),
            "antenna_pitch_deg": r.f32(),
            "pitch_error_deg": r.f32(),
            "range_gate_m": r.f32(),
            "power": r.f32(),
            # 协议第6字写“目标类型，地形固定填9”，但数据类型没有单独写明。
            # 因为所有字均为 32bit，这里按 uint32 解析。
            "target_type": r.u32(),
        }
        points_all.append(point)

    check_word(r.u32(), SYNC["terrain_tail"], "地形点同步尾[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["terrain_tail"], "地形点同步尾[1]", warnings, cfg.strict)

    if count > 5:
        warnings.append(f"地形点个数超范围: {count}, 将只导出前 5 个槽位")
    valid_count = min(count, 5)

    return {
        "count": count,
        "reserved": reserved,
        "points_all": points_all,
        "points": points_all[:valid_count],
    }


def parse_line_segment_word(word: int) -> Dict[str, int]:
    """
    高压线段信息定义：
    bit 0~15: 高压线段 ID
    bit 16~23: 首尾线段标识，0 不是首尾，0xff 是首尾
    bit 24~31: 一条高压线中线段个数，不包含首尾线段
    """
    return {
        "raw": word,
        "segment_id": word & 0xFFFF,
        "head_tail_flag": (word >> 16) & 0xFF,
        "segment_count_in_line": (word >> 24) & 0xFF,
    }


def parse_targets(r: Reader, cfg: ParseConfig, warnings: List[str]) -> Dict[str, Any]:
    check_word(r.u32(), SYNC["target_head"], "目标信息同步头[0]", warnings, cfg.strict)
    check_word(r.u32(), SYNC["target_head"], "目标信息同步头[1]", warnings, cfg.strict)

    target: Dict[str, Any] = {
        "ref_aircraft_lon_deg": r.f32(),
        "ref_aircraft_lat_deg": r.f32(),
        "ref_aircraft_true_heading_deg": r.f32(),
        "ref_aircraft_gps_height_m": r.f32(),

        "powerline_count_internal": r.u32(),  # 0~60
        "line_segment_count": r.u32(),        # 0~60
        "isolated_object_count": r.u32(),     # 0~40
        "dense_area_count": r.u32(),          # 0~6

        "reserved_11_16": r.u32s(6),
    }

    raw_segment_words = r.u32s(60)
    line_segment_infos = [parse_line_segment_word(w) | {"slot": i} for i, w in enumerate(raw_segment_words)]

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
    dense_x_flat = r.f32s(48)  # 6 * 8
    dense_y_flat = r.f32s(48)
    dense_z_flat = r.f32s(48)

    dense_areas_all = []
    for area_idx in range(6):
        n = dense_vertex_counts[area_idx]
        vertices = []
        for v_idx in range(8):
            flat_idx = area_idx * 8 + v_idx
            vertices.append({
                "vertex_index": v_idx,
                "x_north_m": dense_x_flat[flat_idx],
                "y_west_m": dense_y_flat[flat_idx],
                "z_up_m": dense_z_flat[flat_idx],
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

    if target["powerline_count_internal"] > 60:
        warnings.append(f"高压线条数超范围: {target['powerline_count_internal']}")
    if target["line_segment_count"] > 60:
        warnings.append(f"高压线段条数超范围: {target['line_segment_count']}")
    if target["isolated_object_count"] > 40:
        warnings.append(f"孤立物个数超范围: {target['isolated_object_count']}")
    if target["dense_area_count"] > 6:
        warnings.append(f"密集区个数超范围: {target['dense_area_count']}")

    seg_n = min(target["line_segment_count"], 60)
    iso_n = min(target["isolated_object_count"], 40)
    dense_n = min(target["dense_area_count"], 6)

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


def parse_frame(frame: bytes, offset: int, endian: str, strict: bool = True, verbose: bool = False) -> Dict[str, Any]:
    if len(frame) != FRAME_BYTES:
        raise DecodeError(f"单帧长度错误: {len(frame)} != {FRAME_BYTES}")

    cfg = ParseConfig(endian=endian, strict=strict, verbose=verbose)
    warnings: List[str] = []
    r = Reader(frame, endian=endian)

    header = parse_header(r, cfg, warnings)
    if r.tell() != HEADER_WORDS * 4:
        warnings.append(f"数据帧头解析偏移异常: {r.tell()} != {HEADER_WORDS*4}")

    echo = parse_echo(r, cfg, warnings)
    if r.tell() != (HEADER_WORDS + ECHO_WORDS) * 4:
        warnings.append(f"回波数据体解析偏移异常: {r.tell()}")

    terrain = parse_terrain(r, cfg, warnings)
    if r.tell() != (HEADER_WORDS + ECHO_WORDS + TERRAIN_WORDS) * 4:
        warnings.append(f"地形点数据体解析偏移异常: {r.tell()}")

    targets = parse_targets(r, cfg, warnings)
    if r.tell() != FRAME_BYTES:
        warnings.append(f"目标信息体解析后偏移异常: {r.tell()} != {FRAME_BYTES}")

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
# 输出工具
# =========================

def strip_large_arrays(frame_info: Dict[str, Any], keep_echo_stats_only: bool = True) -> Dict[str, Any]:
    """summary.json 默认不放完整 668 点数组，避免过大。"""
    import copy
    obj = copy.deepcopy(frame_info)
    if keep_echo_stats_only and "echo" in obj:
        obj["echo"].pop("sum_channel", None)
        obj["echo"].pop("diff_channel", None)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_rows_for_terrain(parsed_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for frame_idx, f in enumerate(parsed_frames):
        h = f["header"]
        for p in f["terrain"]["points"]:
            row = {
                "frame_index": frame_idx,
                "file_offset": f["offset"],
                "network_frame_no": h.get("network_frame_no"),
                "low_power_frame_no": h.get("low_power_frame_no"),
                **p,
            }
            rows.append(row)
    return rows


def flatten_rows_for_isolated(parsed_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for frame_idx, f in enumerate(parsed_frames):
        h = f["header"]
        for obj in f["targets"]["isolated_objects"]:
            row = {
                "frame_index": frame_idx,
                "file_offset": f["offset"],
                "network_frame_no": h.get("network_frame_no"),
                "low_power_frame_no": h.get("low_power_frame_no"),
                **obj,
            }
            rows.append(row)
    return rows


def flatten_rows_for_powerlines(parsed_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for frame_idx, f in enumerate(parsed_frames):
        h = f["header"]
        infos = f["targets"]["line_segment_infos"]
        poss = f["targets"]["powerline_positions"]
        for i, pos in enumerate(poss):
            info = infos[i] if i < len(infos) else {}
            row = {
                "frame_index": frame_idx,
                "file_offset": f["offset"],
                "network_frame_no": h.get("network_frame_no"),
                "low_power_frame_no": h.get("low_power_frame_no"),
                **info,
                **pos,
            }
            rows.append(row)
    return rows


def flatten_rows_for_dense_areas(parsed_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for frame_idx, f in enumerate(parsed_frames):
        h = f["header"]
        for area in f["targets"]["dense_areas"]:
            for v in area["vertices"]:
                row = {
                    "frame_index": frame_idx,
                    "file_offset": f["offset"],
                    "network_frame_no": h.get("network_frame_no"),
                    "low_power_frame_no": h.get("low_power_frame_no"),
                    "area_index": area["area_index"],
                    "area_vertex_count": area["vertex_count"],
                    **v,
                }
                rows.append(row)
    return rows


def dump_echo_npy(out_dir: Path, parsed_frames: List[Dict[str, Any]]) -> None:
    try:
        import numpy as np
    except ImportError:
        print("[WARN] 未安装 numpy，跳过 echo .npy 导出。可执行: pip install numpy", file=sys.stderr)
        return

    sum_arr = np.asarray([f["echo"]["sum_channel"] for f in parsed_frames], dtype=np.float32)
    diff_arr = np.asarray([f["echo"]["diff_channel"] for f in parsed_frames], dtype=np.float32)

    np.save(out_dir / "sum_channel.npy", sum_arr)
    np.save(out_dir / "diff_channel.npy", diff_arr)


def print_frame_summary(frame_idx: int, f: Dict[str, Any]) -> None:
    h = f["header"]
    t = f["terrain"]
    g = f["targets"]
    e = f["echo"]

    print(
        f"[Frame {frame_idx:06d}] "
        f"offset={f['offset']} endian={f['endian']} "
        f"net_frame={h.get('network_frame_no')} "
        f"low_power={h.get('low_power_frame_no')} "
        f"range_km={h.get('range_km')} "
        f"valid_points={h.get('valid_points')} "
        f"terrain={t.get('count')} "
        f"line_seg={g.get('line_segment_count')} "
        f"isolated={g.get('isolated_object_count')} "
        f"dense={g.get('dense_area_count')} "
        f"sum_mean={e['sum_stats']['mean']} "
        f"diff_mean={e['diff_stats']['mean']} "
        f"warnings={len(f['warnings'])}"
    )


# =========================
# 主程序
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(description="毫米波雷达协议 bin 数据解析/测试脚本")
    ap.add_argument("bin_file", type=Path, help="输入 .bin 文件")
    ap.add_argument("--out", type=Path, default=Path("mmwave_parse_out"), help="输出目录，默认 mmwave_parse_out")
    ap.add_argument("--endian", choices=["auto", "little", "big"], default="auto", help="字节序，默认 auto")
    ap.add_argument("--max-frames", type=int, default=None, help="最多解析多少帧")
    ap.add_argument("--strict", action="store_true", help="严格模式：同步字不匹配立即报错")
    ap.add_argument("--non-strict", action="store_true", help="非严格模式：同步字不匹配时尽量继续")
    ap.add_argument("--dump-echo", action="store_true", help="导出 sum_channel.npy 和 diff_channel.npy")
    ap.add_argument("--verbose", action="store_true", help="打印每帧 warning")
    args = ap.parse_args()

    strict = True
    if args.non_strict:
        strict = False
    if args.strict:
        strict = True

    if not args.bin_file.exists():
        print(f"[ERROR] 文件不存在: {args.bin_file}", file=sys.stderr)
        return 2

    data = args.bin_file.read_bytes()
    print(f"[INFO] 读取文件: {args.bin_file}")
    print(f"[INFO] 文件大小: {len(data)} bytes")
    print(f"[INFO] 协议单帧长度: {FRAME_BYTES} bytes")

    args.out.mkdir(parents=True, exist_ok=True)

    candidates = find_candidate_offsets(data, "<")
    print(f"[INFO] 扫描到候选帧头数量: {len(candidates)}")

    parsed_frames: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for frame_idx, (off, endian, frame) in enumerate(iter_frames(data, args.endian, args.max_frames)):
        try:
            f = parse_frame(frame, offset=off, endian=endian, strict=strict, verbose=args.verbose)
            parsed_frames.append(f)
            print_frame_summary(len(parsed_frames) - 1, f)

            if args.verbose and f["warnings"]:
                for w in f["warnings"]:
                    print(f"  [WARN] {w}")

        except Exception as e:
            err = {
                "offset": off,
                "endian": endian_name(endian),
                "error": repr(e),
            }
            errors.append(err)
            print(f"[ERROR] 解析失败 offset={off}, endian={endian_name(endian)}: {e}", file=sys.stderr)
            if strict:
                # strict 模式下某候选失败并不立即退出，因为可能是误命中的候选帧头
                continue

    print(f"[INFO] 成功解析帧数: {len(parsed_frames)}")
    print(f"[INFO] 解析失败候选数: {len(errors)}")

    summary = {
        "input_file": str(args.bin_file),
        "file_size_bytes": len(data),
        "frame_bytes": FRAME_BYTES,
        "candidate_header_count": len(candidates),
        "parsed_frame_count": len(parsed_frames),
        "error_count": len(errors),
        "errors": errors[:100],
        "frames": [strip_large_arrays(f, keep_echo_stats_only=True) for f in parsed_frames],
    }

    write_json(args.out / "summary.json", summary)

    write_csv(args.out / "terrain_points.csv", flatten_rows_for_terrain(parsed_frames))
    write_csv(args.out / "isolated_objects.csv", flatten_rows_for_isolated(parsed_frames))
    write_csv(args.out / "powerlines.csv", flatten_rows_for_powerlines(parsed_frames))
    write_csv(args.out / "dense_areas.csv", flatten_rows_for_dense_areas(parsed_frames))

    if args.dump_echo:
        dump_echo_npy(args.out, parsed_frames)

    print(f"[INFO] 输出目录: {args.out.resolve()}")
    print("[INFO] 已输出:")
    print(f"  - {args.out / 'summary.json'}")
    print(f"  - {args.out / 'terrain_points.csv'}")
    print(f"  - {args.out / 'isolated_objects.csv'}")
    print(f"  - {args.out / 'powerlines.csv'}")
    print(f"  - {args.out / 'dense_areas.csv'}")
    if args.dump_echo:
        print(f"  - {args.out / 'sum_channel.npy'}")
        print(f"  - {args.out / 'diff_channel.npy'}")

    if not parsed_frames:
        print(
            "\n[HINT] 没有解析出有效帧。常见原因：\n"
            "1. bin 不是纯 UDP payload，而是 pcap 或带 IP/UDP 头；需要先提取 payload。\n"
            "2. 数据不是该协议的 2156*32bit 帧格式。\n"
            "3. 字节序或同步字与文档不一致。\n"
            "4. 文件中只有半帧或丢包导致长度不足。\n",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
