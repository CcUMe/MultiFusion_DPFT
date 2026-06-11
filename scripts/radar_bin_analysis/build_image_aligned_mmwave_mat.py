"""End-to-end mmWave bin -> image-aligned AntFrame MAT export.

This script stitches together the existing radar bin analysis workflow:

1. Read one raw ``*_mmwave_udp.bin`` file.
2. Reconstruct per-packet bag-relative time with the same anchor interpolation
   strategy as ``match_radar_camera_anchor.py``.
3. Build an image <-> AntFrame bridge CSV.
4. Export the matched AntFrames as 1218-style ``.mat`` files.

Outputs:
  - ``match_radar_camera_anchor.csv``: packet-level alignment table.
  - ``image_to_antframe_time_aligned.csv``: image-level bridge table.
  - ``mmwave_mat_1218style/*.mat``: one MAT per matched AntFrame.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import savemat

import match_radar_camera_anchor as anchor_match


N_RANGE_OUT = 666
EL_MIN, EL_MAX, EL_STEP = -10.0, 5.0, 0.5
EL_GRID = np.arange(EL_MIN, EL_MAX + EL_STEP / 2, EL_STEP)
IMAGE_SEQ_RE = re.compile(r"_(\d{4,})_(?=.*_t[\d.]+\.jpg$)")


@dataclass
class AntFrameInfo:
    antframe: int
    pkt_start: int
    pkt_end: int
    part_idx: int
    rel_start: float
    rel_end: float
    rel_center: float
    camera_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build image-aligned mmWave AntFrame MAT files from one raw bin capture."
    )
    parser.add_argument("--cap-dir", type=Path, required=True, help="Capture directory containing the bin and *_part folders.")
    parser.add_argument("--bin-path", type=Path, help="Raw *_mmwave_udp.bin path. Defaults to the only such file under cap-dir.")
    parser.add_argument("--packet-csv", type=Path, help="Output packet-level match CSV path.")
    parser.add_argument("--mapping-csv", type=Path, help="Output image-to-antframe CSV path.")
    parser.add_argument("--mat-out-dir", type=Path, help="Output directory for matched MAT files.")
    parser.add_argument(
        "--export-all-antframes",
        action="store_true",
        help="Export every AntFrame instead of only those selected by the image mapping CSV.",
    )
    return parser.parse_args()


def resolve_bin_path(cap_dir: Path, bin_path: Path | None) -> Path:
    if bin_path is not None:
        return bin_path
    matches = sorted(cap_dir.glob("*_mmwave_udp.bin"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one *_mmwave_udp.bin under {cap_dir}, found {len(matches)}"
        )
    return matches[0]


def build_packet_alignment(bin_path: Path, parts: list[anchor_match.PartData]):
    n_bytes = bin_path.stat().st_size
    n_pkts = n_bytes // anchor_match.PKT
    if n_pkts == 0:
        raise RuntimeError(f"empty bin: {bin_path}")

    mm = np.memmap(
        str(bin_path),
        dtype="<u4",
        mode="r",
        shape=(n_pkts, anchor_match.PKT_WORDS),
    )
    start_mask = (mm[:, 6] == 1).astype(np.int32)
    antframe_arr = np.maximum(np.cumsum(start_mask) - 1, 0)
    gps_cst_all = anchor_match.decode_w12(mm[:, 11])
    del mm

    w12_sec = gps_cst_all.astype(np.int64)
    diff = np.concatenate([[1], np.diff(w12_sec)])
    trans_idx = np.where(diff != 0)[0]
    trans_gps = w12_sec[trans_idx].astype(np.float64)

    anchor_rel, anchor_part_idx = anchor_match.build_anchor_rel_times(trans_idx, trans_gps, parts)
    dense_idx, dense_rel, dense_pi = anchor_match.densify_with_camera_anchors(
        n_pkts, trans_idx, anchor_rel, anchor_part_idx, parts
    )
    rel_time_out, pkt_part_out = anchor_match.interpolate_rel_times(n_pkts, dense_idx, dense_rel, dense_pi)

    cam_names_out = np.empty(n_pkts, dtype=object)
    cam_names_out[:] = ""
    cam_rt_out = np.full(n_pkts, np.nan, dtype=np.float64)
    for pi, part in enumerate(parts):
        if len(part.cam_t) == 0:
            continue
        mask = pkt_part_out == pi
        if not mask.any():
            continue
        pkt_idx = np.where(mask)[0]
        rt = rel_time_out[pkt_idx]
        valid = ~np.isnan(rt)
        if not valid.any():
            continue
        names, crt = anchor_match.nearest_cam_vectorized(rt[valid], part.cam_t, part.cam_n)
        cam_names_out[pkt_idx[valid]] = names
        cam_rt_out[pkt_idx[valid]] = crt

    return antframe_arr, rel_time_out, pkt_part_out, cam_names_out, cam_rt_out


def write_packet_csv(
    out_path: Path,
    antframe_arr: np.ndarray,
    rel_time_out: np.ndarray,
    cam_names_out: np.ndarray,
    cam_rt_out: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rt_strs = np.where(np.isnan(rel_time_out), "", np.char.mod("%.6f", rel_time_out))
    crt_strs = np.where(np.isnan(cam_rt_out), "", np.char.mod("%.4f", cam_rt_out))
    lines = ["antframe,pkt_idx,nav100_rel_time,camera_name,camera_rel_time\n"]
    lines += [
        f"{int(antframe_arr[i])},{i},{rt_strs[i]},{cam_names_out[i]},{crt_strs[i]}\n"
        for i in range(len(antframe_arr))
    ]
    out_path.write_text("".join(lines), encoding="utf-8")


def summarize_antframes(
    antframe_arr: np.ndarray,
    rel_time_out: np.ndarray,
    pkt_part_out: np.ndarray,
    cam_names_out: np.ndarray,
) -> list[AntFrameInfo]:
    starts = np.where(np.concatenate([[True], antframe_arr[1:] != antframe_arr[:-1]]))[0]
    ends = np.concatenate([starts[1:] - 1, [len(antframe_arr) - 1]])
    infos: list[AntFrameInfo] = []

    for ant_idx, (pkt_start, pkt_end) in enumerate(zip(starts, ends)):
        rel_seg = rel_time_out[pkt_start : pkt_end + 1]
        finite = np.isfinite(rel_seg)
        if not finite.any():
            continue
        pkt_part_seg = pkt_part_out[pkt_start : pkt_end + 1]
        valid_parts = pkt_part_seg[pkt_part_seg >= 0]
        part_idx = int(valid_parts[0]) if len(valid_parts) else -1
        if part_idx < 0:
            continue
        rel_valid = rel_seg[finite]
        cam_seg = [c for c in cam_names_out[pkt_start : pkt_end + 1] if c]
        camera_name = cam_seg[len(cam_seg) // 2] if cam_seg else ""
        infos.append(
            AntFrameInfo(
                antframe=ant_idx,
                pkt_start=int(pkt_start),
                pkt_end=int(pkt_end),
                part_idx=part_idx,
                rel_start=float(rel_valid[0]),
                rel_end=float(rel_valid[-1]),
                rel_center=float(rel_valid[len(rel_valid) // 2]),
                camera_name=camera_name,
            )
        )
    return infos


def parse_camera_seq(name: str, fallback_seq: int) -> int:
    m = IMAGE_SEQ_RE.search(name)
    if m:
        return int(m.group(1))
    return fallback_seq


def build_image_bridge_rows(parts: list[anchor_match.PartData], ant_infos: list[AntFrameInfo]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    fallback_seq = 0

    by_part: dict[int, list[AntFrameInfo]] = {}
    for info in ant_infos:
        by_part.setdefault(info.part_idx, []).append(info)

    for part_idx, part in enumerate(parts):
        frames = by_part.get(part_idx, [])
        if not frames or len(part.cam_t) == 0:
            continue
        frames = sorted(frames, key=lambda item: item.rel_center)

        for cam_t, cam_name in zip(part.cam_t, part.cam_n):
            inside = [f for f in frames if f.rel_start <= float(cam_t) <= f.rel_end]
            if inside:
                chosen = min(inside, key=lambda f: abs(f.rel_center - float(cam_t)))
                assignment = "inside"
            else:
                chosen = min(frames, key=lambda f: abs(f.rel_center - float(cam_t)))
                assignment = "nearest"

            rows.append(
                {
                    "image": cam_name,
                    "camera_seq": str(parse_camera_seq(cam_name, fallback_seq)),
                    "camera_t_bag": f"{float(cam_t):.6f}",
                    "antframe": str(chosen.antframe),
                    "mat": "",
                    "mat_fz_start": str(chosen.pkt_start),
                    "mat_fz_end": str(chosen.pkt_end),
                    "assignment": assignment,
                    "part_name": part.name,
                    "antframe_t_start": f"{chosen.rel_start:.6f}",
                    "antframe_t_end": f"{chosen.rel_end:.6f}",
                    "antframe_t_center": f"{chosen.rel_center:.6f}",
                }
            )
            fallback_seq += 1

    rows.sort(key=lambda row: (float(row["camera_t_bag"]), int(row["camera_seq"])))
    return rows


def write_bridge_csv(out_path: Path, rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "camera_seq",
        "camera_t_bag",
        "antframe",
        "mat",
        "mat_fz_start",
        "mat_fz_end",
        "assignment",
        "part_name",
        "antframe_t_start",
        "antframe_t_end",
        "antframe_t_center",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def power_to_db(x: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(x, 1e-3))).astype(np.float32)


def export_antframes_to_mat(
    bin_path: Path,
    out_dir: Path,
    selected_antframes: set[int] | None = None,
) -> dict[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pkt = anchor_match.PKT
    n_total = bin_path.stat().st_size // pkt

    af_start = np.empty(n_total, dtype=np.uint32)
    with bin_path.open("rb") as f:
        chunk = 8192
        done = 0
        while done < n_total:
            m = min(chunk, n_total - done)
            buf = f.read(m * pkt)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(m, pkt)
            af_start[done : done + m] = arr[:, 24:28].copy().view("<u4").ravel()
            done += m

    start_idx = np.where(af_start == 1)[0]
    boundaries = np.r_[start_idx, n_total]
    exported: dict[int, str] = {}

    with bin_path.open("rb") as f:
        for antframe, (i0, i1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if selected_antframes is not None and antframe not in selected_antframes:
                continue

            n_fz = int(i1 - i0)
            f.seek(int(i0) * pkt)
            buf = f.read(n_fz * pkt)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_fz, pkt)

            ant_az = arr[:, 80:84].copy().view("<f4").ravel().astype(np.float64)
            ant_el = arr[:, 84:88].copy().view("<f4").ravel().astype(np.float64)
            lat = arr[:, 52:56].copy().view("<f4").ravel().astype(np.float64)
            lon = arr[:, 48:52].copy().view("<f4").ravel().astype(np.float64)
            hdg = arr[:, 56:60].copy().view("<f4").ravel().astype(np.float64)
            alt = arr[:, 60:64].copy().view("<f4").ravel().astype(np.float64)

            body = arr[:, 256:5632]
            sum_lin = np.frombuffer(body[:, 8 : 8 + 668 * 4].tobytes(), dtype="<f4").reshape(n_fz, 668)
            diff_lin = np.frombuffer(body[:, 2696 : 2696 + 668 * 4].tobytes(), dtype="<f4").reshape(n_fz, 668)
            sum_db_666 = power_to_db(sum_lin)[:, 1 : 1 + N_RANGE_OUT]
            diff_db_666 = power_to_db(diff_lin)[:, 1 : 1 + N_RANGE_OUT]

            el_idx = np.round((ant_el - EL_MIN) / EL_STEP).astype(int)
            valid = (el_idx >= 0) & (el_idx < len(EL_GRID))
            if valid.all():
                grid_use = EL_GRID
            else:
                local_min = float(np.floor(np.min(ant_el) / EL_STEP) * EL_STEP)
                local_max = float(np.ceil(np.max(ant_el) / EL_STEP) * EL_STEP)
                grid_use = np.arange(local_min, local_max + EL_STEP / 2, EL_STEP)
                el_idx = np.round((ant_el - grid_use[0]) / EL_STEP).astype(int)

            data_ori = np.empty((len(grid_use), 1), dtype=object)
            for k in range(len(grid_use)):
                sel = np.where(el_idx == k)[0]
                if len(sel) == 0:
                    el_scalar = np.array([[float(grid_use[k])]], dtype=np.float32)
                    az0 = np.zeros((1, 0), dtype=np.float64)
                    diffd = np.zeros((N_RANGE_OUT, 0), dtype=np.float32)
                    sumd = np.zeros((N_RANGE_OUT, 0), dtype=np.float32)
                    meta = np.zeros((0, 7), dtype=np.float64)
                else:
                    sel = sel[np.argsort(ant_az[sel])]
                    el_scalar = np.array([[float(grid_use[k])]], dtype=np.float32)
                    az0 = ant_az[sel].reshape(1, -1)
                    diffd = diff_db_666[sel].T.copy()
                    sumd = sum_db_666[sel].T.copy()
                    meta = np.zeros((len(sel), 7), dtype=np.float64)
                    meta[:, 1] = lat[sel]
                    meta[:, 2] = lon[sel]
                    meta[:, 3] = hdg[sel]
                    meta[:, 4] = alt[sel]
                    meta[:, 6] = ant_el[sel]

                cell5 = np.empty((1, 5), dtype=object)
                cell5[0, 0] = el_scalar
                cell5[0, 1] = az0
                cell5[0, 2] = diffd
                cell5[0, 3] = sumd
                cell5[0, 4] = meta
                data_ori[k, 0] = cell5

            out_name = f"{bin_path.stem}_AntFrame{antframe:03d}_FZ{int(i0):06d}-{int(i1) - 1:06d}.mat"
            savemat(out_dir / out_name, {"Data_Ori": data_ori}, do_compression=True)
            exported[antframe] = out_name

    return exported


def fill_mat_names(rows: list[dict[str, str]], mat_names: dict[int, str]) -> None:
    for row in rows:
        antframe = int(row["antframe"])
        row["mat"] = mat_names.get(antframe, "")


def main() -> None:
    args = parse_args()
    cap_dir = args.cap_dir.resolve()
    bin_path = resolve_bin_path(cap_dir, args.bin_path.resolve() if args.bin_path else None)

    packet_csv = args.packet_csv.resolve() if args.packet_csv else bin_path.parent / "match_radar_camera_anchor.csv"
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else bin_path.parent / "image_to_antframe_time_aligned.csv"
    mat_out_dir = args.mat_out_dir.resolve() if args.mat_out_dir else bin_path.parent / "mmwave_mat_1218style"

    parts = anchor_match.load_capture(cap_dir)
    if not parts:
        raise RuntimeError(f"no *_part capture folders with nav100/image data found under {cap_dir}")

    antframe_arr, rel_time_out, pkt_part_out, cam_names_out, cam_rt_out = build_packet_alignment(bin_path, parts)
    write_packet_csv(packet_csv, antframe_arr, rel_time_out, cam_names_out, cam_rt_out)

    ant_infos = summarize_antframes(antframe_arr, rel_time_out, pkt_part_out, cam_names_out)
    bridge_rows = build_image_bridge_rows(parts, ant_infos)
    if not bridge_rows:
        raise RuntimeError("no image rows could be matched to any AntFrame")

    if args.export_all_antframes:
        selected_antframes = None
    else:
        selected_antframes = {int(row["antframe"]) for row in bridge_rows}

    mat_names = export_antframes_to_mat(bin_path, mat_out_dir, selected_antframes)
    fill_mat_names(bridge_rows, mat_names)
    write_bridge_csv(mapping_csv, bridge_rows)

    print(f"bin: {bin_path}")
    print(f"parts: {len(parts)}")
    print(f"packet csv: {packet_csv}")
    print(f"bridge csv: {mapping_csv}")
    print(f"mat dir: {mat_out_dir}")
    print(f"matched images: {len(bridge_rows)}")
    print(f"exported mats: {len(mat_names)}")


if __name__ == "__main__":
    main()
