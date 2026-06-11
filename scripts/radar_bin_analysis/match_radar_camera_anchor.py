"""
match_radar_camera_anchor.py  —  方法2：W12 秒边界锚点 + 包内分数插值

逻辑：
  W12 每秒跳变一次，同一 GPS 整秒内所有包的 W12 值相同，
  方法1 因此对同一秒内所有包给出相同 rel_time。

  本方法利用相邻秒边界之间包的序号来做线性插值，获得亚秒精度：

    ① 找出所有 W12 整秒跳变处（即每个新 GPS 秒的第一个包索引）→ 锚点序列
    ② 对每个锚点，通过 nav100 插值计算 rel_time，
       并尝试在 ±150ms 窗口内用最近相机帧的时间戳微调（camera-snap）
    ③ 在相邻两锚点之间，按包序号比例线性插值 rel_time（亚秒分辨率）
    ④ 按插值后 rel_time 在本 part 相机列表中找最近帧

  对 L: 盘数据（W12 稳定，比率 ≤1.10），两种方法结果差异约 ≤ 1ms，
  但本方法可给每包提供独立的、有物理意义的时间估计。

输出 CSV（存于每个 bin 文件同级目录，文件名与脚本名相同）：
  antframe        : 该包所属的 AntFrame 序号（0-based）
  pkt_idx         : 包在 bin 文件中的序号（0-based）
  nav100_rel_time : 锚点插值后的相对时间（秒）
  camera_name     : 最近相机帧文件名
  camera_rel_time : 该相机帧的相对时间（秒）
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np

# ── 常量配置 ─────────────────────────────────────────────────────────────────
PKT       = 8624
PKT_WORDS = PKT // 4

CAM_SUBDIR   = "hikrobot_camera__DA8679037__image_raw"
CAM_SNAP_WIN = 0.030     # 相机 snap 窗口（秒）：在锚点 ±30ms 内找相机帧微调
                         # 30fps 帧间距 ~33ms，±30ms ≈ ±1帧，避免 nav100 误差
                         # 较大时 snap 到错帧（隔帧偏移 ~33ms）

L_ROOT = Path("/mnt/disk1/low_altitude/LH_data_all_sensor/")

_RE_CAM_T  = re.compile(r'_t([\d.]+)\.jpg$')

SCRIPT_STEM = Path(__file__).stem   # "match_radar_camera_anchor"


# ── 工具函数（与 interp 版本相同）────────────────────────────────────────────

def decode_w12(raw_arr: np.ndarray) -> np.ndarray:
    """W12 原始值已是 GPS_CST 整数秒（北京时间当日秒数），直接返回。"""
    return raw_arr.astype(np.float64)


def load_nav_csv(csv_path: Path):
    gps_list, rel_list = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                g = (int(row['gps_hour'])   * 3600
                     + int(row['gps_minute']) * 60
                     + int(row['gps_second'])
                     + int(row['gps_millisecond']) / 1000.0
                     + 28800.0)
                r = float(row['relative_time_sec'])
                gps_list.append(g)
                rel_list.append(r)
            except (KeyError, ValueError):
                pass
    if not gps_list:
        return None
    g_arr = np.array(gps_list, dtype=np.float64)
    r_arr = np.array(rel_list, dtype=np.float64)
    order = np.argsort(g_arr)
    return g_arr[order], r_arr[order]


def load_cams(cam_dir: Path):
    times, names = [], []
    if not cam_dir.exists():
        return np.empty(0, dtype=np.float64), []
    for f in cam_dir.iterdir():
        m = _RE_CAM_T.search(f.name)
        if m:
            times.append(float(m.group(1)))
            names.append(f.name)
    if not times:
        return np.empty(0, dtype=np.float64), []
    order  = np.argsort(times)
    t_arr  = np.array(times, dtype=np.float64)[order]
    n_list = [names[i] for i in order]
    return t_arr, n_list


def nearest_cam_vectorized(rt_arr: np.ndarray, cam_t: np.ndarray, cam_n: list):
    idx   = np.searchsorted(cam_t, rt_arr).clip(0, len(cam_t) - 1)
    idx_l = (idx - 1).clip(0, len(cam_t) - 1)
    best  = np.where(
        np.abs(cam_t[idx_l] - rt_arr) < np.abs(cam_t[idx] - rt_arr),
        idx_l, idx
    )
    cam_n_arr = np.array(cam_n, dtype=object)
    return cam_n_arr[best], cam_t[best]


# ── PartData ─────────────────────────────────────────────────────────────────

class PartData:
    __slots__ = ['name', 'gps_arr', 'rel_arr', 'cam_t', 'cam_n', 'gps_min', 'gps_max']

    def __init__(self, name: str):
        self.name    = name
        self.gps_arr = np.empty(0, dtype=np.float64)
        self.rel_arr = np.empty(0, dtype=np.float64)
        self.cam_t   = np.empty(0, dtype=np.float64)
        self.cam_n: list[str] = []
        self.gps_min = np.inf
        self.gps_max = -np.inf


def load_capture(cap_dir: Path) -> list[PartData]:
    parts: list[PartData] = []
    for pdir in sorted(cap_dir.iterdir()):
        if not pdir.is_dir() or '_part' not in pdir.name:
            continue

        pd = PartData(pdir.name)
        g_segs, r_segs, ct_segs, cn_all = [], [], [], []

        for seg in sorted(pdir.iterdir()):
            if not seg.is_dir() or not seg.name.startswith('segment_'):
                continue
            nav_csv = seg / 'nav100_state' / 'nav100__state' / 'nav100__state.csv'
            if not nav_csv.exists():
                continue
            res = load_nav_csv(nav_csv)
            if res is None:
                continue
            g_segs.append(res[0])
            r_segs.append(res[1])
            ct, cn = load_cams(seg / 'images' / CAM_SUBDIR)
            ct_segs.append(ct)
            cn_all.extend(cn)

        if not g_segs:
            continue

        g_all = np.concatenate(g_segs)
        r_all = np.concatenate(r_segs)
        order = np.argsort(g_all)
        pd.gps_arr = g_all[order]
        pd.rel_arr = r_all[order]

        if ct_segs:
            ct_all = np.concatenate(ct_segs)
            co = np.argsort(ct_all)
            pd.cam_t = ct_all[co]
            pd.cam_n = [cn_all[i] for i in co]

        pd.gps_min = float(pd.gps_arr[0])
        pd.gps_max = float(pd.gps_arr[-1])
        parts.append(pd)

    return parts


# ── 锚点构建 + 亚秒插值 ───────────────────────────────────────────────────────

def build_anchor_rel_times(
    trans_idx: np.ndarray,
    trans_gps: np.ndarray,
    parts: list[PartData],
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个锚点（W12 整秒跳变位置）计算 rel_time 和所属 part。
    先用 nav100 线性插值，再尝试用相机帧 snap 微调（±CAM_SNAP_WIN 秒内）。

    返回：
        anchor_rel      shape (n_trans,) float64，nan 表示在任何 part 范围外
        anchor_part_idx shape (n_trans,) int32，-1 表示无对应 part
    """
    n_trans = len(trans_idx)
    anchor_rel      = np.full(n_trans, np.nan,  dtype=np.float64)
    anchor_part_idx = np.full(n_trans, -1,       dtype=np.int32)

    for pi, p in enumerate(parts):
        if len(p.gps_arr) < 2:
            continue
        in_part = (trans_gps >= p.gps_min) & (trans_gps <= p.gps_max)
        if not in_part.any():
            continue
        anchor_rel[in_part] = np.interp(
            trans_gps[in_part], p.gps_arr, p.rel_arr
        )
        anchor_part_idx[in_part] = pi

        # Camera-snap：在 ±CAM_SNAP_WIN 内找最近相机帧，用其时间代替 nav100 插值结果
        if len(p.cam_t) == 0:
            continue
        snap_mask = in_part & ~np.isnan(anchor_rel)
        for k in np.where(snap_mask)[0]:
            rt = anchor_rel[k]
            ci = int(np.searchsorted(p.cam_t, rt))
            # 检查右侧和左侧候选
            for c in [ci, ci - 1]:
                c = max(0, min(c, len(p.cam_t) - 1))
                if abs(p.cam_t[c] - rt) <= CAM_SNAP_WIN:
                    anchor_rel[k] = p.cam_t[c]   # snap 到相机帧时刻
                    break

    return anchor_rel, anchor_part_idx


def interpolate_rel_times(
    n_pkts: int,
    trans_idx: np.ndarray,
    anchor_rel: np.ndarray,
    anchor_part_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在相邻锚点之间按包序号比例线性插值 rel_time。

    返回：
        rel_time_out  shape (n_pkts,) float64
        pkt_part_out  shape (n_pkts,) int32（每包所属 part 序号，-1 表示未覆盖）
    """
    rel_time_out  = np.full(n_pkts, np.nan, dtype=np.float64)
    pkt_part_out  = np.full(n_pkts, -1,     dtype=np.int32)

    n_trans = len(trans_idx)

    for k in range(n_trans):
        i_start = int(trans_idx[k])
        i_end   = int(trans_idx[k + 1]) if k + 1 < n_trans else n_pkts
        n_int   = i_end - i_start

        r0 = anchor_rel[k]
        pi = anchor_part_idx[k]

        if np.isnan(r0) or pi < 0:
            continue

        # 是否有下一个有效锚点（且属于同一 part）
        if (k + 1 < n_trans
                and not np.isnan(anchor_rel[k + 1])
                and anchor_part_idx[k + 1] == pi):
            r1   = anchor_rel[k + 1]
            frac = np.arange(n_int, dtype=np.float64) / n_int
            rel_time_out[i_start:i_end] = r0 + frac * (r1 - r0)
        else:
            # 只有起点锚，后续分配同一 rel_time（与方法1退化一致）
            rel_time_out[i_start:i_end] = r0

        pkt_part_out[i_start:i_end] = pi

    return rel_time_out, pkt_part_out


def densify_with_camera_anchors(
    n_pkts: int,
    trans_idx: np.ndarray,
    anchor_rel: np.ndarray,
    anchor_part_idx: np.ndarray,
    parts: list[PartData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在相邻 W12 锚点之间插入相机帧时刻作为密集锚点。

    原理：
      当前 W12 锚点间距 ≈ 700 包（1 秒），相机 30fps → 每 23 包一帧。
      以当前 W12 插值结果估算每个相机帧对应的包索引，将
      (pkt_idx, cam_t) 作为额外锚点。
      结果：插值间隔从 ~700 包缩短到 ~23 包，残差误差由 ±nav100_jitter
      降低为 ≤ 半帧（~16ms），边界跨帧误判显著减少。

    返回合并并排序去重后的 (anchor_idx, anchor_rel, anchor_pi)。
    已有 camera-snap 保证 W12 锚点贴近相机帧，新锚点直接使用相机帧时刻，
    两者一致性良好。
    """
    extra_idx: list[int]   = []
    extra_rel: list[float] = []
    extra_pi:  list[int]   = []

    n_trans = len(trans_idx)
    for k in range(n_trans - 1):
        r0  = anchor_rel[k]
        r1  = anchor_rel[k + 1]
        pi  = int(anchor_part_idx[k])
        pi1 = int(anchor_part_idx[k + 1])

        if np.isnan(r0) or np.isnan(r1) or pi < 0 or pi1 != pi or r1 <= r0:
            continue

        i_start  = int(trans_idx[k])
        i_end    = int(trans_idx[k + 1])
        span_pkt = i_end - i_start
        span_t   = r1 - r0

        p = parts[pi]
        if len(p.cam_t) == 0:
            continue

        # 找 (r0, r1) 内的相机帧（不含端点，端点已由 W12 锚点覆盖）
        ci_lo = int(np.searchsorted(p.cam_t, r0, side='right'))
        ci_hi = int(np.searchsorted(p.cam_t, r1, side='left'))

        for ci in range(ci_lo, ci_hi):
            t_cam = float(p.cam_t[ci])
            frac  = (t_cam - r0) / span_t
            i_cam = int(round(i_start + frac * span_pkt))
            # 严格在区间内，不覆盖 W12 边界
            if i_start < i_cam < i_end:
                extra_idx.append(i_cam)
                extra_rel.append(t_cam)
                extra_pi.append(pi)

    if not extra_idx:
        return trans_idx, anchor_rel, anchor_part_idx

    all_idx = np.concatenate([trans_idx,       np.array(extra_idx, dtype=np.int64)])
    all_rel = np.concatenate([anchor_rel,      np.array(extra_rel, dtype=np.float64)])
    all_pi  = np.concatenate([anchor_part_idx, np.array(extra_pi,  dtype=np.int32)])

    order   = np.argsort(all_idx, kind='stable')
    all_idx = all_idx[order]
    all_rel = all_rel[order]
    all_pi  = all_pi[order]

    # 去重：同一包索引保留第一个（W12 锚点已在前，优先保留）
    _, keep = np.unique(all_idx, return_index=True)
    return all_idx[keep], all_rel[keep], all_pi[keep]


# ── 主处理函数 ───────────────────────────────────────────────────────────────

def process_bin(bin_path: Path, parts: list[PartData]) -> Path:
    n_bytes = bin_path.stat().st_size
    n_pkts  = n_bytes // PKT
    if n_pkts == 0:
        return None

    mm = np.memmap(str(bin_path), dtype='<u4', mode='r', shape=(n_pkts, PKT_WORDS))

    start_mask   = (mm[:, 6] == 1).astype(np.int32)
    antframe_arr = np.maximum(np.cumsum(start_mask) - 1, 0)
    gps_cst_all  = decode_w12(mm[:, 11])
    del mm

    # ── 找 W12 整秒跳变锚点 ──
    w12_sec   = gps_cst_all.astype(np.int64)           # 已经是整秒
    diff      = np.concatenate([[1], np.diff(w12_sec)])
    trans_idx = np.where(diff != 0)[0]                  # 每个新 GPS 秒的第一包
    trans_gps = w12_sec[trans_idx].astype(np.float64)

    # ── 为锚点计算 rel_time（nav100 插值 + 相机 snap 微调）──
    anchor_rel, anchor_part_idx = build_anchor_rel_times(trans_idx, trans_gps, parts)

    # ── 用相机帧时刻密化锚点（~700包/锚 → ~23包/锚）──
    dense_idx, dense_rel, dense_pi = densify_with_camera_anchors(
        n_pkts, trans_idx, anchor_rel, anchor_part_idx, parts
    )

    # ── 包内亚秒线性插值（使用密集锚点）──
    rel_time_out, pkt_part_out = interpolate_rel_times(
        n_pkts, dense_idx, dense_rel, dense_pi
    )

    # ── 相机匹配（按 part 分组，避免跨 part 混匹）──
    cam_names_out = np.empty(n_pkts, dtype=object)
    cam_names_out[:] = ''
    cam_rt_out = np.full(n_pkts, np.nan, dtype=np.float64)

    for pi, p in enumerate(parts):
        if len(p.cam_t) == 0:
            continue
        mask = (pkt_part_out == pi)
        if not mask.any():
            continue
        pki   = np.where(mask)[0]
        rt    = rel_time_out[pki]
        valid = ~np.isnan(rt)
        if not valid.any():
            continue
        names, crt = nearest_cam_vectorized(rt[valid], p.cam_t, p.cam_n)
        cam_names_out[pki[valid]] = names
        cam_rt_out[pki[valid]]    = crt

    # ── 写出 CSV（批量构建行列表，一次性写入）──
    out_path = bin_path.parent / f'{SCRIPT_STEM}.csv'
    rt_strs  = np.where(np.isnan(rel_time_out), '', np.char.mod('%.6f', rel_time_out))
    crt_strs = np.where(np.isnan(cam_rt_out),   '', np.char.mod('%.4f', cam_rt_out))
    lines = ['antframe,pkt_idx,nav100_rel_time,camera_name,camera_rel_time\n']
    lines += [
        f'{antframe_arr[i]},{i},{rt_strs[i]},{cam_names_out[i]},{crt_strs[i]}\n'
        for i in range(n_pkts)
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return out_path


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    bin_files = sorted(L_ROOT.rglob('*_mmwave_udp.bin'))
    print(f'共找到 {len(bin_files)} 个 bin 文件\n')

    for bin_path in bin_files:
        cap_dir = bin_path.parent
        print(f'[{cap_dir.parent.name}/{cap_dir.name}]', end='  ', flush=True)

        parts = load_capture(cap_dir)
        if not parts:
            print('⚠ 无 part 目录，跳过')
            continue

        n_pkts = bin_path.stat().st_size // PKT
        print(f'{n_pkts} 包 | {len(parts)} part(s) | 锚点数≈{n_pkts // 700}', flush=True)

        out = process_bin(bin_path, parts)
        if out:
            print(f'  → 写出: {out.name}')

    print('\n全部完成。')


if __name__ == '__main__':
    main()
