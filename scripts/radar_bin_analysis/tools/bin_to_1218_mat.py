"""
把 bin 中 95 个天线帧导出成 95 个 mat 文件, 严格遵循 1218 Data_Ori 格式:

Data_Ori : cell (n_el, 1)
  每元素 = cell (1, 5):
    {1} EL_scalar      float32 (1,1)        本俯仰角 (度)
    {2} AZ0            float64 (1, n_az)    各方位采样的方位角向量 (度)
    {3} DiffDatadB     float32 (666, n_az)  差路 dB
    {4} SumDatadB      float32 (666, n_az)  和路 dB
    {5} meta           float64 (n_az, 7)    每方位元数据:
        [rangeGate, fLatitude, fLongitude, PlaneCouse, HHeight, PlaneEL, RadarEl]
"""
from __future__ import annotations
import os, sys
import numpy as np
from scipy.io import savemat

BIN = r"D:\Dataset\LH_2026-04-27\bag1.1\with_cameras_capture_20260427_151113_mmwave_udp.bin"
OUT_DIR = r"D:\Dataset\LH_2026-04-27\mmwave_mat_1218style"
os.makedirs(OUT_DIR, exist_ok=True)

PKT = 8624
N_RANGE_FULL = 668     # bin 实际距离单元数
N_RANGE_OUT  = 666     # 与 1218 mat 一致 (去掉首尾各 1 个保护单元)
EL_MIN, EL_MAX, EL_STEP = -10.0, 5.0, 0.5  # 1218 默认的俯仰栅格

el_grid = np.arange(EL_MIN, EL_MAX + EL_STEP/2, EL_STEP)  # 31 个

print(f"[准备] 输出目录: {OUT_DIR}")
print(f"       俯仰栅格: {el_grid[0]}..{el_grid[-1]} step={EL_STEP}  共 {len(el_grid)} 级")

# ============ 第 1 遍:扫描所有 FZ, 拿到 ant_frame_start 位置 ============
N_TOTAL = os.path.getsize(BIN) // PKT
print(f"[扫描] {N_TOTAL} 个 FZ ...")

af_start = np.empty(N_TOTAL, dtype=np.uint32)
with open(BIN, "rb") as f:
    chunk = 8192
    done = 0
    while done < N_TOTAL:
        m = min(chunk, N_TOTAL - done)
        buf = f.read(m * PKT)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(m, PKT)
        af_start[done:done+m] = arr[:, 24:28].copy().view("<u4").ravel()
        done += m

start_idx = np.where(af_start == 1)[0]
# 末尾再补一个 N_TOTAL 当作 sentinel, 用于切片
boundaries = np.r_[start_idx, N_TOTAL]
n_ant_frames = len(start_idx)
print(f"[切分] 发现 {n_ant_frames} 个天线帧, 平均 {N_TOTAL/n_ant_frames:.1f} FZ/天线帧")


# ============ 第 2 遍:逐天线帧加载 + 解码 + 写 mat ============
# 用结构化 dtype 一次性 view 头部
HEAD_FIELDS_OFFSETS = {
    "lon":    (48, "<f4"),
    "lat":    (52, "<f4"),
    "hdg":    (56, "<f4"),
    "alt":    (60, "<f4"),
    "ant_az": (80, "<f4"),
    "ant_el": (84, "<f4"),
    "range_km": (32, "<f4"),
    "n_valid": (36, "<u4"),
    "ts_hmsm": (44, "<u4"),
}

def power_to_db(x: np.ndarray) -> np.ndarray:
    """实数功率谱 -> dB. 与 1218 一致的 caxis [60 150] 量级 -> 10*log10."""
    return (10.0 * np.log10(np.maximum(x, 1e-3))).astype(np.float32)


with open(BIN, "rb") as f:
    for ai, (i0, i1) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        n_fz = int(i1 - i0)
        # 跳到该天线帧首
        f.seek(int(i0) * PKT)
        buf = f.read(n_fz * PKT)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_fz, PKT)

        # 提取每 FZ 的关键标量
        ant_az = arr[:, 80:84].copy().view("<f4").ravel().astype(np.float64)
        ant_el = arr[:, 84:88].copy().view("<f4").ravel().astype(np.float64)
        lat    = arr[:, 52:56].copy().view("<f4").ravel().astype(np.float64)
        lon    = arr[:, 48:52].copy().view("<f4").ravel().astype(np.float64)
        hdg    = arr[:, 56:60].copy().view("<f4").ravel().astype(np.float64)
        alt    = arr[:, 60:64].copy().view("<f4").ravel().astype(np.float64)
        rg_km  = arr[:, 32:36].copy().view("<f4").ravel().astype(np.float64)
        ts     = arr[:, 44:48].copy().view("<u4").ravel().astype(np.int64)

        # 解和路/差路 (offsets 见 parse_mmwave_v2.py 验证过)
        # 回波体: 256..5631
        # 和路:    264..264+668*4 = 264..2936 (字节)
        # 差路: 256+2696..256+2696+668*4 = 2952..5624 (字节)
        body = arr[:, 256:5632]  # (n_fz, 5376)
        sum_lin  = np.frombuffer(body[:, 8:8+668*4].tobytes(), dtype="<f4").reshape(n_fz, 668)
        diff_lin = np.frombuffer(body[:, 2696:2696+668*4].tobytes(), dtype="<f4").reshape(n_fz, 668)
        sum_db  = power_to_db(sum_lin)
        diff_db = power_to_db(diff_lin)
        # 裁到 666 距离 (与 1218 一致, 去掉首尾各 1 个)
        sum_db_666  = sum_db[:, 1:1+N_RANGE_OUT]
        diff_db_666 = diff_db[:, 1:1+N_RANGE_OUT]

        # 把 FZ 按俯仰量化到 EL 栅格
        el_idx = np.round((ant_el - EL_MIN) / EL_STEP).astype(int)
        # 只保留落在 [0, len(el_grid)) 范围内的 FZ
        valid = (el_idx >= 0) & (el_idx < len(el_grid))
        if not valid.all():
            n_drop = int((~valid).sum())
            # 不致命: 该天线帧扫描的俯仰可能超出 1218 默认 -10..+5 范围
            # 我们把超出的也保留, 自动扩展栅格
            this_el_min = float(np.min(ant_el))
            this_el_max = float(np.max(ant_el))
            local_grid = np.arange(
                np.floor(this_el_min/EL_STEP)*EL_STEP,
                np.ceil(this_el_max/EL_STEP)*EL_STEP + EL_STEP/2,
                EL_STEP)
            el_idx = np.round((ant_el - local_grid[0]) / EL_STEP).astype(int)
            grid_use = local_grid
        else:
            grid_use = el_grid

        n_el = len(grid_use)

        # 构造 Data_Ori cell
        data_ori = np.empty((n_el, 1), dtype=object)
        for k in range(n_el):
            sel = np.where(el_idx == k)[0]
            if len(sel) == 0:
                # 该俯仰在这一帧没采到 — 留一个空占位, 但保持 1218 的 5-cell 结构
                el_scalar = np.array([[float(grid_use[k])]], dtype=np.float32)
                az0   = np.zeros((1, 0), dtype=np.float64)
                diffd = np.zeros((N_RANGE_OUT, 0), dtype=np.float32)
                sumd  = np.zeros((N_RANGE_OUT, 0), dtype=np.float32)
                meta  = np.zeros((0, 7), dtype=np.float64)
            else:
                # 同俯仰内按方位排序
                sel = sel[np.argsort(ant_az[sel])]
                el_scalar = np.array([[float(grid_use[k])]], dtype=np.float32)
                az0   = ant_az[sel].reshape(1, -1)              # (1, n_az)
                diffd = diff_db_666[sel].T.copy()               # (666, n_az)
                sumd  = sum_db_666[sel].T.copy()                # (666, n_az)
                # meta: [rangeGate, fLatitude, fLongitude, PlaneCouse, HHeight, PlaneEL, RadarEl]
                # rangeGate: 1218 中是 ~0(?) — 实际可能是距离量程, 用 km 转米数 / 距离格数
                meta = np.zeros((len(sel), 7), dtype=np.float64)
                meta[:, 0] = 0.0                       # rangeGate (1218 全 0, 我们也填 0)
                meta[:, 1] = lat[sel]
                meta[:, 2] = lon[sel]
                meta[:, 3] = hdg[sel]                  # PlaneCouse
                meta[:, 4] = alt[sel]                  # HHeight
                meta[:, 5] = 0.0                       # PlaneEL (载机俯仰, bin 里没有)
                meta[:, 6] = ant_el[sel]               # RadarEl

            cell5 = np.empty((1, 5), dtype=object)
            cell5[0, 0] = el_scalar
            cell5[0, 1] = az0
            cell5[0, 2] = diffd
            cell5[0, 3] = sumd
            cell5[0, 4] = meta
            data_ori[k, 0] = cell5

        # 文件名: 反映该天线帧在 bin 中的 FZ 区间
        out_name = f"mmwave_2026-04-27_151113_AntFrame{ai:03d}_FZ{int(i0):06d}-{int(i1)-1:06d}.mat"
        out_path = os.path.join(OUT_DIR, out_name)
        savemat(out_path, {"Data_Ori": data_ori}, do_compression=True)

        if ai < 3 or ai == n_ant_frames - 1 or (ai+1) % 10 == 0:
            print(f"  [{ai+1:3d}/{n_ant_frames}] FZ {i0}..{i1-1} ({n_fz:5d} 帧)  "
                  f"el {ant_el.min():+6.2f}..{ant_el.max():+6.2f}  "
                  f"az {ant_az.min():+7.2f}..{ant_az.max():+7.2f}  "
                  f"-> {out_name}")

print(f"\n[完成] 共导出 {n_ant_frames} 个 mat 到 {OUT_DIR}")
