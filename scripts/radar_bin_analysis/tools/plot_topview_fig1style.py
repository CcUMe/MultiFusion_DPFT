"""按 frameFindAndTimeFreAnalMain.m 的 Figure 1 风格输出单张 PPI 俯视图。

.m 中 Figure 1 的关键代码：
    surf(AZ, g, sumDatadB');  shading interp;  colormap jet;
    caxis([maxP-60, maxP]);   axis([-60 60 0 max(g)]);   view(0,90);
其中 g = 1:iLenA 为距离单元索引（不是米）。

我们的 1218-style mat 含 31 个 EL 层。.m 里用户输入一个 EL（默认 -1）查看
单层；这里为了批量化，自动选每个 mat 中“az 采样点最多”的那一层（即主扫描
面），输出一张 PPI PNG，与 mat 同名，保存到 mmwave_topview_fig1style/。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import loadmat

SRC = Path(r"D:\Dataset\LH_2026-04-27\mmwave_mat_1218style")
DST = Path(r"D:\Dataset\LH_2026-04-27\mmwave_topview_fig1style")
DST.mkdir(parents=True, exist_ok=True)

CAXIS_SPAN = 60.0
XLIM = (-60.0, 60.0)


def pick_layer(data):
    """返回 (k, el_val, az, sumdb, n_az)，sumdb 形状 (range, az)。
    选 az 采样点最多的 EL 层。"""
    best = None
    n_el = data.shape[0]
    for k in range(n_el):
        sub = data[k, 0].ravel()
        el_val = float(sub[0].ravel()[0])
        az = np.asarray(sub[1]).ravel().astype(float)
        sumdb = np.asarray(sub[3]).astype(float)
        if sumdb.size == 0 or az.size == 0:
            continue
        n_az = az.size
        if best is None or n_az > best[4]:
            best = (k, el_val, az, sumdb, n_az)
    return best


def render_one(mat_path: Path, out_path: Path):
    data = loadmat(str(mat_path))["Data_Ori"]
    pick = pick_layer(data)
    if pick is None:
        print(f"[skip] {mat_path.name} 无可绘制 EL 层")
        return
    k, el_val, az, sumdb, n_az = pick
    rows = sumdb.shape[0]
    g = np.arange(1, rows + 1)
    max_p = float(sumdb.max())
    vmin, vmax = max_p - CAXIS_SPAN, max_p

    # AZ 升序排序，使 pcolormesh 显示正确
    order = np.argsort(az)
    az_s = az[order]
    sumdb_s = sumdb[:, order]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    # shading='gouraud' 等价 MATLAB 的 shading interp（顶点间双线性插值），
    # 避免 flat 着色产生的方块感。
    im = ax.pcolormesh(az_s, g, sumdb_s, vmin=vmin, vmax=vmax,
                       shading="gouraud", cmap="jet")
    ax.set_xlim(*XLIM)
    ax.set_ylim(g.min(), g.max())
    ax.set_xlabel("AZ (deg)")
    ax.set_ylabel("g - range cell index")
    ax.set_title(
        f"PPI Top-View  EL={el_val:+.2f}deg  (layer #{k+1}/{data.shape[0]}, "
        f"n_az={n_az}, range_cells={rows}, maxP={max_p:.1f}dB)\n{mat_path.stem}",
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)  caxis=[maxP-60, maxP]")

    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    mats = sorted(SRC.glob("*.mat"))
    print(f"[info] 待处理 mat 数: {len(mats)}, 输出到 {DST}")
    for i, mp in enumerate(mats):
        op = DST / (mp.stem + ".png")
        try:
            render_one(mp, op)
        except Exception as e:
            print(f"[warn] {mp.name} 失败: {e}")
            continue
        if (i + 1) % 10 == 0 or i == len(mats) - 1:
            print(f"  [{i+1}/{len(mats)}] {op.name}")
    print("[done] PPI 俯视图渲染完成。")


if __name__ == "__main__":
    main()
