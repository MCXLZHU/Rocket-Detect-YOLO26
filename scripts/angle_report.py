# -*- coding: utf-8 -*-
"""把测角结果画成时间线图, 并输出分阶段汇总表。

产物: runs/angle/angle_timeline.png + 控制台汇总
用法: python scripts/angle_report.py
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
(CACHE / "mpl").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = PROJECT / "runs" / "angle"


def load():
    d = json.loads((OUT / "angles.json").read_text(encoding="utf-8"))
    fr = d["frames"]
    fps = d["info"]["fps"]
    t = np.array([r["frame"] / fps for r in fr])
    phi = np.array([r["phi_deg"] if r["phi_deg"] is not None else np.nan
                    for r in fr])
    ok = np.array([bool(r["ok"]) for r in fr])
    sm = np.array([r["phi_smooth"] if r["phi_smooth"] is not None else np.nan
                   for r in fr])
    rel = np.array([r["phi_rel"] if r["phi_rel"] is not None else np.nan
                    for r in fr])
    wb = np.array([r["w_body_px"] if r["w_body_px"] is not None else np.nan
                   for r in fr])
    wbox = np.array([r["w_box_px"] if r["w_box_px"] is not None else np.nan
                     for r in fr])
    conf = np.array([r["conf"] for r in fr])
    nb = np.array([r["n_bands"] for r in fr])
    return d, t, phi, ok, sm, rel, wb, wbox, conf, nb


def main():
    d, t, phi, ok, sm, rel, wb, wbox, conf, nb = load()
    fps = d["info"]["fps"]
    ref = d["info"]["ref"]
    print(f"φ_ref(45-66s) = {ref:+.3f}°")

    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    ax[0].axvspan(11, 40, color="#ffe9b0", alpha=0.6, zorder=0,
                  label="有姿态信息的下降段 11-40s")
    ax[0].axvspan(40, 69.4, color="#d8e8d8", alpha=0.6, zorder=0,
                  label="已落地静止 40-69s")
    ax[0].plot(t[ok], phi[ok], ".", ms=4, color="#1f77b4", label="单帧测量 φ")
    ax[0].plot(t, sm, "-", lw=1.3, color="#d62728", label="因果平滑 φ_smooth")
    ax[0].axhline(ref, color="k", ls="--", lw=1.0,
                  label=f"落地基准 φ_ref={ref:+.2f}°")
    ax[0].axhline(0, color="#888", lw=0.6)
    ax[0].set_ylabel("φ (度)\n正=顶端右倾")
    ax[0].legend(loc="upper right", fontsize=8, ncol=2)
    ax[0].set_title("箭体倾角(相对图像竖直) —— 第 3 步输出")
    ax[0].grid(alpha=0.25)

    ax[1].plot(t[ok], rel[ok], "-", lw=1.2, color="#2ca02c", label="相对倾角 φ−φ_ref")
    ax[1].axhline(0, color="#888", lw=0.6)
    ax[1].set_ylabel("相对倾角 (度)")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].grid(alpha=0.25)
    ax[1].set_title("相对倾角(假定着陆时竖直、相机静止)")

    ax[2].plot(t, wbox, ".", ms=2.5, color="#bbbbbb", label="YOLO 框宽")
    ax[2].plot(t[ok], wb[ok], ".", ms=3.5, color="#ff7f0e", label="拟合筒身宽度")
    ax[2].set_ylabel("宽度 (px)")
    ax[2].legend(loc="upper right", fontsize=8)
    ax[2].grid(alpha=0.25)
    ax[2].set_title("框宽 vs 拟合筒身宽度(两者之比≈0.5 ⇒ 框被支腿撑大)")

    st = np.where(ok, np.where(nb >= 12, 2, 1), 0)
    cmap = np.array([[0.75, 0.75, 0.75, 1.0], [1.0, 0.75, 0.3, 1.0],
                     [0.2, 0.65, 0.3, 1.0]])
    ax[3].vlines(t, 0, 1, colors=cmap[st], lw=1.2)
    ax[3].plot([], [], lw=6, color=cmap[2], label="有效(段数≥12)")
    ax[3].plot([], [], lw=6, color=cmap[1], label="有效但段数少")
    ax[3].plot([], [], lw=6, color=cmap[0], label="无效")
    ax[3].set_ylim(0, 1)
    ax[3].set_yticks([])
    ax[3].set_xlabel("时间 (s)")
    ax[3].legend(loc="upper right", fontsize=8, ncol=3)
    ax[3].grid(alpha=0.25)
    ax[3].set_title("测角有效性(配合 tracker 的 angle_ok 门控)")
    fig.tight_layout()
    fig.savefig(OUT / "angle_timeline.png", dpi=130)
    print(f"-> {OUT / 'angle_timeline.png'}")

    print()
    print("分阶段汇总(有效帧)")
    print(f"  {'时段':>12} {'帧数':>6} {'φ中位':>8} {'φ均值':>8} {'σ':>7} "
          f"{'极差':>7} {'w_body中位':>10}")
    for lo, hi in ((11, 15), (15, 20), (20, 25), (25, 30), (30, 35),
                   (35, 40), (40, 45), (45, 69)):
        m = ok & (t >= lo) & (t < hi)
        if m.sum() < 3:
            continue
        v = phi[m]
        print(f"  {f'{lo}-{hi}s':>12} {m.sum():>6} {np.median(v):>+8.2f} "
              f"{np.mean(v):>+8.2f} {np.std(v):>7.3f} "
              f"{v.max() - v.min():>7.2f} {np.nanmedian(wb[m]):>10.1f}")


if __name__ == "__main__":
    main()
