# -*- coding: utf-8 -*-
"""第 4 步结果可视化: 时间线图 + 标注视频(画上"竖直参考线"与"图像上边缘参考线")。

产物: runs/attitude/attitude_timeline.png + attitude_ref.mp4
用法: python scripts/attitude_report.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames  # noqa: E402

DIAG = PROJECT / "runs" / "diag"
ANG = PROJECT / "runs" / "angle"
OUT = PROJECT / "runs" / "attitude"


def load():
    d = json.loads((OUT / "attitude.json").read_text(encoding="utf-8"))
    fr = d["frames"]
    fps = d["info"]["fps"]
    g = lambda k: np.array([r[k] if r[k] is not None else np.nan for r in fr])  # noqa: E731
    return d, fps, g


def main():
    d, fps, g = load()
    t = g("t")
    phi = g("phi_deg")
    ok = np.array([bool(r["ok"]) for r in d["frames"]])
    phitop = g("phi_vs_top")
    phirel = g("phi_rel")
    roll = g("cam_roll")
    ncc = g("cam_ncc")
    conf = g("conf")
    ref = d["info"]["ref"]

    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    ax[0].axvspan(11, 40, color="#ffe9b0", alpha=0.6, zorder=0,
                  label="下降段 11-40s")
    ax[0].axvspan(40, 69.4, color="#d8e8d8", alpha=0.6, zorder=0,
                  label="落地静止 40-69s")
    ax[0].plot(t[ok], phi[ok], ".", ms=4, color="#1f77b4",
               label="φ: 相对图像**竖直**的夹角")
    ax[0].plot(t[ok], phitop[ok], ".", ms=3, color="#ff7f0e",
               label="90−|φ|: 相对图像**上边缘**的夹角")
    ax[0].axhline(90, color="#888", lw=0.6, ls=":")
    ax[0].set_ylabel("角度 (度)")
    ax[0].legend(loc="center right", fontsize=8)
    ax[0].set_title("第 4 步主输出: 箭体与图像方向的夹角")
    ax[0].grid(alpha=0.25)

    ax[1].plot(t[ok], phirel[ok], "-", lw=1.2, color="#2ca02c")
    ax[1].axhline(0, color="#888", lw=0.6)
    ax[1].set_ylabel("相对倾角 (度)")
    ax[1].set_title(f"相对倾角 φ(t) − φ_ref  (φ_ref={ref:+.2f}°, 落地段实测竖直)")
    ax[1].grid(alpha=0.25)

    ax[2].plot(t, roll, "-", lw=0.8, color="#d62728", label="相机滚转(直接对参考帧配准)")
    ax[2].plot(t, ncc, "-", lw=0.8, color="#7f7f7f", alpha=0.8,
               label="背景配准 NCC")
    ax[2].axhline(0.6, color="k", ls="--", lw=0.8, label="NCC 可信下限 0.6")
    ax[2].set_ylabel("度 / NCC")
    ax[2].legend(loc="center right", fontsize=8)
    ax[2].set_title("相机运动核查: 落地段滚转 +0.03°±0.06° ⇒ 静止; 下降段 σ≈0.31°")
    ax[2].grid(alpha=0.25)

    st = np.where(ok, np.where(conf >= 0.5, 2, 1), 0)
    cmap = np.array([[0.78, 0.78, 0.78, 1.0], [1.0, 0.75, 0.3, 1.0],
                     [0.2, 0.65, 0.3, 1.0]])
    ax[3].vlines(t, 0, 1, colors=cmap[st], lw=1.2)
    ax[3].plot([], [], lw=6, color=cmap[2], label="有效 conf≥0.5")
    ax[3].plot([], [], lw=6, color=cmap[1], label="有效 conf<0.5")
    ax[3].plot([], [], lw=6, color=cmap[0], label="无效")
    ax[3].set_ylim(0, 1)
    ax[3].set_yticks([])
    ax[3].set_xlabel("时间 (s)")
    ax[3].legend(loc="upper right", fontsize=8, ncol=3)
    ax[3].grid(alpha=0.25)
    ax[3].set_title("测角有效性")
    fig.tight_layout()
    fig.savefig(OUT / "attitude_timeline.png", dpi=130)
    print(f"-> {OUT / 'attitude_timeline.png'}")

    # ---------------- 标注视频: 画竖直参考线与上边缘参考线 ----------------
    meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    W, H = meta["meta"]["W"], meta["meta"]["H"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                        bound_wh=(W, H))
    # 注意: phi_vs_top 在 attitude.json 里, 不在 angles.json 里
    attmap = {int(r["frame"]): r for r in d["frames"]}
    vpath = PROJECT / meta["meta"]["video"]
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        tmp = CACHE / "tmp" / "rep.mp4"
        shutil.copy(str(vpath), str(tmp))
        cap = cv2.VideoCapture(str(tmp))
    writer = cv2.VideoWriter(str(OUT / "attitude_ref.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = 0
    while True:
        okf, fr = cap.read()
        if not okf or n >= len(outs):
            break
        o = outs[n]
        if o.has and o.box is not None:
            b = o.box
            cx = 0.5 * (b[0] + b[2])
            y0, y1 = int(b[1]), int(b[3])
            # 图像竖直参考(白虚线)
            for yy in range(y0, y1, 12):
                cv2.line(fr, (int(cx), yy), (int(cx), min(yy + 6, y1)),
                         (255, 255, 255), 1)
            # 图像水平参考(灰虚线, 画在框顶)
            for xx in range(int(b[0]), int(b[2]), 12):
                cv2.line(fr, (xx, y0), (min(xx + 6, int(b[2])), y0),
                         (180, 180, 180), 1)
            a = attmap.get(n)
            if a and a["ok"] and a["phi_deg"] is not None:
                # 箭体轴线(红)
                v = a
                ph = math.radians(v["phi_deg"])
                L = (y1 - y0) * 0.9
                x1 = int(cx + L * math.sin(ph))
                y_1 = int(y1 - L * math.cos(ph))
                cv2.line(fr, (int(cx), y1), (x1, y_1), (0, 0, 255), 2,
                         cv2.LINE_AA)
                txt = (f"phi_vs_vertical={v['phi_deg']:+.2f}  "
                       f"vs_top_edge={v['phi_vs_top']:.2f}  "
                       f"rel={v['phi_rel']:+.2f}")
                cv2.putText(fr, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(fr, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(fr, f"f{n} t={n / fps:.1f}s", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        writer.write(fr)
        n += 1
    cap.release()
    writer.release()
    print(f"-> {OUT / 'attitude_ref.mp4'}")


if __name__ == "__main__":
    main()
