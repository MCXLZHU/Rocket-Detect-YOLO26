# -*- coding: utf-8 -*-
"""把稳定化结果画出来: 标注视频 + 时间线曲线。

  runs/diag/tracked.mp4   —— 稳定框(青) + ROI(黄) + 原始被接受框(绿细) + 状态
  runs/diag/timeline.png  —— 原始框 vs 稳定框 的上/下边沿与高度曲线

复用已存盘的检测结果(runs/diag/dets_iou70.json), 不重新推理, 无需 GPU。
用法: python scripts/track_video.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE / "yolo")

import cv2  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DIAG = PROJECT / "runs" / "diag"


def open_cap(path: Path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "src.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def main():
    d = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    FH = [r["b"] for r in d["frames"]]
    meta = d["meta"]
    W, H = meta["W"], meta["H"]
    fps = meta["fps"]
    video = PROJECT / meta["video"]

    cfg = TrackerConfig()
    outs = track_frames(FH, cfg, fps, bound_wh=(W, H))
    print(f"跟踪完成, {len(outs)} 帧")

    # ---------------- 标注视频 ----------------
    cap = open_cap(video)
    writer = cv2.VideoWriter(str(DIAG / "tracked.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    COL_STAB = (255, 255, 0)     # 稳定框 青
    COL_ROI = (0, 200, 255)      # ROI 黄
    COL_RAW = (0, 220, 0)        # 原始框 绿
    n = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n >= len(outs):
            break
        o = outs[n]
        if o.has and o.roi is not None:
            x1, y1, x2, y2 = [int(v) for v in o.roi]
            cv2.rectangle(img, (x1, y1), (x2, y2), COL_ROI, 1)
            x1, y1, x2, y2 = [int(v) for v in o.box]
            col = COL_STAB if o.state == "OBSERVED" else (128, 128, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            if o.raw_box is not None:
                a = [int(v) for v in o.raw_box]
                cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), COL_RAW, 1)
            txt = (f"{o.state}  s={o.score:.2f}  h={o.box[3] - o.box[1]:.0f}"
                   + ("  [MODE]" if o.mode_switched else "")
                   + ("" if o.angle_ok else "  [no-angle]"))
            cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"f{n} t={n / fps:.1f}s", (8, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA)
        writer.write(img)
        n += 1
        if n % 500 == 0:
            print(f"  {n} 帧", flush=True)
    cap.release()
    writer.release()
    print(f"标注视频 -> {DIAG / 'tracked.mp4'}")

    # ---------------- 时间线曲线 ----------------
    raw_y1, raw_y2, trk_y1, trk_y2, states, swit = [], [], [], [], [], []
    for i, o in enumerate(outs):
        bs = [b for b in FH[i] if b["c"] == 1 and b["s"] >= 0.25]
        if bs:
            b = max(bs, key=lambda b: b["s"])["x"]
            raw_y1.append(b[1])
            raw_y2.append(b[3])
        else:
            raw_y1.append(np.nan)
            raw_y2.append(np.nan)
        if o.has and o.box is not None:
            trk_y1.append(o.box[1])
            trk_y2.append(o.box[3])
        else:
            trk_y1.append(np.nan)
            trk_y2.append(np.nan)
        states.append(0 if o.state == "OBSERVED" else
                      (1 if o.state == "COASTED" else 2))
        swit.append(1 if o.mode_switched else 0)
    t = np.arange(len(outs)) / fps

    fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(t, raw_y1, ".", ms=1.5, color="#bbbbbb", label="原始 上边沿 y1")
    ax[0].plot(t, trk_y1, "-", lw=1.4, color="#d62728", label="稳定后 上边沿 y1")
    ax[0].set_ylabel("y1 (px)")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title("箭体框上边沿: 原始检测 vs 时序稳定")
    ax[0].invert_yaxis()
    ax[0].grid(alpha=0.25)

    ax[1].plot(t, raw_y2, ".", ms=1.5, color="#bbbbbb", label="原始 下边沿 y2")
    ax[1].plot(t, trk_y2, "-", lw=1.4, color="#1f77b4", label="稳定后 下边沿 y2")
    ax[1].set_ylabel("y2 (px)")
    ax[1].legend(loc="lower left", fontsize=9)
    ax[1].set_title("箭体框下边沿: 存在 250/400 两种模式, 稳定后只保留连续变化")
    ax[1].invert_yaxis()
    ax[1].grid(alpha=0.25)
    for o in outs:
        if o.mode_switched:
            ax[1].axvline(o.frame / fps, color="orange", lw=1.0, alpha=0.8)

    st = np.array(states)
    cmap = np.array([[0.17, 0.63, 0.17, 1.0],     # OBSERVED 绿
                     [1.00, 0.50, 0.05, 1.0],     # COASTED  橙
                     [0.60, 0.60, 0.60, 1.0]])    # LOST     灰
    ax[2].vlines(t, 0, 1, colors=cmap[st], lw=1.4)
    ax[2].set_ylim(0, 1)
    ax[2].set_yticks([])
    ax[2].set_xlabel("时间 (s)")
    ax[2].plot([], [], lw=6, color=cmap[0], label="真实观测 OBSERVED")
    ax[2].plot([], [], lw=6, color=cmap[1], label="预测补位 COASTED")
    ax[2].plot([], [], lw=6, color=cmap[2], label="无目标 LOST")
    ax[2].legend(loc="upper right", fontsize=9, ncol=3)
    ax[2].set_title("跟踪状态: 只有绿色段存在真实像素, 才允许输出倾角")
    ax[2].grid(alpha=0.25)
    ax[1].annotate("橙色竖线 = 底边模式切换", xy=(0.01, 0.06),
                   xycoords="axes fraction", fontsize=9, color="orange")
    fig.tight_layout()
    fig.savefig(DIAG / "timeline.png", dpi=130)
    print(f"时间线图 -> {DIAG / 'timeline.png'}")


if __name__ == "__main__":
    main()
