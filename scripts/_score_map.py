# -*- coding: utf-8 -*-
"""把"成对边缘"打分图画出来, 看筒身边缘到底是不是一条可靠的脊。"""
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

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames
import rocket_angle as RA

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DIAG = PROJECT / "runs" / "diag"
OUT = PROJECT / "runs" / "angle" / "debug3"

meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                    bound_wh=(W, H))
cap = cv2.VideoCapture(str(PROJECT / meta["meta"]["video"]))
if not cap.isOpened():
    tmp = CACHE / "tmp" / "dbg3.mp4"
    shutil.copy(str(PROJECT / meta["meta"]["video"]), str(tmp))
    cap = cv2.VideoCapture(str(tmp))
cfg = RA.AngleConfig()
OUT.mkdir(parents=True, exist_ok=True)

for IDX in (900, 1200, 1400):
    cap.set(cv2.CAP_PROP_POS_FRAMES, IDX)
    ok, frame = cap.read()
    o = outs[IDX]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cr = RA.crop_roi(gray, RA.angle_roi(o.box, o.roi, cfg, W, H), o.box)
    p0, b0, org = cr
    s = int(np.clip(round(cfg.upscale_target_w / max(b0[2] - b0[0], 6.0)),
                    1, cfg.upscale_max))
    RA.cfg_scale_holder[0] = float(s)
    patch = cv2.resize(p0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    patch = cv2.medianBlur(np.clip(patch, 0, 255).astype(np.uint8), 3).astype(np.float32)
    p8 = cv2.createCLAHE(cfg.clahe_clip, (cfg.clahe_grid, cfg.clahe_grid)).apply(
        np.clip(patch, 0, 255).astype(np.uint8))
    patch = p8.astype(np.float32)
    box = b0 * s
    bx1, by1, bx2, by2 = box
    wb, hb = bx2 - bx1, by2 - by1
    y0 = max(1, int(round(by1 + cfg.trim_top * hb)))
    y1 = min(patch.shape[0] - 1, int(round(by2 - cfg.trim_bottom * hb)))
    ys = np.arange(y0, y1)

    # 对比度诊断: 框内 vs 框外 的灰度差
    inmean = float(patch[y0:y1, int(bx1 + 0.25 * wb):int(bx1 + 0.75 * wb)].mean())
    lmean = float(patch[y0:y1, :max(1, int(bx1 - 0.05 * wb))].mean())
    rmean = float(patch[y0:y1, min(patch.shape[1] - 1, int(bx2 + 0.05 * wb)):].mean())

    fig, ax = plt.subplots(1, 4, figsize=(18, 7),
                           gridspec_kw={"width_ratios": [1.1, 1.2, 1.2, 1.2]})
    ax[0].imshow(cv2.cvtColor(p8, cv2.COLOR_GRAY2RGB))
    ax[0].add_patch(plt.Rectangle((bx1, by1), wb, hb, fill=False, ec="cyan", lw=1.2))
    ax[0].axhline(y0, color="lime", lw=0.8); ax[0].axhline(y1, color="lime", lw=0.8)
    ax[0].set_title(f"帧 {IDX} t={IDX / fps:.1f}s  ×{s}\n框内均值={inmean:.0f} "
                    f"左外={lmean:.0f} 右外={rmean:.0f}", fontsize=9)

    cols = (0.30, 0.46, 0.62, 0.82)
    for k, wr in enumerate(cols):
        w = int(round(wr * wb))
        S = RA._bar_score(patch, w)
        if k < 3:
            a = ax[k + 1]
        Sd = S[ys]
        vmax = np.percentile(np.abs(Sd[Sd > -1e8]), 98) if (Sd > -1e8).any() else 1
        a.imshow(Sd, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                 extent=[0, S.shape[1], ys[-1], ys[0]])
        lo = max(0, int(round(bx1 - 0.15 * wb)))
        hi = min(S.shape[1] - 1, int(round(bx2 - w)))
        sub = Sd[:, lo:hi + 1]
        ii = sub.argmax(axis=1)
        a.plot(lo + ii, ys, "-", color="lime", lw=0.9)
        a.axvline(bx1, color="cyan", lw=0.8); a.axvline(bx2, color="cyan", lw=0.8)
        a.set_title(f"w={w}(={w / s:.1f}px原图) 打分图 第j列=左边缘在j\n"
                    f"绿=逐行argmax(含抽样行)", fontsize=9)
        a.set_xlim(0, S.shape[1])
    fig.tight_layout()
    fig.savefig(OUT / f"score{IDX}.png", dpi=105)
    plt.close(fig)
    print(f"帧 {IDX}: 框内均值 {inmean:.0f} / 左外 {lmean:.0f} / 右外 {rmean:.0f} "
          f"| 筒身→背景对比度 ≈ {inmean - 0.5 * (lmean + rmean):+.0f} 灰阶")
cap.release()
print(f"-> {OUT}")
