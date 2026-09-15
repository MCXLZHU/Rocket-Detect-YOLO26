# -*- coding: utf-8 -*-
"""直接插桩 acquire: 看每行检测到的左边缘位置到底是什么。"""
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

import cv2
sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames
import rocket_angle as RA

DIAG = PROJECT / "runs" / "diag"
meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                    bound_wh=(W, H))

cap = cv2.VideoCapture(str(PROJECT / meta["meta"]["video"]))
if not cap.isOpened():
    tmp = CACHE / "tmp" / "ins.mp4"
    shutil.copy(str(PROJECT / meta["meta"]["video"]), str(tmp))
    cap = cv2.VideoCapture(str(tmp))

cfg = RA.AngleConfig()
IDX = 1400
cap.set(cv2.CAP_PROP_POS_FRAMES, IDX)
ok, frame = cap.read()
o = outs[IDX]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cr = RA.crop_roi(gray, RA.angle_roi(o.box, o.roi, cfg, W, H), o.box)
p0, b0, org = cr
s = int(np.clip(round(cfg.upscale_target_w / max(b0[2] - b0[0], 6.0)), 1, cfg.upscale_max))
RA.cfg_scale_holder[0] = float(s)
patch = cv2.resize(p0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
patch = cv2.medianBlur(np.clip(patch, 0, 255).astype(np.uint8), 3).astype(np.float32)
patch = cv2.createCLAHE(cfg.clahe_clip, (cfg.clahe_grid, cfg.clahe_grid)).apply(
    np.clip(patch, 0, 255).astype(np.uint8)).astype(np.float32)
box = b0 * s
print(f"帧 {IDX}: patch={patch.shape[1]}x{patch.shape[0]}  s={s}  "
      f"box={np.round(box,1)}  框宽={b0[2]-b0[0]:.1f}px")

bx1, by1, bx2, by2 = box
wb, hb = bx2 - bx1, by2 - by1
y0 = max(1, int(round(by1 + cfg.trim_top * hb)))
y1 = min(patch.shape[0] - 1, int(round(by2 - cfg.trim_bottom * hb)))
ys = np.arange(y0, y1)
ysub = ys[::max(1, cfg.row_step)]
print(f"行范围 y={y0}..{y1}  ({len(ys)} 行, 抽样 {len(ysub)})")

for wr in (0.30, 0.46, 0.62, 0.82):
    w = int(round(wr * wb))
    k = max(2, int(round(0.55 * w)))
    for pol in (1, -1):
        S = pol * RA._bar_score(patch, w)
        if S.shape[1] < 3:
            print(f"  wr={wr} pol={pol}: 空")
            continue
        lo = max(0, int(round(bx1 - 0.15 * wb)))
        hi = min(patch.shape[1] - 1, int(round(bx2 - w)))
        if hi - lo < 3:
            print(f"  wr={wr} pol={pol}: 窗口太窄 {lo}..{hi}")
            continue
        sub = S[ysub][:, lo:hi + 1]
        i = sub.argmax(axis=1)
        v = sub[np.arange(len(i)), i]
        xl = lo + i + RA._parabolic(sub, i)
        print(f"  wr={wr:.2f} pol={pol:+d} w={w} k={k} S.cols={S.shape[1]} "
              f"win=[{lo},{hi}] | 唯一峰位={len(np.unique(i))} "
              f"std(xl)={np.std(xl):.3f} 范围[{xl.min():.1f},{xl.max():.1f}] "
              f"峰强中位={np.median(v):.2f}")
        # 手动拟合
        a, b, rms, inm = RA._robust_line(ysub.astype(float), xl, 5, 2.5)
        print(f"        -> 拟合 a={a:.6f} (φ={np.degrees(np.arctan(-a)):+.3f}°) "
              f"rms={rms:.4f} 内点={int(inm.sum())}/{len(inm)}")
