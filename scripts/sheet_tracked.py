# -*- coding: utf-8 -*-
"""从 runs/diag/tracked.mp4 抽关键帧拼一张图, 用于肉眼复核稳定框。

用法: python scripts/sheet_tracked.py
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
(CACHE / "tmp").mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
DIAG = PROJECT / "runs" / "diag"

FRAMES = [780, 1000, 1146, 1165, 1211, 1500]

cap = cv2.VideoCapture(str(DIAG / "tracked.mp4"))
tiles = []
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    if not ok:
        print(f"读不到帧 {f}")
        continue
    img = cv2.resize(img, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
    tiles.append(img)
cap.release()

if tiles:
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    g = np.full((rows * th + (rows - 1) * 4, cols * tw + (cols - 1) * 4, 3),
                30, np.uint8)
    for i, t in enumerate(tiles):
        rr, cc = divmod(i, cols)
        g[rr * (th + 4):rr * (th + 4) + th, cc * (tw + 4):cc * (tw + 4) + tw] = t
    out = DIAG / "tracked_sheet.jpg"
    cv2.imwrite(str(out), g, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"-> {out}  ({g.shape[1]}x{g.shape[0]})")
