# -*- coding: utf-8 -*-
"""把几类关键帧单独导出并放大, 用于人工确认框覆盖范围。

产出 runs/diag/frames/ 下:
  f<idx>_full.jpg  —— 整帧 + 框 (绿=RB conf>=0.25, 橙=RB 低分, 蓝=EF, 白=SP)
  f<idx>_zoom.jpg  —— ROI 4 倍放大
  grid_full.jpg    —— 全部整帧拼一张
用法: python scripts/inspect_frames.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE / "yolo")

import cv2
import numpy as np

DIAG = PROJECT / "runs" / "diag"
OUT = DIAG / "frames"

TARGETS = [780, 900, 1101, 1165, 1211, 1500]
COL = {0: (255, 160, 0), 1: (0, 220, 0), 2: (255, 255, 255)}


def main():
    d = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    frames = d["frames"]
    video = PROJECT / d["meta"]["video"]
    OUT.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        # 中文路径在部分 OpenCV 版本下会失败, 复制成 ascii 名再读
        ascii_v = CACHE / "tmp" / "diag_src.mp4"
        cap.release()
        import shutil
        shutil.copy(str(video), str(ascii_v))
        cap = cv2.VideoCapture(str(ascii_v))

    tiles = []
    for idx in TARGETS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            print(f"[!] 读不到帧 {idx}")
            continue
        full = img.copy()
        roi = None
        for b in frames[idx]["b"]:
            x1, y1, x2, y2 = [int(v) for v in b["x"]]
            c = b["c"]
            if c == 1:
                col = (0, 220, 0) if b["s"] >= 0.25 else (0, 140, 255)
                thick = 2 if b["s"] >= 0.25 else 1
            else:
                col, thick = COL[c], 1
            cv2.rectangle(full, (x1, y1), (x2, y2), col, thick)
            cv2.putText(full, f"{['EF','RB','SP'][c]}{b['s']:.2f}",
                        (max(0, x1 - 2), max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            if c == 1 and (roi is None or b["s"] > roi[0]):
                roi = (b["s"], [x1, y1, x2, y2])
        cv2.putText(full, f"f{idx}  t={idx / d['meta']['fps']:.2f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"f{idx}_full.jpg"), full,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        tiles.append(full)

        if roi is not None:
            x1, y1, x2, y2 = roi[1]
            pw, ph = max(8, (x2 - x1) // 2), max(8, (y2 - y1) // 2)
            cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
            cx2, cy2 = min(img.shape[1], x2 + pw), min(img.shape[0], y2 + ph)
            crop = img[cy1:cy2, cx1:cx2]
            k = max(1, int(1400 / max(crop.shape[:2])))
            z = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
            # 在放大图上叠原始框
            for b in frames[idx]["b"]:
                if b["c"] != 1:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in b["x"]]
                if bx2 < cx1 or bx1 > cx2 or by2 < cy1 or by1 > cy2:
                    continue
                col = (0, 220, 0) if b["s"] >= 0.25 else (0, 140, 255)
                cv2.rectangle(z, ((bx1 - cx1) * k, (by1 - cy1) * k),
                              ((bx2 - cx1) * k, (by2 - cy1) * k), col, 2)
            cv2.imwrite(str(OUT / f"f{idx}_zoom.jpg"), z,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            print(f"  f{idx}: crop {crop.shape[1]}x{crop.shape[0]} -> x{k}")
    cap.release()

    if tiles:
        h = max(t.shape[0] for t in tiles)
        w = sum(t.shape[1] for t in tiles) + 4 * (len(tiles) - 1)
        g = np.zeros((h, w, 3), np.uint8)
        x = 0
        for t in tiles:
            g[:t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 4
        cv2.imwrite(str(OUT / "grid_full.jpg"), g,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"grid -> {OUT / 'grid_full.jpg'}")
    print(f"frames -> {OUT}")


if __name__ == "__main__":
    main()
