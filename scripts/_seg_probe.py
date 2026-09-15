# -*- coding: utf-8 -*-
"""seg 产物探针: 掩码到底圈住了什么, 宽度口径对不对。

方案 B 的成败全在这一点上: 如果掩码圈的是"整箭(含支腿/栅格舵)", 那它给出的宽度
仍然是框宽那一套, 解决不了原方案 8.3 的 w_body 口径问题; 只有圈住**筒身**才有价值。

用法:
    python scripts/_seg_probe.py --tag smoke --frames 1400,1430,1460,1490
    python scripts/_seg_probe.py --tag iou70 --frames 900,1500,2100
输出:
    runs/seg/probe_<tag>.log   —— 逐帧行宽剖面与统计
    runs/seg/probe_<tag>.png   —— 叠加图(掩码轮廓红/绿 + 稳定框青)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames  # noqa: E402

SEG = PROJECT / "runs" / "seg"
DETS = PROJECT / "runs" / "diag"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--dets-tag", default="iou70")
    ap.add_argument("--frames", default="1400,1440,1480")
    args = ap.parse_args()

    meta = json.loads((DETS / f"dets_{args.dets_tag}.json").read_text(
        encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                        fps, bound_wh=(W, H))
    d = np.load(SEG / f"bounds_{args.tag}.npz")
    XL, XR = d["XL"], d["XR"]
    sfx = f"_{args.tag}" if args.tag else ""
    stats = (SEG / f"mask_stats{sfx}.csv").read_text(encoding="utf-8").splitlines()

    lines = []
    frames = [int(v) for v in args.frames.split(",") if v.strip()]
    tiles = []
    for f in frames:
        if f >= XL.shape[0]:
            continue
        xl, xr = XL[f].astype(float), XR[f].astype(float)
        k = xl > 0
        o = outs[f]
        box = o.box if o.box is not None else np.full(4, np.nan)
        lines.append(f"--- frame {f} (t={f / fps:.1f}s) ---")
        lines.append(f"  tracker box = {np.round(box, 1).tolist()}  "
                     f"w_box={box[2] - box[0]:.1f} h_box={box[3] - box[1]:.1f}")
        lines.append(f"  csv: {stats[f + 1] if f + 1 < len(stats) else 'N/A'}")
        if not k.any():
            lines.append("  (无掩码)")
            continue
        ys = np.where(k)[0]
        wr = (xr - xl)[k]
        x1, x2 = xl[k].min(), xr[k].max()
        lines.append(f"  mask rows {ys[0]}..{ys[-1]} ({k.sum()} 行)  "
                     f"x {x1:.0f}..{x2:.0f}  bbox_w={x2 - x1:.1f}")
        lines.append(f"  行宽 w(y): 中位 {np.median(wr):.1f}  "
                     f"均值 {wr.mean():.1f}  最小 {wr.min():.0f} 最大 {wr.max():.0f}")
        # 宽度剖面: 从顶到底取 12 个采样点, 看有没有"突变"(整流罩/支腿)
        samp = np.linspace(0, ys.size - 1, 12).astype(int)
        prof = " ".join(f"{wr[i]:.0f}" for i in samp)
        lines.append(f"  宽度剖面(顶->底 12 点): {prof}")
        dw = np.abs(np.diff(wr))
        lines.append(f"  |dw/dy| 中位 {np.median(dw):.2f} px/行  "
                     f"95分位 {np.percentile(dw, 95):.2f}")
        lines.append(f"  中位宽度 / 框宽 = {np.median(wr) / max(box[2] - box[0], 1):.3f}"
                     f"   (原方案实测筒身/框 ≈ 0.52)")
        # 叠加图
        cap = cv2.VideoCapture(str(PROJECT / vm["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        if ok:
            for y in ys[::3]:
                cv2.circle(img, (int(xl[y]), int(y)), 1, (0, 0, 255), -1)
                cv2.circle(img, (int(xr[y]), int(y)), 1, (0, 0, 255), -1)
            if np.isfinite(box).all():
                cv2.rectangle(img, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (255, 255, 0), 1)
            crop = img[max(0, int(ys[0]) - 20):min(H, int(ys[-1]) + 20),
                       max(0, int(x1) - 40):min(W, int(x2) + 40)]
            crop = cv2.resize(crop, None, fx=3, fy=3,
                              interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, f"f{f} w={np.median(wr):.0f} wbox="
                              f"{box[2] - box[0]:.0f}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                        cv2.LINE_AA)
            tiles.append(crop)
    txt = "\n".join(lines)
    (SEG / f"probe_{args.tag}.txt").write_text(txt, encoding="utf-8")
    print(txt)
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        wsum = sum(t.shape[1] for t in tiles) + 12 * (len(tiles) - 1)
        canvas = np.full((hmax, wsum, 3), 30, np.uint8)
        x = 0
        for t in tiles:
            canvas[:t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 12
        cv2.imwrite(str(SEG / f"probe_{args.tag}.png"), canvas)
        print(f"[写入] {SEG / f'probe_{args.tag}.png'}")


if __name__ == "__main__":
    main()
