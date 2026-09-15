# -*- coding: utf-8 -*-
"""第 3 步调参工具: 打印宽度扫描的决策表, 并把选中的"左/右边缘对"画到 ROI 上。

产出 runs/angle/debug2/f<idx>.png
用法: python scripts/debug_angle2.py --frames 1400 1200 1000 900 700
"""
from __future__ import annotations

import argparse
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
import rocket_angle as RA  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DIAG = PROJECT / "runs" / "diag"
OUT = PROJECT / "runs" / "angle" / "debug2"


def open_cap(path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "dbg2.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def sweep_table(patch, box, cfg, s):
    """复刻 acquire 的打分逻辑, 返回每个 (wr, pol) 的指标。"""
    H, W = patch.shape
    bx1, by1, bx2, by2 = box
    wb, hb = bx2 - bx1, by2 - by1
    y0 = max(1, int(round(by1 + cfg.trim_top * hb)))
    y1 = min(H - 1, int(round(by2 - cfg.trim_bottom * hb)))
    ys_all = np.arange(y0, y1)
    ysub = ys_all[::max(1, cfg.row_step)]
    wmin = max(3, int(cfg.min_width_px * s))
    rows = []
    for wr in cfg.width_ratios:
        w = int(round(wr * wb))
        if w < wmin or w >= W // 3:
            continue
        for pol in (1, -1):
            S = pol * RA._bar_score(patch, w)[ysub]
            if S.size == 0 or S.shape[1] < 3:
                continue
            lo = max(0, int(round(bx1 - 0.15 * wb)))
            hi = min(S.shape[1] - 1, int(round(bx2 - w)))
            if hi - lo < 3:
                continue
            sub = S[:, lo:hi + 1]
            i = sub.argmax(axis=1)
            v = sub[np.arange(len(i)), i]
            thr = cfg.min_peak_ratio * np.median(v)
            m = v >= thr
            if m.sum() < cfg.min_rows:
                continue
            xcm = (lo + i[m] + RA._parabolic(sub, i)[m]) + w / 2.0
            a, b, rms, inm = RA._robust_line(ysub[m].astype(float), xcm,
                                             cfg.irls_iters, cfg.inlier_k)
            ratio = float(inm.sum()) / max(m.sum(), 1)
            score = ratio - 0.12 * (rms / cfg.upscale_max) + 0.02 * wr
            rows.append(dict(wr=wr, pol=pol, w=w, ratio=ratio, rms=rms / s,
                             phi=np.degrees(np.arctan(-a)), score=score,
                             n=int(m.sum())))
    if not rows:
        return None, None
    best = max(rows, key=lambda r: r["score"])
    # 用最佳配置在全行上重算边缘对, 供画图
    w, pol = best["w"], best["pol"]
    S = pol * RA._bar_score(patch, w)[ys_all]
    lo = max(0, int(round(bx1 - 0.15 * wb)))
    hi = min(S.shape[1] - 1, int(round(bx2 - w)))
    sub = S[:, lo:hi + 1]
    i = sub.argmax(axis=1)
    v = sub[np.arange(len(i)), i]
    m = v >= cfg.min_peak_ratio * np.median(v)
    xl = lo + i + RA._parabolic(sub, i)
    return rows, (ys_all[m], xl[m], xl[m] + w, best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[1400, 1200, 1000, 900])
    ap.add_argument("--tag", default="iou70")
    args = ap.parse_args()
    meta = json.loads((DIAG / f"dets_{args.tag}.json").read_text(encoding="utf-8"))
    W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                        bound_wh=(W, H))
    cap = open_cap(PROJECT / meta["meta"]["video"])
    cfg = RA.AngleConfig()
    OUT.mkdir(parents=True, exist_ok=True)

    for idx in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        o = outs[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cr = RA.crop_roi(gray, RA.angle_roi(o.box, o.roi, cfg, W, H), o.box)
        if cr is None:
            print(f"帧 {idx}: ROI 太小")
            continue
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
        rows, sel = sweep_table(patch, box, cfg, s)
        if rows is None:
            print(f"帧 {idx}: 无解")
            continue
        rows.sort(key=lambda r: -r["score"])
        print(f"\n=== 帧 {idx} (t={idx / fps:.1f}s)  patch={patch.shape[1]}x{patch.shape[0]}  "
              f"×{s}  框宽={b0[2] - b0[0]:.1f}px ===")
        print(f"{'wr':>5} {'pol':>4} {'w_patch':>8} {'w_orig':>7} {'内点率':>7} "
              f"{'rms_orig':>9} {'phi':>7} {'行数':>6} {'score':>7}")
        for r in rows[:8]:
            print(f"{r['wr']:>5.2f} {r['pol']:>4} {r['w']:>8} {r['w'] / s:>7.1f} "
                  f"{r['ratio']:>7.3f} {r['rms']:>9.2f} {r['phi']:>7.2f} "
                  f"{r['n']:>6} {r['score']:>7.4f}")
        ys, xl, xr, best = sel
        fig, axpair = plt.subplots(1, 2, figsize=(11, 8),
                                   gridspec_kw={"width_ratios": [1, 2]})
        ax0, ax1 = axpair[0], axpair[1]
        ax0.imshow(cv2.cvtColor(p8, cv2.COLOR_GRAY2RGB))
        ax0.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0],
                                    box[3] - box[1], fill=False, ec="cyan", lw=1.2))
        ax0.plot(xl, ys, ".", ms=2, color="red")
        ax0.plot(xr, ys, ".", ms=2, color="lime")
        ax0.set_title(f"帧 {idx}  ROI×{s}\n青=稳定框 red/lime=左/右边", fontsize=9)
        ax1.imshow(cv2.cvtColor(p8, cv2.COLOR_GRAY2RGB), aspect="auto")
        ax1.plot(xl, ys, "-", color="red", lw=1.4)
        ax1.plot(xr, ys, "-", color="lime", lw=1.4)
        ax1.set_title(f"选中宽度 {best['w']}px(patch) = {best['w'] / s:.1f}px(原图), "
                      f"φ={best['phi']:+.2f}°, 框宽={b0[2] - b0[0]:.1f}px", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"f{idx}.png", dpi=110)
        plt.close(fig)
    cap.release()
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
