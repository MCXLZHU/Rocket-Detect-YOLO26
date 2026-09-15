# -*- coding: utf-8 -*-
"""调试第 3 步: 把 ROI 放大图 + 每行检测到的左右边缘点 + 拟合直线 画出来。

产出 runs/angle/debug/f<idx>.png (单张, 上下两栏):
  上: ROI 放大图, 红/绿点 = 左右边缘候选, 蓝线 = 拟合直线
  下: 每行的水平梯度剖面热力图(白=正梯度, 黑=负梯度), 看边缘峰是否被找到

用法: python scripts/debug_angle.py --frames 1400 1200 900 --pol -1
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
from rocket_angle import AngleConfig, _edge_candidates, crop_roi, robust_line  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
DIAG = PROJECT / "runs" / "diag"
OUT = PROJECT / "runs" / "angle" / "debug"


def open_cap(path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "dbg.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[1400, 1200, 900])
    ap.add_argument("--tag", default="iou70")
    args = ap.parse_args()

    meta = json.loads((DIAG / f"dets_{args.tag}.json").read_text(encoding="utf-8"))
    FH = [r["b"] for r in meta["frames"]]
    W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
    outs = track_frames(FH, TrackerConfig(), fps, bound_wh=(W, H))
    cap = open_cap(PROJECT / meta["meta"]["video"])
    cfg = AngleConfig()
    OUT.mkdir(parents=True, exist_ok=True)

    for idx in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"读不到帧 {idx}")
            continue
        o = outs[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cr = crop_roi(gray, o.roi, o.box)
        if cr is None:
            print(f"帧 {idx}: ROI 太小")
            continue
        patch0, box0, origin = cr
        s = int(np.clip(round(cfg.upscale_target_w / max(box0[2] - box0[0], 6.0)),
                        1, cfg.upscale_max))
        patch = cv2.resize(patch0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        p8 = np.clip(patch, 0, 255).astype(np.uint8)
        p8c = cv2.createCLAHE(cfg.clahe_clip, (cfg.clahe_grid, cfg.clahe_grid)).apply(p8)
        pf = p8c.astype(np.float32)
        box_s = box0 * s

        fig, ax = plt.subplots(3, 1, figsize=(9, 12),
                               gridspec_kw={"height_ratios": [3, 2, 1]})
        ax[0].imshow(cv2.cvtColor(p8c, cv2.COLOR_GRAY2RGB))
        ax[0].add_patch(plt.Rectangle((box_s[0], box_s[1]),
                                      box_s[2] - box_s[0], box_s[3] - box_s[1],
                                      fill=False, ec="cyan", lw=1.5))
        ax[0].set_title(f"帧 {idx} (t={idx / fps:.1f}s)  ROI×{s}  青=稳定框")

        info = []
        for pol, col in ((1, "red"), (-1, "lime")):
            cand = _edge_candidates(pf, box_s, pol, cfg)
            if cand is None:
                info.append(f"pol={pol}: 无候选")
                continue
            ys, xl, xr, n = cand
            al, bl, rmsl, ml = robust_line(ys, xl, cfg.irls_iters, cfg.inlier_k)
            ar, br, rmsr, mr = robust_line(ys, xr, cfg.irls_iters, cfg.inlier_k)
            both = ml & mr
            if both.sum() >= cfg.min_rows:
                al, bl, rmsl, _ = robust_line(ys[both], xl[both], cfg.irls_iters, cfg.inlier_k)
                ar, br, rmsr, _ = robust_line(ys[both], xr[both], cfg.irls_iters, cfg.inlier_k)
            ax[0].plot(xl, ys, ".", ms=2.5, color=col, alpha=0.8)
            ax[0].plot(xr, ys, ".", ms=2.5, color=col, alpha=0.8, marker="x")
            yy = np.array([ys.min(), ys.max()])
            ax[0].plot(al * yy + bl, yy, "-", color=col, lw=1.5)
            ax[0].plot(ar * yy + br, yy, "-", color=col, lw=1.5, ls="--")
            pl = np.degrees(np.arctan(-al))
            pr = np.degrees(np.arctan(-ar))
            info.append(f"pol={pol}: 行数={n} 共同内点={int(both.sum())} "
                        f"φL={pl:+.2f}° φR={pr:+.2f}° 均值={0.5 * (pl + pr):+.2f}° "
                        f"rms={0.5 * (rmsl + rmsr) / s:.2f}px "
                        f"宽度中位={np.median(xr - xl) / s:.1f}px "
                        f"框宽={(box0[2] - box0[0]):.1f}px")

        g = np.gradient(pf, axis=1)
        ncol = g.shape[1]
        vmax = np.percentile(np.abs(g), 99.5)
        ax[1].imshow(g, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax[1].axvline(max(0.0, (box_s[0] + box_s[2]) / 2), color="k", lw=0.8)
        ax[1].set_title("水平梯度 ∂I/∂x (红=正, 蓝=负); 直筒应为 左正右负 或 左负右正")
        prof = np.abs(g).mean(axis=0)
        ax[2].plot(prof, lw=1.0, color="#333")
        ax[2].axvspan(box_s[0], box_s[2], color="cyan", alpha=0.15)
        ax[2].set_title("|∂I/∂x| 的行平均剖面 (青=框范围)")
        for a in ax:
            a.tick_params(labelsize=8)
        ax[0].set_ylabel("y (patch px)")
        ax[2].set_xlabel("x (patch px)")
        fig.suptitle(f"帧 {idx}   " + "\n".join(info), fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / f"f{idx}.png", dpi=110)
        plt.close(fig)
        for line in info:
            print(f"帧 {idx}: {line}")
    cap.release()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
