# -*- coding: utf-8 -*-
"""旋转注入测试(快速迭代用): 对比 本方案 / 逐行梯度峰 / 朴素 minAreaRect。

用法: python scripts/_rot_test.py [--frames 1500 1600 ...]
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

sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames  # noqa: E402
import rocket_angle as RA  # noqa: E402

DIAG = PROJECT / "runs" / "diag"


def open_cap(path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "rt.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+",
                    default=[1500, 1550, 1600, 1700, 1800, 1900, 2000])
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--pad-mul", type=float, default=1.0,
                    help="框水平扩张倍数(1.0=理论最小值)")
    ap.add_argument("--modes", nargs="+", default=["ours", "grad", "naive"])
    args = ap.parse_args()

    cfg = RA.AngleConfig()
    meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                        bound_wh=(W, H))
    cap = open_cap(PROJECT / meta["meta"]["video"])

    patches = {}
    for f in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok or f >= len(outs):
            continue
        o = outs[f]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cr = RA.crop_roi(gray, RA.angle_roi(o.box, o.roi, cfg, W, H), o.box)
        if cr is not None:
            patches[f] = cr
    cap.release()
    print(f"探针帧 {list(patches)}  (共 {len(patches)})")

    def run(mode, ang, p0, b0):
        h, w = p0.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
        rot = np.ascontiguousarray(cv2.warpAffine(
            p0, M, (w, h), flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE))
        # 关键: 框是**轴对齐**的。目标倾斜 θ 后, 若要仍然包住它, 水平方向必须
        # 扩宽 |tanθ|·h/2 —— 这正是真实场景里 YOLO 会给出的框。不同步扩宽的话,
        # 筒身两端会跑出搜索窗, 测出的角被系统性压缩(实测增益只有 0.2~0.8)。
        b = b0.astype(float).copy()
        d = args.pad_mul * abs(np.tan(np.radians(ang))) * (b0[3] - b0[1]) / 2.0
        b[0] -= d
        b[2] += d
        if mode == "ours":
            est = RA.AngleEstimator(cfg)
            r = est.step(rot, b, (0, 0), 0, 1.0)
            return r.phi_deg if r.ok or r.mode == "acquire" else np.nan
        # 另两种需要先预处理
        est = RA.AngleEstimator(cfg)
        patch, box_s, s = est._prepare(rot, b)
        if mode == "grad":
            ge = RA.gradient_edges(patch, box_s, cfg)
            if ge is None:
                return np.nan
            ys, xl, xr, _, _ = ge
            al, _, _, _ = RA._robust_line(ys, xl, cfg.irls_iters, cfg.inlier_k)
            ar, _, _, _ = RA._robust_line(ys, xr, cfg.irls_iters, cfg.inlier_k)
            if not (np.isfinite(al) and np.isfinite(ar)):
                return np.nan
            return 0.5 * (np.degrees(np.arctan(-al)) + np.degrees(np.arctan(-ar)))
        if mode == "naive":
            return RA._naive_minarearect(
                np.clip(patch, 0, 255).astype(np.uint8), box_s)
        return np.nan

    for mode, label in (("ours", "本方案-每帧重新捕获(最难路径)"),
                        ("oursseq", "本方案-连续跟踪(≈真实运行路径)"),
                        ("grad", "逐行梯度峰(朴素边缘)"),
                        ("naive", "Otsu+minAreaRect")):
        if mode not in args.modes:
            continue
        if mode == "oursseq":
            rows = []
            for a in args.angles:
                est = RA.AngleEstimator(cfg)
                vals = []
                for k, (p0, b0, _) in enumerate(patches.values()):
                    h, w = p0.shape
                    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0)
                    rot = np.ascontiguousarray(cv2.warpAffine(
                        p0, M, (w, h), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE))
                    b = b0.astype(float).copy()
                    d = args.pad_mul * abs(np.tan(np.radians(a))) \
                        * (b0[3] - b0[1]) / 2.0
                    b[0] -= d
                    b[2] += d
                    r = est.step(rot, b, (0, 0), k, 1.0)
                    if k >= 2 and (r.mode == "track") and np.isfinite(r.phi_deg):
                        vals.append(r.phi_deg)
                rows.append((a, float(np.mean(vals)) if vals else np.nan,
                             float(np.std(vals)) if vals else np.nan,
                             len(vals)))
            xs = np.array([r[0] for r in rows])
            ys = np.array([r[1] for r in rows])
            ok = np.isfinite(ys)
            print(f"\n=== {label} ===")
            print(f"  {'注入θ':>7} {'测得φ均值':>10} {'σ':>7} {'n':>3}")
            for a, m, s, n in rows:
                print(f"  {a:>7.1f} {m:>10.3f} {s:>7.3f} {n:>3}")
            if ok.sum() >= 3:
                kk = np.polyfit(xs[ok], ys[ok], 1)
                dev = ys[ok] - np.polyval(kk, xs[ok])
                print(f"  增益 = {kk[0]:+.3f}  (理想 -1.000)")
                print(f"  偏置 = {kk[1]:+.3f}°   线性残差σ = {np.std(dev):.3f}°")
            continue
        rows = []
        for a in args.angles:
            vals = [run(mode, a, p0, b0) for p0, b0, _ in patches.values()]
            vals = [v for v in vals if np.isfinite(v)]
            rows.append((a, float(np.mean(vals)) if vals else np.nan,
                         float(np.std(vals)) if vals else np.nan, len(vals)))
        xs = np.array([r[0] for r in rows])
        ys = np.array([r[1] for r in rows])
        ok = np.isfinite(ys)
        print(f"\n=== {label} ===")
        print(f"  {'注入θ':>7} {'测得φ均值':>10} {'σ':>7} {'n':>3}")
        for a, m, s, n in rows:
            print(f"  {a:>7.1f} {m:>10.3f} {s:>7.3f} {n:>3}")
        if ok.sum() >= 3:
            k = np.polyfit(xs[ok], ys[ok], 1)
            dev = ys[ok] - np.polyval(k, xs[ok])
            print(f"  增益 = {k[0]:+.3f}   (理想 -1.000: 注入角为正表示顶端左倾;"
                  f" 绝对值应=1)")
            print(f"  偏置 = {k[1]:+.3f}°   线性残差σ = {np.std(dev):.3f}°")


if __name__ == "__main__":
    main()
