# -*- coding: utf-8 -*-
"""第 3 步验证: (1) 旋转注入精度测试 (2) 落地静止段重复性 (3) 估计器/参数对比

为什么这么做:
  本视频 40s 后火箭落地静止, 真实倾角是常数 ⇒ 那一段的 σ(φ) 就是**重复性**的直接
  证据。但"稳定"不等于"准确" —— 一个对旋转不敏感的估计器同样会给出很小的 σ。
  所以再做一个**旋转注入**测试: 把真实 ROI 人为旋转 θ, 看测出来的 φ 是否正好变化
  θ。由此得到 (增益, 偏置, 残差σ), 增益应≈1.000, 残差σ即真实精度。

产物: runs/angle/validate.txt
用法: python scripts/validate_angle.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
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
OUT = PROJECT / "runs" / "angle"
L: list[str] = []


def p(s=""):
    L.append(s)


def sd(v):
    return float(np.std(v)) if len(v) else float("nan")


def open_cap(path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "val_angle.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


# --------------------------------------------------------------------------
def main():
    cfg = RA.AngleConfig()
    meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                        bound_wh=(W, H))

    # ============ 0) 一次性裁好所有 ROI, 后续所有实验复用(避免反复解码) ======
    cap = open_cap(PROJECT / meta["meta"]["video"])
    cache = {}
    t0 = time.time()
    for i in range(len(outs)):
        ok, frame = cap.read()
        if not ok:
            break
        o = outs[i]
        if not (o.has and o.angle_ok and o.roi is not None and o.box is not None):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cr = RA.crop_roi(gray, RA.angle_roi(o.box, o.roi, cfg, W, H), o.box)
        if cr is not None:
            cache[i] = (cr[0].copy(), cr[1].copy(), o.box.copy(), o.reliability)
    cap.release()
    p("=" * 84)
    p("第 3 步(ROI 内边缘检测 + 倾角) 验证报告")
    p("=" * 84)
    p(f"视频 {meta['meta']['video']}")
    p(f"{W}x{H}  {fps:.2f} fps  {len(outs)} 帧;  可取 ROI 的帧: {len(cache)}")
    p(f"(裁 ROI 用时 {time.time() - t0:.1f}s)")
    p()

    # ============ 1) 旋转注入精度测试 =======================================
    p("-" * 84)
    p("[1] 旋转注入测试: 把真实 ROI 人为旋转 θ, 检验测出的 φ 是否变化 θ")
    p("-" * 84)
    angles = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    probe = [f for f in (1500, 1550, 1600, 1700, 1800, 1900, 2000) if f in cache]
    p(f"  探针帧: {probe} (落地静止段)")
    p(f"  {'注入 θ(°)':>10} {'测得 φ 均值':>12} {'φ 标准差':>10} "
      f"{'θ+φ?':>9} {'帧数':>6}")
    meas = {}
    for a in angles:
        vals = []
        for f in probe:
            p0, b0, _, rel = cache[f]
            h, w = p0.shape
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0)
            rot = np.ascontiguousarray(cv2.warpAffine(
                p0, M, (w, h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE))
            # 框是轴对齐的: 目标倾斜 θ 后要仍然包住它, 水平必须扩宽 |tanθ|·h/2
            # (这正是真实场景里 YOLO 会给出的框)。不扩宽会让筒身两端跑出搜索窗,
            # 测出的倾角被系统性压缩。
            b = b0.astype(float).copy()
            d = abs(np.tan(np.radians(a))) * (b0[3] - b0[1]) / 2.0
            b[0] -= d
            b[2] += d
            est = RA.AngleEstimator(cfg)
            r = est.step(rot, b, (0, 0), f, rel)
            if np.isfinite(r.phi_deg):
                vals.append(r.phi_deg)
        if vals:
            meas[a] = vals
            p(f"  {a:>10.1f} {np.mean(vals):>12.3f} {sd(vals):>10.3f} "
              f"{'':>9} {len(vals):>6}")
    if len(meas) >= 3:
        xs = np.array(sorted(meas))
        ys = np.array([np.mean(meas[a]) for a in xs])
        A = np.polyfit(xs, ys, 1)
        resid = ys - np.polyval(A, xs)
        # 相对零点的偏差(消除"火箭本身是否竖直"的未知量)
        dev = np.array([np.mean(meas[a]) - np.mean(meas[0.0])
                        - a for a in xs])
        p()
        p(f"  线性拟合: 测得φ = {A[0]:.4f} × 注入θ + {A[1]:+.4f}   "
          f"(增益应≈1.000)")
        p(f"  相对 0° 的偏差 (测-注): 均值 {np.mean(dev):+.3f}°  "
          f"标准差 {sd(dev):.3f}°  最大 {np.max(np.abs(dev)):.3f}°")
        p(f"  ⇒ 比例增益 {A[0]:.3f} (偏差 {abs(A[0] - 1) * 100:.2f}%), "
          f"单帧精度 ~{sd(dev):.2f}°")
    p()

    # ============ 2) 落地静止段重复性 =======================================
    p("-" * 84)
    p("[2] 落地静止段(46-66s) 重复性 —— 真实倾角为常数, σ 即重复精度")
    p("-" * 84)
    est = RA.AngleEstimator(cfg)
    res_seq = {}
    for i in sorted(cache):
        p0, b0, _, rel = cache[i]
        res_seq[i] = est.step(p0, b0, (0, 0), i, rel)
    stat = [i for i in sorted(res_seq)
            if 46 <= i / fps < 66 and res_seq[i].ok]
    phi = [res_seq[i].phi_deg for i in stat]
    if phi:
        xs = np.array(stat) / fps
        k = np.polyfit(xs, phi, 1)
        p(f"  有效帧 {len(phi)};  φ 均值 {np.mean(phi):+.3f}°  σ {sd(phi):.3f}°  "
          f"极差 {max(phi) - min(phi):.3f}°")
        p(f"  线性漂移 {k[0]:+.4f} °/s  (整段累计 {k[0] * (xs[-1] - xs[0]):+.3f}°)")
        p(f"  分位: P5 {np.percentile(phi, 5):+.3f}  P50 {np.percentile(phi, 50):+.3f} "
          f"P95 {np.percentile(phi, 95):+.3f}")
    modes = {}
    for i in sorted(res_seq):
        if 11 <= i / fps < 70:
            modes[res_seq[i].mode or res_seq[i].reason] = \
                modes.get(res_seq[i].mode or res_seq[i].reason, 0) + 1
    p(f"  模式分布(11-70s): {modes}")
    wb = [res_seq[i].w_body_px for i in stat if np.isfinite(res_seq[i].w_body_px)]
    bxs = [res_seq[i].w_box_px for i in stat if np.isfinite(res_seq[i].w_box_px)]
    if wb:
        p(f"  筒身宽度 w_body: 中位 {np.median(wb):.1f}px  σ {sd(wb):.2f}px   "
          f"| 框宽中位 {np.median(bxs):.1f}px  ⇒ w_body/w_box = "
          f"{np.median(wb) / np.median(bxs):.3f}")
    p()

    # ============ 3) 估计器与参数对比 ======================================
    p("-" * 84)
    p("[3] 估计器对比(落地静止段 σ, 越小越好)")
    p("-" * 84)

    def run_cfg(name, c):
        e = RA.AngleEstimator(c)
        vals, n_ok, modes2 = [], 0, {}
        for i in sorted(cache):
            p0, b0, _, rel = cache[i]
            r = e.step(p0, b0, (0, 0), i, rel)
            if 46 <= i / fps < 66:
                modes2[r.mode or r.reason] = modes2.get(r.mode or r.reason, 0) + 1
                if r.ok:
                    vals.append(r.phi_deg)
            n_ok += int(r.ok)
        p(f"  {name:<28} 全片有效 {n_ok:>4} 帧 | 静止段 n={len(vals):>3} "
          f"σ={sd(vals):>6.3f}° 均值={np.mean(vals) if vals else float('nan'):+.3f}°"
          f" | 模式 {modes2}")
        return sd(vals), n_ok

    run_cfg("默认(条带先验+分段行平均+锁定)", RA.AngleConfig())
    run_cfg("关闭条带先验(直接在宽窗找左右峰)", RA.AngleConfig(use_bar_prior=False))
    run_cfg("关闭时序锁定", RA.AngleConfig(hold_max=0, lock_min_bands=10 ** 9))
    run_cfg("关闭中值预滤波", RA.AngleConfig(use_box_median=False))
    run_cfg("关闭 CLAHE", RA.AngleConfig(use_clahe=False))
    run_cfg("上采样上限=2", RA.AngleConfig(upscale_max=2))
    run_cfg("上采样上限=1", RA.AngleConfig(upscale_max=1))
    run_cfg("段数=6", RA.AngleConfig(n_bands=6))
    run_cfg("段数=16", RA.AngleConfig(n_bands=16))
    run_cfg("窄带±3px", RA.AngleConfig(band_px=3.0, band_px_min=3.0))
    p()

    # ============ 4) 覆盖区间 ==============================================
    p("-" * 84)
    p("[4] 有效倾角覆盖区间 (whole-video, 默认参数)")
    p("-" * 84)
    groups, cur = [], []
    for i in sorted(res_seq):
        if not res_seq[i].ok:
            continue
        if cur and i - cur[-1] > 3:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    p(f"  {'区间(帧)':>16} {'时间':>18} {'帧数':>6} {'φ 中位':>9} {'φ 范围':>16} "
      f"{'w_body中位':>10}")
    for g in groups:
        if len(g) < 10:
            continue
        v = [res_seq[i].phi_deg for i in g]
        ww = [res_seq[i].w_body_px for i in g if np.isfinite(res_seq[i].w_body_px)]
        p(f"  {f'{g[0]}-{g[-1]}':>16} "
          f"{f'{g[0] / fps:.1f}~{g[-1] / fps:.1f}s':>18} {len(g):>6} "
          f"{np.median(v):>+9.2f} {f'{min(v):+.2f}~{max(v):+.2f}':>16} "
          f"{np.median(ww) if ww else float('nan'):>10.1f}")
    p()

    # ============ 5) 与朴素做法对比 ========================================
    p("-" * 84)
    p("[5] 与朴素做法对比: ROI 内 Otsu + 最大连通域 + minAreaRect")
    p("-" * 84)
    for rng, label in (((46, 66), "落地静止段"), ((15, 30), "下降段(15-30s)")):
        nn, mm = [], []
        for i in sorted(res_seq):
            if not (rng[0] <= i / fps < rng[1]):
                continue
            r = res_seq[i]
            if np.isfinite(r.phi_naive):
                nn.append(r.phi_naive)
            if r.ok:
                mm.append(r.phi_deg)
        if nn:
            p(f"  {label:<16} 朴素 n={len(nn):>3} σ={sd(nn):>7.3f}° "
              f"均值={np.mean(nn):>+8.2f}°   |  本方案 n={len(mm):>3} "
              f"σ={sd(mm):>6.3f}° 均值={np.mean(mm) if mm else float('nan'):+.2f}°")
    p()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validate.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[写入] {OUT / 'validate.txt'}")


if __name__ == "__main__":
    main()
