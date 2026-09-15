# -*- coding: utf-8 -*-
"""合成图单元测试: 画一根**已知倾角**的亮条, 检验测角链路是否准确地还原它。

这是把"估计器本身的几何与实现是否正确"与"真实画面里的干扰"隔离开的必要手段。
合成图里没有扬尘、没有支腿、没有涂装条纹 —— 若这里都不准, 就别谈真实数据。

用法: python scripts/test_angle_synth.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "mpl"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")

import cv2  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts"))
import rocket_angle as RA  # noqa: E402

OUT = PROJECT / "runs" / "angle"
W, H = 420, 1300
BODY_W = 90          # 筒身宽度(patch px)
BODY_LEN = 1150


def make_synth(m: float, body_w: int = BODY_W, bg_noise: float = 2.0,
               contrast: float = 60.0, seed: int = 0):
    """中心线 x = xc + m*(y - yc) 的亮条。返回 (patch, box)"""
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 255, H, dtype=np.float32)[:, None]
    bg = 120 + 40 * yy / 255.0 + rng.normal(0, bg_noise, (H, W))
    img = np.clip(bg, 0, 255).astype(np.uint8)
    xc, yc = W / 2.0, H / 2.0
    y0, y1 = (H - BODY_LEN) / 2.0, (H + BODY_LEN) / 2.0
    xa, xb = xc + m * (y0 - yc), xc + m * (y1 - yc)
    poly = np.array([[xa - body_w / 2, y0], [xa + body_w / 2, y0],
                     [xb + body_w / 2, y1], [xb - body_w / 2, y1]], np.int32)
    img = cv2.fillPoly(img, [poly], int(np.clip(120 + 40 * yc / 255.0
                                                + contrast, 0, 255)))
    # 轴对齐框(真实场景里 YOLO 给的就是这个)
    xs = [xa - body_w / 2, xa + body_w / 2, xb - body_w / 2, xb + body_w / 2]
    box = np.array([min(xs), y0, max(xs), y1], float)
    return img.astype(np.float32), box


def main():
    cfg = RA.AngleConfig()
    print("=" * 78)
    print("合成图单元测试: 已知倾角 → 测量值")
    print("=" * 78)
    print(f"  画面 {W}x{H}, 筒身 {BODY_W}x{BODY_LEN}px, 亮条对比度 +60 灰阶, "
          f"背景 2 灰阶高斯噪声")
    print(f"  {'真实倾角':>9} {'测量φ':>9} {'偏差':>8} {'w_body':>8} "
          f"{'rms':>7} {'段数':>5} {'极性':>5}")
    rows = []
    for tilt in (-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0):
        m = -math.tan(math.radians(tilt))     # φ = atan(-m) = tilt
        img, box = make_synth(m)
        est = RA.AngleEstimator(cfg)
        r = est.step(img, box, (0, 0), 0, 1.0)
        rows.append((tilt, r.phi_deg))
        print(f"  {tilt:>9.1f} {r.phi_deg:>9.3f} {r.phi_deg - tilt:>8.3f} "
              f"{r.w_body_px:>8.1f} {r.rms_px:>7.3f} {r.n_bands:>5} "
              f"{r.polarity:>5}")
    xs = np.array([a for a, b in rows if np.isfinite(b)])
    ys = np.array([b for a, b in rows if np.isfinite(b)])
    if len(xs) >= 3:
        k = np.polyfit(xs, ys, 1)
        dev = ys - np.polyval(k, xs)
        print()
        print(f"  线性拟合: 测得 = {k[0]:.4f} × 真实 + {k[1]:+.4f}")
        print(f"  增益 {k[0]:.4f} (理想 1.000)   线性残差σ {np.std(dev):.3f}°   "
              f"最大偏差 {np.max(np.abs(dev)):.3f}°")

    # 加一根竞品结构(模拟支腿/蒸汽带), 看是否被带偏
    print()
    print("-" * 78)
    print("抗干扰测试: 在亮条旁边加一根更亮的斜条(模拟支腿)")
    print("-" * 78)
    print(f"  {'真实倾角':>9} {'测量φ':>9} {'偏差':>8}")
    for tilt in (-2.0, 0.0, 2.0):
        m = -math.tan(math.radians(tilt))
        img, box = make_synth(m)
        # 右侧加一根倾斜 12° 的亮条
        y0, y1 = 200, 1100
        m2 = -math.tan(math.radians(12.0))
        xa = 300 + m2 * (y0 - H / 2)
        xb = 300 + m2 * (y1 - H / 2)
        poly = np.array([[xa - 6, y0], [xa + 6, y0], [xb + 6, y1],
                         [xb - 6, y1]], np.int32)
        img2 = cv2.fillPoly(img.copy(), [poly], 235)
        est = RA.AngleEstimator(cfg)
        r = est.step(img2, box, (0, 0), 0, 1.0)
        print(f"  {tilt:>9.1f} {r.phi_deg:>9.3f} {r.phi_deg - tilt:>8.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    imgs = []
    for tilt in (-2.0, 0.0, 2.0):
        m = -math.tan(math.radians(tilt))
        img, box = make_synth(m)
        g = cv2.cvtColor(cv2.resize(np.clip(img, 0, 255).astype(np.uint8),
                                    (W // 2, H // 2)), cv2.COLOR_GRAY2BGR)
        cv2.putText(g, f"tilt={tilt:+.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1, cv2.LINE_AA)
        imgs.append(g)
    cv2.imwrite(str(OUT / "synth.jpg"), np.hstack(imgs),
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"\n合成样例图 -> {OUT / 'synth.jpg'}")


if __name__ == "__main__":
    main()
