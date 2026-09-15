# -*- coding: utf-8 -*-
"""查 φ 的真实范围: 全片 min/max/分位, 以及前 25s 的典型值。

为什么查: 用户用屏幕量角器量出早期约 13°, 而曲线图纵轴固定 ±8°, 说明早期数据被截顶了。
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\RocketAttitudeEstimation")
ANG = ROOT / "runs" / "angle"
ANGM = ROOT / "runs" / "angle_mask"


def series(path, key="phi_smooth"):
    d = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    v = np.full(len(d), np.nan)
    for i, r in enumerate(d):
        if r.get("ok") and r.get(key) is not None:
            v[i] = float(r[key])
    return v, d


s512, _ = series(ANGM / "angles_s512.json")
s1024, _ = series(ANGM / "angles_s1024.json")
org, _ = series(ANG / "angles.json")
fps = 30.0

for name, v in (("SAM@512", s512), ("SAM@1024", s1024), ("原方案", org)):
    f = v[np.isfinite(v)]
    print(f"\n{name}  有效 {len(f)}")
    print(f"  min {np.min(f):+.2f}  max {np.max(f):+.2f}  极差 {np.ptp(f):.2f}")
    for q in (1, 2, 50, 98, 99):
        print(f"  p{q:<2} {np.percentile(f, q):+.2f}", end="")
    print()
    print(f"  |φ| p98 = {np.percentile(np.abs(f), 98):.2f}   "
          f"|φ| max = {np.max(np.abs(f)):.2f}")
    # 早期典型值
    for lo, hi in ((7, 12), (11, 15), (15, 20), (20, 25)):
        idx = np.arange(int(lo * fps), int(hi * fps))
        seg = v[idx[np.isfinite(v[idx])]]
        if len(seg):
            print(f"  {lo:>2}-{hi:>2}s: n={len(seg):<4} "
                  f"中位 {np.median(seg):+6.2f}  "
                  f"范围 [{np.min(seg):+6.2f}, {np.max(seg):+6.2f}]")

print("\n--- 落地段 46-66s 的绝对 φ 中位(与报告里 +0.8° 的口径对齐) ---")
for name, v in (("SAM@512", s512), ("SAM@1024", s1024), ("原方案", org)):
    idx = np.arange(int(46 * fps), int(66 * fps))
    seg = v[idx[np.isfinite(v[idx])]]
    if len(seg):
        print(f"  {name:<9} n={len(seg):<4} 中位 {np.median(seg):+6.2f}  "
              f"σ {np.std(seg):.3f}  范围 [{np.min(seg):+.2f}, {np.max(seg):+.2f}]")

print("\n--- 若纵轴取 ±X°, 各路线被截顶的帧数 ---")
for X in (8, 10, 12, 15, 18, 20, 24):
    n = sum(int(np.sum(np.abs(v[np.isfinite(v)]) > X)) for v in (s512, s1024, org))
    tot = sum(int(np.sum(np.isfinite(v))) for v in (s512, s1024, org))
    print(f"  ±{X:>2}°  截顶 {n:>5} / {tot}  ({n / tot * 100:.2f}%)")
