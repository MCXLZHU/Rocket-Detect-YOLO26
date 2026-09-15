# -*- coding: utf-8 -*-
"""Δφ 下段该用多大的刻度? —— 定量比较几种取轴规则。

为什么查: Δφ(t) = φ_SAM - φ_原方案, 只有零点几度; 单个失稳帧会甩出一根 ±3° 的
尖峰。若按"最大值/高分位"取轴, 尖峰一个人就决定了整根轴, 真正关心的落地段差值
(约 +0.8°) 会被压成一条几乎贴零的线。本脚本列出各分位对应的轴半宽, 便于取舍。
"""

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(r"E:\RocketAttitudeEstimation")
ANG = ROOT / "runs" / "angle"
ANGM = ROOT / "runs" / "angle_mask"
FPS = 30.0


def csv_map(path):
    if not path.exists():
        return {}
    with Path(path).open(encoding="utf-8") as fh:
        return {int(r["frame"]): r for r in csv.DictReader(fh)}


def fnum(r, k="phi_smooth"):
    if not r:
        return None
    v = r.get(k)
    if v in (None, "", "None"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


org = {int(r["frame"]): r
       for r in json.loads((ANG / "angles.json").read_text(
           encoding="utf-8"))["frames"]}


def ok(m, f):
    r = m.get(f)
    return bool(r) and str(r.get("ok", "0")) in ("1", "True")


def diffs(tag):
    m = csv_map(ANGM / f"angles_{tag}.csv")
    out = []
    for f, r in sorted(m.items()):
        a, b = fnum(r), fnum(org.get(f))
        if a is not None and b is not None and ok(m, f) and ok(org, f):
            out.append((f, abs(a - b)))
    return out


for tag in ("s512", "s1024"):
    dv = diffs(tag)
    if not dv:
        continue
    v = np.array([x for _, x in dv])
    print(f"\n=== {tag} vs 原方案  样本 {len(v)} 帧 ===")
    print(f"  中位 {np.median(v):.2f}   均值 {v.mean():.2f}   "
          f"max {v.max():.2f}")
    print("  分位: " + "  ".join(
        f"p{q}={np.percentile(v, q):.2f}" for q in (50, 90, 96, 98, 99, 99.5)))
    # 落地段(46-66s)最关心, 单列
    land = np.array([x for f, x in dv if 46 * FPS <= f < 66 * FPS])
    if len(land):
        print(f"  落地段 46-66s: n={len(land)}  中位 {np.median(land):.2f}  "
              f"max {land.max():.2f}")
    print("  各种取轴规则 -> Δφ 轴半宽:")
    for q in (90, 96, 98, 99, 99.5):
        raw = np.percentile(v, q) * 1.25
        print(f"    p{q:<4} 分位 x1.25 = {raw:5.2f}   "
              f"截顶 {int(np.sum(v > raw)):>3} 帧 "
              f"({np.sum(v > raw) / len(v) * 100:.1f}%)")
    print(f"    max x1.0        = {v.max():5.2f}   "
          "截顶   0 帧 (0.0%)")
