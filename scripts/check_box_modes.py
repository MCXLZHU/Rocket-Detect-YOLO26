# -*- coding: utf-8 -*-
"""核查 RB 框底边(y2)是否存在"两种模式"跳变, 以及各时段模式占比。

用法: python scripts/check_box_modes.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DIAG = PROJECT / "runs" / "diag"
L: list[str] = []


def _sd(v):
    m = sum(v) / len(v)
    return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5


def p(s=""):
    L.append(s)


d = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
frames = d["frames"]
fps = d["meta"]["fps"]
H = d["meta"]["H"]


def best(i):
    bs = [b for b in frames[i]["b"] if b["c"] == 1 and b["s"] >= 0.25]
    return max(bs, key=lambda b: b["s"]) if bs else None


# 逐帧 y1 / y2
seq = {}
for i in range(200, 2085):
    b = best(i)
    if b:
        seq[i] = (b["x"][1], b["x"][3], b["s"])

p("=" * 78)
p("[E] RB 框 上/下边沿 的单帧跳变统计 (只统计相邻帧都有框的情况)")
p("=" * 78)
keys = sorted(seq)
dy1, dy2 = [], []
for a, b in zip(keys, keys[1:]):
    if b - a != 1:
        continue
    dy1.append(seq[b][0] - seq[a][0])
    dy2.append(seq[b][1] - seq[a][1])
p(f"  |Δy1|>3px 帧数: {sum(1 for v in dy1 if abs(v) > 3)} / {len(dy1)}"
  f"   最大 {max(dy1, key=abs):+.0f}px")
p(f"  |Δy2|>3px 帧数: {sum(1 for v in dy2 if abs(v) > 3)} / {len(dy2)}"
  f"   最大 {max(dy2, key=abs):+.0f}px")
p(f"  |Δy2|>30px 帧数: {sum(1 for v in dy2 if abs(v) > 30)}  <- 模式跳变")
p()

p("=" * 78)
p("[F] 各时段 y2(框底边) 的分布 —— 看是否存在两个聚集")
p("=" * 78)
p(f"  {'时段':>9} {'帧数':>5} {'y2中位':>7} {'y2最小':>7} {'y2最大':>7} "
  f"{'y2<300占比':>10} {'y2>350占比':>10}")
for s in range(7, 70, 3):
    ys = [seq[i][1] for i in range(200, 2085) if s <= i / fps < s + 3 and i in seq]
    if not ys:
        continue
    lo = sum(1 for v in ys if v < 300) / len(ys) * 100
    hi = sum(1 for v in ys if v > 350) / len(ys) * 100
    sm = sorted(ys)
    p(f"  {f'{s}-{s + 3}s':>9} {len(ys):>5} {sm[len(sm) // 2]:>7.0f} "
      f"{sm[0]:>7.0f} {sm[-1]:>7.0f} {lo:>9.0f}% {hi:>9.0f}%")
p()

p("=" * 78)
p("[G] 落地静止段(42s-68s) 的 y1/y2 波动 —— 用于判断倾角可复现性")
p("=" * 78)
ys1, ys2, ws, hs = [], [], [], []
for i in sorted(seq):
    if not (42 <= i / fps < 68):
        continue
    y1v, y2v, _s = seq[i]
    ys1.append(y1v)
    ys2.append(y2v)
    hs.append(y2v - y1v)
if ys1:
    def stat(v):
        v = sorted(v)
        return (f"中位 {v[len(v) // 2]:.1f}  标准差 {_sd(v):.2f}  "
                f"极差 {v[-1] - v[0]:.0f}  最小 {v[0]:.0f}  最大 {v[-1]:.0f}")

    p(f"  y1(上边沿): {stat(ys1)}")
    p(f"  y2(下边沿): {stat(ys2)}")
    p(f"  高 (y2-y1): {stat(hs)}")
out = DIAG / "box_modes.txt"
out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n[写入] {out}")
