# -*- coding: utf-8 -*-
"""针对性核查: (a) 抖动帧是否就是多框帧 (b) 38-40s 漏检段逐帧细节 (c) 烟尘段框底边趋势

用法: python scripts/trace_frames.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DIAG = PROJECT / "runs" / "diag"
RB, EF, SP = 1, 0, 2
L: list[str] = []


def p(s=""):
    L.append(s)


def load(tag):
    return json.loads((DIAG / f"dets_{tag}.json").read_text(encoding="utf-8"))


def top(rec, cls, thr):
    return [b for b in rec["b"] if b["c"] == cls and b["s"] >= thr]


d = load("iou70")
frames = d["frames"]
fps = d["meta"]["fps"]
H = d["meta"]["H"]

win = range(200, 2085)

# ---- (a) 抖动帧 vs 多框帧 ----
p("=" * 78)
p("[A] 高度跳变>5px 的帧 是否就是 多框帧?")
p("=" * 78)
multi = {i for i in win if len(top(frames[i], RB, 0.25)) >= 2}
jh = set()
for i in range(1, len(frames)):
    a = top(frames[i - 1], RB, 0.25)
    b = top(frames[i], RB, 0.25)
    if a and b:
        ha = a[0]["x"][3] - a[0]["x"][1]
        hb = b[0]["x"][3] - b[0]["x"][1]
        if abs(ha - hb) > 5:
            jh.add(i)
p(f"  多框帧: {len(multi)}   高度跳变>5px 帧: {len(jh)}")
p(f"  两者交集: {len(multi & jh)}   多框帧中发生跳变: "
  f"{len(multi & jh)}/{len(multi)}")
p(f"  跳变帧但不是多框帧: {sorted(jh - multi)}")
p()

# ---- (b) 38-40s 漏检段逐帧细节 ----
p("=" * 78)
p("[B] 帧 1100-1210 (36.7s-40.3s) 逐帧: RB 最高分 / 框位置 / EF,SP")
p("=" * 78)
p(f"  {'帧':>5} {'t':>6} {'RBbest':>7} {'RB框(x1,y1,x2,y2)':>28} "
  f"{'h':>6} {'w':>5} {'EF':>6} {'SP':>6}")
for i in range(1100, 1211):
    r = frames[i]
    rb = sorted(top(r, RB, 0.02), key=lambda b: -b["s"])
    ef = sorted(top(r, EF, 0.02), key=lambda b: -b["s"])
    sp = sorted(top(r, SP, 0.02), key=lambda b: -b["s"])
    if rb:
        bb = rb[0]
        x = bb["x"]
        p(f"  {i:>5} {i / fps:>6.2f} {bb['s']:>7.3f} "
          f"{str([round(v) for v in x]):>28} {x[3] - x[1]:>6.0f} {x[2] - x[0]:>5.0f} "
          f"{ef[0]['s'] if ef else 0:>6.3f} {sp[0]['s'] if sp else 0:>6.3f}")
    else:
        p(f"  {i:>5} {i / fps:>6.2f} {'--':>7} {'(无RB框)':>28} "
          f"{'-':>6} {'-':>5} {ef[0]['s'] if ef else 0:>6.3f} "
          f"{sp[0]['s'] if sp else 0:>6.3f}")
p()

# ---- (c) 烟尘/落地段: 框底边与图像底部关系 ----
p("=" * 78)
p("[C] 落地段(28s-70s) RB 框上下边沿趋势, 每 2 秒取中位")
p("=" * 78)
p(f"  {'时段':>9} {'帧数':>5} {'y1中位':>7} {'y2中位':>7} {'h中位':>7} "
  f"{'w中位':>7} {'y2/H':>6} {'h/w':>6} {'分中位':>7}")
for s in range(28, 70, 2):
    rows = [top(frames[i], RB, 0.25)[0] for i in win
            if s <= i / fps < s + 2 and top(frames[i], RB, 0.25)]
    if not rows:
        continue
    y1 = sorted(b["x"][1] for b in rows)[len(rows) // 2]
    y2 = sorted(b["x"][3] for b in rows)[len(rows) // 2]
    wd = sorted(b["x"][2] - b["x"][0] for b in rows)[len(rows) // 2]
    ht = sorted(b["x"][3] - b["x"][1] for b in rows)[len(rows) // 2]
    sc = sorted(b["s"] for b in rows)[len(rows) // 2]
    p(f"  {f'{s}-{s + 2}s':>9} {len(rows):>5} {y1:>7.1f} {y2:>7.1f} {ht:>7.1f} "
      f"{wd:>7.1f} {y2 / H:>6.2f} {ht / wd:>6.2f} {sc:>7.3f}")
p()

# ---- (d) 前段(7-15s)小框情况 ----
p("=" * 78)
p("[D] 前段 7s-15s RB 框尺寸(判断能否用于测角)")
p("=" * 78)
p(f"  {'时段':>9} {'帧数':>5} {'w中位':>7} {'h中位':>7} {'最短边中位':>10} "
  f"{'w>=20且h>=20帧数':>16}")
for s in range(5, 16, 1):
    rows = [top(frames[i], RB, 0.25)[0] for i in win
            if s <= i / fps < s + 1 and top(frames[i], RB, 0.25)]
    if not rows:
        continue
    wd = sorted(b["x"][2] - b["x"][0] for b in rows)[len(rows) // 2]
    ht = sorted(b["x"][3] - b["x"][1] for b in rows)[len(rows) // 2]
    ok = sum(1 for b in rows
             if (b["x"][2] - b["x"][0]) >= 20 and (b["x"][3] - b["x"][1]) >= 20)
    p(f"  {f'{s}-{s + 1}s':>9} {len(rows):>5} {wd:>7.1f} {ht:>7.1f} "
      f"{min(wd, ht):>10.1f} {ok:>16}")
p()

out = DIAG / "trace.txt"
out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n[写入] {out}")
