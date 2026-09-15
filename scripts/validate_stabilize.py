# -*- coding: utf-8 -*-
"""用已存盘的逐帧检测结果(runs/diag/dets_iou70.json)离线对比:
   A. 原始做法   —— 取分数最高的 RB 框, conf 固定 0.25
   B. 本方案     —— 嵌套抑制 + 双阈值续轨 + 因果中值 + KF 补位
并扫描几个关键参数, 输出 runs/diag/stabilize_report.txt。

不需要重新推理, 几秒跑完。
用法: python scripts/validate_stabilize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, ios_xyxy as ios, track_frames  # noqa: E402

DIAG = PROJECT / "runs" / "diag"
L: list[str] = []


def p(s=""):
    L.append(s)


def sd(v):
    return float(np.std(v)) if len(v) else 0.0


d = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
frames_raw = d["frames"]
meta = d["meta"]
fps = meta["fps"]
W, H = meta["W"], meta["H"]
N = len(frames_raw)
FH = [r["b"] for r in frames_raw]

# ---- 评估窗口: RB 真正存在的区间 ------------------------------------------
rb_frames = [i for i, r in enumerate(frames_raw)
             if any(b["c"] == 1 and b["s"] >= 0.25 for b in r["b"])]
W0, W1 = rb_frames[0], rb_frames[-1]
WIN = list(range(W0, W1 + 1))

p("=" * 82)
p("检测稳定化: 前后对比 (数据源 runs/diag/dets_iou70.json)")
p("=" * 82)
p(f"视频 {meta['video']}")
p(f"{W}x{H}  {fps:.2f} fps  {N} 帧")
p(f"RB 评估窗口: 帧 {W0}..{W1}  (t {W0 / fps:.2f}s..{W1 / fps:.2f}s, "
  f"共 {len(WIN)} 帧)")
p()


def baseline(conf):
    boxes = [None] * N
    for i in WIN:
        bs = [b for b in frames_raw[i]["b"] if b["c"] == 1 and b["s"] >= conf]
        if bs:
            boxes[i] = max(bs, key=lambda b: b["s"])["x"]
    return boxes


def trace_metrics(boxes, label, note="", n_sup=0, n_multi_cand=0, n_coast=0,
                  n_switch=0, roi_cover=None):
    miss = [1 if boxes[i] is None else 0 for i in WIN]
    run = best = 0
    for m in miss:
        run = run + 1 if m else 0
        best = max(best, run)
    seq = [(i, boxes[i]) for i in WIN if boxes[i] is not None]
    dy2, dcx, dh = [], [], []
    oob = 0
    for (a, ba), (b, bb) in zip(seq, seq[1:]):
        if b - a != 1:
            continue
        dy2.append(abs(bb[3] - ba[3]))
        dcx.append(abs((bb[0] + bb[2]) / 2 - (ba[0] + ba[2]) / 2))
        dh.append(abs((bb[3] - bb[1]) - (ba[3] - ba[1])))
    for _, bb in seq:
        if bb[3] > H + 1 or bb[1] < -1:
            oob += 1
    ys1, ys2, cxs, hs = [], [], [], []
    for i in WIN:
        if not (42 <= i / fps < 68) or boxes[i] is None:
            continue
        b = boxes[i]
        ys1.append(b[1])
        ys2.append(b[3])
        cxs.append((b[0] + b[2]) / 2)
        hs.append(b[3] - b[1])
    p(f"  {label:<20} 有框 {len(seq):>4}/{len(WIN)} 缺 {sum(miss):>3} "
      f"连缺 {best:>3} | |Δy2|max {max(dy2, default=0):>4.0f} | 出界 {oob:>3} | "
      f"抑制 {n_sup:>3} | 多框帧 {n_multi_cand:>3} | 预测 {n_coast:>3} | "
      f"模式切换 {n_switch:>2}")
    if ys2:
        p(f"  {'':<20} 静止段 σ(y1)={sd(ys1):.2f} σ(y2)={sd(ys2):.2f} "
          f"σ(cx)={sd(cxs):.2f} y1极差={max(ys1) - min(ys1):.0f} "
          f"y2极差={max(ys2) - min(ys2):.0f} 中位高={np.median(hs):.0f}")
    if roi_cover is not None:
        p(f"  {'':<20} ROI 未包住 raw 高分框的帧: {roi_cover}/{len(WIN)}"
          f"   (应≈0; ROI 宁可大不可裁)")
    if note:
        p(f"  {'':<20} {note}")
    return dict(miss=sum(miss), longest=best, dy2=max(dy2, default=0))


p("-" * 82)
p("[1] 基线: 取分数最高的 RB 框, 固定阈值(不做任何时序处理)")
p("-" * 82)
raw_dup25 = sum(1 for i in WIN
                if len([b for b in frames_raw[i]["b"]
                        if b["c"] == 1 and b["s"] >= 0.25]) >= 2)
b25 = baseline(0.25)
m25 = trace_metrics(b25, "raw conf>=0.25", n_multi_cand=raw_dup25)
trace_metrics(baseline(0.15), "raw conf>=0.15")
trace_metrics(baseline(0.10), "raw conf>=0.10")
trace_metrics(baseline(0.05), "raw conf>=0.05")
p()
p(f"  注: raw conf>=0.25 时共 {raw_dup25} 帧出现 >=2 个 RB 框(用户观察到的现象);")
p("      降阈值能补漏检, 但同时把重复框从 47 帧涨到 115 帧(见 [1] 各行),")
p("      所以不能只靠调阈值 —— 必须把'建轨阈值'和'续轨阈值'分开。")
p()

p("-" * 82)
p("[2] 本方案: 不同参数组合")
p("-" * 82)
variants = [
    ("默认", TrackerConfig()),
    ("sup_keep=larger", TrackerConfig(sup_keep="larger")),
    ("sup_keep=anchor", TrackerConfig(sup_keep="anchor")),
    ("conf_low=0.15", TrackerConfig(conf_low=0.15)),
    ("conf_low=0.05", TrackerConfig(conf_low=0.05)),
    ("median_win=1", TrackerConfig(median_win=1)),
    ("median_win=5", TrackerConfig(median_win=5)),
    ("mode_persist=1", TrackerConfig(mode_persist=1)),
    ("mode_persist=5", TrackerConfig(mode_persist=5)),
    ("max_coast=10", TrackerConfig(max_coast=10)),
]
for name, cfg in variants:
    outs = track_frames(FH, cfg, fps, bound_wh=(W, H))
    boxes = [None] * N
    rois = [None] * N
    n_coast = n_switch = n_sup = n_multi = 0
    for o in outs:
        if o.has and o.box is not None:
            boxes[o.frame] = o.box
            rois[o.frame] = o.roi
            if o.state == "COASTED":
                n_coast += 1
        if o.mode_switched:
            n_switch += 1
        n_sup += o.n_dup_removed
        if o.n_dup_removed > 0:
            n_multi += 1
    # ROI 是否漏裁: 用本帧原始最高分 RB 框与 ROI 的 IoS 判定(>=0.98 视为包住)
    bad = 0
    for i in WIN:
        bs = [b for b in frames_raw[i]["b"] if b["c"] == 1 and b["s"] >= 0.25]
        if not bs or rois[i] is None:
            continue
        rb_box = np.asarray(max(bs, key=lambda b: b["s"])["x"], float)
        if ios(rb_box, rois[i]) < 0.98:
            bad += 1
    trace_metrics(boxes, name, n_sup=n_sup, n_multi_cand=n_multi,
                  n_coast=n_coast, n_switch=n_switch, roi_cover=bad)
    p()

p("-" * 82)
p("[3] 默认参数下的补位行为")
p("-" * 82)
cfg = TrackerConfig()
outs = track_frames(FH, cfg, fps, bound_wh=(W, H))
ev = [o for o in outs if W0 <= o.frame <= W1]
coast = [o for o in ev if o.state == "COASTED"]
p(f"  预测补位(coasting) {len(coast)} 帧: {[o.frame for o in coast]}")
p(f"  底边模式切换 {sum(1 for o in ev if o.mode_switched)} 帧: "
  f"{[o.frame for o in ev if o.mode_switched][:30]}")
p(f"  最低观察分 {min([o.score for o in ev if o.state == 'OBSERVED'], default=0):.3f}"
  "  (conf>=0.25 时这些帧会被判为漏检)")
p()
p("  原方案判为漏检、现方案接上的帧:")
p(f"  {'帧':>6} {'t(s)':>7} {'raw最高分':>10} {'状态':>10} {'宽x高':>12} "
  f"{'y1..y2':>14} {'可靠度':>7}")
k = 0
for o in ev:
    if b25[o.frame] is not None or not o.has:
        continue
    raw = max([b for b in frames_raw[o.frame]["b"] if b["c"] == 1],
              key=lambda b: b["s"], default=None)
    b = o.box
    raw_s = f"{raw['s']:.3f}" if raw else "--"
    p(f"  {o.frame:>6} {o.frame / fps:>7.2f} {raw_s:>10} {o.state:>10} "
      f"{f'{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}':>12} "
      f"{f'{b[1]:.0f}..{b[3]:.0f}':>14} {o.reliability:>7.2f}")
    k += 1
    if k >= 42:
        p("  ...")
        break
p()

p("-" * 82)
p("[4] 输出给第 3 步(测角)的门控")
p("-" * 82)
ok = [o for o in ev if o.angle_ok]
p(f"  angle_ok=True 的帧: {len(ok)}/{len(ev)}")
groups, cur = [], []
for o in ok:
    if cur and o.frame - cur[-1].frame > 2:
        groups.append(cur)
        cur = []
    cur.append(o)
if cur:
    groups.append(cur)
p(f"  {'区间(帧)':>16} {'时间':>18} {'帧数':>6} {'中点框(宽x高)':>16}")
for g in groups:
    if len(g) < 5:
        continue
    b = g[len(g) // 2].box
    p(f"  {f'{g[0].frame}-{g[-1].frame}':>16} "
      f"{f'{g[0].frame / fps:.1f}~{g[-1].frame / fps:.1f}s':>18} {len(g):>6} "
      f"{f'{b[2] - b[0]:.0f}x{b[3] - b[1]:.0f}':>16}")
p()
p("  结论: 可测角区间约从 11s 起; 40s 之后火箭已落地静止, 倾角退化为常数,")
p("        真正携带姿态变化信息的是 11~40s 这一段。")

out = DIAG / "stabilize_report.txt"
out.write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
print(f"\n[写入] {out}")
