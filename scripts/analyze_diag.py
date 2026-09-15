# -*- coding: utf-8 -*-
"""读 runs/diag/dets_*.json, 量化视频检测的三类问题, 输出 runs/diag/analysis.txt。

用法: python scripts/analyze_diag.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DIAG = PROJECT / "runs" / "diag"
RB, EF, SP = 1, 0, 2
NAMES = {0: "Engine Flames", 1: "Rocket Body", 2: "Space"}

L: list[str] = []


def p(s: str = "") -> None:
    L.append(s)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ios(a, b) -> float:
    """交集 / 较小框面积 —— 用于识别"嵌套框"(IoU 会失效)。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    small = min(aa, ba)
    return inter / small if small > 0 else 0.0


def load(tag: str):
    f = DIAG / f"dets_{tag}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def top_boxes(rec, cls, thr):
    return [b for b in rec["b"] if b["c"] == cls and b["s"] >= thr]


def runs_of_zero(flags: list[bool]) -> list[tuple[int, int]]:
    """返回 True 连续区段 [(start, end), ...] (闭区间, 0-based)"""
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(flags) - 1))
    return out


def main() -> None:
    d70 = load("iou70")
    d30 = load("iou30")
    if d70 is None:
        print("缺少 dets_iou70.json")
        return
    meta = d70["meta"]
    frames = d70["frames"]
    fps = meta["fps"]
    N = len(frames)
    W, H = meta["W"], meta["H"]

    p("=" * 78)
    p("视频检测诊断报告")
    p("=" * 78)
    p(f"视频: {meta['video']}")
    p(f"规格: {W}x{H}  {fps:.2f} fps  {N} 帧  {N / fps:.1f} 秒")
    p(f"权重: {meta['weights']}")
    p(f"导出参数: conf={meta['conf']} iou={meta['iou']} imgsz={meta['imgsz']}")
    p()

    # ---------- 0. 有效飞行窗口 ----------
    active = [i for i, r in enumerate(frames)
              if any(b["s"] >= 0.25 for b in r["b"])]
    a0, a1 = active[0], active[-1]
    p("-" * 78)
    p("[0] 有效飞行窗口(任一类 conf>=0.25)")
    p("-" * 78)
    p(f"  帧区间: {a0} .. {a1}  (t = {a0 / fps:.2f}s .. {a1 / fps:.2f}s)")
    p(f"  长度  : {a1 - a0 + 1} 帧 = {(a1 - a0 + 1) / fps:.1f} 秒")
    p(f"  窗口外帧数: {N - (a1 - a0 + 1)} (片头/片尾黑场, 属正常)")
    p()

    # ---------- 1. 置信度阈值 vs RB 检出率 ----------
    p("-" * 78)
    p("[1] Rocket Body 检出率 vs 置信度阈值 (统计窗口内)")
    p("-" * 78)
    p(f"  {'conf':>6} {'RB框数':>7} {'有RB帧':>8} {'缺RB帧':>8} {'帧数>1':>8} "
      f"{'最长连续缺帧':>13}")
    win = range(a0, a1 + 1)
    for thr in (0.25, 0.20, 0.15, 0.10, 0.05, 0.02):
        nb = sum(len(top_boxes(frames[i], RB, thr)) for i in win)
        n1 = sum(1 for i in win if len(top_boxes(frames[i], RB, thr)) >= 1)
        n2 = sum(1 for i in win if len(top_boxes(frames[i], RB, thr)) >= 2)
        miss = [len(top_boxes(frames[i], RB, thr)) == 0 for i in win]
        rr = runs_of_zero(miss)
        longest = max((e - s + 1 for s, e in rr), default=0)
        p(f"  {thr:>6.2f} {nb:>7} {n1:>8} {len(miss) - n1:>8} {n2:>8} {longest:>13}")
    p()

    # ---------- 2. 漏检区间详情 ----------
    p("-" * 78)
    p("[2] RB 漏检区间详情 (以 conf>=0.25 判定)")
    p("-" * 78)
    miss = [len(top_boxes(frames[i], RB, 0.25)) == 0 for i in win]
    rr = [(s + a0, e + a0) for s, e in runs_of_zero(miss)]
    rr = [(s, e) for s, e in rr if e - s + 1 >= 2]
    p(f"  连续缺 RB 的区段(长度>=2 帧), 共 {len(rr)} 段:")
    p(f"  {'区间(帧)':>16} {'时长':>7} {'该段RB最高分':>13} "
      f"{'该段EF最高分':>13} {'该段SP最高分':>13} {'全类最高分':>11}")
    for s, e in rr:
        rbs = [b["s"] for i in range(s, e + 1) for b in top_boxes(frames[i], RB, 0.02)]
        efs = [b["s"] for i in range(s, e + 1) for b in top_boxes(frames[i], EF, 0.02)]
        sps = [b["s"] for i in range(s, e + 1) for b in top_boxes(frames[i], SP, 0.02)]
        alls = [b["s"] for i in range(s, e + 1) for b in frames[i]["b"]]
        p(f"  {f'{s}-{e}':>16} {e - s + 1:>5}帧 "
          f"{max(rbs, default=0):>13.3f} {max(efs, default=0):>13.3f} "
          f"{max(sps, default=0):>13.3f} {max(alls, default=0):>11.3f}")
    p()

    # ---------- 3. 多框(重复/嵌套)分析 ----------
    p("-" * 78)
    p("[3] 同一帧出现 >=2 个 Rocket Body 框的情况")
    p("-" * 78)
    for tag, d in (("iou=0.7(默认)", d70), ("iou=0.3(严格NMS)", d30)):
        if d is None:
            continue
        fr = d["frames"]
        cnt = [(i, top_boxes(fr[i], RB, 0.25)) for i in win
               if len(top_boxes(fr[i], RB, 0.25)) >= 2]
        p(f"  {tag}: {len(cnt)} 帧出现多框")
    p()
    p("  默认参数下多框帧明细 (前 40 条):")
    p(f"  {'帧':>6} {'t(s)':>7} {'#1分':>7} {'#2分':>7} {'#3分':>7} "
      f"{'IoU(1,2)':>9} {'IoS(1,2)':>9} {'面积比':>8}")
    multi = []
    for i in win:
        bs = top_boxes(frames[i], RB, 0.25)
        if len(bs) >= 2:
            multi.append((i, bs))
    for i, bs in multi[:40]:
        a, b = bs[0]["x"], bs[1]["x"]
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ba = (b[2] - b[0]) * (b[3] - b[1])
        ratio = min(aa, ba) / max(aa, ba) if max(aa, ba) > 0 else 0
        s3 = f"{bs[2]['s']:>7.3f}" if len(bs) > 2 else f"{'-':>7}"
        p(f"  {i:>6} {i / fps:>7.2f} {bs[0]['s']:>7.3f} {bs[1]['s']:>7.3f} {s3} "
          f"{iou(a, b):>9.3f} {ios(a, b):>9.3f} {ratio:>8.3f}")
    p()
    if multi:
        iosc = [ios(bs[0]["x"], bs[1]["x"]) for _, bs in multi]
        iouc = [iou(bs[0]["x"], bs[1]["x"]) for _, bs in multi]
        dsc = [bs[0]["s"] - bs[1]["s"] for _, bs in multi]
        p(f"  汇总({len(multi)} 帧):")
        p(f"    IoS(1,2)  中位 {st.median(iosc):.3f}  最小 {min(iosc):.3f}  "
          f"最大 {max(iosc):.3f}   -> IoS>0.8 的帧数 "
          f"{sum(1 for v in iosc if v > 0.8)}")
        p(f"    IoU(1,2)  中位 {st.median(iouc):.3f}  最大 {max(iouc):.3f}   "
          f"-> IoU>0.45 的帧数 {sum(1 for v in iouc if v > 0.45)}")
        p(f"    分差(#1-#2) 中位 {st.median(dsc):.3f}  最小 {min(dsc):.3f}  "
          f"最大 {max(dsc):.3f}")
        p(f"    #2 分数: 中位 {st.median([bs[1]['s'] for _, bs in multi]):.3f}")
    p()

    # ---------- 4. RB 框尺寸随时间 ----------
    p("-" * 78)
    p("[4] Rocket Body 最优框尺寸随时间变化 (每 5 秒聚合)")
    p("-" * 78)
    p(f"  {'时段':>10} {'帧数':>6} {'框宽中位':>9} {'框高中位':>9} "
      f"{'高/宽':>7} {'中心Y中位':>10} {'最高分中位':>11}")
    for s in range(int(a0 / fps), int(a1 / fps) + 1, 5):
        rows = []
        for i in range(a0, a1 + 1):
            if not (s <= i / fps < s + 5):
                continue
            bs = top_boxes(frames[i], RB, 0.25)
            if bs:
                rows.append(bs[0])
        if not rows:
            continue
        wd = [b["x"][2] - b["x"][0] for b in rows]
        ht = [b["x"][3] - b["x"][1] for b in rows]
        cy = [(b["x"][1] + b["x"][3]) / 2 for b in rows]
        p(f"  {f'{s}-{s + 5}s':>10} {len(rows):>6} {st.median(wd):>9.1f} "
          f"{st.median(ht):>9.1f} {st.median(ht) / st.median(wd):>7.2f} "
          f"{st.median(cy):>10.1f} {st.median([b['s'] for b in rows]):>11.3f}")
    p()

    # ---------- 5. 框抖动(对后续测角的影响) ----------
    p("-" * 78)
    p("[5] 最优框逐帧跳变(抖动) —— 直接影响第 3 步倾角估计")
    p("-" * 78)
    seq = []
    for i in win:
        bs = top_boxes(frames[i], RB, 0.25)
        if bs:
            x = bs[0]["x"]
            seq.append((i, (x[0] + x[2]) / 2, (x[1] + x[3]) / 2,
                        x[2] - x[0], x[3] - x[1]))
    dc, dw, dh = [], [], []
    for k in range(1, len(seq)):
        if seq[k][0] - seq[k - 1][0] != 1:
            continue
        dc.append(((seq[k][1] - seq[k - 1][1]) ** 2 +
                   (seq[k][2] - seq[k - 1][2]) ** 2) ** 0.5)
        dw.append(abs(seq[k][3] - seq[k - 1][3]))
        dh.append(abs(seq[k][4] - seq[k - 1][4]))
    if dc:
        p(f"  相邻帧中心位移 |dc|: 中位 {st.median(dc):.2f}px  "
          f"P90 {sorted(dc)[int(len(dc) * 0.9)]:.2f}px  最大 {max(dc):.2f}px")
        p(f"  相邻帧宽度跳变 |dw|: 中位 {st.median(dw):.2f}px  "
          f"P90 {sorted(dw)[int(len(dw) * 0.9)]:.2f}px  最大 {max(dw):.2f}px")
        p(f"  相邻帧高度跳变 |dh|: 中位 {st.median(dh):.2f}px  "
          f"P90 {sorted(dh)[int(len(dh) * 0.9)]:.2f}px  最大 {max(dh):.2f}px")
        p(f"  跳变>5px 的帧数: 中心 {sum(1 for v in dc if v > 5)} / "
          f"高度 {sum(1 for v in dh if v > 5)}  (共 {len(dc)} 对)")
    p()

    # ---------- 6. 各类同时出现的分布 ----------
    p("-" * 78)
    p("[6] 各类随时间共存情况 (conf>=0.25, 每 3 秒)")
    p("-" * 78)
    p(f"  {'时段':>9} {'有EF帧':>7} {'有RB帧':>7} {'有SP帧':>7} {'无任何目标帧':>13}")
    for s in range(int(a0 / fps), int(a1 / fps) + 1, 3):
        idx = [i for i in range(a0, a1 + 1) if s <= i / fps < s + 3]
        if not idx:
            continue
        ef = sum(1 for i in idx if top_boxes(frames[i], EF, 0.25))
        rb = sum(1 for i in idx if top_boxes(frames[i], RB, 0.25))
        sp = sum(1 for i in idx if top_boxes(frames[i], SP, 0.25))
        no = sum(1 for i in idx if not frames[i]["b"])
        p(f"  {f'{s}-{s + 3}s':>9} {ef:>7} {rb:>7} {sp:>7} {no:>13}")
    p()

    out = DIAG / "analysis.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[写入] {out}")


if __name__ == "__main__":
    main()
