# -*- coding: utf-8 -*-
"""seg 体检诊断: 统计不可信帧的原因分布, 并打印指定区间的逐帧指标。

为什么需要它: SAM 路线的"体检"门限决定了多少帧可用。门限太紧会在下降段
(箭体只有十几像素宽)成片误杀, 太松则会把漂移放过去。调门限必须看真实分布,
不能凭感觉。

用法:
    python scripts/_seg_diag.py --tag s512
    python scripts/_seg_diag.py --tag s512 --range 240,275
    python scripts/_seg_diag.py --tag s512 --range 800,830 --step 1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache" / "mpl"))
SEG = PROJECT / "runs" / "seg"


def _b(v) -> bool:
    """CSV 里的布尔可能是 "1"/"0" 也可能是 "True"/"False"(早期版本), 都要认。"""
    return str(v).strip().lower() in ("1", "true")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s512")
    ap.add_argument("--range", default="")
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    lines = []
    with (SEG / f"mask_stats{sfx}.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = len(rows)
    has = [r for r in rows if _b(r["has"])]
    good = [r for r in rows if _b(r["ok"])]
    lines.append(f"总帧 {n} | 有掩码 {len(has)} ({len(has) / n:.1%}) | "
                 f"可信 {len(good)} ({len(good) / n:.1%})")
    lines.append(f"无掩码 {sum(1 for r in rows if r['reason'] == 'no_mask')} | "
                 f"无框 {sum(1 for r in rows if r['reason'] == 'no_box')} | "
                 f"无跟踪 {sum(1 for r in rows if r['reason'] == 'no_track')}")
    bad = [r for r in has if not _b(r["ok"])]
    lines.append(f"有掩码但判不可信 {len(bad)} 帧, 原因分布: "
                 + str(Counter(r["reason"] for r in bad)))
    lines.append("")
    # 按 10s 分档看可用率
    fps = 30.0
    lines.append("按 10s 分档可用率:")
    for s in range(0, 74, 10):
        seg = [r for r in rows if s <= int(r["frame"]) / fps < s + 10]
        if not seg:
            continue
        g = sum(1 for r in seg if _b(r["ok"]))
        h = sum(1 for r in seg if _b(r["has"]))
        lines.append(f"  {s:2d}-{s + 10:2d}s  n={len(seg):4d}  有掩码 {h:4d}  "
                     f"可信 {g:4d} ({g / len(seg):5.1%})")
    # 有掩码的帧里, 面积/宽度随时间的量级(判断门限是否失真)
    lines.append("")
    lines.append("有掩码帧的量级(分档中位): area / w_row_med / area_ratio / contain")
    for s in range(0, 74, 10):
        seg = [r for r in has if s <= int(r["frame"]) / fps < s + 10]
        if not seg:
            continue
        def med(k):
            v = [float(r[k]) for r in seg if r[k] not in ("", None)]
            return np.median(v) if v else float("nan")
        lines.append(f"  {s:2d}-{s + 10:2d}s  area={med('area'):9.0f}  "
                     f"w={med('w_row_med'):6.1f}  ar={med('area_ratio'):5.2f}  "
                     f"cont={med('contain'):5.2f}  coff={med('center_off'):5.2f}")
    if a.range:
        lo, hi = (int(v) for v in a.range.split(","))
        lines.append("")
        lines.append(f"--- 逐帧 f{lo}..f{hi} (step {a.step}) ---")
        lines.append("  frame    t   ok has  reason            area    n_c  "
                     "     ar   cont  coff    w_row  h_rows")
        for r in rows[lo:hi:a.step]:
            f = int(r["frame"])
            lines.append(
                f"  {f:>5} {f / fps:5.1f}   {r['ok']}   {r['has']}  "
                f"{r['reason']:<16} {r['area']:>7}  {r['n_comp']:>3}  "
                f"{r['area_ratio']:>6} {r['contain']:>6} {r['center_off']:>6}  "
                f"{r['w_row_med']:>6}  {r['h_rows']:>5}")
    txt = "\n".join(lines)
    (SEG / f"diag{sfx}.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
