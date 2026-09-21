# -*- coding: utf-8 -*-
"""把 runs/bench/pipeline_speed.json 与 runs/seg/bench_sp*.json 变成一张速度分解图。

为什么单独做一个报告脚本: 速度数据是"带条件"的(冷/热抽帧、512/1024、功耗墙状态),
散在 JSON 里没人看得懂; 图要能一眼回答三个问题:
  1. 每帧时间花在哪个阶段?(堆叠条) —— 决定优化方向
  2. 和 30fps 实时门槛差多少?(虚线)
  3. seg 阶段内部, 是"搬运"贵还是"算"贵?(第二张图) —— 决定改 CPU 还是改 GPU

用法:
    python scripts/bench_report.py                       # 默认 tag
    python scripts/bench_report.py --runs bench512b,bench1024b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT = Path(__file__).resolve().parent.parent
BENCH = PROJECT / "runs" / "bench"
SEG = PROJECT / "runs" / "seg"

CN = {"decode": "视频解码", "stabilize": "检测稳定化", "detect": "YOLO 检测",
      "seg": "SAM 掩码传播", "angle_mask": "掩码测角", "viz": "可视化产出"}
COL = {"decode": "#8c8c8c", "stabilize": "#bdbdbd", "detect": "#ff7f0e",
       "seg": "#1f77b4", "angle_mask": "#2ca02c", "viz": "#9467bd"}


def load_runs() -> list[dict]:
    p = BENCH / "pipeline_speed.json"
    if not p.exists():
        return []
    runs = json.loads(p.read_text(encoding="utf-8"))
    runs = runs if isinstance(runs, list) else runs.get("runs", [])
    # 只保留每个阶段都成功的 run(失败的那次只有几秒, 混进来会把均值拉歪)
    return [r for r in runs
            if r.get("stages") and all(v.get("exit_code", 0) == 0
                                       for v in r["stages"].values())]


def load_model_bench() -> dict:
    out = {}
    for tag in ("sp512", "sp1024"):
        p = SEG / f"bench_{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        d = d if isinstance(d, dict) else d[-1]
        v = (d.get("runs") or [d])[-1]
        vv = v.get("video") or {}
        if "init_state_s" in vv:
            out[v["image_size"]] = {
                "load_ms": round(vv["init_state_s"] * 1000
                                 / max(vv["n_frames_loaded"], 1), 2),
                "prop_ms": vv.get("prop_fwd_ms_median"),
                "prop_fps": vv.get("prop_fps"),
                "vram_mb": vv.get("vram_peak_mb"),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="速度分解图")
    ap.add_argument("--runs", default="",
                    help="要画哪几个 run 的 tag(逗号分隔); 默认全部成功的")
    ap.add_argument("--deadline-ms", type=float, default=1000 / 30,
                    help="实时门槛(默认 30fps ⇒ 33.3ms)")
    a = ap.parse_args()

    runs = load_runs()
    if a.runs:
        want = [x.strip() for x in a.runs.split(",") if x.strip()]
        runs = [r for r in runs if r["tag"] in want]
    if not runs:
        print("[x] 没有可用的测速结果, 先跑 python scripts/bench_pipeline.py")
        return

    # 各次 run 测的阶段不一样(有的只跑 seg), 不能拉成矩阵画 —— 按"链路"合成:
    # 解码/稳定化/YOLO 这些与 SAM 分辨率无关的阶段, 统一取全链路那一次(chain)的实测值,
    # SAM 阶段取各自 run 的值。否则就得为了凑齐阶段重复跑 4 分钟的检测。
    order = ["decode", "stabilize", "detect", "seg", "angle_mask"]
    chain = max(runs, key=lambda r: len(r["stages"]))
    rows: list[tuple[str, dict]] = []
    seen_seg = False
    for r in runs:
        if "seg" not in r["stages"]:
            continue
        st: dict[str, float] = {}
        for k in ("decode", "stabilize", "detect"):
            src = r if k in r["stages"] else chain
            if k in src["stages"]:
                st[k] = src["stages"][k]["ms_per_frame"]
        for k in ("seg", "angle_mask"):
            if k in r["stages"]:
                st[k] = r["stages"][k]["ms_per_frame"]
        # 首次跑 seg 的那次要现抽帧(冷), 之后帧已在 .cache/seg_frames 里(热) ——
        # 这个差别值 62s/全片, 必须在图里区分开, 否则数字互相打架说不清。
        thermal = "冷:含抽帧" if not seen_seg else "热:帧已缓存"
        seen_seg = True
        rows.append((f"SAM@{r['sam_imgsz']}px\n({thermal})", st,
                     r["sam_imgsz"]))
    labels = [x[0] for x in rows]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})

    # ---------- 左: 阶段堆叠(每帧毫秒) ----------
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for k in order:
        v = np.array([r[1].get(k, 0.0) for r in rows])
        if not v.any():
            continue
        ax.barh(y, v, left=left, color=COL[k], label=CN[k], height=0.55)
        for i in range(len(rows)):
            if v[i] > 4:      # 太窄的段写不进字, 就不写
                ax.text(left[i] + v[i] / 2, y[i], f"{v[i]:.1f}", ha="center",
                        va="center", color="white", fontsize=8)
        left += v
    for i, t in enumerate(left):
        ax.text(t + 1.5, y[i], f"合计 {t:.1f} ms/帧\n{1000 / max(t, 1e-9):.1f} fps",
                va="center", fontsize=8.5)
    ax.axvline(a.deadline_ms, color="#d62728", ls="--", lw=1.4)
    ax.text(a.deadline_ms + 1.2, len(rows) - 0.45,
            f"30fps 实时门槛 {a.deadline_ms:.1f} ms", color="#d62728",
            fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("每帧耗时 (ms) —— 越短越好", fontsize=9)
    ax.set_title("整链路每帧耗时分解(本机实测)", fontsize=10.5)
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    ax.set_xlim(0, max(left) * 1.30 + 12)

    # ---------- 右: seg 内部"搬运 vs 计算" ----------
    mb = load_model_bench()
    if mb:
        res = sorted(mb)
        x = np.arange(len(res))
        load = np.array([mb[s]["load_ms"] for s in res])
        prop = np.array([mb[s]["prop_ms"] for s in res])
        # 与 bench_sam2 的"载入+传播"对齐: 取该分辨率下**热抽帧**那一次(帧已就位),
        # 否则中间会白算一段抽帧时间, 三项对不上账。
        stage = np.array(
            [next((r[1].get("seg", np.nan) for r in reversed(rows)
                   if r[2] == s), np.nan) for s in res], float)
        rest = np.clip(stage - load - prop, 0, None)
        w = 0.5
        ax2.bar(x, load, w, color="#ff7f0e", label="帧载入(JPEG 解码, CPU)")
        ax2.bar(x, prop, w, bottom=load, color="#1f77b4",
                label="掩码传播(GPU)")
        ax2.bar(x, rest, w, bottom=load + prop, color="#bbbbbb",
                label="其余(体检/落盘)")
        for i in range(len(res)):
            ax2.text(i, load[i] / 2, f"{load[i]:.1f}", ha="center",
                     color="white", fontsize=8.5)
            ax2.text(i, load[i] + prop[i] / 2, f"{prop[i]:.1f}", ha="center",
                     color="white", fontsize=8.5)
            ax2.text(i, stage[i] + 3, f"实测整段\n{stage[i]:.0f} ms", ha="center",
                     fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"SAM@{s}px\nTP {mb[s]['prop_fps']:.1f} fps"
                             for s in res], fontsize=9)
        ax2.set_ylabel("每帧耗时 (ms)", fontsize=9)
        ax2.set_title("seg 阶段内部: CPU 搬运 vs GPU 计算", fontsize=10.5)
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.25, axis="y")
        ax2.set_ylim(0, max(stage) * 1.28)

    fig.tight_layout()
    p = BENCH / "speed_breakdown.png"
    fig.savefig(p, dpi=140)
    print(f"[写入] {p}")

    # ---------- 文本表 ----------
    L = ["每帧耗时实测 (ms/帧)", "-" * 74,
         f"{'阶段':<14}" + "".join(f"{t:>19}" for t in
                                  [x[0].replace("\n", " ") for x in rows])]
    for k in order:
        if not any(k in x[1] for x in rows):
            continue
        L.append(f"{k:<14}" + "".join(
            (f"{x[1][k]:>19.2f}" if k in x[1] else f"{'-':>19}") for x in rows))
    L.append("-" * 74)
    L.append(f"{'端到端':<14}" + "".join(
        f"{sum(x[1].values()):>19.2f}" for x in rows))
    L.append("")
    for s, v in sorted(load_model_bench().items()):
        L.append(f"SAM@{s}px 纯模型: 载入 {v['load_ms']:.1f} ms/帧, "
                 f"传播 {v['prop_ms']:.1f} ms/帧 ({v['prop_fps']:.1f} fps), "
                 f"显存峰值 {v['vram_mb']:.0f} MB")
    txt = "\n".join(L)
    (BENCH / "speed_table.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"[写入] {BENCH / 'speed_table.txt'}")


if __name__ == "__main__":
    main()
