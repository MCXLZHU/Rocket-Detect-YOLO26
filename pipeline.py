# -*- coding: utf-8 -*-
"""Rocket Attitude Estimation —— 整条流水线的一键入口。

  阶段            做什么                                    产物
  ------------    --------------------------------------    ------------------------------------------
  detect          YOLO 低阈值逐帧推理, 存盘原始检测框         runs/diag/dets_<tag>.json           (GPU)
  stabilize       时序稳定化(嵌套抑制+双阈值续轨+模式状态机)   runs/diag/stabilize_report.txt
  angle           ROI 内边缘检测 → 倾角 φ(t)                  runs/angle/angles.csv / .json
  attitude        相对倾角 + 相机核查 (+可选伪3D β)           runs/attitude/attitude.csv / validate.txt
  report          所有图表                                   runs/angle/angle_timeline.png 等

设计要点:
  * **阶段可缓存**: 每阶段检查自己的产物文件, 已存在就跳过(用 --force 强制重跑)。
    所以 detect 只要跑一次(耗 GPU), 后面几阶段纯 CPU, 反复调参只重跑对应阶段。
  * **阶段可独立运行**: 每个阶段就是一个独立脚本, 既能被本入口编排, 也能单独跑。
  * **各阶段解耦**: 通过 runs/ 下的 JSON/CSV 交换数据, 不共享内存。
  * 所有临时/缓存目录都落在工作区内, 不写 C 盘。

用法:
    python pipeline.py                          # 全跑(已完成的阶段自动跳过)
    python pipeline.py --force                  # 全部重跑
    python pipeline.py --only stabilize,angle   # 只跑指定阶段
    python pipeline.py --from angle             # 从某阶段往后跑
    python pipeline.py --to angle               # 跑到某阶段为止
    python pipeline.py --video-out              # 另外产出带标注的视频(较慢)
    python pipeline.py --l-over-d 17            # 顺带算出参数化的面外角 β
    python pipeline.py --summary                # 只打印汇总(不跑任何阶段)
    python pipeline.py --list                   # 列出阶段与产物
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "pip", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
# 所有子进程共享: 缓存重定向到工作区 + 强制 UTF-8 输出(避免中文在 Windows 控制台崩掉)
CHILD_ENV = dict(os.environ)
CHILD_ENV.update({
    "TEMP": str(CACHE / "tmp"), "TMP": str(CACHE / "tmp"),
    "TMPDIR": str(CACHE / "tmp"), "TORCH_HOME": str(CACHE / "torch"),
    "MPLCONFIGDIR": str(CACHE / "mpl"), "YOLO_CONFIG_DIR": str(CACHE / "yolo"),
    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
})

PY = sys.executable          # 与入口同一个解释器(用户用的是 conda 的 yolo26 环境)
S = PROJECT / "scripts"
LOG = PROJECT / "runs" / "pipeline_log.txt"

DIAG, ANG, ATT = (PROJECT / "runs" / "diag", PROJECT / "runs" / "angle",
                  PROJECT / "runs" / "attitude")


# ==========================================================================
# 阶段定义
# ==========================================================================
class Stage:
    def __init__(self, name, desc, artifacts, build, needs=()):
        self.name = name
        self.desc = desc
        self.artifacts = artifacts      # 用于判断"是否已完成"的文件
        self.build = build              # (args) -> list[list[str]] 命令列表
        self.needs = needs              # 前置阶段


def _stages(dets_tag: str):
    def detect(a):
        cmd = [PY, str(S / "diag_video_detections.py"), "--tag", dets_tag,
               "--conf", str(a.conf), "--iou", str(a.iou)]
        if a.video:
            cmd += ["--video", a.video]
        return [cmd]

    def stabilize(a):
        out = [[PY, str(S / "validate_stabilize.py")]]
        if a.video_out:
            out.append([PY, str(S / "track_video.py")])
        return out

    def angle(a):
        cmd = [PY, str(S / "rocket_angle.py")]
        if a.video_out:
            cmd.append("--make-video")
        return [cmd, [PY, str(S / "angle_report.py")]]

    def attitude(a):
        cmd = [PY, str(S / "validate_attitude.py")]
        if a.l_over_d and math.isfinite(a.l_over_d):
            cmd += ["--l-over-d", str(a.l_over_d)]
        out = [cmd, [PY, str(S / "attitude_report.py")]]
        if math.isfinite(a.l_over_d):
            out.append([PY, str(S / "rocket_attitude.py"), "--l-over-d",
                        str(a.l_over_d), "--tag", "ld"])
        return out

    return [
        Stage("detect", "YOLO 低阈值逐帧推理(需 GPU)",
              [DIAG / f"dets_{dets_tag}.json"], detect),
        Stage("stabilize", "检测时序稳定化 + 前后对比",
              [DIAG / "stabilize_report.txt"], stabilize, ("detect",)),
        Stage("angle", "ROI 边缘检测 → 倾角 φ(t)",
              [ANG / "angles.csv", ANG / "angles.json"], angle, ("detect",)),
        Stage("attitude", "相对倾角 + 相机核查 (+可选 β)",
              [ATT / "attitude.csv", ATT / "validate.txt"], attitude,
              ("angle",)),
    ]


REPORTS = ["angle", "attitude"]      # 这两个阶段自带图表产出


# ==========================================================================
# 执行
# ==========================================================================
def run_cmd(cmd: list[str], log) -> int:
    t0 = time.time()
    log.write("\n$ " + " ".join(cmd) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, cwd=str(PROJECT), env=CHILD_ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         bufsize=1)
    for line in p.stdout:                       # 实时回显 + 落日志
        sys.stdout.write(line)
        log.write(line)
    p.wait()
    log.write(f"[退出码 {p.returncode}  用时 {time.time() - t0:.1f}s]\n")
    log.flush()
    return p.returncode


# ==========================================================================
# 汇总: 只读产物, 不跑任何阶段
# ==========================================================================
def summarize(dets_tag: str) -> str:
    L: list[str] = []

    def p(s=""):
        L.append(s)

    meta_f = DIAG / f"dets_{dets_tag}.json"
    att_f = ATT / "attitude.json"
    ang_f = ANG / "angles.json"
    p("=" * 78)
    p("Rocket Attitude Estimation —— 流水线汇总")
    p("=" * 78)

    if meta_f.exists():
        d = json.loads(meta_f.read_text(encoding="utf-8"))
        vm = d["meta"]
        W, H, fps = vm["W"], vm["H"], vm["fps"]
        p(f"视频: {vm['video']}")
        p(f"{W}x{H} @ {fps:.2f}fps, {len(d['frames'])} 帧 "
          f"({len(d['frames']) / fps:.1f}s)   imgsz={vm['imgsz']}")
        p()
        # --- 第 2 步: 稳定化 ---
        sys.path.insert(0, str(S))
        from rocket_track import TrackerConfig, track_frames
        FH = [r["b"] for r in d["frames"]]
        outs = track_frames(FH, TrackerConfig(), fps, bound_wh=(W, H))
        rb = [i for i, r in enumerate(d["frames"])
              if any(b["c"] == 1 and b["s"] >= 0.25 for b in r["b"])]
        if rb:
            w0, w1 = rb[0], rb[-1]
            raw = sum(1 for i in range(w0, w1 + 1)
                      if any(b["c"] == 1 and b["s"] >= 0.25
                             for b in d["frames"][i]["b"]))
            trk = sum(1 for i in range(w0, w1 + 1)
                      if outs[i].has and outs[i].box is not None)
            rm = sum(1 for i in range(w0, w1 + 1)
                     if len([b for b in d["frames"][i]["b"]
                             if b["c"] == 1 and b["s"] >= 0.25]) >= 2)
            dup = sum(o.n_dup_removed for o in outs)
            coast = sum(1 for i in range(w0, w1 + 1)
                        if outs[i].state == "COASTED")
            p("-" * 78)
            p(f"[2] 检测稳定化   窗口 {w0}-{w1} ({w1 - w0 + 1} 帧)")
            p(f"      原始 conf>=0.25 : 有框 {raw}/{w1 - w0 + 1}  "
              f"缺 {w1 - w0 + 1 - raw}   多框帧 {rm}")
            p(f"      稳定化后        : 有框 {trk}/{w1 - w0 + 1}  "
              f"缺 {w1 - w0 + 1 - trk}   抑制重复框 {dup} 个  "
              f"预测补位 {coast} 帧")
        p()

    if ang_f.exists():
        a = json.loads(ang_f.read_text(encoding="utf-8"))
        fps = a["info"]["fps"]
        ok = [r for r in a["frames"] if r["ok"]]
        p("-" * 78)
        p("[3] 倾角 φ(t)    (相对图像竖直, 正=顶端右倾)")
        p(f"      有效帧 {len(ok)}/{len(a['frames'])}   基准 φ_ref = "
          f"{a['info']['ref']:+.3f}°(落地 45-66s)")
        groups = [(45, 69), (25, 45), (11, 25)]
        for lo, hi in groups:
            m = [r["phi_deg"] for r in ok if lo <= r["frame"] / fps < hi]
            if len(m) < 3:
                continue
            mu = sum(m) / len(m)
            sd = (sum((x - mu) ** 2 for x in m) / len(m)) ** 0.5
            p(f"      {lo:>2}-{hi}s: n={len(m):>3}  φ中位 "
              f"{sorted(m)[len(m) // 2]:+.2f}°  σ {sd:.3f}°  "
              f"极差 {max(m) - min(m):.2f}°")
        p()

    if att_f.exists():
        at = json.loads(att_f.read_text(encoding="utf-8"))
        fps = at["info"]["fps"]
        fr = at["frames"]
        p("-" * 78)
        p("[4] 姿态")
        cv = [r for r in fr if r["cam_ncc"] is not None and r["cam_ncc"] >= 0.6]
        for lo, hi, nm in ((45, 69, "落地段"), (11, 40, "下降段")):
            m = [r["cam_roll"] for r in cv if lo <= r["t"] < hi]
            if len(m) < 5:
                continue
            mu = sum(m) / len(m)
            sd = (sum((x - mu) ** 2 for x in m) / len(m)) ** 0.5
            p(f"      相机滚转 {nm}: {mu:+.3f}° ± {sd:.3f}°  (n={len(m)})")
        ok = [r for r in fr if r["ok"]]
        land = [r for r in ok if 45 <= r["t"] < 69]
        if land:
            v = [r["phi_deg"] for r in land]
            mu = sum(v) / len(v)
            sd = (sum((x - mu) ** 2 for x in v) / len(v)) ** 0.5
            top = sum(r["phi_vs_top"] for r in land) / len(land)
            rel = [r["phi_rel"] for r in land if r["phi_rel"] is not None]
            p(f"      落地段(45-69s) n={len(land)}: φ={mu:+.3f}°±{sd:.3f}°, "
              f"90−|φ|={top:.2f}°(与图像上边缘的夹角), "
              f"相对倾角中位 {sorted(rel)[len(rel) // 2]:+.3f}°")
        if math.isfinite(at["info"].get("l_over_d") or float("nan")):
            bet = [r["beta_deg"] for r in ok if r["beta_deg"] is not None]
            if bet:
                p(f"      面外角 β(参数化, L/D={at['info']['l_over_d']}): "
                  f"中位 {sorted(bet)[len(bet) // 2]:.1f}°  "
                  f"[{min(bet):.1f}, {max(bet):.1f}]  ← 仅供参考, 见文档 §4")
        else:
            p("      面外角 β: 未计算(用 --l-over-d 提供火箭真实长径比)")
        p()

    p("-" * 78)
    p("产物")
    for f in (DIAG / f"dets_{dets_tag}.json", DIAG / "stabilize_report.txt",
              ANG / "angles.csv", ANG / "angle_timeline.png",
              ATT / "attitude.csv", ATT / "validate.txt",
              ATT / "attitude_timeline.png", ATT / "attitude_ref.mp4",
              ATT / "angle_roi.mp4", LOG):
        mark = "✓" if f.exists() else "·"
        p(f"  {mark} {f.relative_to(PROJECT)}")
    p()
    p("文档: DETECTION_STABILIZATION.md / ANGLE_ESTIMATION.md / "
      "ATTITUDE_ESTIMATION.md / TRAINING_REPORT.md")
    return "\n".join(L)


# ==========================================================================
# main
# ==========================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rocket Attitude Estimation 流水线入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1])
    p.add_argument("--video", default=None, help="视频路径, 默认自动找")
    p.add_argument("--only", default=None, help="只跑这些阶段(逗号分隔)")
    p.add_argument("--from", dest="frm", default=None, help="从该阶段开始(含)")
    p.add_argument("--to", default=None, help="跑到该阶段为止(含)")
    p.add_argument("--force", action="store_true", help="忽略缓存, 全部重跑")
    p.add_argument("--video-out", action="store_true",
                   help="另外产出带标注的视频(较慢)")
    p.add_argument("--conf", type=float, default=0.02, help="检测存盘阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    p.add_argument("--tag", default=None, help="检测结果标签, 默认 iou<int(iou*100)>")
    p.add_argument("--l-over-d", type=float, default=float("nan"),
                   help="火箭真实长径比; 给出才计算面外角 β")
    p.add_argument("--summary", action="store_true", help="只打印汇总")
    p.add_argument("--list", action="store_true", help="列出阶段与产物")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dets_tag = a.tag or f"iou{int(round(a.iou * 100))}"
    stages = _stages(dets_tag)

    if a.list:
        print(f"{'阶段':<12} {'说明':<34} 产物")
        for s in stages:
            print(f"{s.name:<12} {s.desc:<34} "
                  f"{', '.join(str(x.relative_to(PROJECT)) for x in s.artifacts)}")
        print(f"\n检测结果标签: --tag {dets_tag}")
        return

    if a.summary:
        print(summarize(dets_tag))
        return

    names = [s.name for s in stages]
    sel = list(names)
    if a.only:
        want = [x.strip() for x in a.only.split(",") if x.strip()]
        bad = [x for x in want if x not in names]
        if bad:
            print(f"[x] 未知阶段: {bad}; 可选 {names}")
            sys.exit(2)
        sel = want
    else:
        if a.frm:
            if a.frm not in names:
                print(f"[x] 未知阶段: {a.frm}; 可选 {names}")
                sys.exit(2)
            sel = names[names.index(a.frm):]
        if a.to:
            if a.to not in names:
                print(f"[x] 未知阶段: {a.to}; 可选 {names}")
                sys.exit(2)
            sel = [x for x in sel if names.index(x) <= names.index(a.to)]

    LOG.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Rocket Attitude Estimation —— 流水线")
    print("=" * 78)
    print(f"  待执行阶段: {' -> '.join(sel)}")
    print(f"  检测标签  : {dets_tag}   (conf={a.conf}, iou={a.iou})")
    print(f"  标注视频  : {'开' if a.video_out else '关(--video-out 打开)'}")
    print(f"  长径比    : {a.l_over_d if math.isfinite(a.l_over_d) else '未提供(不算 β)'}")
    print(f"  日志      : {LOG.relative_to(PROJECT)}")
    print()

    t_all = time.time()
    failed = []
    with LOG.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 78 + f"\n运行 {time.strftime('%Y-%m-%d %H:%M:%S')}  "
                  f"阶段 {sel}\n")
        for s in stages:
            if s.name not in sel:
                continue
            if not a.force and all(f.exists() for f in s.artifacts):
                print(f"[跳过] {s.name:<10} 产物已存在("
                      f"{s.artifacts[0].name}); 用 --force 强制重跑")
                continue
            miss = [d for d in s.needs if d not in sel]
            need_files = [f for d in s.needs for f in
                          next(x for x in stages if x.name == d).artifacts]
            if any(not f.exists() for f in need_files):
                print(f"[跳过] {s.name:<10} 缺少前置产物 "
                      f"{[f.name for f in need_files if not f.exists()]}; "
                      f"先跑 {s.needs or miss}")
                continue
            print(f"\n{'=' * 78}\n[阶段] {s.name} —— {s.desc}\n{'=' * 78}")
            t0 = time.time()
            ok = True
            for cmd in s.build(a):
                if run_cmd(cmd, log) != 0:
                    ok = False
                    break
            el = time.time() - t0
            if ok:
                print(f"[完成] {s.name}  ({el:.1f}s)")
            else:
                failed.append(s.name)
                print(f"[失败] {s.name}  ({el:.1f}s) 详见 {LOG.name}")
                break

    print()
    print(summarize(dets_tag))
    print()
    print(f"总用时 {time.time() - t_all:.1f}s" +
          (f"   失败阶段: {failed}" if failed else "   ✅ 全部完成"))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
