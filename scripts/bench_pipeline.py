# -*- coding: utf-8 -*-
"""本机(SAM 2.1 掩码路线)处理速度实测 —— 按**真实流水线**分阶段计时。

为什么不用纯微基准: 论文/汇报里要回答的是"这段视频从视频文件到 φ(t) 要多久",
而不是"卷积核跑多少 TFLOPS"。所以这里直接按 pipeline.py 的真实命令行逐阶段跑,
只额外做两件事:
  1. 每个阶段跑的同时用 nvidia-smi 轮询采样(GPU 利用率/显存/功耗/SM 时钟/温度),
     这样能看出"慢在哪、是不是被功耗墙压住";
  2. 顺带记录功耗墙与降频原因(SW Power Cap / SW Thermal Slowdown),
     否则测出来的 fps 会被误当成硬件极限。

阶段(与 pipeline.py 一致, 用独立 tag 跑, 不碰正式产物):
    decode      读视频(纯 CPU 解码)          —— 内置
    stabilize   检测时序稳定化(track_frames) —— 内置(纯 CPU)
    detect      YOLO26s 逐帧推理             —— 子进程(GPU)
    seg         SAM 2.1 掩码传播             —— 子进程(GPU)
    angle_mask  掩码轮廓 → 倾角 φ(t)         —— 子进程(CPU)
    viz         叠加视频 + 曲线图产出        —— 子进程(CPU)

用法:
    python scripts/bench_pipeline.py --tag bench                   # 512, 全阶段
    python scripts/bench_pipeline.py --sam-imgsz 1024 --tag b1024  # 1024 对比
    python scripts/bench_pipeline.py --stages decode,seg --tag b
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "pip", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
CHILD_ENV = dict(os.environ)
CHILD_ENV.update({
    "TEMP": str(CACHE / "tmp"), "TMP": str(CACHE / "tmp"),
    "TMPDIR": str(CACHE / "tmp"), "TORCH_HOME": str(CACHE / "torch"),
    "MPLCONFIGDIR": str(CACHE / "mpl"), "YOLO_CONFIG_DIR": str(CACHE / "yolo"),
    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
})
sys.path.insert(0, str(PROJECT / "scripts"))

PY = sys.executable
S = PROJECT / "scripts"
DIAG = PROJECT / "runs" / "diag"
BENCH = PROJECT / "runs" / "bench"

NVSMI_FIELDS = ["utilization.gpu", "memory.used", "power.draw", "clocks.sm",
                "temperature.gpu"]


# ==========================================================================
# GPU 采样
# ==========================================================================
def nvidia_query(fields: list[str]) -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return {}
    if not out:
        return {}
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    r = {}
    for k, v in zip(fields, vals):
        try:
            r[k] = float(v)
        except ValueError:
            r[k] = v or None          # 文本字段(如 GPU 名称)原样保留
    return r


def _flag_active(txt: str, key: str):
    """在 nvidia-smi -q 文本里取某个降频原因的状态。

    别用 `txt.split(key)[1][:40]` 这种窗口: 该行有 35 个空格的对齐填充,
    窗口太小会把 "Active" 截成 "Acti" ⇒ 误判成"未限功耗"(实际踩过)。
    """
    for ln in txt.splitlines():
        if key in ln and ":" in ln:
            v = ln.split(":", 1)[1].strip().lower()
            if "active" in v:
                return not v.startswith("not")
    return None


def nvidia_static() -> dict:
    """功耗墙 / 降频原因 —— 不记这些, 测出来的 fps 会被误当硬件极限。"""
    q = nvidia_query(["name", "memory.total", "clocks.max.sm", "power.draw"])
    info = {"name": q.get("name"), "vram_total_mb": q.get("memory.total"),
            "sm_max_mhz": q.get("clocks.max.sm")}
    try:
        txt = subprocess.run(["nvidia-smi", "-q", "-d", "POWER,PERFORMANCE"],
                             capture_output=True, text=True,
                             timeout=20).stdout
        for key, tag in (("Current Power Limit", "power_limit_w"),
                         ("Default Power Limit", "power_default_w"),
                         ("Max Power Limit", "power_max_w")):
            for ln in txt.splitlines():
                if key in ln and "GPU Ceiling" not in ln:
                    try:
                        info[tag] = float(ln.split(":")[-1].replace("W", ""))
                    except ValueError:
                        pass
                    break
        info["power_cap_active"] = _flag_active(txt, "SW Power Cap")
        info["thermal_slowdown_active"] = _flag_active(txt, "SW Thermal Slowdown")
    except Exception:
        pass
    return info


class GpuSampler(threading.Thread):
    """后台按 interval 秒轮询 nvidia-smi, 记录阶段内的 GPU 侧曲线。"""

    def __init__(self, interval: float = 0.3):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[dict] = []
        self._ev = threading.Event()
        self._ev.set()

    def run(self) -> None:
        while self._ev.is_set():
            r = nvidia_query(NVSMI_FIELDS)
            if r:
                self.samples.append(r)
            self._ev.wait(self.interval)

    def stop(self) -> None:
        self._ev.clear()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        def col(k):
            v = [s[k] for s in self.samples if s.get(k) is not None]
            return v
        out = {"n_samples": len(self.samples)}
        for k, nm in (("utilization.gpu", "gpu_util"), ("memory.used", "vram_mb"),
                      ("power.draw", "power_w"), ("clocks.sm", "sm_mhz"),
                      ("temperature.gpu", "temp_c")):
            v = col(k)
            if v:
                out[f"{nm}_mean"] = round(sum(v) / len(v), 1)
                out[f"{nm}_max"] = round(max(v), 1)
                out[f"{nm}_min"] = round(min(v), 1)
        return out


# ==========================================================================
# 阶段
# ==========================================================================
def run_subprocess(cmd: list[str], frames: int, log) -> dict:
    sp = GpuSampler()
    sp.start()
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(PROJECT), env=CHILD_ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         bufsize=1)
    tail: list[str] = []
    for line in p.stdout:
        log.write(line)
        log.flush()
        tail.append(line.rstrip())
        if len(tail) > 40:
            tail.pop(0)
    p.wait()
    el = time.time() - t0
    sp.stop()
    sp.join(timeout=2)
    r = {"wall_s": round(el, 2), "n_frames": frames,
         "ms_per_frame": round(el * 1000 / max(frames, 1), 2),
         "fps": round(frames / max(el, 1e-9), 2),
         "exit_code": p.returncode, "gpu": sp.summary()}
    r["log_tail"] = tail[-6:]
    return r


def bench_decode(video: Path, log) -> dict:
    """纯解码: 打开视频逐帧读到底(不算任何算法)。"""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = time.time()
    k = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        k += 1
    el = time.time() - t0
    cap.release()
    log.write(f"[decode] {k} 帧 {el:.2f}s\n")
    return {"wall_s": round(el, 2), "n_frames": k,
            "ms_per_frame": round(el * 1000 / max(k, 1), 2),
            "fps": round(k / max(el, 1e-9), 2), "exit_code": 0, "gpu": {}}


def bench_stabilize(frames_json: Path, log) -> dict:
    """检测稳定化: 直接用产物里的框跑 track_frames(与流水线同一个函数)。"""
    from rocket_track import TrackerConfig, track_frames
    meta = json.loads(frames_json.read_text(encoding="utf-8"))
    vm = meta["meta"]
    boxes = [r["b"] for r in meta["frames"]]
    n = len(boxes)
    t0 = time.time()
    outs = track_frames(boxes, TrackerConfig(), vm["fps"],
                        bound_wh=(vm["W"], vm["H"]))
    el = time.time() - t0
    log.write(f"[stabilize] {n} 帧 {el:.2f}s\n")
    return {"wall_s": round(el, 2), "n_frames": n,
            "ms_per_frame": round(el * 1000 / max(n, 1), 3),
            "fps": round(n / max(el, 1e-9), 1), "exit_code": 0, "gpu": {},
            "note": f"稳定输出 {sum(1 for o in outs if o.has)} 帧"}


# ==========================================================================
# main
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="SAM 路线本机处理速度实测")
    ap.add_argument("--tag", default="bench", help="本次基准用的独立标签")
    ap.add_argument("--dets-tag", default="iou70", help="正式检测产物标签(只读)")
    ap.add_argument("--det-source", default="",
                    help="seg/angle_mask/viz 引用的检测产物标签; 默认用 --tag。"
                         "对比不同 SAM 分辨率时必须指定同一个已有标签(如 iou70), "
                         "否则会去找 dets_<tag>.json 而报 FileNotFoundError")
    ap.add_argument("--stages", default="decode,stabilize,detect,seg,angle_mask,viz")
    ap.add_argument("--sam-model", default="tiny")
    ap.add_argument("--sam-imgsz", type=int, default=512)
    ap.add_argument("--sam-chunk", type=int, default=200)
    ap.add_argument("--interval", type=float, default=0.3)
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()

    tag = a.tag
    dsrc = a.det_source or tag          # SAM 阶段引用的检测产物标签
    BENCH.mkdir(parents=True, exist_ok=True)
    # 用 .log: 原始日志含大量进度条噪声, .gitignore 里 *.log 已排除;
    # 入库的证据是 pipeline_speed.json 与 speed_table.txt 这两个小文件。
    LOG = BENCH / f"log_{tag}.log"
    meta_f = DIAG / f"dets_{a.dets_tag}.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8"))
    vm = meta["meta"]
    n_frame = len(meta["frames"])
    video = PROJECT / vm["video"]

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if a.no_viz and "viz" in stages:
        stages.remove("viz")

    gpu0 = nvidia_static()
    print("=" * 78)
    print("本机处理速度实测 —— SAM 2.1 掩码测角路线")
    print("=" * 78)
    print(f"  视频      : {vm['video']}")
    print(f"  规格      : {vm['W']}x{vm['H']} @ {vm['fps']:.2f}fps, "
          f"{n_frame} 帧 ({n_frame / vm['fps']:.1f}s)")
    print(f"  GPU       : {gpu0.get('name')}  "
          f"{gpu0.get('vram_total_mb')} MB  SM 最高 {gpu0.get('sm_max_mhz')} MHz")
    print(f"  功耗墙    : 当前 {gpu0.get('power_limit_w')} W / 默认 "
          f"{gpu0.get('power_default_w')} W / 上限 {gpu0.get('power_max_w')} W"
          f"    SW Power Cap: "
          f"{'Active(正被限功耗!)' if gpu0.get('power_cap_active') else '不活跃'}")
    print(f"  SAM       : {a.sam_model} @ {a.sam_imgsz}px fp16")
    print(f"  检测来源  : {dsrc}   (SAM 阶段引用它, 结果写到 tag={tag})")
    print(f"  阶段      : {', '.join(stages)}")
    print(f"  日志      : {LOG.relative_to(PROJECT)}")
    print()

    res: dict = {}
    with LOG.open("w", encoding="utf-8") as log:
        log.write(f"# bench {time.strftime('%Y-%m-%d %H:%M:%S')}  tag={tag}\n")
        for st in stages:
            print(f"[阶段] {st:<11}", end="", flush=True)
            t0 = time.time()
            if st == "decode":
                r = bench_decode(video, log)
            elif st == "stabilize":
                r = bench_stabilize(meta_f, log)
            elif st == "detect":
                r = run_subprocess(
                    [PY, str(S / "diag_video_detections.py"), "--tag", tag,
                     "--conf", "0.02", "--iou", "0.7"], n_frame, log)
            elif st == "seg":
                r = run_subprocess(
                    [PY, str(S / "rocket_seg.py"), "--dets-tag", dsrc,
                     "--tag", tag, "--model", a.sam_model, "--image-size",
                     str(a.sam_imgsz), "--chunk", str(a.sam_chunk)],
                    n_frame, log)
            elif st == "angle_mask":
                r = run_subprocess(
                    [PY, str(S / "rocket_mask_angle.py"), "--tag", tag,
                     "--dets-tag", dsrc], n_frame, log)
            elif st == "viz":
                r = run_subprocess(
                    [PY, str(S / "sam_angle_viz.py"), "--tag", tag,
                     "--dets-tag", dsrc, "--out",
                     str(BENCH / f"overlay_{tag}.mp4")], n_frame, log)
            else:
                print(f"  未知阶段, 跳过")
                continue
            r["label"] = st
            res[st] = r
            g = r.get("gpu") or {}
            print(f"  {r['wall_s']:7.1f}s  {r['ms_per_frame']:7.2f} ms/帧  "
                  f"{r['fps']:7.2f} fps" +
                  (f"   GPU {g.get('gpu_util_mean', '-')}%  "
                   f"显存 {g.get('vram_mb_max', '-')}MB  "
                   f"功耗 {g.get('power_w_mean', '-')}W  "
                   f"SM {g.get('sm_mhz_mean', '-')}MHz" if g else ""))
        log.write(f"\n[GPU 静态] {json.dumps(gpu0, ensure_ascii=False)}\n")
        gpu1 = nvidia_static()
        log.write(f"[GPU 结束时] {json.dumps(gpu1, ensure_ascii=False)}\n")

    e2e_stages = [s for s in stages if s != "viz"]
    tot = sum(res[s]["wall_s"] for s in e2e_stages if s in res)
    print("-" * 78)
    print(f"  端到端({' + '.join(e2e_stages)})  {tot:.1f}s  "
          f"({n_frame} 帧 ⇒ {n_frame / max(tot, 1e-9):.2f} fps, "
          f"每帧 {tot * 1000 / n_frame:.1f} ms)")
    if "viz" in res:
        print(f"  (可视化产出另需 {res['viz']['wall_s']:.1f}s, 不计入检测链路)")
    print()

    out = {"tag": tag, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "video": vm["video"], "W": vm["W"], "H": vm["H"],
           "fps_video": round(vm["fps"], 3), "n_frame": n_frame,
           "sam_model": a.sam_model, "sam_imgsz": a.sam_imgsz,
           "sam_chunk": a.sam_chunk, "gpu": gpu0, "stages": res,
           "e2e_stages": e2e_stages, "e2e_wall_s": round(tot, 2),
           "e2e_fps": round(n_frame / max(tot, 1e-9), 2),
           "e2e_ms_per_frame": round(tot * 1000 / max(n_frame, 1), 2)}
    p = BENCH / "pipeline_speed.json"
    prev = []
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
            prev = prev if isinstance(prev, list) else prev.get("runs", [])
        except Exception:
            prev = []
    p.write_text(json.dumps(prev + [out], ensure_ascii=False, indent=1),
                 encoding="utf-8")
    print(f"[写入] {p}")
    print(f"[写入] {LOG.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
