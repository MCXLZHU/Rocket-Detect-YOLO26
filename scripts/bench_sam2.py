# -*- coding: utf-8 -*-
"""SAM 2.1 在本机的推理性能基准 —— 回答"能不能实时"。

为什么单独做这个: 官方标称的 FPS (tiny 47.2 / small 43.3) 是 **A100 + torch 2.5.1**
的数字, 与"RTX 4060 Laptop 8GB + 默认 55W 功耗墙"完全不是一回事。
实时性必须在本机实测, 否则方案选型(prompt 一次传播全片 vs 逐帧提示)会选错。

测三件事:
  A. 视频模式(方案 B)   init_state -> add box prompt -> propagate 的逐帧耗时
  B. 图像模式(方案 A)   set_image + predict 的逐帧耗时(每帧独立, 无记忆)
  C. 降分辨率/编译的收益  image_size 512 与 compile_image_encoder 的加速比

注意 init_state 会把整段视频预处理成 image_size×image_size 的 float32 张量并常驻:
    1024² × 3ch × 4B = 12.6 MB/帧  =>  2203 帧要 27.7 GB(远超本机 15.7GB 内存 + 8GB 显存)
所以**必须分块**(chunk)处理, 本脚本默认只取一段做基准, 真实跑用 rocket_seg.py 的分块逻辑。

用法:
    python scripts/bench_sam2.py                      # tiny, 1024, 120 帧
    python scripts/bench_sam2.py --model small
    python scripts/bench_sam2.py --image-size 512
    python scripts/bench_sam2.py --half --compile
    python scripts/bench_sam2.py --mode image
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["TMPDIR"] = str(CACHE / "tmp")
os.environ["TORCH_HOME"] = str(CACHE / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE / "yolo")

# sam2 以 sys.path 方式接入(不做 pip 安装: 构建隔离要联网拉构建依赖, 会卡住)
SAM2_ROOT = PROJECT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_ROOT))
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402
import torch  # noqa: E402

from sam2.build_sam import build_sam2, build_sam2_video_predictor  # noqa: E402

WEIGHTS = PROJECT / "weights"
DIAG = PROJECT / "runs" / "diag"
OUT_DIR = PROJECT / "runs" / "seg"

CFG_FILE = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
}
CKPT_FILE = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base": "sam2.1_hiera_base_plus.pt",
}


# ==========================================================================
# 工具
# ==========================================================================
@contextmanager
def autocast_ctx(enabled: bool):
    """fp16 推理。注意 autocast 只影响矩阵乘类算子, 显存不会显著下降。"""
    if enabled:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            yield
    else:
        with torch.inference_mode():
            yield


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def vram_mb() -> tuple[float, float]:
    """返回 (当前已分配, 峰值已分配) MB。"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 2 ** 20,
            torch.cuda.max_memory_allocated() / 2 ** 20)


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {}
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, total_mem_gb=round(p.total_memory / 2 ** 30, 2),
                cc=f"{p.major}.{p.minor}", sm=p.multi_processor_count)


def extract_frames(video: Path, out_dir: Path, start: int, n: int,
                   quality: int = 95) -> int:
    """把 [start, start+n) 抽成 <idx>.jpg 到 out_dir (SAM2 要求数字命名的 JPEG)。

    文件名用**视频全局帧号**而不是 0..n-1, 这样分块跑的时候帧号能直接对齐,
    不会出现"第 3 块的 0 号帧到底是全片第几帧"这种错位。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n_ok = 0
    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{start + k:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        n_ok += 1
    cap.release()
    return n_ok


# ==========================================================================
# A. 视频模式(方案 B) 逐帧耗时
# ==========================================================================
def bench_video(model, model_name: str, frames_dir: Path, box: np.ndarray,
                size: int, half: bool, warmup: int, n_prop: int,
                offload: bool) -> dict:
    r = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.time()
    state = model.init_state(str(frames_dir), offload_video_to_cpu=offload)
    sync()
    r["init_state_s"] = round(time.time() - t0, 3)
    r["n_frames_loaded"] = int(state["num_frames"])
    r["offload_video_to_cpu"] = offload
    r["vram_after_init_mb"] = round(vram_mb()[1], 1)

    # 单次 box prompt (会立刻在该帧出 mask, 属"交互延迟")
    sync()
    t0 = time.time()
    with autocast_ctx(half):
        model.add_new_points_or_box(state, frame_idx=int(box[0]), obj_id=1,
                                    box=box[1])
    sync()
    r["prompt_s"] = round(time.time() - t0, 4)

    # 传播
    r["n_prop"] = 0
    r["prop_s"] = 0.0
    r["prop_fps"] = 0.0
    r["prop_fwd_ms_mean"] = 0.0
    r["prop_fwd_ms_median"] = 0.0
    r["prop_fwd_ms_min"] = 0.0
    r["prop_fwd_ms_max"] = 0.0
    r["vram_peak_mb"] = 0.0
    per_frame = []
    if n_prop > 0:
        with autocast_ctx(half):
            for _ in range(warmup):
                pass
            t0 = time.time()
            for i, (_fidx, _oids, _masks) in enumerate(
                    model.propagate_in_video(state)):
                sync()
                if i >= 1:
                    per_frame.append(time.time() - t_prev)
                t_prev = time.time()
                if i + 1 >= n_prop:
                    break
        el = time.time() - t0
        r["n_prop"] = len(per_frame) + 1
        r["prop_s"] = round(el, 3)
        r["prop_fps"] = round(r["n_prop"] / max(el, 1e-9), 2)
        if per_frame:
            ms = np.array(per_frame) * 1000.0
            r["prop_fwd_ms_mean"] = round(float(ms.mean()), 2)
            r["prop_fwd_ms_median"] = round(float(np.median(ms)), 2)
            r["prop_fwd_ms_min"] = round(float(ms.min()), 2)
            r["prop_fwd_ms_max"] = round(float(ms.max()), 2)
        r["vram_peak_mb"] = round(vram_mb()[1], 1)
    del state
    torch.cuda.empty_cache()
    return r


# ==========================================================================
# B. 图像模式(方案 A) 逐帧耗时
# ==========================================================================
def bench_image(model_cfg: str, ckpt: Path, frames_dir: Path, box_xyxy: np.ndarray,
                size: int, half: bool, n: int) -> dict:
    """每帧独立: set_image(跑图像编码器) + predict(box)。"""
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import compose
    from omegaconf import OmegaConf

    r = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = compose(config_name=model_cfg,
                  overrides=[f"++model.image_size={size}"])
    OmegaConf.resolve(cfg)
    from hydra.utils import instantiate
    m = instantiate(cfg.model, _recursive_=True)
    m = m.to("cuda")
    m.eval()
    pred = SAM2ImagePredictor(m)

    names = sorted(frames_dir.glob("*.jpg"))[:n]
    gray_box = box_xyxy
    ts_set, ts_pred = [], []
    for k, p in enumerate(names):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        sync()
        t0 = time.time()
        with autocast_ctx(half):
            pred.set_image(img)
        sync()
        ts_set.append(time.time() - t0)
        t0 = time.time()
        with autocast_ctx(half):
            pred.predict(box=gray_box[None, :], multimask_output=False)
        sync()
        ts_pred.append(time.time() - t0)
        if k == 2:
            torch.cuda.reset_peak_memory_stats()   # 排除首次分配的抖动
    ts_all = np.array(ts_set) + np.array(ts_pred)
    r["img_set_image_ms"] = round(float(np.median(ts_set[2:]) * 1000), 2)
    r["img_predict_ms"] = round(float(np.median(ts_pred[2:]) * 1000), 2)
    r["img_total_ms"] = round(float(np.median(ts_all[2:]) * 1000), 2)
    r["img_fps"] = round(1.0 / max(float(np.median(ts_all[2:])), 1e-9), 2)
    r["vram_peak_mb"] = round(vram_mb()[1], 1)
    del pred, m
    torch.cuda.empty_cache()
    return r


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="SAM 2.1 本机实时性基准")
    ap.add_argument("--model", default="tiny", choices=list(CFG_FILE))
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--start", type=int, default=1400,
                    help="抽帧起点(默认 1400 = 落地段, 箭体清晰)")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--prop", type=int, default=60, help="传播测多少帧")
    ap.add_argument("--half", action="store_true", help="fp16 autocast")
    ap.add_argument("--compile", action="store_true",
                    help="compile_image_encoder(首次编译很慢)")
    ap.add_argument("--offload", action="store_true",
                    help="帧张量放 CPU(省显存, 略降速度)")
    ap.add_argument("--mode", default="both", choices=["video", "image", "both"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    vm = meta["meta"]
    video = PROJECT / vm["video"]

    # 抽帧(用全局帧号命名)。
    # 注意: 目录名带上区间, 且**从不删除** —— 一次性删 >50 个文件会被宿主的安全策略
    # 拦成 SAFE_DELETE_BULK_CONFIRM_REQUIRED, 整条命令直接作废。
    fdir = CACHE / f"frames_bench_{a.start}_{a.start + a.frames}"
    have = len(list(fdir.glob("*.jpg"))) if fdir.exists() else 0
    if have == a.frames:
        n = have
        print(f"[抽帧] 复用已有 {n} 帧 @ {a.start} -> {fdir}", flush=True)
    else:
        n = extract_frames(video, fdir, a.start, a.frames)
        print(f"[抽帧] {n} 帧 @ {a.start} -> {fdir}", flush=True)

    # prompt: 用该帧的 Rocket Body 检测框(cls==1)
    rb = [d for d in meta["frames"][a.start]["b"] if d["c"] == 1]
    if not rb:
        for k in range(a.start, min(a.start + 60, len(meta["frames"]))):
            rb = [d for d in meta["frames"][k]["b"] if d["c"] == 1]
            if rb:
                print(f"[提示] {a.start} 帧无 RB, 退到 {k}", flush=True)
                break
    box = np.asarray(rb[0]["x"], float)

    ovr = [f"++model.image_size={a.image_size}"]
    if a.compile:
        ovr.append("++model.compile_image_encoder=True")

    print(f"[构建] sam2.1_hiera_{a.model}  image_size={a.image_size}  "
          f"half={a.half}  compile={a.compile}", flush=True)
    t0 = time.time()
    model = build_sam2_video_predictor(CFG_FILE[a.model],
                                       str(WEIGHTS / CKPT_FILE[a.model]),
                                       device="cuda", hydra_overrides_extra=ovr)
    sync()
    build_s = time.time() - t0
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[构建] {build_s:.1f}s  参数 {n_par / 1e6:.1f}M", flush=True)

    out = dict(model=a.model, image_size=a.image_size, half=a.half,
               compile=a.compile, build_s=round(build_s, 2),
               params_M=round(n_par / 1e6, 1), gpu=gpu_info(),
               prompt_box=[round(float(v), 1) for v in box],
               prompt_frame=a.start, n_frames_extracted=n)

    if a.mode in ("video", "both"):
        # 注意: bench 里 init_state 用的是"全段同时载入", 帧号是全局帧号,
        # 所以 prompt 的 frame_idx 要用相对下标
        b2 = (0, box)
        print("[A] 视频模式 ...", flush=True)
        out["video"] = bench_video(model, a.model, fdir, b2,
                                   a.image_size, a.half, 0, a.prop, a.offload)
        v = out["video"]
        print(f"    init_state {v['init_state_s']}s | prompt {v['prompt_s']}s | "
              f"传播 {v['prop_fps']} fps ({v['prop_fwd_ms_median']} ms/帧中位) | "
              f"峰值显存 {v['vram_peak_mb']} MB", flush=True)
        del model
        torch.cuda.empty_cache()

    if a.mode in ("image", "both"):
        print("[B] 图像模式 ...", flush=True)
        out["image"] = bench_image(CFG_FILE[a.model], WEIGHTS / CKPT_FILE[a.model],
                                   fdir, box, a.image_size, a.half, min(n, 40))
        i = out["image"]
        print(f"    set_image {i['img_set_image_ms']}ms + predict "
              f"{i['img_predict_ms']}ms = {i['img_total_ms']}ms/帧 "
              f"=> {i['img_fps']} fps | 峰值显存 {i['vram_peak_mb']} MB", flush=True)

    sfx = f"_{a.tag}" if a.tag else ""
    p = OUT_DIR / f"bench{sfx}.json"
    prev = []
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                prev = prev.get("runs", [])
        except Exception:
            prev = []
    (OUT_DIR / f"bench{sfx}.json").write_text(
        json.dumps({"runs": prev + [out]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[写入] {p}", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
