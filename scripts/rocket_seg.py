# -*- coding: utf-8 -*-
"""第 3 步(替代路线): SAM 2.1 视频模式分割箭体轮廓 —— 方案 B 的具体落实。

=============================== 方案 B 是什么 ==============================
一次提示 + 流式记忆传播 + YOLO 框周期体检:
  1. 在**块首帧**用第 2 步的稳定框作为 box prompt, 让 SAM 2.1 分割出箭体;
  2. `propagate_in_video()` 靠流式记忆把 mask 传到整块;
  3. 逐帧拿**稳定框**做体检(掩码是不是还在框里 / 面积比有没有突变);
  4. 体检不过 ⇒ 在坏段**起点重新下提示**, 从该帧继续传播(即"重锚定")。

与方案 A(逐帧独立提示)的区别: A 每帧都要重跑图像编码器且帧间无约束; B 有记忆,
时序一致性好, 但会**平滑漂移**(漂移是渐变的, 单帧看不像错误) —— 所以第 3、4 步的
体检与重锚定不是可选项, 是必需品。

=============================== 为什么必须分块 ==============================
`init_state()` 会把整段视频预处理成 image_size×image_size 的 float32 张量常驻:
    1024² × 3ch × 4B = 12.6 MB/帧
    2203 帧 = 27.7 GB   ≫   本机内存 15.7 GB + 显存 8 GB
所以按 `--chunk`(默认 200 帧)切块, 每块单独 init_state, 块间靠"块首重新提示"衔接。
本机 8 GB 显存下若 OOM 会自动把块长减半重试。

=============================== 一个实测坑: 不能用 IoU 判漂移 ==============
落地段稳定框宽 35 px 而筒身只有 ~18 px(框被支腿撑大), 掩码 bbox 与框的 IoU
天然只有 ~0.5。若用"IoU < 0.5 就算漂移", 会天天误报。所以判据用:
  * containment = 交集 / 掩码bbox面积  —— 掩码是否**落在框内**(核心判据)
  * area_ratio  = 掩码面积 / 框面积     —— 相对本块中位的倍数
  * center_off  = 质心偏移 / 框对角线
且必须**连续 K 帧**不过才重锚定(单帧孤立跳变不算)—— 与第 2 步"底边模式状态机"同思路。

=============================== 落盘什么 ==============================
* `runs/seg/bounds_<tag>.npz`   —— 每帧的**逐行左右边界** int16 (n,H)×2, -1=无掩码。
  这是上游测角真正需要的量, 约 2×2203×480×2B ≈ 4 MB。存下它, 后处理调参就不必再上 GPU。
* `runs/seg/mask_stats_<tag>.csv` —— 逐帧体检与几何统计(便于画时间线)。
* `runs/seg/seg_meta_<tag>.json`  —— 运行配置与汇总。
* `--save-masks` 额外把打包位掩码顺序写到 `runs/seg/masks_<tag>.bin`(约 113 MB), 供事后诊断。
* `--make-video` 落叠加视频(掩码bbox + 稳定框 + 重锚定标记)。

用法:
    python scripts/rocket_seg.py                      # tiny, 全片, 分块传播
    python scripts/rocket_seg.py --start 1400 --end 1600
    python scripts/rocket_seg.py --model small --chunk 150
    python scripts/rocket_seg.py --save-masks --make-video
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SAM2_ROOT = PROJECT / "third_party" / "sam2"
sys.path.insert(0, str(SAM2_ROOT))
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402
import torch  # noqa: E402

from rocket_track import TrackerConfig, iou_xyxy, track_frames  # noqa: E402

DETS_DIR = PROJECT / "runs" / "diag"
OUT_DIR = PROJECT / "runs" / "seg"
WEIGHTS = PROJECT / "weights"

CFG_FILE = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
}
CKPT_FILE = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
}


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class SegConfig:
    model: str = "tiny"
    image_size: int = 1024
    half: bool = True                  # fp16 autocast
    chunk: int = 200                   # 每块帧数(受内存/显存约束)
    min_chunk: int = 50                # OOM 时能退到的最小块
    offload_video_to_cpu: bool = True   # 帧张量放内存, 省显存(仅小幅降速)
    obj_id: int = 1
    # --- 体检门限(见文件头"不能用 IoU 判漂移") ---
    # contain 只作**兜底**且在极小目标段会误杀(箭体仅 6~8px 宽时, 掩码越界 1~2 px
    # 就让 contain 掉到 0.74~0.81, 卡在 0.82 门限之下 → 实测成片误报)。主判据改用
    # 尺度相关的"越界像素数" escape: 越界超过 max(escape_px_min, escape_frac*框宽)
    # 才算漂移。这才和"掩码跑出框"这个语义一致。
    escape_px_min: float = 3.0
    escape_frac: float = 0.35
    min_containment: float = 0.45    # 兜底
    area_lo: float = 0.28              # 相对本块面积比中位的倍数下限
    area_hi: float = 2.60
    center_max: float = 0.75           # 质心偏移 / 框对角线
    bad_persist: int = 3               # 连续 K 帧不过才重锚定
    min_area_px: float = 12.0
    max_reanchor_per_chunk: int = 8
    # --- 帧抽取 ---
    jpeg_quality: int = 95
    # --- 产物 ---
    save_masks: bool = False
    make_video: bool = False
    verbose: bool = True


@dataclass
class FrameSeg:
    """一帧的分割结果。xl/xr 是逐行左右边界(全帧坐标, NaN=该行无掩码)。"""
    frame: int = 0
    has: bool = False
    ok: bool = False
    reason: str = ""
    area: float = float("nan")
    n_comp: int = 0
    x1: float = float("nan")
    y1: float = float("nan")
    x2: float = float("nan")
    y2: float = float("nan")
    cx: float = float("nan")
    cy: float = float("nan")
    iou_box: float = float("nan")
    contain: float = float("nan")
    area_ratio: float = float("nan")
    center_off: float = float("nan")
    escape: float = float("nan")       # 掩码越出稳定框的最大像素数(主判据)
    esc_tol: float = float("nan")      # 该帧的越界容差
    w_row_med: float = float("nan")
    h_rows: int = 0
    prompted: bool = False
    reanchored: bool = False
    bad_run: int = 0
    xl: np.ndarray | None = None
    xr: np.ndarray | None = None


# ==========================================================================
# mask 几何: 最大连通域 + 逐行左右边界
# ==========================================================================
def mask_geom(m: np.ndarray, y_pad: int = 1) -> dict:
    """取**最大连通域**的几何量(全帧坐标)。

    为什么必须取最大连通域: SAM 在落地段偶尔会把发动机喷流或地面亮斑一起分割进来,
    直接对整张掩码求 bbox 会被这些孤岛拽偏。
    y_pad: 上下各丢弃若干行(边界行常只覆盖一两个像素, 会污染宽度统计)。
    """
    H, W = m.shape
    m8 = m.astype(np.uint8)
    n_comp, lab, stats, cent = cv2.connectedComponentsWithStats(m8, 8)
    if n_comp <= 1:
        return dict(ok=False, n_comp=0)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[k, cv2.CC_STAT_AREA])
    if area < 1:
        return dict(ok=False, n_comp=int(n_comp - 1))
    x1i, y1i = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
    wi, hi = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
    sub = (lab[y1i:y1i + hi, x1i:x1i + wi] == k)
    xl = np.full(H, np.nan)
    xr = np.full(H, np.nan)
    for j in range(hi):
        row = np.flatnonzero(sub[j])
        if row.size:
            xl[y1i + j] = x1i + row[0]
            xr[y1i + j] = x1i + row[-1]
    keep = np.isfinite(xl)
    if y_pad > 0:
        idx = np.where(keep)[0]
        if idx.size > 2 * y_pad + 2:
            keep[idx[:y_pad]] = False
            keep[idx[-y_pad:]] = False
    wr = (xr - xl)[keep]
    return dict(ok=True, n_comp=int(n_comp - 1), area=area,
                x1=float(x1i), y1=float(y1i),
                x2=float(x1i + wi), y2=float(y1i + hi),
                cx=float(cent[k][0]), cy=float(cent[k][1]),
                xl=xl, xr=xr, rows=keep,
                w_row_med=float(np.median(wr)) if wr.size else float("nan"),
                h_rows=int(keep.sum()))


def health(g: dict, box: np.ndarray, cfg: SegConfig, area_ref: float) -> dict:
    """掩码 vs 稳定框的体检。判据**不用 IoU**(见文件头)。area_ref=本块面积比中位。"""
    if not g.get("ok"):
        return dict(bad=True, iou=float("nan"), contain=float("nan"),
                    area_ratio=float("nan"), center_off=float("nan"),
                    reason="no_mask")
    mb = np.array([g["x1"], g["y1"], g["x2"], g["y2"]])
    bw, bh = box[2] - box[0], box[3] - box[1]
    barea = max(bw * bh, 1.0)
    ix1, iy1 = max(mb[0], box[0]), max(mb[1], box[1])
    ix2, iy2 = min(mb[2], box[2]), min(mb[3], box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    marea = max((mb[2] - mb[0]) * (mb[3] - mb[1]), 1.0)
    contain = inter / marea
    ar = g["area"] / barea
    diag = max(float(np.hypot(bw, bh)), 1.0)
    off = float(np.hypot(g["cx"] - 0.5 * (box[0] + box[2]),
                         g["cy"] - 0.5 * (box[1] + box[3]))) / diag
    # 主判据: 掩码越出稳定框的最大像素数(尺度相关, 不会在小目标段误杀)
    escape = max(box[0] - mb[0], box[1] - mb[1], mb[2] - box[2], mb[3] - box[3])
    esc_tol = max(cfg.escape_px_min, cfg.escape_frac * bw)
    reason = ""
    if g["area"] < cfg.min_area_px:
        reason = "tiny_mask"
    elif escape > esc_tol:
        reason = "mask_escaped_box"
    elif contain < cfg.min_containment:
        reason = "mask_outside_box"
    elif np.isfinite(area_ref) and area_ref > 0 \
            and not (cfg.area_lo <= ar / area_ref <= cfg.area_hi):
        reason = "area_jump"
    elif off > cfg.center_max:
        reason = "center_drift"
    return dict(bad=bool(reason), iou=float(iou_xyxy(mb, box)), contain=contain,
                area_ratio=ar, center_off=off, reason=reason,
                escape=float(escape), esc_tol=float(esc_tol))


# ==========================================================================
# 主体: 分块 + 传播 + 体检 + 重锚定
# ==========================================================================
class Sam2Segmenter:
    def __init__(self, cfg: SegConfig | None = None):
        self.cfg = cfg or SegConfig()
        self.predictor = None

    def load(self):
        from sam2.build_sam import build_sam2_video_predictor
        c = self.cfg
        ovr = [f"++model.image_size={c.image_size}"]
        self.predictor = build_sam2_video_predictor(
            CFG_FILE[c.model], str(WEIGHTS / CKPT_FILE[c.model]),
            device="cuda", hydra_overrides_extra=ovr)
        return self.predictor

    def segment_chunk(self, frames_dir: Path, local_of: dict[int, int],
                      boxes: dict[int, np.ndarray], chunk_area_ref: list
                      ) -> tuple[dict[int, FrameSeg], dict[int, bytes]]:
        """local_of: {全局帧号: 局部索引}; boxes: {全局帧号: 稳定框}。

        返回 ({全局帧号: FrameSeg}, {全局帧号: 打包位掩码 bytes})。
        掩码用 dict 收集而不是边传边写文件: 重锚定会让 propagate 把同一段重放一遍,
        边写边追加会产生**重复帧**; dict 天然覆盖去重, 且内存只有
        chunk×51KB(200 帧约 10 MB)。
        """
        cfg = self.cfg
        want_mask = cfg.save_masks
        packed: dict[int, bytes] = {}
        idx_of_local = {v: k for k, v in local_of.items()}
        # ★ 锚点必须是"块内**第一个有稳定框**的帧"。
        # 本视频前 ~10s 箭体还没被检出(检测器只给出 Space 类), 那一段 local_of 有帧
        # 但 boxes 为空; 若拿它当锚点, propagate_in_video 会直接抛
        # "cannot propagate... no prompt"。整块无框时直接跳过, 连 init_state 都不做。
        anchor = next((l for l in sorted(idx_of_local)
                       if idx_of_local[l] in boxes), None)
        if anchor is None:
            return {}, {}
        state = self.predictor.init_state(
            str(frames_dir), offload_video_to_cpu=cfg.offload_video_to_cpu)
        res: dict[int, FrameSeg] = {}
        area_ref = chunk_area_ref[0] if chunk_area_ref else float("nan")
        n_reanchor = 0
        last_stop = -1
        last_local = max(idx_of_local)
        while True:
            g_anchor = idx_of_local[anchor]
            box_a = boxes.get(g_anchor)
            if box_a is not None:
                with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.float16, enabled=cfg.half):
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=anchor, obj_id=cfg.obj_id,
                        box=torch.as_tensor(box_a, dtype=torch.float32))
                res.setdefault(g_anchor, FrameSeg(frame=g_anchor)).prompted = True
            bad_run = 0
            bad_start = None
            stop_at = None
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=cfg.half):
                for fidx, _oids, masks in self.predictor.propagate_in_video(
                        state, start_frame_idx=anchor):
                    if fidx not in idx_of_local:
                        continue
                    g_f = idx_of_local[fidx]
                    # masks: (num_obj, 1, H, W) 的 logits
                    m = (masks[0, 0] > 0.0).cpu().numpy()
                    if want_mask:
                        packed[g_f] = np.packbits(
                            np.ascontiguousarray(m).reshape(-1)).tobytes()
                    b = boxes.get(g_f)
                    fs = res.setdefault(g_f, FrameSeg(frame=g_f))
                    if b is None:
                        fs.reason = "no_box"
                        continue
                    g = mask_geom(m)
                    h = health(g, b, cfg, area_ref)
                    fs.has = bool(g.get("ok"))
                    fs.reason = h["reason"]
                    if not g.get("ok"):
                        fs.ok, bad_run = False, bad_run + 1
                        if bad_start is None:
                            bad_start = fidx
                        continue
                    fs.area, fs.n_comp = g["area"], g["n_comp"]
                    for k in ("x1", "y1", "x2", "y2", "cx", "cy"):
                        fs.__dict__[k] = g[k]
                    fs.xl, fs.xr = g["xl"], g["xr"]
                    fs.w_row_med, fs.h_rows = g["w_row_med"], g["h_rows"]
                    fs.iou_box, fs.contain = h["iou"], h["contain"]
                    fs.area_ratio, fs.center_off = h["area_ratio"], h["center_off"]
                    fs.escape, fs.esc_tol = h["escape"], h["esc_tol"]
                    fs.ok = not h["bad"]
                    if h["bad"]:
                        bad_run += 1
                        if bad_start is None:
                            bad_start = fidx
                    else:
                        bad_run, bad_start = 0, None
                    fs.bad_run = bad_run
                    # 面积比中位: 用已确认良好的帧在线更新(自适应口径)
                    if fs.ok:
                        chunk_area_ref[0] = (fs.area_ratio if not np.isfinite(
                            area_ref) else 0.9 * area_ref + 0.1 * fs.area_ratio)
                        area_ref = chunk_area_ref[0]
                    if bad_run >= cfg.bad_persist:
                        if n_reanchor < cfg.max_reanchor_per_chunk:
                            stop_at = bad_start if bad_start is not None else fidx
                        break
            if stop_at is None:
                break
            # ★ 两个必须的收尾规则(实测踩坑):
            # (1) 重锚定额度用尽后**不能直接 break** —— 那会把本块剩下的帧整块丢掉
            #     (实测 chunk 200-400 只产出 30 帧)。改成不再补提示、但继续靠记忆
            #     把余下的帧走完, 让它们被如实标成不可信。
            # (2) 不许原地打转: 若新锚点没有比上次失败点更靠后, 说明补提示救不回来,
            #     强制往前跳一段(否则会在同一帧上反复重锚 8 次后仍停在原处)。
            if n_reanchor >= cfg.max_reanchor_per_chunk:
                nxt = int(min(stop_at + 1, last_local))
                if nxt <= last_stop:
                    break
                anchor = nxt
                continue
            n_reanchor += 1
            cand = int(min(stop_at, last_local))
            if cand <= last_stop:
                cand = int(min(stop_at + max(2 * cfg.bad_persist, 4), last_local))
            if cand <= anchor and anchor >= last_local:
                break
            last_stop = stop_at
            anchor = cand
            fs = res.setdefault(idx_of_local[anchor], FrameSeg(
                frame=idx_of_local[anchor]))
            fs.reanchored = True
            if cfg.verbose:
                print(f"      [重锚定 #{n_reanchor}] 局部 {anchor} "
                      f"(全局 {idx_of_local[anchor]})", flush=True)
        self.predictor.reset_state(state)
        del state
        torch.cuda.empty_cache()
        return res, packed


# ==========================================================================
# 跑全片
# ==========================================================================
def run(cfg: SegConfig | None = None, dets_tag: str = "iou70",
        start: int = 0, end: int | None = None, est=None) -> tuple:
    cfg = cfg or SegConfig()
    meta = json.loads((DETS_DIR / f"dets_{dets_tag}.json").read_text(
        encoding="utf-8"))
    vm = meta["meta"]
    W, H_, fps = vm["W"], vm["H"], vm["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]],
                        TrackerConfig(), fps, bound_wh=(W, H_))
    n_all = len(outs)
    end = n_all if end is None else min(end, n_all)

    video = PROJECT / vm["video"]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        tmp = CACHE / "tmp" / "src_seg.mp4"
        shutil.copy(str(video), str(tmp))
        cap = cv2.VideoCapture(str(tmp))

    seg = est or Sam2Segmenter(cfg)
    if seg.predictor is None:
        print(f"[构建] sam2.1_hiera_{cfg.model} image_size={cfg.image_size} "
              f"half={cfg.half}", flush=True)
        t0 = time.time()
        seg.load()
        print(f"[构建] {time.time() - t0:.1f}s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_fp = open(OUT_DIR / "masks.bin", "wb") if cfg.save_masks else None
    rec_bytes = (H_ * W + 7) // 8          # 每帧打包位掩码字节数

    frames_root = CACHE / "seg_frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    all_res: dict[int, FrameSeg] = {}
    writer = None
    if cfg.make_video:
        writer = cv2.VideoWriter(str(OUT_DIR / "seg_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H_))

    t_start = time.time()
    chunk = cfg.chunk
    i = start
    while i < end:
        # 面积比参考**每块重置**: 掩码面积/框面积 这个比值随飞行阶段变化
        # (下降段框紧贴箭体, 落地段框被支腿撑大), 跨块沿用会把块边界误判成面积突变。
        area_ref = [float("nan")]
        j = min(i + chunk, end)
        # 帧目录按**区间**命名, 且从不删除 —— 一次性删 >50 个文件会被宿主安全策略拦成
        # SAFE_DELETE_BULK_CONFIRM_REQUIRED, 整条命令作废(实测踩过)。这些 JPEG 都在
        # .cache 下且被 git 忽略, 全片约 70 MB; 留着还能让重跑免于重复抽帧。
        fdir = frames_root / f"c{i:05d}_{j:05d}"
        fdir.mkdir(parents=True, exist_ok=True)
        local_of, boxes, frames_cache = {}, {}, {}
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        for k in range(i, j):
            ok, fr = cap.read()
            if not ok:
                break
            fp = fdir / f"{k:05d}.jpg"
            if not fp.exists():
                cv2.imwrite(str(fp), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            local_of[k] = k - i
            o = outs[k]
            if o.has and o.box is not None:
                boxes[k] = np.asarray(o.box, float)
            if writer is not None:
                frames_cache[k] = fr
        print(f"[块 {i}-{j}) {len(local_of)} 帧, 有框 {len(boxes)}", flush=True)
        if not local_of:
            break
        while True:
            try:
                r, packed = seg.segment_chunk(fdir, local_of, boxes, area_ref)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk <= cfg.min_chunk:
                    raise
                chunk = max(cfg.min_chunk, chunk // 2)
                j = min(i + chunk, end)
                print(f"    [OOM] 块长降到 {chunk}, 换更短区间重来", flush=True)
                fdir = frames_root / f"c{i:05d}_{j:05d}"
                fdir.mkdir(parents=True, exist_ok=True)
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                for k in range(i, j):
                    ok, fr = cap.read()
                    if not ok:
                        break
                    fp = fdir / f"{k:05d}.jpg"
                    if not fp.exists():
                        cv2.imwrite(str(fp), fr,
                                    [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
                local_of = {k: k - i for k in range(i, j)}
                boxes = {k: v for k, v in boxes.items() if i <= k < j}
        all_res.update(r)
        if mask_fp is not None:
            # 按帧号顺序写, 缺掩码的帧写全零占位, 保证"记录号 = 帧号 - start"
            for k in range(i, j):
                mask_fp.write(packed.get(k) or bytes(rec_bytes))
        print(f"    完成 {len(r)} 帧, 可信 {sum(1 for v in r.values() if v.ok)}, "
              f"重锚定 {sum(1 for v in r.values() if v.reanchored)}", flush=True)
        if writer is not None:
            for k, fr in frames_cache.items():
                writer.write(_draw(fr, outs[k], all_res.get(k), fps))
        i = j
    cap.release()
    if writer is not None:
        writer.release()
    if mask_fp is not None:
        mask_fp.close()
    el = time.time() - t_start
    print(f"[完成] {len(all_res)} 帧, {el:.1f}s "
          f"({len(all_res) / max(el, 1e-9):.1f} fps)", flush=True)
    info = dict(W=W, H=H_, fps=fps, video=vm["video"], n_frame=n_all,
                model=cfg.model, image_size=cfg.image_size, half=cfg.half,
                chunk=chunk, elapsed_s=round(el, 1),
                n_track=sum(1 for o in outs if o.has),
                n_ok=sum(1 for v in all_res.values() if v.ok),
                n_reanchor=sum(1 for v in all_res.values() if v.reanchored),
                start=start, end=end, save_masks=cfg.save_masks,
                mask_rec_bytes=rec_bytes)
    return all_res, outs, info


def _draw(frame, out, fs: FrameSeg | None, fps: float) -> np.ndarray:
    img = frame.copy()
    if out is not None and out.box is not None:
        b = [int(v) for v in out.box]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
    if fs is not None and fs.has and np.isfinite(fs.x1):
        col = (0, 255, 0) if fs.ok else (0, 140, 255)
        if fs.xl is not None:
            ys = np.where(np.isfinite(fs.xl))[0]
            if ys.size:
                for y in (ys[0], ys[-1]):
                    cv2.line(img, (int(fs.xl[y]), int(y)), (int(fs.xr[y]), int(y)),
                             col, 1)
        cv2.rectangle(img, (int(fs.x1), int(fs.y1)), (int(fs.x2), int(fs.y2)),
                      col, 1)
        cv2.circle(img, (int(fs.cx), int(fs.cy)), 2, col, -1)
    if fs is None or not fs.has:
        txt = "no-mask"
    else:
        txt = (f"{'OK ' if fs.ok else 'BAD ' + fs.reason} area={fs.area:.0f} "
               f"cont={fs.contain:.2f} ar={fs.area_ratio:.2f} "
               f"w={fs.w_row_med:.1f}")
        if fs.reanchored:
            txt += "  <<REANCHOR"
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ==========================================================================
# 落盘
# ==========================================================================
COLS = ["frame", "t", "ok", "has", "reason", "area", "n_comp", "x1", "y1",
        "x2", "y2", "cx", "cy", "iou_box", "contain", "area_ratio",
        "center_off", "escape", "esc_tol", "w_row_med", "h_rows", "prompted",
        "reanchored", "bad_run"]
# CSV 里"fram/t"之后的所有列都直接按 FrameSeg 的同名字段导出(避免下标硬编码错位)
FIELDS = COLS[2:]


def save(all_res: dict[int, FrameSeg], outs: list, info: dict,
         tag: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    n, H, W = info["n_frame"], info["H"], info["W"]
    fps = info["fps"]

    XL = np.full((n, H), -1, np.int16)
    XR = np.full((n, H), -1, np.int16)
    for f, fs in all_res.items():
        if fs.xl is None:
            continue
        k = np.isfinite(fs.xl)
        XL[f, k] = np.round(fs.xl[k]).astype(np.int16)
        XR[f, k] = np.round(fs.xr[k]).astype(np.int16)
    np.savez_compressed(OUT_DIR / f"bounds{sfx}.npz", XL=XL, XR=XR)
    print(f"[写入] {OUT_DIR / f'bounds{sfx}.npz'}  shape={XL.shape}", flush=True)

    def g(v):
        if isinstance(v, bool):
            # ★ 必须显式转 int: csv 会把 Python 的 True/False 写成字符串 "True"/"False",
            # 下游按 "1"/"0" 解析就全部读成 0(实测踩过, 诊断脚本报"可用率 0%")。
            return int(v)
        if isinstance(v, float):
            return round(v, 4) if np.isfinite(v) else ""
        return v

    with (OUT_DIR / f"mask_stats{sfx}.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for f in range(n):
            fs = all_res.get(f)
            if fs is None:
                # 该帧没进过任何块(块内无框被跳过, 或本身不在 [start,end) 内)
                w.writerow([f, round(f / fps, 4), 0, 0, "no_track"]
                           + [""] * (len(FIELDS) - 3))
                continue
            w.writerow([f, round(f / fps, 4)]
                       + [g(getattr(fs, k)) for k in FIELDS])
    (OUT_DIR / f"seg_meta{sfx}.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[写入] {OUT_DIR / f'mask_stats{sfx}.csv'} / seg_meta{sfx}.json",
          flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 2.1 视频模式分割箭体(方案 B)")
    p.add_argument("--dets-tag", default="iou70")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--model", default="tiny", choices=list(CFG_FILE))
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=200)
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--save-masks", action="store_true")
    p.add_argument("--make-video", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = SegConfig(model=a.model, image_size=a.image_size, chunk=a.chunk,
                    half=not a.no_half, save_masks=a.save_masks,
                    make_video=a.make_video)
    res, outs, info = run(cfg, dets_tag=a.dets_tag, start=a.start,
                          end=(a.end or None))
    save(res, outs, info, a.tag)


if __name__ == "__main__":
    main()
