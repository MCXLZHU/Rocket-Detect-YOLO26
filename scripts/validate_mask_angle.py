# -*- coding: utf-8 -*-
"""SAM 路线的准入门槛: 旋转注入测试 + 落地段重复性 + 与原方案对照。

=============================== 为什么必须要这一关 ==============================
原方案在选型时踩过一个坑: **Otsu+minAreaRect 的 σ 比最终方案还小, 但旋转增益只有
0.196** —— 它对旋转几乎不响应, "稳定"是假象。
SAM 路线的落地段重复性 σ≈0.04°(比原方案小 6 倍), 但这个数字同样可能来自
**视频模式的记忆平滑**(记忆让掩码帧间过于一致, 反而掩盖了真实运动)。
所以唯一硬指标还是旋转注入: 把图人为旋转 θ, 测出的 φ 必须正好变化 θ, 增益≈1。

=============================== 测试怎么做 ==============================
用**单帧序列 + 视频预测器**而不是 SAM2ImagePredictor:
  * 需求上: 旋转注入要求每个角度独立, 视频模式的记忆会把"未旋转"的信息带进来;
    单帧序列没有跨帧记忆, 等价于独立分割, 又走的是**实际部署同一条代码路径**。
  * 工程上: 本机 torch 2.14 下 SAM2ImagePredictor.set_image 会因
    `permute(...).view(...)` 报 "view size is not compatible with input tensor's
    size and stride"(新 torch 不再允许对非连续张量用 view) —— 不改第三方源码。
对每个 θ ∈ {-3,-1.5,-1,-0.5,0,+0.5,+1,+1.5,+3}°:
  1. 以**框中心**为轴旋转整帧(绕框中心而非画面中心, 目标才不会跑出画面);
  2. 把框的四个角一起旋转, 取旋转后的**轴对齐外接框**作为新提示框;
     ★ 这一步就是原方案文档里第 7 条坑("只旋转不扩框 ⇒ 目标两端跑出搜索窗 ⇒
       增益被系统性压缩"), 取外接框天然包含了水平扩宽 |tanθ|·h/2;
  3. SAM 分割 -> 逐行边界 -> 同一套中心线拟合 -> φ;
  4. 对 φ(θ) 做线性回归: 斜率 = 旋转增益, 截距 = 偏置, 残差 = 随机误差。

用法:
    python scripts/validate_mask_angle.py --tag s512 --frames 1400,1500,1600,1800,2000
    python scripts/validate_mask_angle.py --tag s512 --image-size 512 --angles -3,-1.5,-1,-0.5,0,0.5,1,1.5,3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["TORCH_HOME"] = str(CACHE / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
sys.path.insert(0, str(PROJECT / "third_party" / "sam2"))
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402
import torch  # noqa: E402

from rocket_mask_angle import MaskAngleConfig, estimate  # noqa: E402
from rocket_seg import CFG_FILE, CKPT_FILE, mask_geom  # noqa: E402

DETS = PROJECT / "runs" / "diag"
OUT = PROJECT / "runs" / "angle_mask"
WEIGHTS = PROJECT / "weights"


def rot_box(box: np.ndarray, center: np.ndarray, ang_deg: float) -> np.ndarray:
    """把框四角绕 center 旋转后取轴对齐外接框(= 原方案文档第 7 条坑的正确做法)。"""
    th = math.radians(ang_deg)
    c, s = math.cos(th), math.sin(th)
    pts = np.array([[box[0], box[1]], [box[2], box[1]],
                    [box[2], box[3]], [box[0], box[3]]]) - center
    r = np.stack([c * pts[:, 0] - s * pts[:, 1],
                  s * pts[:, 0] + c * pts[:, 1]], axis=1) + center
    return np.array([r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()])


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM 路线旋转注入验证")
    ap.add_argument("--tag", default="s512", help="seg 产物标签(用于读框与对照)")
    ap.add_argument("--model", default="tiny", choices=list(CFG_FILE))
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--angles", default="-3,-1.5,-1,-0.5,0,0.5,1,1.5,3")
    ap.add_argument("--frames", default="1400,1500,1600,1800,2000")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--no-half", dest="half", action="store_false")
    a = ap.parse_args()

    from sam2.build_sam import build_sam2_video_predictor

    meta = json.loads((DETS / "dets_iou70.json").read_text(encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]

    print(f"[构建] video predictor(单帧序列) sam2.1_hiera_{a.model} "
          f"image_size={a.image_size}  fp16={a.half}", flush=True)
    model = build_sam2_video_predictor(
        CFG_FILE[a.model], str(WEIGHTS / CKPT_FILE[a.model]),
        device="cuda",
        hydra_overrides_extra=[f"++model.image_size={a.image_size}"])

    angles = [float(v) for v in a.angles.split(",") if v.strip()]
    frames = [int(v) for v in a.frames.split(",") if v.strip()]
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))
    macfg = MaskAngleConfig()
    rot_root = CACHE / "rot_test"
    rot_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in frames:
        rb = [q for q in meta["frames"][f]["b"] if q["c"] == 1]
        if not rb:
            continue
        box = np.asarray(rb[0]["x"], float)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        for th in angles:
            M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])),
                                        th, 1.0)
            rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
            b2 = rot_box(box, center, th)
            d = rot_root / f"f{f}_{int(round(th * 100)):+05d}"
            d.mkdir(parents=True, exist_ok=True)
            # 单帧序列: 文件名 00000.jpg
            cv2.imwrite(str(d / "00000.jpg"), rot,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            state = model.init_state(str(d), offload_video_to_cpu=True)
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=a.half):
                _, _ids, masks = model.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    box=torch.as_tensor(b2, dtype=torch.float32))
            m = (masks[0, 0] > 0.0).cpu().numpy()
            model.reset_state(state)
            del state
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST) > 0
            g = mask_geom(m)
            if not g.get("ok"):
                rows.append((f, th, float("nan"), float("nan"), 0))
                continue
            e = estimate(g["xl"], g["xr"], macfg)
            if e is None:
                rows.append((f, th, float("nan"), float("nan"), 0))
                continue
            rows.append((f, th, e["phi"], e["w_body"], e["n"]))
    cap.release()
    del model
    torch.cuda.empty_cache()

    phi = np.array([r[2] for r in rows], float)
    ths = np.array([r[1] for r in rows], float)
    fr_arr = np.array([r[0] for r in rows])
    good = np.isfinite(phi)
    L = ["=== SAM 路线 旋转注入测试 ===",
         f"模型 sam2.1_hiera_{a.model}  image_size={a.image_size}  "
         f"fp16={a.half}  (单帧序列 + 视频预测器, 逐角度独立分割)",
         f"注入角度: {angles}",
         f"测试帧: {frames}",
         f"有效样本 {int(good.sum())}/{len(rows)}", ""]
    if good.sum() >= 5:
        # ★ 符号约定: cv2.getRotationMatrix2D 的正角 = 逆时针。逆时针转 θ 会让箭体
        #   顶端向左倾, 而 φ 的定义是"正 = 顶端向右倾", 所以**理想增益是 −1**。
        #   (原方案文档里他们的注入方向与自己的 φ 同号, 所以那边理想是 +1。)
        L += ["符号约定: 正角 = 逆时针(顶端左倾) ⇒ 理想增益 = −1.0000"]
        per, resid_all = [], []
        for f in frames:
            m = good & (fr_arr == f)
            if m.sum() < 4:
                continue
            kk, bb = np.polyfit(ths[m], phi[m], 1)
            rr = phi[m] - (kk * ths[m] + bb)
            per.append((f, kk, bb, float(rr.std()), int(m.sum())))
            resid_all.extend(rr.tolist())
        if per:
            ks = np.array([p[1] for p in per])
            bs = np.array([p[2] for p in per])
            L += ["",
                  f"★ 逐帧旋转增益(应≈−1): "
                  + "  ".join(f"f{f}:{k:+.4f}" for f, k, _b, _s, _n in per),
                  f"  增益中位 {np.median(ks):+.4f}  |增益|中位 "
                  f"{np.median(np.abs(ks)):.4f}  "
                  f"帧间 σ {ks.std():.4f}  (即增益的帧间一致性)",
                  f"  偏置中位 {np.median(bs):+.4f}°",
                  f"  帧内线性残差(合并) σ {np.std(resid_all):.4f}°  "
                  f"最大 {np.abs(resid_all).max() if resid_all else 0:.4f}°"]
            # 合并回归(仅供参考) —— 会把帧间截距差算进残差, 偏大
            k_all, b_all = np.polyfit(ths[good], phi[good], 1)
            r_all = phi[good] - (k_all * ths[good] + b_all)
            L.append(f"  (合并回归, 仅供参考: 增益 {k_all:+.4f}, "
                     f"偏置 {b_all:+.4f}°, 残差 σ {r_all.std():.4f}°"
                     f" —— 残差被帧间截距差放大, 不是随机误差)")
        L += ["",
              "  对照(原方案文档 §5.2 实测): 增益 0.898, 偏置 -0.229°, "
              "残差 σ 0.187°",
              "  对照: 逐行梯度峰 0.712 | Otsu+minAreaRect 0.196"]
    else:
        L.append("有效样本不足, 无法回归")
    per_angle = []
    L += ["", "逐角度中位 φ:"]
    for th in angles:
        m = np.array([r[1] for r in rows]) == th
        v = phi[m]
        v = v[np.isfinite(v)]
        if v.size:
            L.append(f"  θ={th:+6.2f}°  n={v.size}  φ 中位 {np.median(v):+.4f}°  "
                     f"σ {v.std():.4f}°")
            per_angle.append((th, float(np.median(v)), float(v.std())))
    txt = "\n".join(L)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rot_test_{a.tag}.txt").write_text(txt, encoding="utf-8")
    (OUT / f"rot_test_{a.tag}.json").write_text(json.dumps(
        dict(tag=a.tag, model=a.model, image_size=a.image_size,
             angles=angles, frames=frames,
             rows=[dict(frame=int(r[0]), theta=r[1], phi=r[2],
                        w_body=r[3], n_rows=r[4]) for r in rows],
             per_angle=per_angle), ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(txt, flush=True)
    print(f"[写入] {OUT / f'rot_test_{a.tag}.txt'}", flush=True)


if __name__ == "__main__":
    main()
