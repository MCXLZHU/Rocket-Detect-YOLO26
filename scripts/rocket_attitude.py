# -*- coding: utf-8 -*-
"""第 4 步: 姿态反算。

=============================== 先给结论 ===============================
本视频**没有相机云台/内参参数**, 且是**单视角**。在这个前提下:

  ✅ **可行**: 箭体轴线 **相对图像方向的夹角**(第 3 步已给出 φ), 以及
     **相对倾角** φ−φ_land。这是纯几何量, 不需要任何相机参数。
     —— 前提之一是"相机静止"。本模块用背景配准**实测验证**了这一点(见 §验证),
     并且用**注入已知运动**的方法证明该验证本身是灵敏的(否则"静止"可能只是
     估计器不敏感造成的假象)。

  ⚠️ **不可靠**: 完整的 3D 姿态。单视角下"面外角 β"只能通过
        cos β = (投影长度 / 投影宽度) × (D / L)
     反解, 它同时依赖 (a) 火箭真实长径比 L/D —— 本视频无从得知;
     (b) 投影宽度必须量到**筒身轮廓** —— 实测该量在 0.34~0.82 倍框宽之间波动
     (σ=4.0px), 且框宽被展开的支腿/栅格舵撑大; (c) β 的符号(朝向/远离相机)无法区分。
     实测灵敏度: 宽度取 18.4px 推出 β≈0°, 取 24px 推出 β≈37° —— **同一个画面给出
     0° 到 37° 的结果**, 所以这个量只能作为"参数化的假想姿态", 不能当结论用。

因此本模块的主输出 = **箭体与图像竖直/图像水平边的夹角**, 3D 姿态作为可选、
显式参数化并附灵敏度表。

=============================== 输入 ===============================
**SAM 2.1 掩码路线**的测角结果 `runs/angle_mask/angles_<src-tag>.json`
(默认 s512)。字段与旧的原方案产物同构, 所以这里只换路径, 几何换算完全复用。
原来的输入 `runs/angle/angles.json`(ROI 梯度边缘)已随原方案一起下线。

=============================== 输出 ===============================
  runs/attitude/attitude.csv / .json   逐帧: 各种夹角 + 相机运动 + 可选 β
  runs/attitude/attitude_timeline.png  时间线
  runs/attitude/camera_check.txt       相机静止性验证报告(含注入运动校验)
用法:
    python scripts/rocket_attitude.py
    python scripts/rocket_attitude.py --src-tag s1024     # 换另一套分割配置
    python scripts/rocket_attitude.py --l-over-d 17      # 给出长径比才算 β
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")

import cv2  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts"))
from video_io import resolve_video  # noqa: E402

DIAG = PROJECT / "runs" / "diag"
ANGM = PROJECT / "runs" / "angle_mask"    # 掩码路线测角结果(本步的输入)
OUT = PROJECT / "runs" / "attitude"


# ==========================================================================
# 配置
# ==========================================================================
@dataclass
class AttitudeConfig:
    # 注: 原来这里有 ref_range=(45,66) —— "片尾落地静止段"当竖直基准, 那是只对
    # 某一段视频成立的先验, 已删除。本步只输出**绝对夹角**(轴线 vs 图像竖直/上边缘),
    # 不需要任何参考段; 相对倾角 φ_rel 随之废弃。
    cam_ref_frac: float = 0.5         # 相机配准参考帧: 取有效区间的该分位处(0~1)
    l_over_d: float = float("nan")    # 火箭真实长径比; 不给就不算面外角
    ld_scan: tuple = (8.0, 11.0, 14.0, 17.0, 20.0)   # 灵敏度扫描用
    cam_scale: float = 0.5            # 背景配准的降采样倍数(提速)
    cam_max_deg: float = 2.0          # 单帧滚转搜索范围
    cam_mask_pad: float = 0.7         # 屏蔽火箭区域的额外边距(×框宽/高)
    static_tol_deg: float = 0.30      # 判定"相机静止"的滚转阈值
    cam_ncc_min: float = 0.60         # 背景配准可信的 NCC 下限
    min_n_bands: int = 12
    min_w_body: float = 6.0
    # 相机估计器自身的灵敏度校验(注入已知运动)
    inj_rot_deg: tuple = (0.25, 0.5, 1.0)
    inj_shift_px: tuple = (0.5, 1.0, 2.0)


@dataclass
class FrameAtt:
    frame: int = 0
    t: float = 0.0
    ok: bool = False
    phi_deg: float = float("nan")          # 相对图像竖直(有符号, 正=顶端右倾)
    phi_vs_top: float = float("nan")       # 相对图像上边缘(水平方向)的夹角
    phi_vs_top_signed: float = float("nan")
    phi_rel: float = float("nan")          # 相对"落地竖直基准"
    tilt_total: float = float("nan")       # acos(cosβ·cosφ), 需要 β
    beta_deg: float = float("nan")         # 面外角(需要 L/D)
    aspect: float = float("nan")           # 投影长度/投影宽度
    axis_len_px: float = float("nan")
    w_body_px: float = float("nan")
    w_box_px: float = float("nan")
    conf: float = 0.0
    n_bands: int = 0
    dphi_dt: float = float("nan")
    cam_dtheta: float = float("nan")       # 本帧相机滚转增量
    cam_roll: float = float("nan")         # 累计相机滚转(相对首帧)
    cam_dx: float = float("nan")
    cam_dy: float = float("nan")
    cam_ncc: float = float("nan")
    reason: str = ""


# ==========================================================================
# 相机运动估计: 背景配准(屏蔽火箭区域)
# ==========================================================================
def bg_mask(frame_gray: np.ndarray, box, cfg: AttitudeConfig) -> np.ndarray:
    """把火箭所在区域涂掉, 只留背景。返回 float32。"""
    g = frame_gray.astype(np.float32)
    H, W = g.shape
    if box is not None:
        wb = max(box[2] - box[0], 4.0)
        hb = max(box[3] - box[1], 4.0)
        x0 = int(max(0, box[0] - cfg.cam_mask_pad * wb))
        x1 = int(min(W, box[2] + cfg.cam_mask_pad * wb))
        y0 = int(max(0, box[1] - 0.15 * hb))
        y1 = int(min(H, box[3] + 0.15 * hb))
        g[y0:y1, x0:x1] = 0.0
    return g


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-9
    return float((a * b).sum()) / d


def estimate_frame_motion(prev: np.ndarray, cur: np.ndarray,
                          cfg: AttitudeConfig) -> tuple[float, float, float, float]:
    """返回 (dtheta_deg, dx, dy, ncc)。

    1) 小角度旋转搜索(粗→精)得到滚转增量 —— 相机滚转才是会**旋转箭体像**的分量;
    2) 相位相关得到平移 —— 纯平移**不会**改变直线的方向, 只作诊断。

    注意纯水平 pan(绕世界竖直轴转)在水平相机下不会让竖直线倾斜, 所以这里测的是
    图像域的刚体旋转, 对"箭体像是否被相机转歪"来说正是需要的量。
    """
    s = cfg.cam_scale
    a0 = cv2.resize(prev, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b0 = cv2.resize(cur, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    h, w = a0.shape
    c = (w / 2.0, h / 2.0)
    best = (0.0, -2.0)

    def sc(th):
        M = cv2.getRotationMatrix2D(c, th, 1.0)
        r = cv2.warpAffine(a0, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return _ncc(r, b0)

    for th in np.arange(-cfg.cam_max_deg, cfg.cam_max_deg + 1e-9, 0.25):
        v = sc(float(th))
        if v > best[1]:
            best = (float(th), v)
    t0 = best[0]
    for th in np.arange(t0 - 0.25, t0 + 0.25 + 1e-9, 0.02):
        v = sc(float(th))
        if v > best[1]:
            best = (float(th), v)
    try:
        (dx, dy), _ = cv2.phaseCorrelate(a0, b0)
    except Exception:
        dx = dy = float("nan")
    return best[0], float(dx) / s, float(dy) / s, best[1]


def validate_motion_estimator(frame_gray: np.ndarray, box, cfg: AttitudeConfig):
    """注入已知旋转/平移, 检验相机运动估计器是否灵敏。

    为什么必须做: "测得相机静止"有两种可能 —— 真相机静止, 或估计器根本不敏感。
    只有注入已知运动并成功恢复, 才能排除后者(这与第 3 步"稳定≠准确"是同一个道理)。
    """
    bg = bg_mask(frame_gray, box, cfg)
    h, w = bg.shape
    c = (w / 2.0, h / 2.0)
    rows_rot, rows_sh = [], []
    for inj in cfg.inj_rot_deg:
        M = cv2.getRotationMatrix2D(c, inj, 1.0)
        rotated = cv2.warpAffine(bg, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        th, _, _, _ = estimate_frame_motion(bg, rotated, cfg)
        rows_rot.append((inj, th))
    for inj in cfg.inj_shift_px:
        M = np.float32([[1, 0, -inj], [0, 1, 0]])
        shifted = cv2.warpAffine(bg, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        _, dx, _, _ = estimate_frame_motion(bg, shifted, cfg)
        rows_sh.append((inj, dx))
    return rows_rot, rows_sh


# ==========================================================================
# 角度换算
# ==========================================================================
def angle_conventions(phi_deg: float) -> tuple[float, float]:
    """由"相对图像竖直的有符号倾角"换算出其它常用口径。

    phi_vs_top_signed = 90 − φ  (相对图像**上边缘/水平方向**的带符号夹角)
    phi_vs_top        = 90 − |φ| (轴线与水平边的锐角, 0~90)
    """
    if not np.isfinite(phi_deg):
        return float("nan"), float("nan")
    return 90.0 - abs(phi_deg), 90.0 - phi_deg


def beta_from_aspect(axis_len: float, w_body: float,
                     l_over_d: float) -> tuple[float, float]:
    """"面外角"反解: cosβ = (投影长度/投影宽度) · (D/L)。

    推导: 轴与像面夹角 β ⇒ 投影长度 = (f/Z)·L·cosβ;
          圆柱剪影宽度 ≈ (f/Z)·D (与倾角无关);
          两式相除 ⇒ 长宽比 = (L/D)·cosβ。
    返回 (beta_deg, cosβ_raw)。cosβ_raw>1 说明几何不自洽(通常意味着
    w_body 量的不是筒身轮廓, 而是更窄的涂装条纹)。
    """
    if not (np.isfinite(axis_len) and np.isfinite(w_body)
            and np.isfinite(l_over_d)) or w_body <= 0 or l_over_d <= 0:
        return float("nan"), float("nan")
    raw = (axis_len / w_body) / l_over_d
    return math.degrees(math.acos(min(max(raw, 0.0), 1.0))), raw


def total_tilt(phi_deg: float, beta_deg: float) -> float:
    """轴线相对世界竖直的总夹角: cos(tilt) = cosβ·cosφ (在"相机≈世界"假设下)。"""
    if not (np.isfinite(phi_deg) and np.isfinite(beta_deg)):
        return float("nan")
    return math.degrees(math.acos(min(max(
        math.cos(math.radians(beta_deg)) * math.cos(math.radians(phi_deg)),
        -1.0), 1.0)))


# ==========================================================================
# 主流程
# ==========================================================================
def open_cap(path: Path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "att_src.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def run(cfg: AttitudeConfig | None = None, dets_tag: str = "iou70",
        verbose: bool = True, src_tag: str = "s512"):
    """src_tag: 第 3 步测角结果的标签。

    输入是 **SAM 2.1 掩码路线**的 `runs/angle_mask/angles_<src_tag>.json`。
    本步只输出**绝对量**: φ(t)(轴线 vs 图像竖直)、90−|φ|(vs 图像上边缘)、角速率、
    相机滚转核查、参数化面外角 β。**不输出相对倾角 φ_rel** —— 它需要一个"参考姿态",
    而唯一能拿到的是"片尾落地静止即竖直", 那是针对特定视频的先验(已废弃)。
    """
    cfg = cfg or AttitudeConfig()
    meta = json.loads((DIAG / f"dets_{dets_tag}.json").read_text(
        encoding="utf-8"))
    vmeta = meta["meta"]
    W, H, fps = vmeta["W"], vmeta["H"], vmeta["fps"]
    vpath = resolve_video(vmeta)     # 支持项目外的视频(见 video_io.py)

    # --- 读第 3 步(SAM 掩码路线)的结果: 拿 phi / w_body / conf, box 另由 tracker 复现 ---
    src = ANGM / f"angles_{src_tag}.json"
    if not src.exists():
        raise FileNotFoundError(
            f"找不到掩码路线测角结果 {src}; 先跑 "
            f"rocket_mask_angle.py --tag {src_tag}")
    ang = json.loads(src.read_text(encoding="utf-8"))
    ang_fr = ang["frames"]

    # 用 tracker 重新拿每帧的 box(背景屏蔽要用)
    sys.path.insert(0, str(PROJECT / "scripts"))
    from rocket_track import TrackerConfig, track_frames
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                        fps, bound_wh=(W, H))

    res: list[FrameAtt] = []
    for a in ang_fr:
        r = FrameAtt(frame=int(a["frame"]), t=float(a["frame"]) / fps,
                     ok=bool(a["ok"]), reason=a.get("reason") or "",
                     phi_deg=a["phi_deg"] if a["phi_deg"] is not None else
                     float("nan"),
                     phi_rel=float("nan"),   # 相对倾角已废弃(见 run 的说明)
                     axis_len_px=a["axis_len_px"] if a.get("axis_len_px")
                     is not None else float("nan"),
                     w_body_px=a["w_body_px"] if a["w_body_px"] is not None
                     else float("nan"),
                     w_box_px=a["w_box_px"] if a["w_box_px"] is not None
                     else float("nan"),
                     conf=float(a["conf"] or 0.0), n_bands=int(a["n_bands"]),
                     dphi_dt=a["dphi_dt"] if a["dphi_dt"] is not None
                     else float("nan"))
        res.append(r)

    # --- 相机静止性: 逐帧背景配准(**直接**对参考帧配准, 不累加) ---
    # 为什么不能逐帧累加: 单帧误差 σ≈0.018°, 2202 帧随机游走会累出 5.9° 的假漂移。
    # 直接对固定参考帧配准, 误差不累积。
    # 参考帧不写死(原来是 2000): 取"有稳定框的区间的 cam_ref_frac 分位处"。
    # 任何背景占比足够的帧都能当配准参考, 取区间中位是为了让各帧与参考帧的
    # 外观差异不要太大(时间上离得太远会拉低 NCC)。
    valid_idx = [i for i, o in enumerate(outs) if o.has and o.box is not None]
    ref_idx = (valid_idx[int(round(cfg.cam_ref_frac * (len(valid_idx) - 1)))]
               if valid_idx else 0)
    ref_idx = max(0, min(ref_idx, len(outs) - 1))
    if verbose:
        print(f"[参考帧] 相机配准参考帧 = {ref_idx} "
              f"({ref_idx / fps:.1f}s, 有效区间 "
              f"{valid_idx[0] if valid_idx else '-'}.."
              f"{valid_idx[-1] if valid_idx else '-'})", flush=True)
    cap = open_cap(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    okf, frref = cap.read()
    ref_bg = None
    if okf:
        gref = cv2.cvtColor(frref, cv2.COLOR_BGR2GRAY)
        ref_bg = bg_mask(gref, outs[ref_idx].box if outs[ref_idx].has else None,
                         cfg)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_valid = 0
    for i in range(len(res)):
        okf, fr = cap.read()
        if not okf:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        box = outs[i].box if outs[i].has else None
        bg = bg_mask(g, box, cfg)
        if i == ref_idx:
            res[i].cam_dtheta, res[i].cam_roll = 0.0, 0.0
            res[i].cam_dx = res[i].cam_dy = 0.0
            res[i].cam_ncc = 1.0
        elif ref_bg is not None:
            th, dx, dy, ncc = estimate_frame_motion(ref_bg, bg, cfg)
            res[i].cam_dtheta, res[i].cam_roll = th, th
            res[i].cam_dx, res[i].cam_dy, res[i].cam_ncc = dx, dy, ncc
            n_valid += int(ncc >= cfg.cam_ncc_min)
        if verbose and i and i % 600 == 0:
            print(f"  相机配准 {i}/{len(res)}  NCC={res[i].cam_ncc:.4f}",
                  flush=True)

    # --- 相机估计器灵敏度校验(注入已知运动) ---
    # 用参考帧做校验(不写死帧号): 保证该帧一定有稳定框, 且与配准用的背景一致
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
    okf, fr = cap.read()
    inj_rot, inj_sh = ([], [])
    if okf:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        inj_rot, inj_sh = validate_motion_estimator(
            g, outs[ref_idx].box if outs[ref_idx].has else None, cfg)
    cap.release()

    # --- 角度换算 + 可选 β ---
    for r in res:
        r.phi_vs_top, r.phi_vs_top_signed = angle_conventions(r.phi_deg)
        if np.isfinite(r.axis_len_px) and np.isfinite(r.w_body_px) and \
                r.w_body_px > 0:
            r.aspect = r.axis_len_px / r.w_body_px
        if np.isfinite(cfg.l_over_d):
            r.beta_deg, _raw = beta_from_aspect(r.axis_len_px, r.w_body_px,
                                                cfg.l_over_d)
            # 用**绝对**倾角 φ(相对图像竖直)与面外角 β 合成总倾角;
            # 原实现用相对倾角 φ_rel, 那个参考段已废弃
            r.tilt_total = total_tilt(r.phi_deg, r.beta_deg)

    info = dict(W=W, H=H, fps=fps, l_over_d=cfg.l_over_d,
                video=vmeta["video"], n_frame=len(res),
                cam_ref_frame=ref_idx, cam_n_valid=n_valid,
                angle_src=f"angle_mask/angles_{src_tag}.json",
                src_tag=src_tag, dets_tag=dets_tag)
    return res, info, dict(inj_rot=inj_rot, inj_shift=inj_sh,
                           outs=outs, cfg=cfg, vpath=vpath)


# ==========================================================================
# 落盘
# ==========================================================================
COLS = ["frame", "t", "ok", "phi_deg", "phi_vs_top", "phi_vs_top_signed",
        "phi_rel", "beta_deg", "tilt_total", "aspect", "axis_len_px",
        "w_body_px", "w_box_px", "conf", "n_bands", "dphi_dt", "cam_dtheta",
        "cam_roll", "cam_dx", "cam_dy", "cam_ncc", "reason"]


def save(res: list[FrameAtt], info: dict, tag: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    def g(r, k):
        v = getattr(r, k)
        if isinstance(v, float):
            return round(v, 4) if np.isfinite(v) else ""
        return v

    with (OUT / f"attitude{sfx}.csv").open("w", newline="",
                                           encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in res:
            w.writerow([g(r, k) for k in COLS])
    (OUT / f"attitude{sfx}.json").write_text(
        json.dumps({"info": info,
                    "frames": [{k: (None if isinstance(v, float)
                                    and not np.isfinite(v) else v)
                                for k, v in r.__dict__.items()} for r in res]},
                   ensure_ascii=False), encoding="utf-8")
    print(f"[写入] {OUT / f'attitude{sfx}.csv'} / attitude{sfx}.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="第 4 步: 姿态反算")
    p.add_argument("--l-over-d", type=float, default=float("nan"),
                   help="火箭真实长径比; 不给就只输出角度, 不算面外角")
    p.add_argument("--tag", default="")
    p.add_argument("--dets-tag", default="iou70")
    p.add_argument("--src-tag", default="s512",
                   help="第 3 步掩码路线的标签(读 angles_<src-tag>.json), 默认 s512")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = AttitudeConfig(l_over_d=a.l_over_d)
    res, info, extra = run(cfg, dets_tag=a.dets_tag, src_tag=a.src_tag)
    save(res, info, a.tag)


if __name__ == "__main__":
    main()
