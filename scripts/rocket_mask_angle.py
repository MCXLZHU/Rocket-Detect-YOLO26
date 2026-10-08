# -*- coding: utf-8 -*-
"""第 3 步(SAM 路线): 从 SAM 2.1 掩码轮廓反算箭体倾角。

=============================== 为什么"掩码的中心线就是轴线" ==============================
箭体是**回旋体**。回旋体的剪影(silhouette)在任意视角下, 左右两条边界的中点连线
就是**三维轴线在像面上的投影**。所以不需要知道箭体直径、不需要相机内参, 只要掩码
是"完整的剪影", 逐行取中点再做鲁棒直线拟合就得到倾角。

这是 SAM 路线相对原方案(梯度边缘)的本质优势: 原方案必须靠"左右边互相平行 +
间距恒定"两条先验去**猜**哪两条边属于同一根筒子; SAM 直接给出剪影, 不需要猜。

=============================== 但不能直接用整张掩码 ==============================
掩码两端会包含:
  * 顶部整流罩是锥面 —— 锥的母线是直线但**不平行**, 会污染"不平行度"指标;
  * 底部发动机段收窄, 偶尔把支腿一起圈进来 —— 会造成剪影不对称。
所以先做"**筒身段定位**": 用行宽 w(y) 的鲁棒中位剔除明显发散/收缩的行
(支腿让 w 突增, 锥面让 w 单调变化), 只在剩余行上拟合中心线。

=============================== 与原方案的关系 ==============================
原方案输出 `runs/angle/angles.csv`; 本脚本输出 `runs/angle_mask/angles_<tag>.csv`,
**列名与语义完全对齐**, 便于逐帧对拍(`--compare`)。

用法:
    python scripts/rocket_mask_angle.py --tag iou70 --compare
    python scripts/rocket_mask_angle.py --tag s1024 --compare --vs s512   # 与另一套 SAM 配置对比
    python scripts/rocket_mask_angle.py --tag smoke --compare
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
sys.path.insert(0, str(PROJECT / "scripts"))

# 共享内核(结果容器 / IRLS 拟合 / 时序平滑 / 落盘)。原先从 rocket_angle(原方案的
# 梯度边缘模块)里借, 现在原方案下线, 这条依赖已迁到中立的 angle_core。
from angle_core import (  # noqa: E402
    AngleResult, SmoothConfig, robust_line, save as save_angles,
    smooth_and_reference,
)

SEG = PROJECT / "runs" / "seg"
DETS = PROJECT / "runs" / "diag"
OUT_DIR = PROJECT / "runs" / "angle_mask"


@dataclass
class MaskAngleConfig:
    """门限分两类, 换视频时**只有第一类需要留意**:

    * 比例/角度量(trim_*/w_tol/max_parallel/sanity_max_deg) —— 与分辨率无关, 直接通用;
    * 像素绝对量(min_w/min_rows/max_rms) —— 原值是按 852x480、箭体几十像素宽调的。
      换视频后由 `_eff_limits()` 按**当前这帧剪影的尺寸**自动放宽(**只放宽, 不收紧**):
      目标比"基准"小就按比例松, 大就保持原值。所以 480p 上的行为与改动前完全一致,
      而小目标/低分辨率视频不会再被门限全拒。
    """
    trim_top: float = 0.02            # 剪影上下各裁掉的比例(边界行只有几像素)
    trim_bottom: float = 0.06
    w_tol: float = 0.18               # 行宽偏离中位多少比例就剔除(支腿/锥面)
    w_tol_passes: int = 2             # 按收紧后的中位再剔一轮
    min_w: float = 6.0                # 筒身行宽下限(px, 480p 基准)
    min_rows: int = 24                # 参与拟合的最少行数(480p 基准)
    max_rms: float = 1.20             # px(480p 基准)
    max_parallel: float = 0.012       # rad ≈ 0.7°
    smooth_median: int = 5
    smooth_alpha: float = 0.35
    sanity_max_deg: float = 45.0


def _eff_limits(cfg: MaskAngleConfig, span: float, w_max: float,
                w_body: float = float("nan")) -> tuple[int, float, float]:
    """按剪影尺寸给出这一帧的有效门限 (min_rows, min_w, max_rms)。

    规则: 以 480p 上的典型尺寸(剪影高 ~300px / 宽 ~40px / 筒身宽 ~25px)为基准,
    目标更小就**按比例放宽**, 更大就保持配置值 —— 即"只放宽、不收紧",
    保证同一段视频里大目标帧的结果与调参时一致。
    """
    min_rows = min(cfg.min_rows, max(6, int(round(0.15 * max(span, 1.0)))))
    min_w = min(cfg.min_w, max(2.0, 0.5 * max(w_max, 1.0)))
    if not np.isfinite(w_body):
        max_rms = cfg.max_rms
    else:
        max_rms = cfg.max_rms * max(1.0, 12.0 / max(w_body, 1e-6))
    return int(min_rows), float(min_w), float(max_rms)


def body_rows(xl: np.ndarray, xr: np.ndarray, cfg: MaskAngleConfig):
    """筒身段定位: 返回 (ys, xl, xr, w), 已剔除发散/收缩的行。"""
    k = np.isfinite(xl) & np.isfinite(xr)
    ys = np.where(k)[0]
    if ys.size < 6:                    # 绝对下限: 少于 6 行不可能拟合出直线
        return None
    span = float(ys[-1] - ys[0])
    w_max = float(np.max(xr[ys] - xl[ys]))
    min_rows, min_w, _ = _eff_limits(cfg, span, w_max)   # 随目标尺寸自适应
    if ys.size < min_rows:
        return None
    y0 = ys[0] + int(round(cfg.trim_top * span))
    y1 = ys[-1] - int(round(cfg.trim_bottom * span))
    ys = ys[(ys >= y0) & (ys <= y1)]
    if ys.size < min_rows:
        return None
    xl, xr = xl[ys], xr[ys]
    w = xr - xl
    for _ in range(max(1, cfg.w_tol_passes)):
        wm = float(np.median(w))
        keep = (w >= min_w) & (np.abs(w - wm) <= cfg.w_tol * wm)
        if keep.sum() < min_rows:
            return None
        if keep.all():
            break
        ys, xl, xr, w = ys[keep], xl[keep], xr[keep], w[keep]
    return ys, xl, xr, w


def estimate(xl: np.ndarray, xr: np.ndarray, cfg: MaskAngleConfig):
    """逐行左右边界 -> (phi, w_body, rms, 行数, 内点率, 不平行度, 跨度, 端点)。"""
    br = body_rows(xl, xr, cfg)
    if br is None:
        return None
    ys, xl2, xr2, w = br
    ysf = ys.astype(float)
    xc = 0.5 * (xl2 + xr2)
    min_rows, _, _ = _eff_limits(cfg, float(ysf[-1] - ysf[0]),
                                 float(np.max(w)), float(np.median(w)))
    a, b, rms, m = robust_line(ysf, xc, 5, 2.5)
    if not np.isfinite(a) or m.sum() < min_rows:
        return None
    a, b, rms, m2 = robust_line(ysf[m], xc[m], 5, 2.5)
    idx = np.where(m)[0][m2]
    if idx.size < min_rows:
        return None
    al, bl, _, _ = robust_line(ysf[idx], xl2[idx], 5, 2.5)
    ar, br_, _, _ = robust_line(ysf[idx], xr2[idx], 5, 2.5)
    par = abs(al - ar) if (np.isfinite(al) and np.isfinite(ar)) else float("nan")
    yt, yb = float(ysf[idx].min()), float(ysf[idx].max())
    return dict(phi=float(np.degrees(np.arctan(-a))), w_body=float(np.median(w[idx])),
                rms=float(rms), n=int(idx.size),
                inlier=float(idx.size) / max(len(ysf), 1),
                parallel=float(par), span=float(yb - yt),
                w_sigma=float(np.std(w[idx])), y_top=yt, y_bot=yb,
                l_top=((al * yt + bl), yt) if np.isfinite(al) else (np.nan, np.nan),
                l_bot=((al * yb + bl), yb) if np.isfinite(al) else (np.nan, np.nan),
                r_top=((ar * yt + br_), yt) if np.isfinite(ar) else (np.nan, np.nan),
                r_bot=((ar * yb + br_), yb) if np.isfinite(ar) else (np.nan, np.nan))


def run(tag: str, dets_tag: str = "iou70",
        cfg: MaskAngleConfig | None = None) -> tuple:
    cfg = cfg or MaskAngleConfig()
    meta = json.loads((DETS / f"dets_{dets_tag}.json").read_text(
        encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]
    n = len(meta["frames"])
    d = np.load(SEG / f"bounds_{tag}.npz")
    XL, XR = d["XL"], d["XR"]
    res = [AngleResult(frame=i) for i in range(n)]
    n_no = n_fit = 0
    for i in range(n):
        r = res[i]
        xl = XL[i].astype(float)
        xr = XR[i].astype(float)
        xl[xl < 0] = np.nan
        xr[xr < 0] = np.nan
        if not np.isfinite(xl).any():
            r.reason = "no_mask"
            n_no += 1
            continue
        e = estimate(xl, xr, cfg)
        if e is None:
            r.reason = "fit_failed"
            n_fit += 1
            continue
        r.ok, r.mode = True, "mask"
        r.phi_deg, r.phi_l, r.phi_r = e["phi"], float("nan"), float("nan")
        r.w_body_px, r.rms_px, r.n_bands = e["w_body"], e["rms"], e["n"]
        r.inlier_ratio, r.parallel = e["inlier"], e["parallel"]
        r.axis_len_px = e["span"]
        r.l_top, r.l_bot, r.r_top, r.r_bot = (e["l_top"], e["l_bot"],
                                              e["r_top"], e["r_bot"])
        rb = [q for q in meta["frames"][i]["b"] if q["c"] == 1]
        if rb:
            x = np.asarray(rb[0]["x"], float)
            r.w_box_px = float(x[2] - x[0])
        # 残差门限随筒身宽自适应(小目标放宽), 见 _eff_limits
        _, _, max_rms_eff = _eff_limits(cfg, float(r.axis_len_px),
                                        float(r.w_body_px), float(r.w_body_px))
        if float(r.rms_px) > max_rms_eff:
            r.ok, r.reason = False, "rms_too_big"
        if abs(float(r.phi_deg)) > cfg.sanity_max_deg:
            r.ok, r.reason = False, "angle_out_of_range"
        r.conf = float(np.clip(0.4 * min(1.0, r.n_bands / 60.0)
                               + 0.3 * r.inlier_ratio
                               + 0.3 * (1.0 - min(1.0, r.rms_px / 1.5)), 0, 1))
    # 只做时序平滑(phi_smooth / dphi_dt); 不做"相对倾角", 见 angle_core 的说明
    scfg = SmoothConfig(smooth_median=cfg.smooth_median,
                        smooth_alpha=cfg.smooth_alpha)
    smooth_and_reference(res, fps, scfg)
    info = dict(W=W, H=H, fps=fps, video=vm["video"], n_frame=n,
                source=f"sam2_mask:{tag}", n_no_mask=n_no, n_fit_failed=n_fit,
                angle_kind="absolute_vs_image_vertical")
    # 带上 seg 阶段的运行配置(模型规格/输入边长/分块), 方便 summary 里一眼看清用的是哪套
    sm = SEG / f"seg_meta_{tag}.json"
    if sm.exists():
        try:
            s = json.loads(sm.read_text(encoding="utf-8"))
            for k in ("model", "image_size", "half", "chunk", "n_reanchor"):
                if k in s:
                    info[k] = s[k]
        except Exception:
            pass
    return res, info


def compare(res: list[AngleResult], dets_tag: str = "iou70",
            vs: str = "s512") -> str:
    """与**另一套 SAM 配置**逐帧对拍(例如 s1024 vs s512, 量化分辨率带来的差异)。

    注: 原方案(ROI 梯度边缘)已下线, 原来的 vs="original" 分支随之删除 ——
    对拍对象只剩下"另一套分割配置"。

    这里只做**绝对角 φ 的差值**统计。相对倾角 φ_rel 已废弃(它依赖"片尾竖直"
    这个只对某些视频成立的先验), 因此不再有"落地段 σ"这类按固定时间窗切出来的指标;
    对拍只看全部同时有效的帧, 并按 10s 一档自适应地分时段观察。
    """
    p = OUT_DIR / f"angles_{vs}.json"
    label = f"另一套 SAM 配置 tag={vs}"
    if not p.exists():
        return f"(未找到 {p}, 跳过对拍)"
    a = json.loads(p.read_text(encoding="utf-8"))["frames"]
    fps = json.loads((DETS / f"dets_{dets_tag}.json").read_text(
        encoding="utf-8"))["meta"]["fps"]
    n = min(len(a), len(res))
    dp, dw, t_list = [], [], []
    for i in range(n):
        ra, rb = a[i], res[i]
        pa, pb = ra.get("phi_deg"), rb.phi_deg
        if ra.get("ok") and rb.ok and np.isfinite(pa) and np.isfinite(pb):
            dp.append(pb - pa)
            t_list.append(i / fps)
            aa, bb = ra.get("w_body_px"), rb.w_body_px
            if aa and np.isfinite(bb):
                dw.append(bb / float(aa))
    L = ["", f"=== 与 {label} 对拍(只比绝对角 φ) ===",
         f"  两套配置同时有效的帧数: {len(dp)} / {n}"]
    if dp:
        d = np.array(dp)
        L.append(f"  Δφ = 本配置 − 对照:  中位 {np.median(d):+.3f}°  "
                 f"均值 {d.mean():+.3f}°  σ {d.std():.3f}°  "
                 f"|Δφ|95% {np.percentile(np.abs(d), 95):.3f}°  "
                 f"极差 {d.max() - d.min():.3f}°")
        # 分时段(每 10s 一档): 档数由视频实际长度决定, 不写死
        t_arr = np.array(t_list)
        d_arr = np.array(d)
        L.append("  分时段 Δφ (中位/σ):")
        t_max = float(t_arr.max())
        for s in range(0, int(t_max) + 1, 10):
            m = (t_arr >= s) & (t_arr < s + 10)
            if m.sum() >= 5:
                L.append(f"    {s:2d}-{s + 10:2d}s  n={int(m.sum()):4d}  "
                         f"{np.median(d_arr[m]):+.3f}° / {d_arr[m].std():.3f}°")
    if dw:
        r = np.array(dw)
        L.append(f"  w_body(本配置 / 对照):  中位 {np.median(r):.2f}  σ {r.std():.2f}  "
                 f"范围 {r.min():.2f}~{r.max():.2f}")
        L.append("    ⇒ 两套配置量的是同一套剪影口径, 比值应≈1;")
        L.append("      明显偏离说明分辨率改变了掩码边界的落点(细目标尤甚)。")
    return "\n".join(L)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAM 掩码轮廓 -> 箭体倾角")
    p.add_argument("--tag", default="iou70")
    p.add_argument("--dets-tag", default="iou70")
    p.add_argument("--compare", action="store_true",
                   help="与另一套 SAM 配置对拍(需先有那套配置的 angles_<vs>.json)")
    p.add_argument("--vs", default="s512",
                   help="对拍对象: 另一个 seg tag(如 s1024 / s512)")
    p.add_argument("--out-tag", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    res, info = run(a.tag, a.dets_tag)
    print(f"[完成] 有效 {info['n_frame'] - info['n_no_mask'] - info['n_fit_failed']}"
          f"/{info['n_frame']} 帧  (无掩码 {info['n_no_mask']}, "
          f"拟合失败 {info['n_fit_failed']})", flush=True)
    print("       输出: 箭体轴线相对**图像竖直**的绝对倾角 φ(t)"
          "(不做相对姿态, 无参考段先验)", flush=True)
    save_angles(res, info, a.out_tag or a.tag, out_dir=OUT_DIR)
    if a.compare:
        txt = compare(res, a.dets_tag, a.vs)
        print(txt, flush=True)
        (OUT_DIR / f"compare_{a.tag}_vs_{a.vs}.txt").write_text(
            txt, encoding="utf-8")


if __name__ == "__main__":
    main()
