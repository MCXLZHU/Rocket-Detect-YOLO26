# -*- coding: utf-8 -*-
"""第 3 步: 在 ROI 内做边缘检测, 估计箭体倾角。

=============================== 物理先验决定算法 =============================
箭体在画面里是一根**细长直筒**。这个事实给出两条很强的约束:
  (1) 左右两条轮廓边在投影里都是直线, 且互相**平行**;
  (2) 两条边的**间距 = 筒身投影宽度**, 在帧内基本恒定(透视收缩极小)。
所以倾角不该用框的长宽比去推(轴对齐框本身不含角度), 而应:
  * 逐行找"左右成对边缘"(成对约束天然排除单条毛刺);
  * 对成对边缘的中点做**鲁棒直线拟合**, 其斜率就是筒身轴线方向;
  * 左右边的不平行度与拟合残差直接作为质量指标。

主估计器 = **成对边缘的匹配滤波(矩形脉冲/差分盒算子)**: 对每一行、每一个
候选宽度 w, 打分
        S(x; w) = mean(I[x : x+w]) − 0.5·( mean(I[x−w : x]) + mean(I[x+w : x+2w]) )
取 S 的最大值位置作为"左边缘", 右边缘即 x+w。用行前缀和实现, O(1) 每窗口。
它比"直接取梯度峰"稳健得多: 白筒 vs 亮扬尘这种低对比场景下, 梯度峰是个位数
灰阶, 而均值差在整段宽度上积分, 信噪比高一个量级。

辅助估计器 = 逐行梯度峰 + 独立左右鲁棒拟合(即最朴素的"边缘检测"做法),
仅用于交叉校验与降级。

=============================== 两个实测坑 ===============================
1) **有多条近竖直强边缘**: 落地段 ROI 里同时存在 箭体 / 展开的支腿 / 着陆坪
   边缘 / 高亮蒸汽带。宽窗口取梯度峰会**逐行锁到不同结构**, 实测左右边拟合出
   2° 的不平行度(直筒理论应为 0)。→ 用成对约束 + 宽度扫描解决。
2) **框宽会被支腿撑大**(落地段框 37.6px, 而筒身实测只有 ~23px, 比值 0.61),
   所以宽度扫描区间要覆盖 0.3~0.9 倍框宽, 不能假定框≈筒身。

=============================== 客观验证基准 =============================
本视频 40s 之后火箭已落地静止 ⇒ 真实倾角是常数。那一段的 **σ(φ)** 与线性漂移
就是测量精度的直接证据, 不需要人工标注真值。

输出: runs/angle/angles.csv / angles.json
用法:
    python scripts/rocket_angle.py                    # 全片
    python scripts/rocket_angle.py --make-video
    python scripts/rocket_angle.py --start 1400 --end 1600
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["TMPDIR"] = str(CACHE / "tmp")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE / "yolo")

import cv2  # noqa: E402

sys.path.insert(0, str(PROJECT / "scripts"))
from rocket_track import TrackerConfig, track_frames  # noqa: E402

DIAG = PROJECT / "runs" / "diag"
OUT_DIR = PROJECT / "runs" / "angle"

# 当前帧的上采样比例, 供 fit_band_axis / refine 把 band 由原始像素换算到 patch 像素
cfg_scale_holder = [1.0]


# ==========================================================================
# 配置与结果
# ==========================================================================
@dataclass
class AngleConfig:
    # --- 预处理 ---
    upscale_target_w: float = 160.0    # 上采样后筒身约 160px 宽
    upscale_max: int = 4
    roi_pad_w: float = 0.75            # 测角 ROI 在水平方向额外放宽(×框宽)
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    use_box_median: bool = True        # 先减 3x3 中值, 压制压缩伪影
    # --- 分段行平均的边缘定位(主估计器) ---
    n_bands: int = 16                  # 行范围切成几段; 每段做行平均→压噪声 √n 倍
    band_pad: float = 0.10             # 搜索窗在框外再放宽的比例
    min_band_rows: int = 12            # 每段最少行数
    min_bands: int = 4                 # 最少有效段数
    min_peak_ratio: float = 0.30       # 段峰强度门槛(相对该侧中位)
    peak_half: int = 6                 # 峰质心窗口半宽(patch 像素)
    # --- 先定条带再精修(消除"左edge取自结构A、右edge取自结构B") ---
    use_bar_prior: bool = True
    width_ratios: tuple = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
    bar_bg_ratio: float = 0.55         # 条带匹配滤波的背景窗比例
    edge_refine: float = 0.06          # 精修窄窗半宽(× 该侧搜索窗宽)
    # --- 行范围裁剪 ---
    trim_top: float = 0.03
    trim_bottom: float = 0.10
    # --- 鲁棒直线 ---
    irls_iters: int = 5
    inlier_k: float = 2.5
    # --- 锁定/降级(时序) ---
    band_px: float = 5.0               # 跟踪窄带半宽(原始像素)
    band_px_min: float = 3.0
    lock_min_bands: int = 6
    lock_min_inlier: float = 0.55
    lock_max_rms: float = 1.0          # 原始像素
    lock_max_parallel: float = 0.012   # rad ≈ 0.7°
    hold_max: int = 40                 # 连续 hold 上限后重新捕获
    # --- 交叉校验 ---
    cross_check: bool = True
    # --- 时序平滑 / 输出 ---
    smooth_median: int = 5
    smooth_alpha: float = 0.35
    min_conf: float = 0.30
    min_w_body: float = 6.0            # 拟合出的条带宽度下限(低于此不认为是筒身)
    sanity_max_deg: float = 45.0
    ref_range: tuple = (45.0, 66.0)


@dataclass
class LineFit:
    """一条直线 x = a*y + b (patch 坐标) 及质量。"""
    a: float = float("nan")
    b: float = float("nan")
    n_rows: int = 0
    inlier_ratio: float = 0.0
    rms_px: float = float("nan")      # patch 像素
    strength: float = float("nan")


@dataclass
class AngleResult:
    frame: int = 0
    ok: bool = False
    reason: str = ""
    mode: str = ""                    # acquire / track / hold
    phi_deg: float = float("nan")
    phi_l: float = float("nan")
    phi_r: float = float("nan")
    phi_grad: float = float("nan")    # 辅助估计器(梯度峰)
    phi_hough: float = float("nan")
    phi_naive: float = float("nan")   # minAreaRect 朴素对照
    n_bands: int = 0                  # 参与拟合的行段数
    inlier_ratio: float = 0.0
    rms_px: float = float("nan")
    parallel: float = float("nan")
    w_body_px: float = float("nan")
    w_box_px: float = float("nan")
    axis_len_px: float = float("nan")  # 参与拟合的段沿轴方向的跨度(第 4 步用)
    w_patch: float = float("nan")     # 内部: patch 尺度下的宽度
    polarity: int = 0
    conf: float = 0.0
    upscale: int = 1
    l_top: tuple = (0.0, 0.0)
    l_bot: tuple = (0.0, 0.0)
    r_top: tuple = (0.0, 0.0)
    r_bot: tuple = (0.0, 0.0)
    track_rel: float = 0.0
    phi_smooth: float = float("nan")
    phi_rel: float = float("nan")
    dphi_dt: float = float("nan")


# ==========================================================================
# 几何小工具
# ==========================================================================
def _robust_line(ys: np.ndarray, xs: np.ndarray, iters: int = 5,
                 inlier_k: float = 2.5,
                 min_scale: float = 0.15) -> tuple[float, float, float, np.ndarray]:
    """IRLS(Huber) 拟合 x = a*y + b, 返回 (a, b, rms, inlier_mask)。

    min_scale: MAD 尺度的下限(px)。没有它的话, 当拟合极好(rms≈0.02px)时内点
    判据会变得荒谬地紧(2.5×MAD≈0.04px), 把正常点全判成离群 —— 实测会让
    合成图在个别倾角上直接返回 None。
    """
    n = len(ys)
    if n < 4:
        return float("nan"), float("nan"), float("nan"), np.zeros(n, bool)
    a, b = np.polyfit(ys, xs, 1)
    for _ in range(iters):
        r = xs - (a * ys + b)
        s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))),
                min_scale, 1e-6)
        w = 1.0 / (1.0 + (r / (1.5 * s)) ** 2)
        a, b = np.polyfit(ys, xs, 1, w=np.sqrt(w))   # polyfit 的 w 作用在残差平方根
    r = xs - (a * ys + b)
    s = max(1.4826 * float(np.median(np.abs(r - np.median(r)))), min_scale, 1e-6)
    m = np.abs(r) <= inlier_k * s
    if m.sum() >= 4:
        a, b = np.polyfit(ys[m], xs[m], 1)
        r = xs - (a * ys + b)
        s2 = max(1.4826 * float(np.median(np.abs(r[m] - np.median(r[m])))),
                 min_scale, 1e-6)
        m = np.abs(r) <= inlier_k * s2
    rms = float(np.sqrt(np.mean(r[m] ** 2))) if m.sum() else float("nan")
    return float(a), float(b), rms, m


def _parabolic(vals: np.ndarray, i: np.ndarray,
               fill: float = -1e8) -> np.ndarray:
    """抛物线插值求亚像素峰位偏移(以样本为单位)。无效邻点视为与峰等高。"""
    n = vals.shape[1]
    a0 = vals[np.arange(len(i)), np.clip(i - 1, 0, n - 1)]
    a1 = vals[np.arange(len(i)), i]
    a2 = vals[np.arange(len(i)), np.clip(i + 1, 0, n - 1)]
    a0 = np.where(a0 < fill, a1, a0)
    a2 = np.where(a2 < fill, a1, a2)
    den = a0 - 2.0 * a1 + a2
    off = np.where(np.abs(den) > 1e-9,
                   0.5 * (a0 - a2) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    return np.clip(off, -0.5, 0.5)


def _peak_off(seg: np.ndarray, i: int) -> float:
    """单个峰的抛物线亚像素偏移。"""
    if i <= 0 or i >= len(seg) - 1:
        return 0.0
    a0, a1, a2 = float(seg[i - 1]), float(seg[i]), float(seg[i + 1])
    den = a0 - 2.0 * a1 + a2
    if abs(den) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (a0 - a2) / den, -0.5, 0.5))


def _peak_centroid(seg: np.ndarray, i: int, half: int = 6) -> float:
    """峰位 = 局部窗口内"超出半高部分"的质心。

    为什么不用 argmax + 抛物线:
      段内做了行平均, 而倾斜的边缘在段内会被"抹开": 斜率 a、段高 Δy 的边缘,
      其梯度峰是一个宽度 |a|·Δy 的**平顶**(矩形)。argmax 取的是平顶的第一个
      位置, 偏差与斜率成正比(实测使旋转增益压到 0.76, 即系统性低估 24%)。
      质心对平顶是**无偏**的: 平顶质心正好在矩形中心。
    """
    n = len(seg)
    lo = max(0, i - half)
    hi = min(n, i + half + 1)
    v = seg[lo:hi].astype(float)
    if v.size == 0:
        return float(i)
    w = np.clip(v - 0.5 * float(seg[i]), 0.0, None)
    s = float(w.sum())
    if s <= 1e-9:
        return float(i)
    return float(lo + (w * np.arange(w.size)).sum() / s)


# ==========================================================================
# 主估计器: 分段行平均的边缘定位 + 双向鲁棒直线拟合
# ==========================================================================
# 为什么不用"逐行 argmax":
#   实测逐行找梯度峰, argmax 会在若干条近竖直结构之间乱跳(落地段 ROI 里同时有
#   箭体/支腿/着陆坪/蒸汽带), 拟合残差高达 8~45px, 测出的倾角恒为 0(退化)。
# 改成把行范围切成 K 段, 每段对水平梯度做**行平均**后再找峰:
#   * 噪声按 √行数 下降 (每段 ~120 行 → 约 11 倍)
#   * 峰位由整段共同决定, 不会因个别行的干扰而跳变
#   * 最后只有 K 个点(默认 10 个)参与直线拟合, 一个坏段可以直接被鲁棒拟合剔除
# ==========================================================================
def _bar_1d(iprof: np.ndarray, lo: int, hi: int, w_cands, bg_ratio: float):
    """在 1 维灰度剖面上找"最佳条带": 返回 (score, x_left, w)。

    这是 2 维矩形脉冲滤波的 1 维版 —— 只用来**定条带**(给出左右边缘大概在哪),
    精确定位仍交给梯度峰。
    """
    n = len(iprof)
    best = None
    C = np.concatenate([[0.0], np.cumsum(iprof)])
    for w in w_cands:
        w = int(w)
        k = max(2, int(round(bg_ratio * w)))
        if w < 3 or n < w + 2 * k + 2:
            continue
        mw = (C[w:] - C[:-w]) / w
        mk = (C[k:] - C[:-k]) / k
        s0, s1 = k, n - w - k
        if s1 - s0 < 2:
            continue
        sc = mw[s0:s1 + 1] - 0.5 * (mk[s0 - k:s1 - k + 1]
                                   + mk[s0 + w:s1 + w + 1])
        a0 = max(s0, lo)
        a1 = min(s1, hi - w)
        if a1 - a0 < 1:
            continue
        seg = sc[a0 - s0:a1 - s0 + 1]
        j = int(seg.argmax())
        if best is None or seg[j] > best[0]:
            best = (float(seg[j]), a0 + j, w)
    return best


def band_edges(patch: np.ndarray, y0: int, y1: int, pol: int, cfg: AngleConfig,
               lwin: np.ndarray, rwin: np.ndarray, w_cands):
    """在每段内对 ∂I/∂x 做行平均并定位左/右边缘。

    lwin / rwin: (n_bands, 2) 数组, 每段的搜索窗 [lo, hi] (patch 坐标)。
    w_cands: 候选条带宽度(patch 像素)。

    两步走(关键):
      1) **先定条带**: 在段内灰度剖面上做 1 维矩形脉冲匹配(pol 决定明暗),
         得到一个"左边缘大致位置 x 与宽度 w";
      2) **再精修**: 只在 [x-m, x+m] 和 [x+w-m, x+w+m] 两个窄窗里找梯度峰。
    为什么不直接在宽窗里各找左右峰: 那样左边缘可能取自箭体、右边缘取自支腿,
    两者倾角不同, 测出的轴线是"混合"结果 —— 实测这正是旋转增益被压到 0.7~0.8
    (合成图本应为 1.00) 的原因。
    """
    g = pol * np.gradient(patch, axis=1)
    ed = np.linspace(y0, y1, cfg.n_bands + 1).astype(int)
    ys, xl, xr, sl, sr = [], [], [], [], []
    for k in range(cfg.n_bands):
        a, b = int(ed[k]), int(ed[k + 1])
        if b - a < cfg.min_band_rows:
            continue
        prof = g[a:b].mean(axis=0)
        iprof = pol * patch[a:b].mean(axis=0)
        lo, hi = int(lwin[k, 0]), int(lwin[k, 1])
        lo2, hi2 = int(rwin[k, 0]), int(rwin[k, 1])
        if hi - lo < 3 or hi2 - lo2 < 3:
            continue
        margin = max(4, int(round(0.70 * np.median(np.diff(w_cands))))) \
            if len(w_cands) > 1 else max(4, int(round(cfg.edge_refine
                                                      * (hi2 - lo))))
        if cfg.use_bar_prior:
            bar = _bar_1d(iprof, lo, hi2, w_cands, cfg.bar_bg_ratio)
            if bar is None:
                continue
            _sc, xb, wb = bar
            wl = (max(0, xb - margin), min(len(prof) - 1, xb + margin))
            wr = (max(0, xb + wb - margin),
                  min(len(prof) - 1, xb + wb + margin))
            if wl[1] - wl[0] < 2 or wr[1] - wr[0] < 2:
                continue
        else:
            wl, wr = (lo, hi), (lo2, hi2)
        se = prof[wl[0]:wl[1] + 1]
        i = int(se.argmax())
        v = float(se[i])
        if v <= 0:
            continue
        se2 = -prof[wr[0]:wr[1] + 1]
        j = int(se2.argmax())
        v2 = float(se2[j])
        if v2 <= 0:
            continue
        ys.append(0.5 * (a + b))
        xl.append(wl[0] + _peak_centroid(se, i, cfg.peak_half))
        xr.append(wr[0] + _peak_centroid(se2, j, cfg.peak_half))
        sl.append(v)
        sr.append(v2)
    if not ys:
        return (np.zeros(0),) * 5
    return (np.asarray(ys, float), np.asarray(xl), np.asarray(xr),
            np.asarray(sl), np.asarray(sr))


def _default_windows(patch: np.ndarray, box: np.ndarray, cfg: AngleConfig):
    H, W = patch.shape
    bx1, by1, bx2, by2 = box
    xc = 0.5 * (bx1 + bx2)
    wb = bx2 - bx1
    pad = cfg.band_pad * wb
    l = (max(1, int(round(bx1 - pad))), int(round(xc - 0.02 * wb)))
    r = (int(round(xc + 0.02 * wb)), min(W - 2, int(round(bx2 + pad))))
    return l, r


def fit_band_axis(patch: np.ndarray, box: np.ndarray, cfg: AngleConfig, pol: int,
                  lwin=None, rwin=None):
    """返回 (phi, w_body, rms, n_bands, inlier_ratio, lines, pts, snr)。

    轴线用**筒身中心线**拟合 —— 对"左右对称的直筒"而言中心线就是轴线, 而且
    左边缘与右边缘的随机误差在取中点时相互抵消, 比"左右各拟合一次再平均"更稳。
    左右边仍各自拟合一次, 只用于"不平行度"这个质量指标。
    """
    H, W = patch.shape
    bx1, by1, bx2, by2 = box
    hb = by2 - by1
    y0 = max(1, int(round(by1 + cfg.trim_top * hb)))
    y1 = min(H - 1, int(round(by2 - cfg.trim_bottom * hb)))
    if y1 - y0 < cfg.min_bands * cfg.min_band_rows:
        return None
    nb = cfg.n_bands
    ed = np.linspace(y0, y1, nb + 1).astype(int)
    if lwin is None or rwin is None:
        l, r = _default_windows(patch, box, cfg)
        lwin = np.tile(np.array(l, float), (nb, 1))
        rwin = np.tile(np.array(r, float), (nb, 1))
    lwin = np.clip(np.asarray(lwin, float), 0, W - 1)
    rwin = np.clip(np.asarray(rwin, float), 0, W - 1)
    w_cands = np.unique(np.maximum(
        3, np.round((bx2 - bx1) * np.array(cfg.width_ratios))).astype(int))
    ys, xl, xr, sl, sr = band_edges(patch, y0, y1, pol, cfg, lwin, rwin,
                                    w_cands)
    if len(ys) < cfg.min_bands:
        return None
    # 段峰强度门槛(分侧, 相对该侧中位)
    keep = ((sl >= cfg.min_peak_ratio * np.median(sl))
            & (sr >= cfg.min_peak_ratio * np.median(sr)))
    if keep.sum() < cfg.min_bands:
        return None
    ys, xl, xr = ys[keep], xl[keep], xr[keep]
    xc = 0.5 * (xl + xr)
    a, b, rms, m = _robust_line(ys, xc, cfg.irls_iters, cfg.inlier_k)
    idx = np.where(m)[0]
    if idx.size < cfg.min_bands:
        return None
    # 在第一次内点上再拟合一次(并把掩码映射回原数组下标)
    a, b, rms, m2 = _robust_line(ys[idx], xc[idx], cfg.irls_iters, cfg.inlier_k)
    idx = idx[m2]
    if idx.size < cfg.min_bands:
        return None
    phi = math.degrees(math.atan(-a))
    w = float(np.median(xr[idx] - xl[idx]))
    al, bl, _, _ = _robust_line(ys[idx], xl[idx], cfg.irls_iters, cfg.inlier_k)
    ar, br, _, _ = _robust_line(ys[idx], xr[idx], cfg.irls_iters, cfg.inlier_k)
    snr = float(np.median(np.minimum(sl, sr)[keep]))
    # 沿轴方向的跨度: 行跨度 / cos φ (φ 小时≈行跨度)
    dy_span = float(ys[idx].max() - ys[idx].min())
    axis_len = dy_span / max(math.cos(math.radians(phi)), 1e-3)
    return (phi, w, rms, int(idx.size), float(idx.size) / len(ys),
            (al, bl, ar, br), (ys[idx], xl[idx], xr[idx]), snr, axis_len)


# ==========================================================================
# 备用工具: 成对边缘匹配滤波(矩形脉冲)响应图。
# 主流程不用它 —— 实测在低对比 + 多条近竖直结构的场景下 argmax 会跳变。
# 保留是因为它在"背景干净、目标对比度高"时定位更精确, 也用于宽度先验。
# ==========================================================================
def _bar_score(patch: np.ndarray, w: int, bg_ratio: float = 0.55,
               fill: float = -1e9) -> np.ndarray:
    """成对边缘(矩形脉冲)响应图, 形状 (nrow, ncol)。

    第 j 列 = 左边缘落在 j 列时的响应:
        S[:, j] = mean(I[j : j+w]) − 0.5·( mean(I[j−k : j]) + mean(I[j+w : j+w+k]) )
    没有足够背景窗的列填 fill。
    """
    nrow, ncol = patch.shape
    k = max(2, int(round(bg_ratio * w)))
    s0, s1 = k, ncol - w - k
    out = np.full((nrow, ncol), fill, dtype=np.float32)
    if s1 - s0 < 2:
        return out
    C = np.cumsum(patch, axis=1)
    C = np.concatenate([np.zeros((nrow, 1), C.dtype), C], axis=1)

    def mlen(L):
        return (C[:, L:] - C[:, :-L]) / L

    Mw, Mk = mlen(w), mlen(k)
    out[:, s0:s1 + 1] = (Mw[:, s0:s1 + 1]
                         - 0.5 * (Mk[:, s0 - k:s1 - k + 1]
                                  + Mk[:, s0 + w:s1 + w + 1]))
    return out

# ==========================================================================
# 辅助 / 交叉校验: 逐行梯度峰(最朴素的"边缘检测"做法)
# ==========================================================================
def gradient_edges(patch: np.ndarray, box: np.ndarray, cfg: AngleConfig):
    """返回 (ys, xl, xr, n, pol) 或 None。仅用于交叉校验。"""
    H, W = patch.shape
    bx1, by1, bx2, by2 = box
    xc = 0.5 * (bx1 + bx2)
    wb, hb = bx2 - bx1, by2 - by1
    y0 = max(1, int(round(by1 + cfg.trim_top * hb)))
    y1 = min(H - 1, int(round(by2 - cfg.trim_bottom * hb)))
    if y1 - y0 < 24:
        return None
    l0, l1 = max(1, int(xc - 0.60 * wb)), int(xc - 0.08 * wb)
    r0, r1 = int(xc + 0.08 * wb), min(W - 2, int(xc + 0.60 * wb))
    if l1 - l0 < 2 or r1 - r0 < 2:
        return None
    ys = np.arange(y0, y1)
    hit = None
    for pol in (1, -1):
        g = pol * np.gradient(patch[y0:y1], axis=1)
        SL, SR = g[:, l0:l1 + 1], -g[:, r0:r1 + 1]
        il = SL.argmax(axis=1); vl = SL[np.arange(len(il)), il]
        ir = SR.argmax(axis=1); vr = SR[np.arange(len(ir)), ir]
        m = ((vl >= cfg.min_peak_ratio * max(np.median(vl), 1e-6))
             & (vr >= cfg.min_peak_ratio * max(np.median(vr), 1e-6)))
        if m.sum() < 24:
            continue
        xl = l0 + il + _parabolic(SL, il)
        xr = r0 + ir + _parabolic(SR, ir)
        if hit is None or m.sum() > hit[3]:
            hit = (ys[m].astype(float), xl[m], xr[m], int(m.sum()), pol)
    return hit


def refine(patch: np.ndarray, box: np.ndarray, cfg: AngleConfig, pol: int,
           lf: LineFit, rf: LineFit):
    """窄带跟踪: 在上一帧两条直线附近 ±band 内做同样的分段行平均定位。"""
    H, W = patch.shape
    bx1, by1, bx2, by2 = box
    hb = by2 - by1
    y0 = max(1, int(round(by1 + cfg.trim_top * hb)))
    y1 = min(H - 1, int(round(by2 - cfg.trim_bottom * hb)))
    if y1 - y0 < cfg.min_bands * cfg.min_band_rows:
        return None
    nb = cfg.n_bands
    band = max(cfg.band_px, cfg.band_px_min) * cfg_scale_holder[0]
    ed = np.linspace(y0, y1, nb + 1).astype(int)
    lw = np.zeros((nb, 2))
    rw = np.zeros((nb, 2))
    for k in range(nb):
        yc = 0.5 * (ed[k] + ed[k + 1])
        lw[k] = (lf.a * yc + lf.b - band, lf.a * yc + lf.b + band)
        rw[k] = (rf.a * yc + rf.b - band, rf.a * yc + rf.b + band)
    return fit_band_axis(patch, box, cfg, pol, lw, rw)


# ==========================================================================
# 逐帧流程
# ==========================================================================
def angle_roi(box: np.ndarray, roi: np.ndarray, cfg: AngleConfig,
              W: int, H: int) -> np.ndarray:
    """测角专用 ROI。

    跟踪器给的 ROI 只比框宽 ~16%, 而"成对边缘"打分需要左右各留背景窗
    (k = 0.55w), 所以可用宽度被卡死在 ROI/2.1 ≈ 0.38 倍框宽 —— 实测正是这个
    原因让宽度扫描永远选到最窄的档。这里在水平方向把 ROI 放宽到 ±(0.5+pad)·框宽。
    """
    xc = 0.5 * (box[0] + box[2])
    wb = max(box[2] - box[0], 4.0)
    half = (0.5 + cfg.roi_pad_w) * wb
    return np.array([max(0.0, xc - half), max(0.0, float(roi[1])),
                     min(float(W), xc + half), min(float(H), float(roi[3]))])


def crop_roi(gray: np.ndarray, roi: np.ndarray, box: np.ndarray):
    """返回 (patch0 float32, box_in_patch, (x0,y0)) 或 None。"""
    H, W = gray.shape
    x0 = max(0, int(math.floor(roi[0])))
    y0 = max(0, int(math.floor(roi[1])))
    x1 = min(W, int(math.ceil(roi[2])))
    y1 = min(H, int(math.ceil(roi[3])))
    if x1 - x0 < 8 or y1 - y0 < 16:
        return None
    patch0 = gray[y0:y1, x0:x1].astype(np.float32)
    b = np.array([max(box[0], x0) - x0, max(box[1], y0) - y0,
                  min(box[2], x1) - x0, min(box[3], y1) - y0], float)
    if b[2] - b[0] < 5 or b[3] - b[1] < 16:
        return None
    return patch0, b, (x0, y0)


class AngleEstimator:
    """因果逐帧测角。

    两条路径:
      acquire —— 用整框两侧的宽窗做分段行平均, 估计左右边缘并拟合轴线;
      track   —— 已锁定后, 只在上一帧直线 ±band 的窄带内定位, 抗干扰;
      两者都失败则 hold(沿用上一帧直线, 但**不**计入角度序列)。
    """

    def __init__(self, cfg: AngleConfig | None = None):
        self.cfg = cfg or AngleConfig()
        self.lock: tuple | None = None      # (LineFit_l, LineFit_r, pol) 全帧坐标
        self.hold = 0
        self.a_l = self.b_l = self.a_r = self.b_r = float("nan")

    # ---- 预处理 ----
    def _prepare(self, patch0, box0):
        cfg = self.cfg
        wb0 = box0[2] - box0[0]
        s = int(np.clip(round(cfg.upscale_target_w / max(wb0, 6.0)),
                        1, cfg.upscale_max))
        cfg_scale_holder[0] = float(s)
        patch = (cv2.resize(patch0, None, fx=s, fy=s,
                            interpolation=cv2.INTER_CUBIC) if s > 1
                 else patch0.copy())
        if cfg.use_box_median and s > 1:
            patch = cv2.medianBlur(np.clip(patch, 0, 255).astype(np.uint8), 3
                                   ).astype(np.float32)
        if cfg.use_clahe:
            p8 = np.clip(patch, 0, 255).astype(np.uint8)
            p8 = cv2.createCLAHE(cfg.clahe_clip,
                                 (cfg.clahe_grid, cfg.clahe_grid)).apply(p8)
            patch = p8.astype(np.float32)
        return patch, box0 * s, s

    # ---- 主入口 ----
    def step(self, patch0: np.ndarray, box0: np.ndarray, origin=(0, 0),
             frame: int = 0, track_rel: float = 0.0) -> AngleResult:
        cfg = self.cfg
        res = AngleResult(frame=frame, track_rel=track_rel)
        patch, box_s, s = self._prepare(patch0, box0)
        res.upscale = s

        # ---------- 1) 已锁定 → 先试窄带跟踪 ----------
        if self.lock is not None:
            lf, rf, pol = self.lock
            if self._try_refine(patch, box_s, origin, lf, rf, pol, s, res) \
                    and self._lockable(res):
                self.hold = 0
                res.mode = "track"
                self._finalize(res, cfg, patch, box_s, s)
                return res
            self.hold += 1
            if self.hold <= cfg.hold_max:
                res.ok, res.reason, res.mode = False, "hold", "hold"
                res.phi_deg = 0.5 * (math.degrees(math.atan(-lf.a))
                                     + math.degrees(math.atan(-rf.a)))
                res.phi_l = math.degrees(math.atan(-lf.a))
                res.phi_r = math.degrees(math.atan(-rf.a))
                res.conf = 0.0
                return res
            self.lock, self.hold = None, 0

        # ---------- 2) 全局捕获(两个极性各试一次) ----------
        best = None
        for pol in (1, -1):
            out = fit_band_axis(patch, box_s, cfg, pol)
            if out is None:
                continue
            phi, w, rms, nb, ratio, lines, _pts, snr, alen = out
            if not np.isfinite(phi):
                continue
            # 打分: 内点率 + 峰强度(信噪比) − 残差惩罚。
            # 只用内点率会让"极性选错但恰好拟合出一条斜结构"的解胜出, 加信噪比
            # 项后实测合成图 9/9 全部选对极性。
            sc = (ratio + 0.10 * min(1.0, snr / 8.0)
                  - 0.05 * max(0.0, rms - 0.3))
            if best is None or sc > best[0]:
                best = (sc, pol, phi, w, rms, nb, ratio, lines, alen)
        if best is None:
            res.reason = "acquire_failed"
            return res
        (_, pol, phi, w, rms, nb, ratio, lines, alen) = best
        a_l, b_l, a_r, b_r = lines
        res.polarity, res.mode, res.phi_deg = pol, "acquire", phi
        res.phi_l = math.degrees(math.atan(-a_l))
        res.phi_r = math.degrees(math.atan(-a_r))
        res.parallel = abs(a_l - a_r)
        res.n_bands, res.inlier_ratio = nb, ratio
        res.rms_px, res.w_body_px = rms / s, w / s
        res.axis_len_px = alen / s
        self.a_l, self.b_l, self.a_r, self.b_r = a_l, b_l, a_r, b_r
        self._finalize(res, cfg, patch, box_s, s)
        if self._lockable(res):
            self.lock = (LineFit(a_l, (b_l / s + origin[0]) - a_l * origin[1]),
                         LineFit(a_r, (b_r / s + origin[0]) - a_r * origin[1]),
                         pol)
        else:
            self.lock = None
        return res

    # ---- 窄带跟踪 ----
    def _try_refine(self, patch, box_s, origin, lf, rf, pol, s, res) -> bool:
        x0, y0 = origin
        pl = LineFit(lf.a, (lf.b - x0) * s + lf.a * y0 * s)
        pr = LineFit(rf.a, (rf.b - x0) * s + rf.a * y0 * s)
        out = refine(patch, box_s, self.cfg, pol, pl, pr)
        if out is None:
            return False
        phi, w, rms, nb, ratio, lines, _pts, _snr, alen = out
        a_l, b_l, a_r, b_r = lines
        res.phi_deg, res.n_bands = phi, nb
        res.inlier_ratio, res.rms_px, res.w_body_px = ratio, rms / s, w / s
        res.axis_len_px = alen / s
        res.polarity = pol
        res.phi_l = math.degrees(math.atan(-a_l))
        res.phi_r = math.degrees(math.atan(-a_r))
        res.parallel = abs(a_l - a_r)
        self.a_l, self.b_l, self.a_r, self.b_r = a_l, b_l, a_r, b_r
        self.lock = (LineFit(a_l, (b_l / s + x0) - a_l * y0),
                     LineFit(a_r, (b_r / s + x0) - a_r * y0), pol)
        return True

    # ---- 质量判定与收尾 ----
    def _lockable(self, res: AngleResult) -> bool:
        c = self.cfg
        if not np.isfinite(res.phi_deg):
            return False
        if res.n_bands < c.lock_min_bands or res.inlier_ratio < c.lock_min_inlier:
            return False
        if not np.isfinite(res.rms_px) or res.rms_px > c.lock_max_rms:
            return False
        if np.isfinite(res.parallel) and res.parallel > c.lock_max_parallel:
            return False
        return True

    def _finalize(self, res: AngleResult, cfg: AngleConfig, patch, box_s, s):
        a_l, b_l, a_r, b_r = self.a_l, self.b_l, self.a_r, self.b_r
        # 线条端点(全帧坐标由调用方补 origin; 这里用 patch 坐标即可用于叠加)
        if np.isfinite(a_l):
            y_top = box_s[1]
            y_bot = box_s[3] - cfg.trim_bottom * (box_s[3] - box_s[1])
            for a, b, d1, d2 in ((a_l, b_l, "l_top", "l_bot"),
                                 (a_r, b_r, "r_top", "r_bot")):
                res.__dict__[d1] = (float((a * y_top + b) / s), float(y_top / s))
                res.__dict__[d2] = (float((a * y_bot + b) / s), float(y_bot / s))
        res.w_box_px = float(box_s[2] - box_s[0]) / s
        if cfg.cross_check:
            ge = gradient_edges(patch, box_s, cfg)
            if ge is not None:
                ys_g, xl_g, xr_g, _, _ = ge
                al, _, _, _ = _robust_line(ys_g, xl_g, cfg.irls_iters, cfg.inlier_k)
                ar, _, _, _ = _robust_line(ys_g, xr_g, cfg.irls_iters, cfg.inlier_k)
                if np.isfinite(al) and np.isfinite(ar):
                    res.phi_grad = 0.5 * (math.degrees(math.atan(-al))
                                          + math.degrees(math.atan(-ar)))
        res.phi_naive = _naive_minarearect(
            np.clip(patch, 0, 255).astype(np.uint8), box_s)
        c_row = min(1.0, res.n_bands / max(cfg.lock_min_bands * 1.6, 1))
        c_in = float(np.clip((res.inlier_ratio - cfg.lock_min_inlier)
                             / (1.0 - cfg.lock_min_inlier), 0, 1))
        c_rms = float(np.clip(1.0 - res.rms_px / 1.5, 0, 1))
        c_par = 1.0 if not np.isfinite(res.parallel) else float(np.clip(
            1.0 - res.parallel / (cfg.lock_max_parallel * 2), 0, 1))
        c_ge = 1.0
        if np.isfinite(res.phi_grad):
            c_ge = float(np.clip(1.0 - abs(res.phi_deg - res.phi_grad) / 3.0,
                                 0, 1))
        res.conf = float(np.clip(0.15 * c_row + 0.25 * c_in + 0.25 * c_rms
                                 + 0.15 * c_par + 0.20 * c_ge, 0, 1))
        if abs(res.phi_deg) > cfg.sanity_max_deg:
            res.ok, res.reason = False, "angle_out_of_range"
        elif not np.isfinite(res.w_body_px) or res.w_body_px < cfg.min_w_body:
            # 拟合出的"条带"太窄 ⇒ 多半抓到的是涂装条纹而不是筒身, 不可信
            res.ok, res.reason = False, "body_too_narrow"
        elif res.conf < cfg.min_conf:
            res.ok, res.reason = False, "low_conf"
        else:
            res.ok, res.reason = True, ""


# ==========================================================================
# 朴素对照: ROI 内 Otsu + 最大连通域 + minAreaRect
# ==========================================================================
def _naive_minarearect(patch8u: np.ndarray, box: np.ndarray) -> float:
    bx1, by1, bx2, by2 = [int(round(v)) for v in box]
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2 = min(patch8u.shape[1], bx2)
    by2 = min(patch8u.shape[0], by2)
    if bx2 - bx1 < 6 or by2 - by1 < 20:
        return float("nan")
    sub = patch8u[by1:by2, bx1:bx2]
    _, m = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return float("nan")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_HEIGHT] < 0.4 * sub.shape[0]:
        return float("nan")
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float("nan")
    (_, _), (w, h), ang = cv2.minAreaRect(cnts[0])
    th = math.radians(ang)
    if w >= h:                       # 长边方向
        d = (math.cos(th), math.sin(th))
    else:
        d = (-math.sin(th), math.cos(th))
    if d[1] > 0:
        d = (-d[0], -d[1])           # 统一指向画面"上"
    return math.degrees(math.atan2(d[0], -d[1]))


# ==========================================================================
# 时序后处理
# ==========================================================================
def smooth_and_reference(results: list[AngleResult], fps: float,
                         cfg: AngleConfig | None = None):
    cfg = cfg or AngleConfig()
    prev = None
    for i, r in enumerate(results):
        if not r.ok:
            continue
        win = [results[j].phi_deg for j in range(max(0, i - cfg.smooth_median + 1),
                                                i + 1) if results[j].ok]
        v = float(np.median(win))
        prev = v if prev is None else cfg.smooth_alpha * v + (1 - cfg.smooth_alpha) * prev
        r.phi_smooth = prev
        if i > 0 and np.isfinite(results[i - 1].phi_smooth):
            r.dphi_dt = (r.phi_smooth - results[i - 1].phi_smooth) * fps
    rr = cfg.ref_range
    ref = [r.phi_smooth for r in results
           if r.ok and rr[0] <= r.frame / fps < rr[1]
           and np.isfinite(r.phi_smooth)]
    ref_val = float(np.median(ref)) if ref else float("nan")
    for r in results:
        if np.isfinite(r.phi_smooth) and np.isfinite(ref_val):
            r.phi_rel = r.phi_smooth - ref_val
    return ref_val


# ==========================================================================
# 跑视频
# ==========================================================================
def open_cap(path: Path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "src_angle.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def run(cfg: AngleConfig | None = None, tcfg: TrackerConfig | None = None,
        dets_tag: str = "iou70", start: int = 0, end: int | None = None,
        make_video: bool = False, verbose: bool = True, est=None):
    cfg = cfg or AngleConfig()
    meta = json.loads((DIAG / f"dets_{dets_tag}.json").read_text(encoding="utf-8"))
    vmeta = meta["meta"]
    W, H, fps = vmeta["W"], vmeta["H"], vmeta["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], tcfg or TrackerConfig(),
                        fps, bound_wh=(W, H))
    n_all = len(outs)
    end = n_all if end is None else min(end, n_all)
    results = [AngleResult(frame=i) for i in range(n_all)]
    est = est or AngleEstimator(cfg)

    cap = open_cap(PROJECT / vmeta["video"])
    writer = None
    if make_video:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(OUT_DIR / "angle_roi.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t0 = time.time()
    n_ok = 0
    i = 0
    while i < n_all:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= end:
            break
        o = outs[i]
        if i >= start and o.has and o.angle_ok and o.roi is not None \
                and o.box is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cr = crop_roi(gray, angle_roi(o.box, o.roi, cfg, W, H), o.box)
            if cr is None:
                results[i].reason = "roi_too_small"
            else:
                p0, b0, org = cr
                results[i] = est.step(p0, b0, org, i, o.reliability)
                n_ok += int(results[i].ok)
        else:
            results[i].reason = "tracker_gate"
        if writer is not None:
            writer.write(_draw(frame, outs[i], results[i], fps))
        if verbose and i and i % 500 == 0:
            el = time.time() - t0
            print(f"  {i} 帧  {el:.1f}s ({i / max(el, 1e-6):.1f} fps)  "
                  f"成功 {n_ok}", flush=True)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    el = time.time() - t0
    ref_val = smooth_and_reference(results, fps, cfg)
    if verbose:
        print(f"测角完成: {i} 帧, {el:.1f}s ({i / max(el, 1e-6):.1f} fps), "
              f"成功 {n_ok} 帧")
        print(f"静止参考角 phi_ref(45-66s) = {ref_val:.3f} deg")
    return results, outs, dict(W=W, H=H, fps=fps, ref=ref_val,
                               video=vmeta["video"], n_frame=n_all)


def _draw(frame, out, res: AngleResult, fps: float) -> np.ndarray:
    img = frame.copy()
    if out.roi is not None:
        a = [int(v) for v in out.roi]
        cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (0, 200, 255), 1)
    if out.box is not None:
        b = [int(v) for v in out.box]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 1)
    if np.isfinite(res.phi_deg) and res.l_top[0] > 0:
        for p, q in ((res.l_top, res.l_bot), (res.r_top, res.r_bot)):
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     (0, 0, 255), 1, cv2.LINE_AA)
    txt = (f"{res.mode} phi={res.phi_deg:+.2f} rel={res.phi_rel:+.2f} "
           f"conf={res.conf:.2f} w={res.w_body_px:.1f} n={res.n_bands}"
           if np.isfinite(res.phi_deg) else
           f"[{res.reason or 'no-angle'}]")
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),
                4, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"f{res.frame} t={res.frame / fps:.1f}s", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ==========================================================================
# 落盘
# ==========================================================================
COLS = ["frame", "t", "ok", "mode", "reason", "phi_deg", "phi_smooth",
        "phi_rel", "dphi_dt", "phi_l", "phi_r", "phi_grad", "phi_naive",
        "n_bands", "inlier_ratio", "rms_px", "parallel", "w_body_px",
        "axis_len_px", "w_box_px", "polarity", "conf", "upscale", "track_rel"]


def save(results: list[AngleResult], info: dict, tag: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    def g(r, k):
        v = getattr(r, k)
        if isinstance(v, float):
            return round(v, 4) if np.isfinite(v) else ""
        return v

    with (OUT_DIR / f"angles{sfx}.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in results:
            w.writerow([r.frame, round(r.frame / info["fps"], 4),
                        int(r.ok), r.mode] + [g(r, k) for k in COLS[4:]])
    (OUT_DIR / f"angles{sfx}.json").write_text(
        json.dumps({"info": info,
                    "frames": [{k: (None if isinstance(v, float)
                                    and not np.isfinite(v) else v)
                                for k, v in r.__dict__.items()}
                               for r in results]},
                   ensure_ascii=False), encoding="utf-8")
    print(f"[写入] {OUT_DIR / f'angles{sfx}.csv'} / angles{sfx}.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROI 内边缘检测 + 箭体倾角")
    p.add_argument("--dets-tag", default="iou70")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=0)
    p.add_argument("--make-video", action="store_true")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    res, outs, info = run(start=a.start, end=(a.end or None),
                          make_video=a.make_video)
    save(res, info, a.tag)


if __name__ == "__main__":
    main()
