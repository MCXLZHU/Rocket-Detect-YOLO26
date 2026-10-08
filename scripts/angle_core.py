# -*- coding: utf-8 -*-
"""测角链路的**共享内核**: 结果容器 + 鲁棒拟合 + 时序后处理 + 落盘。

为什么单独拆出这一层:
  这套东西原本长在 `rocket_angle.py`(原方案: ROI 梯度边缘)里面, 但**掩码路线同样要用** ——
  `rocket_mask_angle.py` 需要 IRLS 直线拟合、结果容器、时序平滑(φ_smooth/φ_rel/dφ/dt)
  与 CSV/JSON 落盘。原方案下线后, 若直接删 `rocket_angle.py` 会连带把掩码路线打断,
  所以先把这条共享内核独立出来, 与"用什么手段拿到左右边界"彻底解耦。

  一句话: **本模块不关心边界是"梯度边缘检测"得来的还是"SAM 掩码剪影"得来的**,
  它只负责"拿到逐行左右边界之后"的公共数学与数据契约。

数据契约(改这里等于改 CSV 表头, 下游 `sam_angle_viz.py` / `_*_check.py` 都会受影响):
  `COLS` + `AngleResult` 的字段集合 = `runs/angle_mask/angles_<tag>.csv` 的列定义。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
# 掩码路线的产物目录(原方案下线后, 这条链路只剩掩码路线一个出口)
OUT_DIR = PROJECT / "runs" / "angle_mask"


# ==========================================================================
# 结果容器
# ==========================================================================
@dataclass
class AngleResult:
    """逐帧测角结果。

    字段里有几个是**给原方案留的、掩码路线恒为空**: `phi_grad`(梯度峰辅助估计器)、
    `phi_naive`(minAreaRect 朴素对照)、`phi_hough`、`w_patch`。之所以不删:
    它们同时是 CSV/JSON 的**数据契约**, 删掉会让已入库的
    `runs/angle_mask/angles_*.csv` 与新代码列数不一致。
    """

    frame: int = 0
    ok: bool = False
    reason: str = ""
    mode: str = ""                    # acquire / track / hold
    phi_deg: float = float("nan")
    phi_l: float = float("nan")
    phi_r: float = float("nan")
    phi_grad: float = float("nan")    # (原方案用)辅助估计器: 梯度峰
    phi_hough: float = float("nan")
    phi_naive: float = float("nan")   # (原方案用)minAreaRect 朴素对照
    n_bands: int = 0                  # 参与拟合的行段数
    inlier_ratio: float = 0.0
    rms_px: float = float("nan")
    parallel: float = float("nan")
    w_body_px: float = float("nan")
    w_box_px: float = float("nan")
    axis_len_px: float = float("nan")  # 参与拟合的段沿轴方向的跨度(姿态换算用)
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
# 鲁棒直线拟合
# ==========================================================================
def robust_line(ys: np.ndarray, xs: np.ndarray, iters: int = 5,
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


# ==========================================================================
# 时序后处理
# ==========================================================================
@dataclass
class SmoothConfig:
    """只保留时序平滑/参考段需要的三个参数(原 AngleConfig 有 30 多个字段,
    其余都是梯度边缘估计器专用的, 掩码路线用不到)。"""
    smooth_median: int = 5
    smooth_alpha: float = 0.35
    ref_range: tuple = (45.0, 66.0)   # 落地静止段, 作为相对倾角的零点


def smooth_and_reference(results: list[AngleResult], fps: float,
                         cfg: SmoothConfig | None = None) -> float:
    """中值窗 + 一阶低通 → `phi_smooth`, 逐帧差分 → `dphi_dt`,
    再以落地段中位为基准 → `phi_rel`。返回基准值 φ_ref。

    注意 `dphi_dt` 是**相邻帧**差分×fps: 逐帧噪声会被放大 30 倍, 画图/报数时
    应改用窗内中心差分(见 `sam_angle_viz.windowed_rate`)。
    """
    cfg = cfg or SmoothConfig()
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
# 落盘
# ==========================================================================
COLS = ["frame", "t", "ok", "mode", "reason", "phi_deg", "phi_smooth",
        "phi_rel", "dphi_dt", "phi_l", "phi_r", "phi_grad", "phi_naive",
        "n_bands", "inlier_ratio", "rms_px", "parallel", "w_body_px",
        "axis_len_px", "w_box_px", "polarity", "conf", "upscale", "track_rel"]


def save(results: list[AngleResult], info: dict, tag: str = "",
         out_dir: Path | None = None) -> None:
    """写 `angles[_<tag>].csv` + `.json`。

    `out_dir` 显式传参, 不再靠"临时改写模块级 OUT_DIR"来换目录
    (原实现是 `RA.OUT_DIR = ...; RA.save(...); RA.OUT_DIR = old` 的猴补丁)。
    """
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""

    def g(r, k):
        v = getattr(r, k)
        if isinstance(v, float):
            return round(v, 4) if np.isfinite(v) else ""
        return v

    with (out_dir / f"angles{sfx}.csv").open("w", newline="",
                                             encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in results:
            w.writerow([r.frame, round(r.frame / info["fps"], 4),
                        int(r.ok), r.mode] + [g(r, k) for k in COLS[4:]])
    (out_dir / f"angles{sfx}.json").write_text(
        json.dumps({"info": info,
                    "frames": [{k: (None if isinstance(v, float)
                                    and not np.isfinite(v) else v)
                                for k, v in r.__dict__.items()}
                               for r in results]},
                   ensure_ascii=False), encoding="utf-8")
    print(f"[写入] {out_dir / f'angles{sfx}.csv'} / angles{sfx}.json")
