# -*- coding: utf-8 -*-
"""SAM 2.1 掩码测角路线 —— **独立成果可视化(不含与原方案的对拍)**。

和 `compare_video.py` 的区别: 那个是"两条路线对拍"的调试材料(左边绿线是原方案的
梯度边线, 下段曲线是 Δφ), 这里是**只讲 SAM 路线自己**的成果件 —— 拿出去汇报时
不需要解释另一条路线。

产物(默认全部落在 runs/sam/):
  1. `overlay_<tag>.mp4`    逐帧叠加: 掩码剪影 + 拟合轴线 + 筒身段标记 + 检出框
                            + 中文读数(φ / φ_rel / dφ/dt / 筒身宽 / 体检结论)
  2. `timeline_<tag>.png`   全片曲线: 上段 φ(t) 中轴, 下段 dφ/dt(角速率)
                            + 三条逐帧健康度微条带(测角有效 / 分割可信 / 重锚定)
                            + 0-25s 尺度受限区 与 45-69s 落地段的区间标注
  3. `stills_<tag>.png`     N 帧抽帧拼图(--stills N), 不打开视频也能快速看
  4. `summary_<tag>.txt`    关键数字: 有效帧、分段统计、落地段绝对 φ、推理配置

产物只写不删(帧目录/视频都按 tag 命名, 复用不清理)。

用法:
    python scripts/sam_angle_viz.py                           # s512 全片
    python scripts/sam_angle_viz.py --no-video                # 只出图, 快
    python scripts/sam_angle_viz.py --start 1400 --end 1800   # 只看落地段
    python scripts/sam_angle_viz.py --tag s1024 --stills 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE / "tmp")
os.environ["TMP"] = str(CACHE / "tmp")
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

# 复用已在用的工具(字体回退链 / CSV 读取 / 数值解析), 不再复制一份;
# 缓存重定向与 sys.path 也由它一并完成。
from compare_video import (  # noqa: E402
    C_AXIS, C_BODY, C_BOX, C_SIL, fnum, load_font, read_csv_map,
)
from rocket_track import TrackerConfig, track_frames  # noqa: E402

DETS = PROJECT / "runs" / "diag"
ANGM = PROJECT / "runs" / "angle_mask"
SEGOUT = PROJECT / "runs" / "seg"
OUT = PROJECT / "runs" / "sam"

# 配色(BGR) —— 只属于 SAM 路线自己
C_OK = (60, 220, 60)          # 测角有效
C_CONF = (200, 200, 60)       # 分割可信(青)
C_REA = (0, 140, 235)         # 重锚定点(橙)
C_RAW = (110, 110, 110)       # 逐帧未平滑 φ(灰)
C_RAWR = (62, 62, 62)         # 逐帧原始差分角速率(更暗: 它是噪声参考, 不该抢眼)
C_RATE = (230, 190, 60)       # 角速率曲线(浅蓝)
C_SHADE = (34, 34, 34)        # 区间底色
C_REF = (225, 225, 225)       # 落地基准线

UNRELIABLE_S = 25.0           # 该时刻之前的 φ 受"目标太小"的尺度偏差影响, 不可采信
LANDING = (45.0, 69.0)        # 落地稳定段(报告口径)


def windowed_rate(amap: dict[int, dict], fps: float,
                  half: int = 5) -> dict[int, float]:
    """中心差分角速率 (deg/s): (φ(f+half) − φ(f−half)) / (2·half/fps)。

    为什么不直接用 CSV 里的 `dphi_dt`: 那是**相邻帧**差分再乘 fps, 逐帧噪声被放大
    30 倍(平滑残差 σ≈0.02° ⇒ ±0.6°/s 的毛刺), 整段画出来全是噪声, 看不出趋势。
    取 ±5 帧(≈0.33s)窗口即可把噪声压掉 3 倍多, 又不会抹掉真实机动。
    窗口内跨过缺口(无效帧)的帧一律不算, 不硬补。
    """
    valid = [f for f, r in sorted(amap.items())
             if str(r.get("ok", "0")) in ("1", "True")
             and fnum(r, "phi_smooth") is not None]
    val = {f: fnum(amap[f], "phi_smooth") for f in valid}
    out: dict[int, float] = {}
    for i, f in enumerate(valid):
        j0, j1 = i - half, i + half
        if j0 < 0 or j1 >= len(valid):
            continue
        f0, f1 = valid[j0], valid[j1]
        if f1 - f0 > 2 * half:
            continue
        out[f] = (val[f1] - val[f0]) / ((f1 - f0) / fps)
    return out


# ==========================================================================
# 全片曲线带(只渲染一次)
# ==========================================================================
def build_timeline(n: int, fps: float, W: int, H: int,
                   amap: dict[int, dict], mmap: dict[int, dict],
                   phi_lo: float, phi_hi: float, rate_max: float,
                   phi_ref: float | None,
                   rmap: dict[int, float] | None = None) -> np.ndarray:
    """返回 (H, W, 3) uint8。上段 φ(t), 下段 dφ/dt(t)。"""
    strip = np.full((H, W, 3), 20, np.uint8)
    pad_l, pad_r = 48, 10
    w = W - pad_l - pad_r
    # 顶部 1-12px 是三条健康度微条带, 13-24px 专留给"区间标注",
    # 曲线从 y1_top 才开始 —— 否则标注文字会被早期高达 +14° 的 φ 曲线穿过。
    y1_top = 27
    avail = max(H - y1_top - 16, 40)
    h1 = int(avail * 0.60)
    h2 = max(avail - h1 - 8, 20)
    y2_top = y1_top + h1 + 8

    def x_of(f):
        return int(pad_l + w * f / max(n - 1, 1))

    def mk_y(y_top, y_bot, lo, hi):
        def y_of(p):
            c = min(max(p, lo), hi)
            return int(y_top + (y_bot - y_top) *
                       (1.0 - (c - lo) / max(hi - lo, 1e-6)))
        return y_of

    y_phi = mk_y(y1_top, y1_top + h1 - 8, phi_lo, phi_hi)
    y_rate = mk_y(y2_top, y2_top + h2 - 8, -rate_max, rate_max)

    def put(s, x, y, col=(200, 200, 200), sc=0.38, th=1):
        cv2.putText(strip, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th,
                    cv2.LINE_AA)

    # ---- 区间底色: 0-25s 尺度受限 / 45-69s 落地段 ----
    y_lo_row, y_hi_row = y1_top, y2_top + h2 - 8
    xa = x_of(int(UNRELIABLE_S * fps))
    strip[y_lo_row:y_hi_row, pad_l:xa] = np.clip(
        0.55 * strip[y_lo_row:y_hi_row, pad_l:xa].astype(np.float32)
        + 0.45 * np.array(C_SHADE, np.float32), 0, 255).astype(np.uint8)
    xb0 = x_of(int(LANDING[0] * fps))
    xb1 = x_of(min(int(LANDING[1] * fps), n - 1))
    strip[y_lo_row:y_hi_row, xb0:xb1] = np.clip(
        0.72 * strip[y_lo_row:y_hi_row, xb0:xb1].astype(np.float32)
        + 0.28 * np.array((0, 55, 0), np.float32), 0, 255).astype(np.uint8)
    # 区间名写在专门的标注行, 不压在曲线上
    put(f"0-{UNRELIABLE_S:.0f}s : scale-limited (target only a few px wide)",
        pad_l + 4, 21, (150, 150, 150), 0.34)
    put(f"{LANDING[0]:.0f}-{LANDING[1]:.0f}s : landing (most reliable)",
        xb0 + 4, 21, (120, 200, 120), 0.34)

    # ---- 上段: φ 网格 + 时间轴 ----
    span = max(phi_hi - phi_lo, 1e-6)
    step = 10.0
    for cand in (1.0, 2.0, 5.0, 10.0):
        if h1 * cand / span >= 9.0:
            step = cand
            break
    for p in np.arange(np.ceil(phi_lo / step) * step, phi_hi + 1e-6, step):
        y = y_phi(p)
        cv2.line(strip, (pad_l, y), (W - pad_r, y),
                 (86, 86, 86) if abs(p) > 1e-6 else (130, 130, 130), 1,
                 cv2.LINE_AA)
        put("0" if abs(p) < 1e-6 else f"{p:+.0f}", 4, y + 4, (185, 185, 185))
    # 落地基准线(φ_ref): 相对倾角都以它为 0
    if phi_ref is not None:
        y = y_phi(phi_ref)
        for x in range(pad_l, W - pad_r, 9):
            cv2.line(strip, (x, y), (x + 5, y), C_REF, 1, cv2.LINE_AA)
        s = f"phi_ref = {phi_ref:+.2f} (landing ref = phi_rel zero)"
        tw = int(4.6 * len(s)) + 6
        cv2.rectangle(strip, (pad_l + 2, y - 13), (pad_l + 2 + tw, y - 1),
                      (18, 18, 18), -1)          # 垫底, 免得被曲线糊住
        put(s, pad_l + 6, y - 4, C_REF, 0.34)
    for t in range(0, int(n / fps) + 1, 10):
        x = x_of(int(t * fps))
        cv2.line(strip, (x, y1_top), (x, y2_top + h2), (50, 50, 50), 1,
                 cv2.LINE_AA)
        put(f"{t}s", x + 2, H - 5, (165, 165, 165), 0.36)

    # ---- 下段: dφ/dt 网格 ----
    for p in (-rate_max, -rate_max / 2, 0.0, rate_max / 2, rate_max):
        y = y_rate(p)
        cv2.line(strip, (pad_l, y), (W - pad_r, y),
                 (130, 130, 130) if abs(p) < 1e-9 else (86, 86, 86), 1,
                 cv2.LINE_AA)
        put("0.0" if abs(p) < 1e-9 else f"{p:+.1f}", 2, y + 4, (185, 185, 185),
            0.34)
    cv2.line(strip, (pad_l, y2_top - 5), (W - pad_r, y2_top - 5),
             (95, 95, 95), 1, cv2.LINE_AA)

    # ---- 顶部三条健康度微条带 ----
    for f in range(n):
        x = x_of(f)
        a = amap.get(f)
        m = mmap.get(f)
        if a and str(a.get("ok", "0")) in ("1", "True"):
            strip[1:4, x] = C_OK
        if m and str(m.get("ok", "0")) in ("1", "True"):
            strip[5:8, x] = C_CONF
        if m and str(m.get("reanchored", "0")) in ("1", "True"):
            strip[9:12, x] = C_REA

    def is_ok(r) -> bool:
        return bool(r) and str(r.get("ok", "0")) in ("1", "True")

    def curve(y_of, key, color, thick=1):
        """只连"该帧被评为有效"的点, 无效处断开(不跨跃缺口画直线)。"""
        pts = []
        for f in range(n):
            r = amap.get(f)
            v = fnum(r, key) if is_ok(r) else None
            if v is None:
                if len(pts) > 1:
                    cv2.polylines(strip, [np.array(pts, np.int32)], False,
                                  color, thick, cv2.LINE_AA)
                pts = []
            else:
                pts.append((x_of(f), y_of(v)))
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, color,
                          thick, cv2.LINE_AA)

    def curve_vals(y_of, vals: dict, color, thick=1):
        pts = []
        for f in range(n):
            v = vals.get(f)
            if v is None:
                if len(pts) > 1:
                    cv2.polylines(strip, [np.array(pts, np.int32)], False,
                                  color, thick, cv2.LINE_AA)
                pts = []
            else:
                pts.append((x_of(f), y_of(v)))
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, color,
                          thick, cv2.LINE_AA)

    curve(y_phi, "phi_deg", C_RAW)                 # 逐帧未平滑(能看到跳变)
    curve(y_phi, "phi_smooth", C_AXIS, 1)          # 平滑后的主曲线
    curve(y_rate, "dphi_dt", C_RAWR)               # 逐帧差分(暗灰, 噪声参考)
    if rmap:
        curve_vals(y_rate, rmap, C_RATE)           # 窗内差分(主曲线)

    # 单位只标在图区右上角(那个位置曲线已落到 0 附近, 不会撞上);
    # 其余说明一律放图外的页脚, 避免压住曲线。
    put("phi [deg]", W - 92, y1_top + 10, (150, 150, 150), 0.34)
    put("dphi/dt [deg/s]", W - 132, y2_top + 10, (150, 150, 150), 0.34)
    return strip


# ==========================================================================
# 单帧叠加
# ==========================================================================
def draw_frame(img: np.ndarray, f: int, XL, XR, sm: dict | None,
               ms: dict | None, tb, font_s, t: float,
               phi_ref: float | None, tag: str) -> np.ndarray:
    H, W = img.shape[:2]
    if tb is not None:
        b = [int(v) for v in tb]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), C_BOX, 1)
    # ---- 掩码剪影(逐行边界重建) ----
    xl, xr = XL[f], XR[f]
    valid = (xl >= 0) & (xr > xl)
    n_sil = 0
    if valid.any():
        for y in np.where(valid)[0]:
            a, b = int(xl[y]), int(xr[y])
            img[y, a:b + 1] = (0.60 * img[y, a:b + 1]
                               + 0.40 * np.array(C_SIL)).astype(np.uint8)
        n_sil = int(valid.sum())
    # ---- 拟合出的中轴 + 筒身段(真正参与拟合的行范围) ----
    if sm:
        lt, lb = sm.get("l_top"), sm.get("l_bot")
        rt, rb = sm.get("r_top"), sm.get("r_bot")
        if lt and rt and float(lt[0]) > 0:
            y_t, y_b = float(lt[1]), float(lb[1])
            x_t = 0.5 * (float(lt[0]) + float(rt[0]))
            x_b = 0.5 * (float(lb[0]) + float(rb[0]))
            dy = (y_b - y_t) * 0.12
            dx = (x_b - x_t) / max(abs(y_b - y_t), 1e-6) * dy
            cv2.line(img, (int(x_t - dx), int(y_t - dy)),
                     (int(x_b + dx), int(y_b + dy)), C_AXIS, 1, cv2.LINE_AA)
            for y in (y_t, y_b):
                cv2.line(img, (int(x_t - 26), int(y)), (int(x_t + 26), int(y)),
                         C_BODY, 1, cv2.LINE_AA)
    # ---- 读数(PIL: 中文) ----
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    ps = fnum(sm, "phi_smooth") if sm else None
    pr = fnum(sm, "phi_rel") if sm else None
    rt = fnum(sm, "dphi_dt") if sm else None
    lines = [f"t = {t:5.1f}s    frame {f}",
             f"SAM 2.1 掩码测角  ({tag})"]
    if ps is not None:
        s = f"φ = {ps:+.2f}°"
        if pr is not None:
            s += f"      相对落地基准 φ_rel = {pr:+.2f}°"
        lines.append(s)
        lines.append("角速率 dφ/dt = %s°/s      筒身宽 %.1fpx"
                     % (f"{rt:+.2f}" if rt is not None else "  -- ",
                        fnum(sm, "w_body_px") or float("nan")))
        ir = fnum(sm, "inlier_ratio")
        lines.append("筒身段 %s 段   内点率 %s"
                     % (sm.get("n_bands", "?"),
                        f"{ir:.2f}" if ir is not None else "--"))
    else:
        rc = (sm or {}).get("reason", "") or (ms or {}).get("reason", "")
        lines.append(f"本帧无有效倾角  ({rc})")
    if ms:
        if str(ms.get("has", "0")) in ("0", "False"):
            lines.append(f"分割体检: 不可用 ({ms.get('reason', '')})")
        elif str(ms.get("ok", "1")) in ("0", "False"):
            lines.append(f"分割体检: 判为不可信 ({ms.get('reason', '')})")
        else:
            lines.append("分割体检: 可信")
        if str(ms.get("reanchored", "0")) in ("1", "True"):
            lines.append("★ 本帧为重锚定点(掩码漂移后重新下提示)")
    bx, by = 8, 6
    d.rectangle([bx - 4, by - 3, bx + 372, by + 14 * len(lines) + 6],
                fill=(0, 0, 0))
    for i, s in enumerate(lines):
        d.text((bx, by + 14 * i), s, font=font_s,
               fill=(255, 255, 255) if i else (255, 235, 120))
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    txt = f"mask silhouette rows = {n_sil}   (no cross-method comparison)"
    cv2.putText(img, txt, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ==========================================================================
# 数字摘要
# ==========================================================================
def write_summary(tag: str, info: dict, amap: dict[int, dict],
                  pad: str = "") -> Path:
    fps = info["fps"]
    n = info["n_frame"]
    good = [r for r in amap.values()
            if str(r.get("ok", "0")) in ("1", "True")]
    v = np.array([fnum(r, "phi_smooth") for r in good], float)
    v = v[np.isfinite(v)]
    # 分段名按机理命名, 不按"高/低"命名 —— 15-25s 的 σ 很大是因为箭体正在
    # 快速翻转(dφ/dt 峰值段), 不是估计变差; 这一点必须让人从表里就看出来。
    groups = [("落地稳定段", LANDING[0], LANDING[1]), ("姿态收敛段", 25.0, 45.0),
              ("翻转过渡段", 15.0, 25.0), ("尺度受限段", 7.0, 15.0)]
    L = ["=" * 74,
         "SAM 2.1 掩码测角路线 —— 结果摘要(独立, 不含与原方案的对拍)",
         "=" * 74,
         f"分割标签      : {tag}",
         f"模型/输入尺寸 : SAM 2.1 {info.get('model', '?')} @ "
         f"{info.get('image_size', '?')}px, "
         f"{'fp16' if info.get('half') else 'fp32'}, chunk={info.get('chunk')}",
         f"视频          : {info['W']}x{info['H']} @ {fps:.2f} fps, "
         f"{n} 帧 ({n / fps:.1f}s)",
         f"测角有效帧    : {len(good)}/{n} ({100 * len(good) / n:.1f}%)   "
         f"无掩码 {info.get('n_no_mask', 0)} 帧, "
         f"拟合失败 {info.get('n_fit_failed', 0)} 帧, "
         f"重锚定 {info.get('n_reanchor', 0)} 次",
         f"落地基准 φ_ref: {info.get('ref', float('nan')):+.3f}°  "
         f"(45-66s, 相对倾角 φ_rel 的零点)",
         "",
         f"全片 φ 范围   : [{v.min():+.2f}, {v.max():+.2f}]  "
         f"中位 {np.median(v):+.2f}°",
         "-" * 74,
         f"{'时段':<12}{'n':>6}{'中位':>10}{'均值':>10}{'σ':>9}{'极差':>9}",
         "-" * 74]
    for nm, lo, hi in groups:
        seg = np.array([fnum(r, "phi_smooth") for r in good
                        if lo <= r["frame"] / fps < hi], float)
        seg = seg[np.isfinite(seg)]
        if len(seg) < 3:
            continue
        L.append(f"{nm + f' {lo:.0f}-{hi:.0f}s':<12}{len(seg):>6}"
                 f"{np.median(seg):>+10.2f}{seg.mean():>+10.3f}"
                 f"{seg.std():>9.3f}{np.ptp(seg):>9.2f}")
    L += ["-" * 74, "",
          "可交付量(单视角能给的):",
          "  * φ(t) 箭体轴线相对图像竖直的倾角(正=顶端右倾)",
          "  * φ_rel = φ − φ_ref, 相对落地姿态的倾角变化",
          "  * 90 − |φ|, 箭体与图像上边缘的夹角",
          "  * dφ/dt, 角速率(见 timeline 下段)",
          "",
          f"注意: 前 {UNRELIABLE_S:.0f}s 箭体在画面中仅几像素宽(条带 7~10px),",
          "      φ 的估计误差随目标变小而放大, 该区间不作为结论使用;",
          "      可信区间从约 25s 起, 最可信为落地段 45-69s。",
          ""]
    p = OUT / f"summary_{tag}.txt"
    p.write_text("\n".join(L) + pad, encoding="utf-8")
    return p


# ==========================================================================
# main
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="SAM 2.1 掩码测角路线的独立成果可视化(无对拍)")
    ap.add_argument("--tag", default="s512", help="分割标签(默认 s512, 已够用)")
    ap.add_argument("--dets-tag", default="iou70")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--phi-range", default="auto",
                    help='"min,max" φ 轴范围; auto = 按数据自动定(不截顶)')
    ap.add_argument("--rate-range", default="auto",
                    help='"X" 下段 dφ/dt 轴半宽(deg/s); auto = 稳健分位')
    ap.add_argument("--strip-h", type=int, default=200)
    ap.add_argument("--stills", type=int, default=0)
    a = ap.parse_args()

    meta = json.loads((DETS / f"dets_{a.dets_tag}.json").read_text(
        encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]
    n_all = len(meta["frames"])
    i0 = a.start
    i1 = min(a.end or n_all, n_all)

    aj = json.loads((ANGM / f"angles_{a.tag}.json").read_text(encoding="utf-8"))
    info = aj["info"]
    amap = {int(r["frame"]): r for r in aj["frames"]}
    mmap = read_csv_map(SEGOUT / f"mask_stats_{a.tag}.csv")
    if not mmap:      # 退回 JSON 侧的健康位(没有 mask_stats 的旧产物)
        mmap = {f: {k: r.get(k) for k in
                    ("ok", "has", "reason", "reanchored")}
                for f, r in amap.items()}
    d = np.load(SEGOUT / f"bounds_{a.tag}.npz")
    XL, XR = d["XL"], d["XR"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                        fps, bound_wh=(W, H))

    # ---- φ 轴: 按数据自适应, 不强制对称 ----
    allv = np.array([fnum(r, "phi_smooth") for r in amap.values()], float)
    allv = allv[np.isfinite(allv)]
    if a.phi_range.strip().lower() == "auto" and allv.size:
        phi_lo = float(np.floor((allv.min() - 1.0) / 2) * 2)
        phi_hi = float(np.ceil((allv.max() + 1.0) / 2) * 2)
    elif allv.size:
        phi_lo, phi_hi = (float(x) for x in a.phi_range.split(","))
    else:
        phi_lo, phi_hi = -8.0, 8.0
    # ---- 角速率: 只用窗内差分定轴(逐帧差分噪声太大会把轴撑到 ±几度/秒) ----
    rmap = windowed_rate(amap, fps)
    rv = np.array(list(rmap.values()), float)
    rv = rv[np.isfinite(rv)]
    if a.rate_range.strip().lower() == "auto":
        rate_max = (float(np.ceil(np.percentile(np.abs(rv), 98) * 1.25 * 2) / 2)
                    if rv.size else 1.0)
    else:
        rate_max = float(a.rate_range)
    rate_max = max(0.5, rate_max)
    n_clip = int(np.sum(np.abs(rv) > rate_max)) if rv.size else 0
    phi_ref = info.get("ref")
    print(f"[刻度] φ 轴 {phi_lo:+.0f}..{phi_hi:+.0f}°   "
          f"dφ/dt 轴 ±{rate_max:.1f}°/s (窗内差分 n={rv.size}, "
          f"截顶 {n_clip} / {100 * n_clip / max(rv.size, 1):.1f}%)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    strip = build_timeline(n_all, fps, W, a.strip_h, amap, mmap, phi_lo, phi_hi,
                           rate_max, phi_ref, rmap)
    sh = strip.shape[0]
    n_ok = sum(1 for r in amap.values() if str(r.get("ok")) in ("1", "True"))
    foot = [
        f"SAM 2.1 {info.get('model', '?')} @ {info.get('image_size', '?')}px"
        f"{' fp16' if info.get('half') else ''}   |   "
        f"angle valid {n_ok}/{n_all} frames   |   "
        f"phi_ref {phi_ref:+.2f} deg (landing 45-66s)",
        "curves:  magenta = smoothed phi   grey = per-frame raw   "
        "light-blue = dphi/dt (0.33s window)   grey thin = per-frame diff",
        "top bands:  angle-valid=green   mask-confident=cyan   "
        "reanchored=orange",
    ]
    # 高度要够: 页脚最后一行基线在 32+sh+20+17*(k-1), 少 1px 就会被裁掉半行
    fig = np.full((sh + 46 + 17 * len(foot), W, 3), 255, np.uint8)
    fig[32:32 + sh] = strip
    cv2.putText(fig, f"SAM 2.1 mask route - standalone result  (tag={a.tag})",
                (52, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (35, 35, 35), 1,
                cv2.LINE_AA)
    for i, s in enumerate(foot):
        cv2.putText(fig, s, (52, 32 + sh + 20 + 17 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1,
                    cv2.LINE_AA)
    png = OUT / f"timeline_{a.tag}.png"
    cv2.imwrite(str(png), fig)
    print(f"[写入] {png}", flush=True)

    sp = write_summary(a.tag, info, amap)
    print(f"[写入] {sp}", flush=True)

    if a.no_video:
        return

    # ---------------- 视频 ----------------
    font_s = load_font(14)
    out_path = Path(a.out) if a.out else (OUT / f"overlay_{a.tag}.mp4")
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W, H + sh))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    # 抽帧只在"画面里真有箭体"的区间里均匀取 —— 视频首尾是没有目标的空镜与
    # 片尾卡, 按整段等距取会把宝贵的格子浪费在空帧上。
    with_sil = [f for f in range(i0, i1) if int(np.sum(XL[f] >= 0)) > 0]
    want = ([int(v) for v in np.linspace(with_sil[0], with_sil[-1], a.stills)]
            if (a.stills > 0 and with_sil) else [])
    stills: list = []
    t0 = time.time()
    for f in range(i0, i1):
        ok, fr = cap.read()
        if not ok:
            break
        img = draw_frame(fr, f, XL, XR, amap.get(f), mmap.get(f),
                         outs[f].box if outs[f].has else None, font_s,
                         f / fps, phi_ref, a.tag)
        canvas = np.full((H + sh, W, 3), 16, np.uint8)
        canvas[:H] = img
        canvas[H:] = strip
        cv2.line(canvas, (0, H), (W, H), (90, 90, 90), 1)
        x = int(48 + (W - 58) * f / max(n_all - 1, 1))
        cv2.line(canvas, (x, H + 15), (x, H + sh), (255, 255, 255), 1,
                 cv2.LINE_AA)
        writer.write(canvas)
        if want and f >= want[0]:
            stills.append(canvas.copy())
            want.pop(0)
        if f and (f - i0) % 400 == 0:
            print(f"  {f - i0} 帧 / {time.time() - t0:.0f}s ...", flush=True)
    cap.release()
    writer.release()
    print(f"[写入] {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{time.time() - t0:.1f}s)", flush=True)
    if stills:
        hh = max(s.shape[0] for s in stills)
        ww = sum(s.shape[1] for s in stills) + 8 * (len(stills) - 1)
        mont = np.full((hh, ww, 3), 25, np.uint8)
        x = 0
        for s in stills:
            mont[:s.shape[0], x:x + s.shape[1]] = s
            x += s.shape[1] + 8
        sj = OUT / f"stills_{a.tag}.png"
        cv2.imwrite(str(sj), mont)
        print(f"[写入] {sj}", flush=True)


if __name__ == "__main__":
    main()
