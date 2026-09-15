# -*- coding: utf-8 -*-
"""把两条测角路线的结果叠加到视频上, 出一份可以直接"看效果"的对照材料。

产出两样东西:
1. `runs/seg/compare_overlay_<tag>[_vs_<vs2>].mp4`
   每一帧同时画出:
     * SAM 掩码剪影(半透明填充) —— 直观看到"分割圈住了什么"
     * 第二套 SAM 配置的剪影(紫色描边, --vs2) —— 同屏对比分辨率
     * SAM 拟合出的轴线(品红)   —— 由剪影中心线拟合而来
     * 筒身段标记(黄色短横)     —— 真正参与拟合的行范围(剔除锥面/支腿)
     * 原方案的左右边(绿色)     —— 来自 runs/angle/angles.json
     * 第 2 步稳定框(青色) + 本帧两条路线的 φ 读数与 Δφ
   底部曲线带(默认 190px)**分两段**:
     * 上段 φ(t): 品红=SAM, 绿=原方案, 蓝=第二套配置; 顶部还有逐帧有效性条带
     * 下段 **Δφ(t) = SAM − 原方案**(自己的刻度) —— 两条线只差 0.7~0.8°, 画在
       同一根 ±8° 轴上几乎重合, 单独放大才看得清差异出现在什么时候
2. `runs/seg/compare_timeline_<tag>[_vs_<vs2>].png`
   全片曲线对照图(上段 φ + 下段 Δφ + 有效性条带)。

设计要点(为什么这么做):
  * 剪影只能用**逐行边界**重建 —— 落盘时为了省空间只存了 bounds_*.npz 而不是整张掩码,
    这里用"逐行填 xl..xr"把它还原成剪影, 精度足够做可视化。
  * 曲线带**预先渲染一次**成一张 numpy 图, 每帧只拷一份 + 画游标,
    避免逐帧跑 matplotlib(那样 2200 帧要几分钟)。
  * 原方案边线的坐标是 **ROI patch 局部**坐标, 必须用 angle_roi + crop_roi 复现原点再平移。
  * 中文用 PIL + 微软雅黑渲染; **cv2.putText 只支持 ASCII, 直接写中文会变乱码并与相邻
    文字重叠**(上一版在曲线带标题上踩过, 现已全部改为 ASCII)。

用法:
    python scripts/compare_video.py --tag s1024
    python scripts/compare_video.py --tag s512 --vs2 s1024         # 两种分辨率同屏
    python scripts/compare_video.py --tag s512 --start 1400 --end 2203
    python scripts/compare_video.py --tag s1024 --no-video         # 只出曲线对照图
"""

from __future__ import annotations

import argparse
import csv
import json
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
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from rocket_angle import AngleConfig, angle_roi, crop_roi  # noqa: E402
from rocket_mask_angle import SEG  # noqa: E402
from rocket_track import TrackerConfig, track_frames  # noqa: E402

DETS = PROJECT / "runs" / "diag"
ANG = PROJECT / "runs" / "angle"
ANGM = PROJECT / "runs" / "angle_mask"
OUT = PROJECT / "runs" / "seg"

# 配色(BGR)
C_SIL = (255, 170, 60)      # 掩码剪影填充(淡蓝)
C_SIL2 = (255, 90, 200)     # 第二套 SAM 配置的剪影(紫)
C_AXIS = (255, 0, 255)      # SAM 轴线(品红)
C_BODY = (0, 220, 255)      # 筒身段标记(黄)
C_EDGE = (0, 220, 0)        # 原方案左右边(绿)
C_BOX = (200, 200, 0)       # 稳定框(青)
C_SAM2 = (230, 130, 60)     # 第二套 SAM 配置(蓝)
C_ORANGE = (0, 140, 235)    # 原方案有效性条带(橙; BGR 是红多蓝少, 别写反)
C_REF = (200, 200, 200)

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def load_font(size: int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def read_csv_map(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {int(r["frame"]): r for r in csv.DictReader(fh)}


def fnum(r: dict | None, k: str):
    if not r:
        return None
    v = r.get(k)
    if v in (None, "", "None"):
        return None
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except ValueError:
        return None


# ==========================================================================
# 时间线底图(只渲染一次)
# ==========================================================================
def build_timeline(n: int, fps: float, W: int, H: int,
                   sam: dict[int, dict], org: dict[int, dict],
                   phi_lo: float, phi_hi: float, dphi_max: float = 0.0,
                   sam2: dict[int, dict] | None = None) -> np.ndarray:
    """全片曲线带: **上段 φ(t) + 下段 Δφ(t)**, 返回 (H, W, 3) uint8。

    为什么必须有 Δφ 那一段: 两条路线的 φ 只差 0.7~0.8°, 画在 ±8° 的同一根轴上
    几乎重合, 看不出差异; 单独给 Δφ 一个刻度才能看清它在什么时候、差多少。

    ★ 所有文字都用 cv2.putText ⇒ **只能 ASCII**, 写中文必然变成乱码方块并与相邻文字
      重叠(上一版就是在这里踩的)。中文要显示得走 PIL, 见 draw_frame。
    """
    strip = np.full((H, W, 3), 22, np.uint8)
    pad_l, pad_r = 46, 10
    w = W - pad_l - pad_r
    # 上段 φ, 下段 Δφ; 中间 8px 分隔
    h1 = int((H - 30) * 0.62)
    h2 = H - 30 - h1
    y1_top, y2_top = 14, 14 + h1 + 8

    def x_of(f):
        return int(pad_l + w * f / max(n - 1, 1))

    def mk_y(y_top, hh, lo, hi):
        def y_of(p):
            c = min(max(p, lo), hi)
            return int(y_top + hh * (1.0 - (c - lo) / max(hi - lo, 1e-6)))
        return y_of

    y_phi = mk_y(y1_top, h1, phi_lo, phi_hi)
    y_dphi = mk_y(y2_top, h2, -dphi_max, dphi_max)

    # ---- 上段: φ 网格 + 时间轴 ----
    for p in np.arange(np.ceil(phi_lo), phi_hi + 1, 2.0):
        y = y_phi(p)
        col = (70, 70, 70) if abs(p) > 1e-6 else (120, 120, 120)
        cv2.line(strip, (pad_l, y), (W - pad_r, y), col, 1, cv2.LINE_AA)
        cv2.putText(strip, f"{p:+.0f}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (180, 180, 180), 1, cv2.LINE_AA)
    for t in range(0, int(n / fps) + 1, 10):
        x = x_of(int(t * fps))
        cv2.line(strip, (x, y1_top), (x, y2_top + h2), (52, 52, 52), 1,
                 cv2.LINE_AA)
        cv2.putText(strip, f"{t}s", (x + 2, H - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (160, 160, 160), 1, cv2.LINE_AA)
    # ---- 下段: Δφ 网格 ----
    if dphi_max > 0:
        for p in (-dphi_max, -dphi_max / 2, 0.0, dphi_max / 2, dphi_max):
            y = y_dphi(p)
            col = (120, 120, 120) if abs(p) < 1e-9 else (70, 70, 70)
            cv2.line(strip, (pad_l, y), (W - pad_r, y), col, 1, cv2.LINE_AA)
            cv2.putText(strip, f"{p:+.1f}", (2, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1,
                        cv2.LINE_AA)
    cv2.line(strip, (pad_l, y2_top - 4), (W - pad_r, y2_top - 4),
             (95, 95, 95), 1, cv2.LINE_AA)

    # ---- 有效性条带(最顶 12px): SAM 绿 / 原方案 橙 / SAM2 蓝 ----
    for f in range(n):
        x = x_of(f)
        s, o = sam.get(f), org.get(f)
        if s and str(s.get("ok", "0")) in ("1", "True"):
            strip[1:4, x] = (60, 220, 60)
        if o and str(o.get("ok", "0")) in ("1", "True"):
            # ★ BGR! 图例写 orange, 而 (230,160,60) 是"蓝多红少" ⇒ 渲染成蓝色,
            #   和 SAM2 条带的 (230,130,60) 几乎分不清。真橙色 = 红多蓝少。
            strip[5:8, x] = C_ORANGE
        if sam2 is not None and str(sam2.get(f, {}).get("ok", "0")) in ("1", "True"):
            strip[9:12, x] = (230, 130, 60)

    def valid(r):
        return bool(r) and str(r.get("ok", "0")) in ("1", "True")

    def curve(y_of, valfn, color):
        """valfn(f) 返回该帧的值; None 表示无效, 曲线在无效处断开。"""
        pts = []
        for f in range(n):
            v = valfn(f)
            if v is None:
                if len(pts) > 1:
                    cv2.polylines(strip, [np.array(pts, np.int32)], False,
                                  color, 1, cv2.LINE_AA)
                pts = []
            else:
                pts.append((x_of(f), y_of(v)))
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, color, 1,
                          cv2.LINE_AA)

    def grep(m, f, key):
        r = m.get(f)
        return fnum(r, key) if valid(r) else None

    curve(y_phi, lambda f: grep(sam, f, "phi_smooth"), C_AXIS)
    curve(y_phi, lambda f: grep(org, f, "phi_smooth"), C_EDGE)
    if sam2 is not None:
        curve(y_phi, lambda f: grep(sam2, f, "phi_smooth"), C_SAM2)
    if dphi_max > 0:
        def d1(f):
            a, b = grep(sam, f, "phi_smooth"), grep(org, f, "phi_smooth")
            return None if (a is None or b is None) else a - b
        curve(y_dphi, d1, C_AXIS)
        if sam2 is not None:
            def d2(f):
                a, b = grep(sam2, f, "phi_smooth"), grep(org, f, "phi_smooth")
                return None if (a is None or b is None) else a - b
            curve(y_dphi, d2, C_SAM2)

    # ---- 标题(ASCII only!) ----
    lab = "phi(t):  magenta = SAM mask      green = gradient-edge"
    if sam2 is not None:
        lab += "      blue = SAM2"
    cv2.putText(strip, lab, (pad_l + 4, y1_top + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 205, 205), 1, cv2.LINE_AA)
    if dphi_max > 0:
        lab2 = "delta-phi = SAM - gradient   (own scale, +-%.1f deg)" % dphi_max
        cv2.putText(strip, lab2, (pad_l + 4, y2_top + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (205, 205, 205), 1,
                    cv2.LINE_AA)
    return strip


# ==========================================================================
# 单帧绘制
# ==========================================================================
def draw_frame(img: np.ndarray, f: int, XL, XR, sm: dict | None, om: dict | None,
               gm: dict | None, tb, roi_org, font_s, t: float,
               XL2=None, XR2=None, sm2: dict | None = None) -> np.ndarray:
    H, W = img.shape[:2]
    # ---- 稳定框 ----
    if tb is not None:
        b = [int(v) for v in tb]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), C_BOX, 1)
    # ---- SAM 剪影(逐行填充); 第二套配置用描边区分, 避免互相盖住 ----
    if XL2 is not None:
        xl2, xr2 = XL2[f], XR2[f]
        v2 = (xl2 >= 0) & (xr2 > xl2)
        for y in np.where(v2)[0]:
            a, b = int(xl2[y]), int(xr2[y])
            for xx in (a, b):
                if 0 <= xx < W:
                    img[y, xx] = C_SIL2
    xl, xr = XL[f], XR[f]
    valid = (xl >= 0) & (xr > xl)
    n_sil = 0
    if valid.any():
        ys = np.where(valid)[0]
        for y in ys:
            a, b = int(xl[y]), int(xr[y])
            img[y, a:b + 1] = (0.60 * img[y, a:b + 1]
                               + 0.40 * np.array(C_SIL)).astype(np.uint8)
        n_sil = int(valid.sum())
    # ---- 原方案左右边(需要 ROI 原点把 patch 坐标转回全帧) ----
    if om and roi_org is not None:
        ox, oy = roi_org
        for key, col in (("l", C_EDGE), ("r", C_EDGE)):
            p = om.get(f"{key}_top")
            q = om.get(f"{key}_bot")
            if not p or not q:
                continue
            if float(p[0]) <= 0 or abs(float(p[1]) - float(q[1])) < 5:
                continue
            cv2.line(img, (int(float(p[0]) + ox), int(float(p[1]) + oy)),
                     (int(float(q[0]) + ox), int(float(q[1]) + oy)), col, 1,
                     cv2.LINE_AA)
    # ---- SAM 轴线 + 筒身段标记 ----
    if sm:
        lt, lb = sm.get("l_top"), sm.get("l_bot")
        rt, rb = sm.get("r_top"), sm.get("r_bot")
        if lt and rt and float(lt[0]) > 0:
            y_t, y_b = float(lt[1]), float(lb[1])
            x_t = 0.5 * (float(lt[0]) + float(rt[0]))
            x_b = 0.5 * (float(lb[0]) + float(rb[0]))
            # 两端各延长 12% 便于看清方向
            dy = (y_b - y_t) * 0.12
            dx = (x_b - x_t) / max(abs(y_b - y_t), 1e-6) * dy
            cv2.line(img, (int(x_t - dx), int(y_t - dy)),
                     (int(x_b + dx), int(y_b + dy)), C_AXIS, 1, cv2.LINE_AA)
            for y in (y_t, y_b):        # 筒身段范围(参与拟合的行)
                cv2.line(img, (int(x_t - 26), int(y)), (int(x_t + 26), int(y)),
                         C_BODY, 1, cv2.LINE_AA)
    # ---- 文字(PIL, 支持中文) ----
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    ps = fnum(sm, "phi_smooth") if sm else None
    po = fnum(om, "phi_smooth") if om else None
    lines = [f"t = {t:5.1f}s   frame {f}"]
    if ps is not None:
        lines.append(f"SAM 掩码路线   φ = {ps:+.2f}°   "
                     f"筒身宽 {fnum(sm, 'w_body_px') or float('nan'):.1f}px")
    else:
        lines.append("SAM 掩码路线   本帧无有效掩码")
    if po is not None:
        lines.append(f"原方案梯度边缘 φ = {po:+.2f}°   "
                     f"拟合条带 {fnum(om, 'w_body_px') or float('nan'):.1f}px")
    else:
        lines.append("原方案梯度边缘 本帧未锁定/被门控剔除")
    if ps is not None and po is not None:
        lines.append(f"Δφ (SAM − 原)  = {ps - po:+.2f}°")
    if gm and str(gm.get("has", "0")) in ("0", "False"):
        lines.append(f"分割体检: 不可用 ({gm.get('reason', '')})")
    elif gm and str(gm.get("ok", "1")) in ("0", "False"):
        lines.append(f"分割体检: 判为不可信 ({gm.get('reason', '')})")
    if gm and str(gm.get("reanchored", "0")) in ("1", "True"):
        lines.append("★ 本帧为重锚定点(掩码漂移后重新下提示)")
    bx, by = 8, 6
    d.rectangle([bx - 4, by - 3, bx + 322, by + 13 * len(lines) + 6],
                fill=(0, 0, 0))
    for i, s in enumerate(lines):
        d.text((bx, by + 13 * i), s, font=font_s,
               fill=(255, 255, 255) if i else (255, 235, 120))
    img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    txt = f"silhouette rows = {n_sil}"      # cv2.putText 只支持 ASCII
    cv2.putText(img, txt, (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ==========================================================================
# 主流程
# ==========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="两条测角路线的视频对照可视化")
    ap.add_argument("--tag", default="s1024", help="SAM 分割标签")
    ap.add_argument("--dets-tag", default="iou70")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--phi-range", default="-8,8")
    ap.add_argument("--vs2", default="",
                    help="第二套 SAM 配置标签(如 s1024), 同屏对比分辨率")
    ap.add_argument("--strip-h", type=int, default=190,
                    help="底部曲线带高度(默认 190: 上段 φ, 下段 Δφ)")
    ap.add_argument("--stills", type=int, default=0,
                    help=">0 时额外抽 N 帧拼成一张 png(不打开视频也能快速看)")
    a = ap.parse_args()

    meta = json.loads((DETS / f"dets_{a.dets_tag}.json").read_text(
        encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]
    n_all = len(meta["frames"])
    i0 = a.start
    i1 = min(a.end or n_all, n_all)

    d = np.load(SEG / f"bounds_{a.tag}.npz")
    XL, XR = d["XL"], d["XR"]
    smap = read_csv_map(ANGM / f"angles_{a.tag}.csv")
    ojson = json.loads((ANG / "angles.json").read_text(encoding="utf-8"))["frames"]
    omap = {int(r["frame"]): r for r in ojson}
    gmap = read_csv_map(SEG / f"mask_stats_{a.tag}.csv")
    if a.vs2:
        d2 = np.load(SEG / f"bounds_{a.vs2}.npz")
        XL2, XR2 = d2["XL"], d2["XR"]
        s2map = read_csv_map(ANGM / f"angles_{a.vs2}.csv")
    else:
        XL2 = XR2 = s2map = None
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                        fps, bound_wh=(W, H))

    # 原方案边线的 patch 原点(逐帧复现; 只需在能算出边线的帧上做)
    acfg = AngleConfig()
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))
    need = [f for f in range(i0, i1)
            if omap.get(f, {}).get("l_top") and float(
                omap[f]["l_top"][0] or 0) > 0]
    print(f"[准备] 需要复现 ROI 原点的帧数: {len(need)}", flush=True)
    roi_org: dict[int, tuple] = {}
    for f in need:
        o = outs[f]
        if not (o.has and o.box is not None and o.roi is not None):
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        cr = crop_roi(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY),
                      angle_roi(o.box, o.roi, acfg, W, H), o.box)
        if cr is not None:
            roi_org[f] = (cr[2][0], cr[2][1])
    cap.release()

    lo, hi = (float(v) for v in a.phi_range.split(","))
    # Δφ 自己的刻度: 用数据的 96 分位(至少 0.5°) —— 差异只有零点几度, 必须放大才看得见
    dv = []
    for f in range(n_all):
        p = fnum(smap.get(f), "phi_smooth")
        q = fnum(omap.get(f), "phi_smooth")
        if p is not None and q is not None:
            dv.append(abs(p - q))
    dphi_max = float(np.clip(np.percentile(dv, 96) * 1.25, 0.5, 8.0)) if dv else 0.0
    print(f"[刻度] Δφ 轴 ±{dphi_max:.2f}°  (样本 {len(dv)} 帧)", flush=True)
    strip = build_timeline(n_all, fps, W, a.strip_h, smap, omap, lo, hi,
                           dphi_max, s2map)
    print(f"[时间线] 底图 {strip.shape} 完成", flush=True)

    # ---------------- 静态对照图 ----------------
    OUT.mkdir(parents=True, exist_ok=True)
    sh = strip.shape[0]
    fig = np.full((sh + 60, W, 3), 255, np.uint8)
    fig[30:30 + sh] = strip
    ttl = ("SAM 2.1 mask route (magenta) vs gradient-edge route (green)"
           + (f" + {a.vs2} (blue)" if a.vs2 else ""))
    cv2.putText(fig, ttl.encode("ascii", "ignore").decode(), (52, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(fig, ("frame-level validity: SAM=green | gradient=orange"
                      + (" | SAM2=blue" if a.vs2 else "")),
                (52, 30 + sh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (120, 120, 120), 1, cv2.LINE_AA)
    png = OUT / (f"compare_timeline_{a.tag}"
                 + (f"_vs_{a.vs2}" if a.vs2 else "") + ".png")
    cv2.imwrite(str(png), fig)
    print(f"[写入] {png}", flush=True)

    if a.no_video:
        return

    # ---------------- 视频 ----------------
    font_s = load_font(14)
    out_path = Path(a.out) if a.out else (
        OUT / (f"compare_overlay_{a.tag}" + (f"_vs_{a.vs2}" if a.vs2 else "")
               + ".mp4"))
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W, H + sh))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    stills: list = []
    want = ([int(v) for v in np.linspace(i0, i1 - 1, a.stills)]
            if a.stills > 0 else [])
    for f in range(i0, i1):
        ok, fr = cap.read()
        if not ok:
            break
        img = draw_frame(fr, f, XL, XR, smap.get(f), omap.get(f), gmap.get(f),
                         outs[f].box if outs[f].has else None,
                         roi_org.get(f), font_s, f / fps,
                         XL2, XR2, s2map.get(f) if s2map else None)
        canvas = np.full((H + sh, W, 3), 18, np.uint8)
        canvas[:H] = img
        canvas[H:] = strip
        cv2.line(canvas, (0, H), (W, H), (90, 90, 90), 1)
        x = int(46 + (W - 56) * f / max(n_all - 1, 1))
        cv2.line(canvas, (x, H + 14), (x, H + sh), (255, 255, 255), 1,
                 cv2.LINE_AA)
        writer.write(canvas)
        if want and f >= want[0]:
            stills.append(canvas.copy())
            want.pop(0)
        if f and (f - i0) % 400 == 0:
            print(f"  {f - i0} 帧 ...", flush=True)
    cap.release()
    writer.release()
    print(f"[写入] {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)
    if stills:
        hh = max(s.shape[0] for s in stills)
        ww = sum(s.shape[1] for s in stills) + 8 * (len(stills) - 1)
        mont = np.full((hh, ww, 3), 25, np.uint8)
        x = 0
        for s in stills:
            mont[:s.shape[0], x:x + s.shape[1]] = s
            x += s.shape[1] + 8
        spng = OUT / (f"compare_stills_{a.tag}"
                      + (f"_vs_{a.vs2}" if a.vs2 else "") + ".png")
        cv2.imwrite(str(spng), mont)
        print(f"[写入] {spng}", flush=True)


if __name__ == "__main__":
    main()
