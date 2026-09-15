# -*- coding: utf-8 -*-
"""把两条测角路线的结果叠加到视频上, 出一份可以直接"看效果"的对照材料。

产出两样东西:
1. `runs/seg/compare_overlay_<tag>.mp4`
   每一帧同时画出:
     * SAM 掩码剪影(半透明填充) —— 直观看到"分割圈住了什么"
     * SAM 拟合出的轴线(品红)   —— 由剪影中心线拟合而来
     * 筒身段标记(黄色短横)     —— 真正参与拟合的行范围(剔除锥面/支腿)
     * 原方案的左右边(绿色)     —— 来自 runs/angle/angles.json
     * 第 2 步稳定框(青色) + 本帧 φ 读数
   底部 150px 是**全片 φ(t) 时间线**, 带当前时刻游标, 两条曲线一眼可比。
2. `runs/seg/compare_timeline_<tag>.png`
   全片 φ(t) 对照图 + Δφ(t) + 有效性条带(三栏)。

设计要点(为什么这么做):
  * 剪影只能用**逐行边界**重建 —— 落盘时为了省空间只存了 bounds_*.npz 而不是整张掩码,
    这里用"逐行填 xl..xr"把它还原成剪影, 精度足够做可视化。
  * 时间线**预先渲染一次**成一张 numpy 图, 每帧只拷一份 + 画游标,
    避免逐帧跑 matplotlib(那样 2200 帧要几分钟)。
  * 原方案边线的坐标是 **ROI patch 局部**坐标, 必须用 angle_roi + crop_roi 复现原点再平移。
  * 中文用 PIL + 微软雅黑渲染(cv2.putText 只有 ASCII, 直接写中文会变乱码)。

用法:
    python scripts/compare_video.py --tag s1024
    python scripts/compare_video.py --tag s512 --start 1400 --end 2203
    python scripts/compare_video.py --tag s1024 --no-video      # 只出时间线图
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
C_AXIS = (255, 0, 255)      # SAM 轴线(品红)
C_BODY = (0, 220, 255)      # 筒身段标记(黄)
C_EDGE = (0, 220, 0)        # 原方案左右边(绿)
C_BOX = (200, 200, 0)       # 稳定框(青)
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
                   phi_lo: float, phi_hi: float) -> np.ndarray:
    """画出全片 φ(t) 的双曲线 + Δφ; 返回 (H, W, 3) uint8。"""
    strip = np.full((H, W, 3), 22, np.uint8)
    pad_l, pad_r, pad_t, pad_b = 46, 10, 16, 18
    w = W - pad_l - pad_r
    h = H - pad_t - pad_b

    def x_of(f):
        return int(pad_l + w * f / max(n - 1, 1))

    def y_of(p):
        c = min(max(p, phi_lo), phi_hi)
        return int(pad_t + h * (1.0 - (c - phi_lo) / max(phi_hi - phi_lo, 1e-6)))

    # 网格与纵轴刻度
    for p in np.arange(np.ceil(phi_lo), phi_hi + 1, 2.0):
        y = y_of(p)
        col = (70, 70, 70) if abs(p) > 1e-6 else (120, 120, 120)
        cv2.line(strip, (pad_l, y), (W - pad_r, y), col, 1, cv2.LINE_AA)
        cv2.putText(strip, f"{p:+.0f}", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (180, 180, 180), 1, cv2.LINE_AA)
    for t in range(0, int(n / fps) + 1, 10):
        x = x_of(int(t * fps))
        cv2.line(strip, (x, pad_t), (x, H - pad_b), (58, 58, 58), 1, cv2.LINE_AA)
        cv2.putText(strip, f"{t}s", (x + 2, H - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (160, 160, 160), 1, cv2.LINE_AA)
    # 有效性条带(顶部 5px): SAM 绿 / 原方案 蓝
    for f in range(n):
        x = x_of(f)
        s = sam.get(f)
        o = org.get(f)
        if s and str(s.get("ok", "0")) in ("1", "True"):
            strip[2:6, x] = (60, 220, 60)
        if o and str(o.get("ok", "0")) in ("1", "True"):
            strip[7:11, x] = (230, 160, 60)

    # 曲线: 无效段断开
    def draw(key, arr_map, color, smooth_key=None):
        pts, prev = [], None
        for f in range(n):
            r = arr_map.get(f)
            v = None
            if r and str(r.get("ok", "0")) in ("1", "True"):
                v = fnum(r, smooth_key or "phi_deg")
            if v is None:
                if len(pts) > 1:
                    cv2.polylines(strip, [np.array(pts, np.int32)], False, color,
                                  1, cv2.LINE_AA)
                pts = []
            else:
                pts.append((x_of(f), y_of(v)))
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, color, 1,
                          cv2.LINE_AA)

    draw("sam", sam, (230, 90, 230), "phi_smooth")
    draw("org", org, C_EDGE, "phi_smooth")
    cv2.putText(strip, f"SAM phi (magenta) vs 原方案 phi (green)  y={phi_lo:.0f}..{phi_hi:.0f}deg",
                (pad_l + 4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (200, 200, 200), 1, cv2.LINE_AA)
    return strip


# ==========================================================================
# 单帧绘制
# ==========================================================================
def draw_frame(img: np.ndarray, f: int, XL, XR, sm: dict | None, om: dict | None,
               gm: dict | None, tb, roi_org, font_s, t: float) -> np.ndarray:
    H, W = img.shape[:2]
    # ---- 稳定框 ----
    if tb is not None:
        b = [int(v) for v in tb]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), C_BOX, 1)
    # ---- SAM 剪影(逐行填充) ----
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
    strip = build_timeline(n_all, fps, W, 150, smap, omap, lo, hi)
    print(f"[时间线] 底图 {strip.shape} 完成", flush=True)

    # ---------------- 静态对照图 ----------------
    OUT.mkdir(parents=True, exist_ok=True)
    fig = np.full((210, W, 3), 255, np.uint8)
    fig[30:180] = strip
    cv2.putText(fig, "SAM 2.1 mask route (magenta) vs gradient-edge route (green)"
                .encode("ascii", "ignore").decode(), (52, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(fig, "frame-level validity: SAM=green | gradient=blue".encode(
        "ascii", "ignore").decode(), (52, 196), cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (120, 120, 120), 1, cv2.LINE_AA)
    png = OUT / f"compare_timeline_{a.tag}.png"
    cv2.imwrite(str(png), fig)
    print(f"[写入] {png}", flush=True)

    if a.no_video:
        return

    # ---------------- 视频 ----------------
    font_s = load_font(14)
    out_path = Path(a.out) if a.out else (OUT / f"compare_overlay_{a.tag}.mp4")
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W, H + 150))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    for f in range(i0, i1):
        ok, fr = cap.read()
        if not ok:
            break
        img = draw_frame(fr, f, XL, XR, smap.get(f), omap.get(f), gmap.get(f),
                         outs[f].box if outs[f].has else None,
                         roi_org.get(f), font_s, f / fps)
        canvas = np.full((H + 150, W, 3), 18, np.uint8)
        canvas[:H] = img
        canvas[H:] = strip
        cv2.line(canvas, (0, H), (W, H), (90, 90, 90), 1)
        x = int(46 + (W - 56) * f / max(n_all - 1, 1))
        cv2.line(canvas, (x, H + 16), (x, H + 150 - 18), (255, 255, 255), 1,
                 cv2.LINE_AA)
        writer.write(canvas)
        if f and (f - i0) % 400 == 0:
            print(f"  {f - i0} 帧 ...", flush=True)
    cap.release()
    writer.release()
    print(f"[写入] {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)


if __name__ == "__main__":
    main()
