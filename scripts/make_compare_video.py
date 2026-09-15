# -*- coding: utf-8 -*-
"""生成"看得懂效果"的对照视频: 两条路线的边线同时叠加 + 实时 φ(t) 曲线。

为什么需要它: 光看数字(Δφ +0.76°、σ 0.095° vs 0.223°)很难判断"到底差在哪"。
把两条路线量到的**边**画在同一帧上, 一眼就能看出谁在剪影上、谁在剪影里面。

画面构成(原视频下方接一条 140px 的曲线带):
  红点 = SAM 掩码的逐行左右边界(剪影)
  蓝点 = 另一套 SAM 配置(可选, --vs2), 用于对比分辨率
  绿线 = 原方案在 ROI 内拟合出的左右边缘直线
  黄框 = 第 2 步的稳定检测框
  底部曲线 = 全程 φ(t), 竖线是当前帧; 绿=原方案, 红=SAM, 蓝=SAM2

用法:
    python scripts/make_compare_video.py --tag s512
    python scripts/make_compare_video.py --tag s512 --vs2 s1024     # 两种分辨率同屏
    python scripts/make_compare_video.py --tag s1024 --start 1300 --end 1600 --slow 2
输出:
    runs/angle_mask/compare_overlay_<tag>[_vs_<vs2>].mp4
    runs/angle_mask/compare_stills_<tag>[_vs_<vs2>].png   (6 帧抽帧拼图, 便于快速看)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / ".cache"
for _s in ("tmp", "torch", "mpl", "yolo"):
    (CACHE / _s).mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE / "mpl")
os.environ["TORCH_HOME"] = str(CACHE / "torch")
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402

from rocket_angle import AngleConfig, angle_roi, crop_roi  # noqa: E402
from rocket_track import TrackerConfig, track_frames  # noqa: E402

SEG = PROJECT / "runs" / "seg"
ANG = PROJECT / "runs" / "angle"
ANGM = PROJECT / "runs" / "angle_mask"
DETS = PROJECT / "runs" / "diag"

CH = 140          # 曲线带高度
C_RED = (0, 0, 255)
C_BLUE = (255, 200, 0)
C_GREEN = (0, 200, 0)
C_BOX = (0, 255, 255)


def _series(frames: list[dict], key: str = "phi_deg") -> np.ndarray:
    v = np.full(len(frames), np.nan)
    for i, r in enumerate(frames):
        if r.get("ok") and r.get(key) is not None:
            v[i] = float(r[key])
        elif r.get(key) is not None:
            v[i] = float(r[key])
    return v


def _draw_panel(canvas: np.ndarray, y0: int, y1: int,
                series: list[tuple[np.ndarray, tuple]], i: int | None, n: int,
                vmax: float, tag: str = "") -> None:
    """在 canvas[y0:y1] 内画时间序列。i=None 表示只画静态部分(不画游标)。"""
    W = canvas.shape[1]
    mid = (y0 + y1) // 2
    half = max((y1 - y0) / 2 - 4, 4)
    cv2.line(canvas, (0, mid), (W, mid), (70, 70, 70), 1, cv2.LINE_AA)
    for d in (-vmax, vmax):
        y = int(mid - d / vmax * half)
        cv2.line(canvas, (0, y), (W, y), (42, 42, 42), 1)
        cv2.putText(canvas, f"{d:+.1f}", (3, max(y0 + 9, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (105, 105, 105), 1)
    for v, col in series:
        pts = []
        for x in range(W):
            f = int(x * (n - 1) / max(W - 1, 1))
            y = v[f]
            if np.isfinite(y):
                pts.append((x, int(mid - float(np.clip(y, -vmax, vmax))
                                   / vmax * half)))
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts)], False, col, 1, cv2.LINE_AA)
    if tag:
        cv2.putText(canvas, tag, (W - 96, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1,
                    cv2.LINE_AA)


def _build_chart_bg(W: int, n: int, series_phi, series_dphi, vmax: float,
                    dmax: float, legend: str) -> np.ndarray:
    """曲线带的**静态**背景: 全程曲线只画一次, 之后每帧只需贴图 + 画游标。

    逐帧重画 852×4 条折线在 2203 帧上会明显拖慢生成(纯 Python 循环),
    预渲染后每帧成本降到一次 memcpy。
    """
    chart = np.zeros((CH, W, 3), np.uint8)
    chart[:] = 18
    ysplit = 94
    _draw_panel(chart, 0, ysplit, series_phi, None, n, vmax, "φ(t)")
    _draw_panel(chart, ysplit, CH, series_dphi, None, n, dmax, "Δφ=SAM−grad")
    cv2.line(chart, (0, ysplit), (W, ysplit), (90, 90, 90), 1)
    _text_row(chart, W - 330, CH - 4, legend, (170, 170, 170), 0.38)
    return chart


def _text_row(img, x, y, txt, col, scale=0.5):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1,
                cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成两路线对照视频")
    ap.add_argument("--tag", default="s512")
    ap.add_argument("--vs2", default="", help="第二套 SAM 配置(如 s1024)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--slow", type=int, default=1, help=">1 时每帧重复, 放慢播放")
    ap.add_argument("--stills", type=int, default=6)
    a = ap.parse_args()

    meta = json.loads((DETS / "dets_iou70.json").read_text(encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                        fps, bound_wh=(W, H))
    acfg = AngleConfig()

    d1 = np.load(SEG / f"bounds_{a.tag}.npz")
    XL1, XR1 = d1["XL"], d1["XR"]
    n = XL1.shape[0]
    if a.vs2:
        d2 = np.load(SEG / f"bounds_{a.vs2}.npz")
        XL2, XR2 = d2["XL"], d2["XR"]

    ang_orig = json.loads((ANG / "angles.json").read_text(
        encoding="utf-8"))["frames"]
    ang_sam = json.loads((ANGM / f"angles_{a.tag}.json").read_text(
        encoding="utf-8"))["frames"]
    if a.vs2:
        ang_sam2 = json.loads((ANGM / f"angles_{a.vs2}.json").read_text(
            encoding="utf-8"))["frames"]

    s_orig = _series(ang_orig)
    s_sam = _series(ang_sam)
    s_sam2 = _series(ang_sam2) if a.vs2 else None
    s_dphi = s_sam - s_orig          # Δφ = SAM − grad, 这才是关键差异
    s_dphi2 = (s_sam2 - s_orig) if s_sam2 is not None else None
    stack = [s_orig[~np.isnan(s_orig)], s_sam[~np.isnan(s_sam)]]
    if s_sam2 is not None:
        stack.append(s_sam2[~np.isnan(s_sam2)])
    allv = np.concatenate(stack) if stack else np.array([-5.0, 5.0])
    vmax = float(np.clip(np.percentile(np.abs(allv), 96) * 1.25, 3.0, 22.0))
    dv = s_dphi[np.isfinite(s_dphi)]
    dmax = float(np.clip(np.percentile(np.abs(dv), 96) * 1.3, 0.5, 6.0)) \
        if dv.size else 1.0
    C_D = (255, 0, 255)

    a_start = a.start
    a_end = a.end or n
    out_name = (f"compare_overlay_{a.tag}"
                + (f"_vs_{a.vs2}" if a.vs2 else "") + ".mp4")
    out_path = ANGM / out_name
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / max(a.slow, 1), (W, H + CH))
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))

    stills, want = [], [int(v) for v in
                        np.linspace(a_start, a_end - 1, a.stills)] if a.stills else []
    chart_bg = _build_chart_bg(
        W, n,
        [(s_orig, C_GREEN), (s_sam, C_RED)]
        + ([(s_sam2, C_BLUE)] if s_sam2 is not None else []),
        [(s_dphi, C_D)] + ([(s_dphi2, C_BLUE)] if s_dphi2 is not None else []),
        vmax, dmax,
        "green=grad  red=SAM" + ("  blue=" + a.vs2 if a.vs2 else "")
        + "  magenta=Δφ")
    t0 = __import__("time").time()
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if i < a_start:
            continue
        if i >= a_end:
            break
        o = outs[i]
        # ---------- 原方案的左右边缘直线(patch 局部坐标 -> 全帧) ----------
        if o.has and o.box is not None and o.roi is not None and ang_orig[i].get("ok"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cr = crop_roi(gray, angle_roi(o.box, o.roi, acfg, W, H), o.box)
            if cr is not None:
                ox, oy = float(cr[2][0]), float(cr[2][1])
                for side in ("l", "r"):
                    p = np.asarray(ang_orig[i].get(f"{side}_top", [0.0, 0.0]), float)
                    q = np.asarray(ang_orig[i].get(f"{side}_bot", [0.0, 0.0]), float)
                    if p[0] <= 0 or abs(p[1] - q[1]) < 5:
                        continue
                    cv2.line(frame,
                             (int(p[0] + ox), int(p[1] + oy)),
                             (int(q[0] + ox), int(q[1] + oy)),
                             C_GREEN, 1, cv2.LINE_AA)
        if o.box is not None:
            b = [int(v) for v in o.box]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), C_BOX, 1)
        # ---------- SAM 掩码边界 ----------
        for (XL, XR, col) in ([(XL1, XR1, C_RED)] +
                              ([(XL2, XR2, C_BLUE)] if a.vs2 else [])):
            xl, xr = XL[i].astype(float), XR[i].astype(float)
            xl[xl < 0] = np.nan
            xr[xr < 0] = np.nan
            ys = np.where(np.isfinite(xl))[0]
            for y in ys[::3]:
                cv2.circle(frame, (int(xl[y]), int(y)), 1, col, -1)
                cv2.circle(frame, (int(xr[y]), int(y)), 1, col, -1)
        # ---------- 文字 ----------
        po, ps = ang_orig[i].get("phi_deg"), ang_sam[i].get("phi_deg")
        txt1 = ("SAM  " + (f"φ={ps:+.2f}°" if ps is not None else "φ=—")
                + (f"  w={ang_sam[i].get('w_body_px', float('nan')):.1f}px"
                   if ang_sam[i].get("w_body_px") else ""))
        txt2 = ("grad " + (f"φ={po:+.2f}°" if po is not None else "φ=—")
                + (f"  w={ang_orig[i].get('w_body_px', float('nan')):.1f}px"
                   if ang_orig[i].get("w_body_px") else ""))
        txt3 = ""
        if po is not None and ps is not None:
            txt3 = f"Δφ = {ps - po:+.2f}°"
        _text_row(frame, 8, 20, txt1, C_RED)
        _text_row(frame, 8, 38, txt2, C_GREEN)
        if txt3:
            _text_row(frame, 8, 56, txt3, (255, 255, 255), 0.55)
        _text_row(frame, 8, H - 8, f"t={i / fps:.1f}s  f{i}  "
                                   f"({a.tag}{' vs ' + a.vs2 if a.vs2 else ''})",
                  (200, 200, 200), 0.5)
        # ---------- 曲线带: 上段 φ(t), 下段 Δφ = SAM − grad (静态背景 + 游标) ----------
        chart = chart_bg.copy()
        xc = int(i * (W - 1) / max(n - 1, 1))
        cv2.line(chart, (xc, 0), (xc, 94), (255, 255, 255), 1)
        cv2.line(chart, (xc, 94), (xc, CH), (255, 255, 255), 1)
        out = np.vstack([frame, chart])
        for _ in range(a.slow):
            writer.write(out)
        if want and i >= want[0]:
            stills.append(out.copy()); want.pop(0)
        if i and i % 400 == 0:
            print(f"  {i}/{a_end}  {__import__('time').time() - t0:.0f}s",
                  flush=True)
    cap.release()
    writer.release()
    print(f"[写入] {out_path}  ({a_end - a_start} 帧, "
          f"{__import__('time').time() - t0:.0f}s)", flush=True)
    if stills:
        hh = max(s.shape[0] for s in stills)
        ww = sum(s.shape[1] for s in stills) + 8 * (len(stills) - 1)
        canvas = np.full((hh, ww, 3), 25, np.uint8)
        x = 0
        for s in stills:
            canvas[:s.shape[0], x:x + s.shape[1]] = s
            x += s.shape[1] + 8
        p = ANGM / out_name.replace(".mp4", "_stills.png")
        cv2.imwrite(str(p), canvas)
        print(f"[写入] {p}", flush=True)


if __name__ == "__main__":
    main()
