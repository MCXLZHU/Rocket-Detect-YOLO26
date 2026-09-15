# -*- coding: utf-8 -*-
"""对拍: SAM 掩码轮廓 vs 原方案(ROI 梯度)拟合出的左右边。

这是方案对比里最关键的一张证据: 两条路线量到的**边**是不是同一条。
  红点 = SAM 掩码的逐行左右边界
  绿线 = 原方案在 ROI 内拟合出的左右边缘直线(runs/angle/angles.json)
若两者重合 ⇒ 两种方法口径一致, 差别只在数值细节;
若系统性错开 ⇒ 说明其中一方量错了(例如原方案量到了涂装条纹)。

用法:
    python scripts/_seg_vs_grad.py --tag smoke --frames 1400,1440,1480
输出:
    runs/seg/cmp_mask_vs_grad_<tag>.png   (放大叠加图)
    runs/seg/cmp_mask_vs_grad_<tag>.log   (逐帧数值对照)
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
sys.path.insert(0, str(PROJECT / "scripts"))

import cv2  # noqa: E402

from rocket_angle import AngleConfig, angle_roi, crop_roi  # noqa: E402
from rocket_track import TrackerConfig, track_frames  # noqa: E402

SEG = PROJECT / "runs" / "seg"
ANG = PROJECT / "runs" / "angle"
DETS = PROJECT / "runs" / "diag"


def _edge_line(d: dict, side: str, ox: float = 0.0, oy: float = 0.0):
    """把 angles.json 里的 (top, bot) 两点变成 y->x 线性函数。

    ★ 坐标是 **ROI patch 局部**坐标, 不是全帧 —— 见 rocket_angle._finalize 的注释
    ("线条端点(全帧坐标由调用方补 origin; 这里用 patch 坐标即可用于叠加)")。
    所以要按 ROI 原点 (ox, oy) 平移出去, 才能和掩码的全帧边界比。
    """
    p = np.asarray(d.get(f"{side}_top", [0.0, 0.0]), float)
    q = np.asarray(d.get(f"{side}_bot", [0.0, 0.0]), float)
    if not (np.isfinite(p).all() and np.isfinite(q).all()) or p[0] <= 0 \
            or abs(p[1] - q[1]) < 5:
        return None
    return np.polyfit([p[1] + oy, q[1] + oy], [p[0] + ox, q[0] + ox], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--frames", default="900,1400,1440,1480,1800,2100")
    ap.add_argument("--zoom", type=float, default=4.0)
    a = ap.parse_args()

    d = np.load(SEG / f"bounds_{a.tag}.npz")
    XL, XR = d["XL"], d["XR"]
    ang = json.loads((ANG / "angles.json").read_text(
        encoding="utf-8"))["frames"]
    meta = json.loads((DETS / "dets_iou70.json").read_text(encoding="utf-8"))
    vm = meta["meta"]
    W, H, fps = vm["W"], vm["H"], vm["fps"]
    # 复现原方案的 ROI 原点: 边线是 patch 局部坐标, 必须加上 (x0,y0) 才是全帧
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                        fps, bound_wh=(W, H))
    acfg = AngleConfig()
    cap = cv2.VideoCapture(str(PROJECT / vm["video"]))

    lines, tiles = [], []
    for f in [int(v) for v in a.frames.split(",") if v.strip()]:
        if f >= XL.shape[0]:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        xl, xr = XL[f].astype(float), XR[f].astype(float)
        k = xl > 0
        A = ang[f]
        ox = oy = 0.0
        o = outs[f]
        if o.has and o.box is not None and o.roi is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cr = crop_roi(gray, angle_roi(o.box, o.roi, acfg, W, H), o.box)
            if cr is not None:
                ox, oy = float(cr[2][0]), float(cr[2][1])
        gl, gr = _edge_line(A, "l", ox, oy), _edge_line(A, "r", ox, oy)
        lines.append(f"--- frame {f} (t={f / fps:.1f}s) ---")
        lines.append(f"  原方案 phi_deg={A.get('phi_deg')} "
                     f"w_body_px={A.get('w_body_px')} "
                     f"w_box_px={A.get('w_box_px')} mode={A.get('mode')}")
        if k.any():
            ys = np.where(k)[0]
            y0, y1 = int(ys[0]), int(ys[-1])
            wr = (xr - xl)[k]
            lines.append(f"  SAM: 行 {y0}..{y1}, 宽度中位 {np.median(wr):.1f}px, "
                         f"x {xl[k].min():.0f}..{xr[k].max():.0f}")
        else:
            lines.append("  SAM: 该帧无掩码")
        if gl is not None and gr is not None and k.any():
            yy = np.arange(y0, y1 + 1)
            gxl, gxr = np.polyval(gl, yy), np.polyval(gr, yy)
            dl, dr = xl[y0:y1 + 1] - gxl, xr[y0:y1 + 1] - gxr
            lines.append(f"  原方案: 宽度中位 {np.median(gxr - gxl):.1f}px")
            lines.append(f"  SAM - 原方案: 左边 {np.median(dl):+.2f}px, "
                         f"右边 {np.median(dr):+.2f}px  (中位; 正=SAM 更靠右)")
            lines.append(f"  掩码宽 / 原方案宽 = "
                         f"{np.median(wr) / max(np.median(gxr - gxl), 1e-6):.3f}")
            # ★ 偏移是否随 y 变化 —— 这才是影响倾角的量。
            #   常数偏移(平行错开)不影响斜率 ⇒ 不影响 φ; 随 y 线性变化才是角度差。
            n = yy.size
            if n >= 20:
                q = n // 5
                dl_t, dl_b = float(np.median(dl[:q])), float(np.median(dl[-q:]))
                dr_t, dr_b = float(np.median(dr[:q])), float(np.median(dr[-q:]))
                span = max(float(yy[-1] - yy[0]), 1.0)
                dphi_l = np.degrees(np.arctan((dl_b - dl_t) / span))
                dphi_r = np.degrees(np.arctan((dr_b - dr_t) / span))
                lines.append(
                    f"  偏移沿 y 的变化: 左边 {dl_t:+.2f}->{dl_b:+.2f}px "
                    f"(等效角度差 {dphi_l:+.3f}°), 右边 {dr_t:+.2f}->{dr_b:+.2f}px "
                    f"({dphi_r:+.3f}°)")
                lines.append("    ⇒ 若这两个角度差接近 0, 说明两者只是**平行错开**"
                             "(原方案的倾角仍然可信, 只是 w_body 口径错了); "
                             "若明显非 0, 说明原方案的边根本不是同一条结构。")
        elif k.any():
            lines.append("  原方案: 该帧没有可用的左右边(未锁定/门控剔除)")
        # ---- 叠加图 ----
        for side, fn in (("l", gl), ("r", gr)):
            if fn is None:
                continue
            yv = np.arange(max(0, y0 - 30), min(H, y1 + 30))
            for y in yv:
                x = int(round(np.polyval(fn, y)))
                if 0 <= x < W:
                    cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        if k.any():
            for y in np.where(k)[0][::2]:
                cv2.circle(img, (int(xl[y]), int(y)), 1, (0, 0, 255), -1)
                cv2.circle(img, (int(xr[y]), int(y)), 1, (0, 0, 255), -1)
        x0 = int(max(0, (xl[k].min() if k.any() else 400) - 34))
        x1 = int(min(W, (xr[k].max() if k.any() else 440) + 34))
        y0c = int(max(0, (y0 if k.any() else 40) - 10))
        y1c = int(min(H, (y1 if k.any() else 400) + 10))
        crop = img[y0c:y1c, x0:x1]
        z = cv2.resize(crop, None, fx=a.zoom, fy=a.zoom,
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(z, f"f{f} red=SAM green=gradient", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(z)
    cap.release()

    txt = "\n".join(lines)
    (SEG / f"cmp_mask_vs_grad_{a.tag}.txt").write_text(txt, encoding="utf-8")
    print(txt, flush=True)
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        wsum = sum(t.shape[1] for t in tiles) + 10 * (len(tiles) - 1)
        canvas = np.full((hmax, wsum, 3), 25, np.uint8)
        x = 0
        for t in tiles:
            canvas[:t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 10
        p = SEG / f"cmp_mask_vs_grad_{a.tag}.png"
        cv2.imwrite(str(p), canvas)
        print(f"[写入] {p}", flush=True)


if __name__ == "__main__":
    main()
