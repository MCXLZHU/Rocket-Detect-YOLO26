# -*- coding: utf-8 -*-
"""第 4 步前置探测: 背景里到底有没有可用的相机运动参考?

要回答三件事:
  A. 画面里能不能找到**地平线**(近似水平的长边)? 若能, 它的倾角 = 相机滚转角(绝对值)。
  B. 相邻帧背景的**平移**能否用相位相关稳定估出? 峰值有多尖?
  C. 相邻帧背景的**滚转**能否用"小角度旋转搜索"估出?
同时输出对比度拉伸后的背景图, 供人工确认。

产物: runs/attitude/probe.txt + runs/attitude/bg/f<idx>.jpg
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
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
from rocket_track import TrackerConfig, track_frames  # noqa: E402

DIAG = PROJECT / "runs" / "diag"
OUT = PROJECT / "runs" / "attitude"
L: list[str] = []


def p(s=""):
    L.append(s)


def open_cap(path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap
    tmp = CACHE / "tmp" / "att.mp4"
    shutil.copy(str(path), str(tmp))
    return cv2.VideoCapture(str(tmp))


def horizon_candidates(gray, y_lo_frac=0.25, y_hi_frac=0.75, max_tilt=15.0,
                       min_len_frac=0.30):
    """在指定纵向范围内找"近似水平"的长直线, 返回 [(tilt_deg, y, length)]。"""
    H, W = gray.shape
    y0, y1 = int(H * y_lo_frac), int(H * y_hi_frac)
    sub = gray[y0:y1]
    # 竖直线不会被检出: 用较低的阈值 + 长度约束
    edges = cv2.Canny(sub, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40,
                            minLineLength=int(min_len_frac * W), maxLineGap=20)
    out = []
    if lines is None:
        return out
    for ln in lines[:, 0]:
        x1, y1_, x2, y2_ = [float(v) for v in ln]
        dx, dy = x2 - x1, y2_ - y1_
        if abs(dx) < 1e-6:
            continue
        tilt = math.degrees(math.atan2(dy, dx))
        if abs(tilt) > max_tilt:
            continue
        out.append((tilt, 0.5 * (y1_ + y2_) + y0, math.hypot(dx, dy)))
    out.sort(key=lambda r: -r[2])
    return out


def rot_search(prev, cur, max_deg=2.0, coarse=0.25, fine=0.05):
    """小角度旋转搜索: 找使 NCC 最大的 θ (围绕图像中心旋转 prev)。"""
    h, w = prev.shape
    c = (w / 2.0, h / 2.0)
    prev32 = prev.astype(np.float32)
    cur32 = cur.astype(np.float32)
    best = (None, -2.0)

    def score(th):
        M = cv2.getRotationMatrix2D(c, th, 1.0)
        r = cv2.warpAffine(prev32, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        a = r - r.mean()
        b = cur32 - cur32.mean()
        d = math.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-9
        return float((a * b).sum()) / d

    ths = np.arange(-max_deg, max_deg + 1e-9, coarse)
    for th in ths:
        s = score(float(th))
        if s > best[1]:
            best = (float(th), s)
    t0 = best[0]
    for th in np.arange(t0 - coarse, t0 + coarse + 1e-9, fine):
        s = score(float(th))
        if s > best[1]:
            best = (float(th), s)
    return best


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=None)
    args = ap.parse_args()
    meta = json.loads((DIAG / "dets_iou70.json").read_text(encoding="utf-8"))
    W, H, fps = meta["meta"]["W"], meta["meta"]["H"], meta["meta"]["fps"]
    outs = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(), fps,
                        bound_wh=(W, H))
    cap = open_cap(PROJECT / meta["meta"]["video"])
    (OUT / "bg").mkdir(parents=True, exist_ok=True)

    p("=" * 80)
    p("第 4 步前置探测: 背景可用性")
    p("=" * 80)
    p(f"{W}x{H} @ {fps:.2f}fps")
    p()

    probes = [500, 700, 900, 1100, 1200, 1500, 1900, 2050]
    p("-" * 80)
    p("[A] 地平线候选(近似水平长直线)")
    p("-" * 80)
    p(f"  {'帧':>5} {'t(s)':>7} {'候选数':>7} {'最长线倾角':>11} {'其y位置':>9} "
      f"{'长度':>7} {'次长线倾角':>11}")
    frames = {}
    for f in probes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames[f] = g
        hs = horizon_candidates(g)
        if hs:
            p(f"  {f:>5} {f / fps:>7.1f} {len(hs):>7} "
              f"{hs[0][0]:>11.3f} {hs[0][1]:>9.0f} {hs[0][2]:>7.0f} "
              f"{(hs[1][0] if len(hs) > 1 else float('nan')):>11.3f}")
        else:
            p(f"  {f:>5} {f / fps:>7.1f} {0:>7} {'--':>11} {'--':>9} {'--':>7} "
              f"{'--':>11}")
    p()

    p("-" * 80)
    p("[B] 相邻帧背景的滚转估计(小角度旋转搜索) + 平移(相位相关)")
    p("-" * 80)
    p(f"  {'帧区间':>15} {'滚转θ':>9} {'NCC峰':>8} {'平移dx':>8} {'平移dy':>8} "
      f"{'背景纹理强度':>12}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev = None
    seq = {}
    for i in range(len(outs)):
        ok, fr = cap.read()
        if not ok:
            break
        seq[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    for f in (args.pairs if args.pairs else
              [699, 899, 1199, 1499, 1899]):
        if f not in seq or f + 1 not in seq:
            continue
        o = outs[f]
        a, b = seq[f].copy(), seq[f + 1].copy()
        # 屏蔽火箭所在区域(留 1.5 倍框宽/高的安全边)
        if o.box is not None:
            bx = o.box
            x0 = int(max(0, bx[0] - 0.5 * (bx[2] - bx[0])))
            x1 = int(min(W, bx[2] + 0.5 * (bx[2] - bx[0])))
            y0 = int(max(0, bx[1] - 20))
            y1 = int(min(H, bx[3] + 20))
            for m in (a, b):
                m[y0:y1, x0:x1] = 0
        th, ncc = rot_search(a, b)
        tx, ty = 0.0, 0.0
        try:
            (tx, ty), _ = cv2.phaseCorrelate(a.astype(np.float32),
                                             b.astype(np.float32))
        except Exception:
            pass
        tex = float(np.std(cv2.Laplacian(a, cv2.CV_32F)))
        p(f"  {f}-{f + 1:>6} {th:>+9.3f} {ncc:>8.4f} {tx:>8.2f} {ty:>8.2f} "
          f"{tex:>12.3f}")

    p()
    p("-" * 80)
    p("[C] 对比度拉伸后的背景(人工确认是否有地平线)")
    p("-" * 80)
    for f, g in frames.items():
        cl = cv2.createCLAHE(3.0, (8, 8)).apply(g)
        rgb = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
        hs = horizon_candidates(g)
        for i, (tilt, y, ln) in enumerate(hs[:3]):
            # 把该直线画出来(近似: 过 y、斜率 tan(tilt) 的整幅线)
            x0 = 0
            x1 = W
            yy0 = int(y - math.tan(math.radians(tilt)) * (W / 2))
            yy1 = int(y + math.tan(math.radians(tilt)) * (W / 2))
            col = (0, 0, 255) if i == 0 else (0, 200, 255)
            cv2.line(rgb, (x0, yy0), (x1, yy1), col, 1)
        cv2.putText(rgb, f"f{f} t={f / fps:.1f}s", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / "bg" / f"f{f}.jpg"), rgb,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cap.release()
    p(f"  已输出 {len(frames)} 张 -> {OUT / 'bg'}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[写入] {OUT / 'probe.txt'}")


if __name__ == "__main__":
    main()
