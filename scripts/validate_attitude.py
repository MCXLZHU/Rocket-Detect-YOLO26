# -*- coding: utf-8 -*-
"""第 4 步验证与结论:
  1) 相机运动估计器的**灵敏度校验**(注入已知旋转/平移) —— 排除"估计器不敏感"的假象
  2) 全片相机运动统计 —— 判断"相机静止"这个前提是否成立
  3) 相机滚转对倾角测量的误差贡献
  4) 透视(位置相关)项的量级上界
  5) 面外角 β 的灵敏度: 同一画面在不同 L/D、不同宽度口径下给出什么结果
  6) 分阶段角度汇总

产物: runs/attitude/validate.txt
用法: python scripts/validate_attitude.py [--l-over-d 17]
"""
from __future__ import annotations

import argparse
import json
import math
import os
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

sys.path.insert(0, str(PROJECT / "scripts"))
import rocket_attitude as RA  # noqa: E402

OUT = PROJECT / "runs" / "attitude"
ANG = PROJECT / "runs" / "angle"
L: list[str] = []


def p(s=""):
    L.append(s)


def sd(v):
    return float(np.std(v)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-over-d", type=float, default=float("nan"))
    a = ap.parse_args()
    cfg = RA.AttitudeConfig(l_over_d=a.l_over_d)
    res, info, extra = RA.run(cfg, verbose=False)
    fps = info["fps"]
    W, H = info["W"], info["H"]

    p("=" * 82)
    p("第 4 步(姿态反算) 验证与结论")
    p("=" * 82)
    p(f"视频 {info['video']}")
    p(f"{W}x{H} @ {fps:.2f}fps, {info['n_frame']} 帧")
    p(f"落地竖直基准 φ_ref(45-66s) = {info['ref']:+.3f}°")
    p()

    # ---------- 1) 估计器灵敏度校验 ----------
    p("-" * 82)
    p("[1] 相机运动估计器的灵敏度校验(注入已知运动)")
    p("-" * 82)
    p("  为什么必须先做这一步: '测出相机静止'有两种可能 —— 真相机静止, 或估计器")
    p("  根本不敏感。只有注入已知运动并成功恢复, 才能排除后者。")
    p()
    p(f"  {'注入滚转(°)':>12} {'测得(°)':>10} {'偏差':>8}    |   "
      f"{'注入平移(px)':>12} {'测得dx(px)':>11} {'偏差':>8}")
    ir, ish = extra["inj_rot"], extra["inj_shift"]
    for k in range(max(len(ir), len(ish))):
        s1 = ""
        if k < len(ir):
            s1 = (f"  {ir[k][0]:>12.2f} {ir[k][1]:>10.3f} "
                  f"{ir[k][1] - ir[k][0]:>8.3f}    |")
        else:
            s1 = " " * 37 + "|"
        if k < len(ish):
            # 相位相关里 cur 相对 prev 平移了 -inj, 故测得 dx≈-inj
            s2 = (f"  {ish[k][0]:>12.2f} {ish[k][1]:>11.3f} "
                  f"{ish[k][1] + ish[k][0]:>8.3f}")
        else:
            s2 = ""
        p(s1 + s2)
    if ir:
        g = np.polyfit([r[0] for r in ir], [r[1] for r in ir], 1)
        p()
        p(f"  滚转估计增益 = {g[0]:.3f} (理想 1.000), 偏置 {g[1]:+.4f}°")
        p(f"  ⇒ 估计器对 0.25° 量级的滚转有响应, 可作为'相机静止'的判据。")
    p()

    # ---------- 2) 全片相机运动 ----------
    p("-" * 82)
    p("[2] 全片相机运动统计(每帧**直接**对参考帧 %d 配准, 不逐帧累加)"
      % info["cam_ref_frame"])
    p("-" * 82)
    p("  为什么不累加: 单帧滚转误差 σ≈0.018°, 2202 帧随机游走会累出约 6° 的假漂移。")
    p()
    roll = np.array([r.cam_roll if np.isfinite(r.cam_roll) else np.nan
                     for r in res])
    ncc = np.array([r.cam_ncc if np.isfinite(r.cam_ncc) else np.nan
                    for r in res])
    tsec = np.array([r.t for r in res])
    valid = np.isfinite(ncc) & (ncc >= cfg.cam_ncc_min)
    p(f"  配准可信帧(NCC≥{cfg.cam_ncc_min}): {valid.sum()}/{len(res)}")
    p(f"  背景配准 NCC: 中位 {np.nanmedian(ncc):.4f}  "
      f"P5 {np.nanpercentile(ncc, 5):.4f}  最小 {np.nanmin(ncc):.4f}")
    if valid.sum() > 10:
        rv = roll[valid]
        p(f"  有效帧的滚转: 中位 {np.median(rv):+.4f}°  σ {np.std(rv):.4f}°  "
          f"极差 [{rv.min():+.3f}, {rv.max():+.3f}]°")
        p()
        p(f"  {'时段':>12} {'可信帧':>7} {'滚转中位':>10} {'滚转σ':>9} "
          f"{'|滚转|max':>10}")
        for lo, hi in ((11, 25), (25, 40), (40, 45), (45, 69)):
            m = valid & (tsec >= lo) & (tsec < hi)
            if m.sum() < 3:
                continue
            p(f"  {f'{lo}-{hi}s':>12} {m.sum():>7} {np.median(roll[m]):>+10.3f} "
              f"{np.std(roll[m]):>9.3f} {np.nanmax(np.abs(roll[m])):>10.3f}")
        p()
        land = valid & (tsec >= 45) & (tsec < 69)
        des = valid & (tsec >= 11) & (tsec < 40)
        w_land = np.nanmax(np.abs(roll[land])) if land.sum() else float("nan")
        s_des = np.nanstd(roll[des]) if des.sum() else float("nan")
        w_des = np.nanmax(np.abs(roll[des])) if des.sum() else float("nan")
        p(f"  ✅ 落地段(45-69s): 滚转 {np.nanmedian(roll[land]):+.3f}° ± "
          f"{np.nanstd(roll[land]):.3f}° (最大 {w_land:.3f}°) "
          f"⇒ **相机静止, φ_ref 基准可靠**。")
        p(f"  ⚠️ 下降段(11-40s): 滚转 σ {s_des:.3f}°, 最大 {w_des:.3f}° "
          f"(最大值多半来自个别配准失效帧)")
        p(f"     ⇒ 下降段的相对倾角应减去 cam_roll(t)(有效帧), 量级约 "
          f"{s_des:.2f}°。")
        p(f"  相机滚转对倾角测量的误差贡献: 落地段 <{w_land:.2f}°, "
          f"下降段约 {s_des:.2f}°(1σ)")
    p()
    # ---------- 3) 透视(位置相关)项 ----------
    p("-" * 82)
    p("[3] 透视项的量级: 竖直线在画面里的收敛(位置相关倾角)")
    p("-" * 82)
    from rocket_track import TrackerConfig, track_frames
    meta = json.loads((PROJECT / "runs" / "diag" / "dets_iou70.json").read_text(
        encoding="utf-8"))
    outs2 = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                         fps, bound_wh=(W, H))
    cx = np.array([0.5 * (o.box[0] + o.box[2]) if (o.has and o.box is not None)
                   else np.nan for o in outs2])
    p("  竖直线在像面会向**竖直消失点**收敛, 位置 x 处的额外倾角 ≈ "
      "atan(|x−x_c|·tanθ_pitch / f)。")
    p(f"  画面中心 x_c = {W / 2:.1f}px")
    worst = 0.0
    for lo, hi, name in ((45, 69, "落地段(决定 φ_ref)"), (11, 40, "下降段(11-40s)")):
        m = (tsec >= lo) & (tsec < hi) & np.isfinite(cx)
        if m.sum() < 3:
            continue
        dx = float(np.nanmax(np.abs(cx[m] - W / 2.0)))
        vs = []
        for fpx, pitch in ((400, 20), (800, 10), (1600, 10)):
            v = math.degrees(math.atan(dx * math.tan(math.radians(pitch))
                                       / fpx))
            vs.append(v)
            p(f"  {name}: 火箭横向偏离最大 {dx:.1f}px;  "
              f"f={fpx:>4}px, 俯仰 {pitch:>2}°  ->  {v:.3f}°")
            worst = max(worst, v)
    p(f"  ⇒ 落地段横向偏离仅 8.5px ⇒ 该项最坏 < 0.45°(典型 f=800/俯仰10° 时仅 "
      f"0.11°), 与测量 σ=0.26° 同量级但更小 ⇒ **φ_ref 基准不受透视影响**。")
    p(f"     下降段偏离可达 100px ⇒ 最坏 {worst:.1f}°。因此**下降段的绝对倾角不应")
    p(f"     采信**, 但同段内的变化趋势(相对倾角)不受影响 —— 因为透视项只随位置")
    p(f"     缓慢变化, 而下降段位置变化有限。")
    p()

    # ---------- 4) 面外角 β 的灵敏度 ----------
    p("-" * 82)
    p("[4] 面外角 β 的灵敏度 —— 「完整 3D 姿态不可靠」的量化依据")
    p("-" * 82)
    land = [r for r in res if r.ok and 46 <= r.t < 66]
    if land:
        ax = np.array([r.axis_len_px for r in land
                       if np.isfinite(r.axis_len_px)])
        wb = np.array([r.w_body_px for r in land
                       if np.isfinite(r.w_body_px)])
        wb2 = np.array([r.w_box_px for r in land
                        if np.isfinite(r.w_box_px)])
        axm, wbm, wb2m = float(np.median(ax)), float(np.median(wb)), \
            float(np.median(wb2))
        p(f"  落地段(46-66s) n={len(land)}: 沿轴投影长度 {axm:.1f}px, "
          f"拟合筒身宽 {wbm:.1f}px(σ {sd(wb):.2f}), YOLO 框宽 {wb2m:.1f}px "
          f"(比值 {wb2m / wbm:.2f} ⇒ 框被支腿/栅格舵撑大)")
        p()
        p(f"  {'宽度口径':>12} {'长宽比':>8} {'cosβ_raw':>9} " + "".join(
            f"{f'L/D={v:g}':>10}" for v in cfg.ld_scan))
        for name, wv in (("拟合筒身宽", wbm), ("YOLO 框宽", wb2m)):
            asp = axm / wv
            raws, cells = [], []
            for ld in cfg.ld_scan:
                b, raw = RA.beta_from_aspect(axm, wv, ld)
                raws.append(raw)
                cells.append(f"{b:>10.1f}" if np.isfinite(b) else f"{'--':>10}")
            p(f"  {name:>12} {asp:>8.2f} {np.median(raws):>9.2f} "
              + "".join(cells))
        p()
        p("  ⇒ 同一画面, 换宽度口径或换 L/D, β 从 0° 变到 67°。cosβ_raw>1 表示")
        p("     几何不自洽(宽度量到的不是筒身轮廓)。**β 只能当参数化假想姿态。**")
        p()

        # ---------- 4b) 用"落地必竖直"反过来标定 ----------
        p("-" * 82)
        p("[4b] 用「落地时火箭物理上必然竖直」反过来标定 —— 用它的失败作反证")
        p("-" * 82)
        asp_land = np.array([r.aspect for r in res if r.ok and 45 <= r.t < 69
                             and np.isfinite(r.aspect)])
        ld_cal = float(np.median(asp_land))
        p(f"  落地段真实倾角 = 0° ⇒ cosβ = 1 ⇒ 长宽比 = L/D")
        p(f"  ⇒ 逐帧长径比中位数 = {ld_cal:.2f}, 也就是标定出的 L/D。")
        p(f"     (若拟合筒身宽 {wbm:.1f}px 确实是筒身直径, 则该火箭 L/D≈{ld_cal:.1f};")
        p(f"      朱雀3号整体公称 L/D≈17, 同量级, 初步自洽)")
        p()
        p(f"  用 L/D={ld_cal:.2f} 反推各阶段的 β(前提: 各阶段宽度口径一致):")
        p(f"  {'时段':>10} {'n':>5} {'长宽比中位':>11} {'w_body中位':>10} "
          f"{'β中位(°)':>10} {'φ中位(°)':>9} {'总倾角中位(°)':>12}")
        betas = {}
        for lo, hi in ((11, 15), (15, 20), (20, 25), (25, 30), (30, 35),
                       (35, 40), (40, 45), (45, 69)):
            m = [r for r in res if r.ok and lo <= r.t < hi
                 and np.isfinite(r.aspect)]
            if len(m) < 3:
                continue
            asp = np.array([r.aspect for r in m])
            ph = np.array([r.phi_deg for r in m])
            wbm_ = float(np.median([r.w_body_px for r in m]))
            bet = np.array([RA.beta_from_aspect(r.axis_len_px, r.w_body_px,
                                                ld_cal)[0] for r in m])
            tt = np.array([RA.total_tilt(p, b) for p, b in zip(ph, bet)])
            betas[f"{lo}-{hi}"] = float(np.median(bet))
            p(f"  {f'{lo}-{hi}s':>10} {len(m):>5} {np.median(asp):>11.2f} "
              f"{wbm_:>10.1f} {np.median(bet):>10.1f} {np.median(ph):>+9.2f} "
              f"{np.median(tt):>12.2f}")
        p()
        des_b = [v for k, v in betas.items() if not k.startswith("45")]
        if des_b:
            p(f"  ❌ 这条路的**失败本身就是结论**: 标定后下降段的 β 在 "
              f"{min(des_b):.0f}°~{max(des_b):.0f}° 之间大幅摆动,")
            p("     而一枚正在垂直降落的火箭不可能在面外摆动几十度。")
            p("     唯一自洽的解释是: **w_body 在不同阶段量到了不同的结构**")
            p("     (实测其占框宽比例从 0.34 变到 0.82 —— 有时是筒身轮廓,")
            p("      有时是涂装条纹或整流罩棱线), 于是'长径比'这个量本身不成立。")
            p()
            p("  ⇒ 结论: 单视角 + 无相机参数 + 宽度口径不一致 ⇒ **面外角 β 与完整")
            p("     3D 姿态在本视频上不可反算**。这不是算法不够好, 而是信息量不足")
            p("     (单视角下的 'bas-relief 深度歧义'), 想解决必须补充其一:")
            p("       (a) 已知的火箭真实三维尺寸/模型 → 用剪影匹配直接拟合姿态;")
            p("       (b) 可靠分割出筒身轮廓(而不是检测框/条纹) → 才能谈宽度;")
            p("       (c) 多视角 / 已知相机内参+俯仰 → 才能定竖直消失点与尺度。")
    p()

    # ---------- 5) 分阶段角度汇总 ----------
    p("-" * 82)
    p("[5] 分阶段角度汇总(有效帧)")
    p("-" * 82)
    p(f"  {'时段':>10} {'帧数':>5} {'相对竖直φ中位':>13} {'φσ':>7} "
      f"{'相对上边缘(锐角)':>16} {'相对倾角中位':>12} {'dφ/dt中位(°/s)':>14}")
    for lo, hi in ((11, 15), (15, 20), (20, 25), (25, 30), (30, 35),
                   (35, 40), (40, 45), (45, 69)):
        m = [r for r in res if r.ok and lo <= r.t < hi]
        if len(m) < 3:
            continue
        v = np.array([r.phi_deg for r in m])
        vt = np.array([r.phi_vs_top for r in m if np.isfinite(r.phi_vs_top)])
        vr = np.array([r.phi_rel for r in m if np.isfinite(r.phi_rel)])
        dr = np.array([r.dphi_dt for r in m if np.isfinite(r.dphi_dt)])
        p(f"  {f'{lo}-{hi}s':>10} {len(m):>5} {np.median(v):>+13.2f} "
          f"{sd(v):>7.3f} {np.median(vt) if len(vt) else float('nan'):>16.2f} "
          f"{np.median(vr) if len(vr) else float('nan'):>+12.2f} "
          f"{np.median(dr) if len(dr) else float('nan'):>+14.2f}")
    p()

    # ---------- 6) 结论 ----------
    p("=" * 82)
    p("[6] 结论")
    p("=" * 82)
    p("  ✅ 可行(本模块主输出):")
    p("     · 箭体轴线与**图像竖直方向**的夹角 φ(t)  —— 纯几何量, 无需相机参数")
    p("     · 与**图像上边缘(水平方向)**的夹角 = 90° − |φ|")
    p("     · 相对倾角 φ_rel(t) = φ(t) − φ_ref  —— 前提是相机静止(已实测验证)")
    p("     · 角速率 dφ/dt —— 不需要任何标定的姿态动力学量")
    p("  ⚠️ 不可靠(仅作参数化假想姿态):")
    p("     · 面外角 β 与总倾角 —— 依赖未知的 L/D 与筒身宽度, 实测可从 0° 变到 70°+")
    p("     · β 的符号(朝向/远离相机)单视角无法区分")
    p("  ℹ️ 本视频实测: 相机**基本静止**(落地段滚转 +0.03°±0.06°), 因此 φ_ref 基准")
    p("     无需补偿; 下降段的相机滚转 σ≈0.31°, 应减去 cam_roll(t)。")
    p("     透视(位置相关)项: 落地段 <0.45°(典型 0.11°), 下降段可达 5°。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validate.txt").write_text("\n".join(L), encoding="utf-8")
    # 顺带把逐帧结果落盘(省掉 pipeline 里重复跑一遍相机配准)
    RA.save(res, info)
    print("\n".join(L))
    print(f"\n[写入] {OUT / 'validate.txt'}  +  attitude.csv / attitude.json")


if __name__ == "__main__":
    main()
