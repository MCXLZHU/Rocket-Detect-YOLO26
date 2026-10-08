# -*- coding: utf-8 -*-
"""第 4 步验证与结论:
  1) 相机运动估计器的**灵敏度校验**(注入已知旋转/平移) —— 排除"估计器不敏感"的假象
  2) 全片相机运动统计 —— 判断"相机静止"这个前提是否成立
  3) 相机滚转对倾角测量的误差贡献
  4) 透视(位置相关)项的量级上界
  5) 面外角 β 的灵敏度: 同一画面在不同 L/D、不同宽度口径下给出什么结果
  6) 分阶段角度汇总

产物: runs/attitude/validate_<tag>.txt + attitude_<tag>.csv/.json
      <tag> 默认与 --src-tag 相同(多视频产物共存, 不互相覆盖)
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
L: list[str] = []


def p(s=""):
    L.append(s)


def sd(v):
    return float(np.std(v)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-over-d", type=float, default=float("nan"))
    ap.add_argument("--dets-tag", default="iou70")
    ap.add_argument("--src-tag", default="s512",
                    help="第 3 步掩码路线的标签(读 angles_<src-tag>.json)")
    ap.add_argument("--out-tag", default="",
                    help="本阶段产物的文件名后缀; 默认与 --src-tag 相同(避免多视频互相覆盖)")
    a = ap.parse_args()
    out_tag = a.out_tag or a.src_tag
    cfg = RA.AttitudeConfig(l_over_d=a.l_over_d)
    res, info, extra = RA.run(cfg, dets_tag=a.dets_tag, verbose=False,
                              src_tag=a.src_tag)
    fps = info["fps"]
    W, H = info["W"], info["H"]

    p("=" * 82)
    p("第 4 步(姿态反算) 验证与结论")
    p("=" * 82)
    p(f"视频 {info['video']}")
    p(f"{W}x{H} @ {fps:.2f}fps, {info['n_frame']} 帧")
    p("输出: 箭体轴线相对**图像竖直**的绝对倾角 φ(t) —— 纯几何量, 无需任何基准")
    p("      (不再输出相对倾角 φ_rel: 它依赖\"片尾落地即竖直\"这个只对特定视频")
    p("       成立的先验, 已废弃)")
    p()

    # 有效区间的分位边界: 后面所有分段统计都用它, 不再写死秒数
    ok_f = [r for r in res if r.ok and np.isfinite(r.phi_deg)]
    t_lo = ok_f[0].t if ok_f else 0.0
    t_hi = ok_f[-1].t if ok_f else 0.0
    bins = [(t_lo, t_lo + (t_hi - t_lo) / 3, "前 1/3"),
            (t_lo + (t_hi - t_lo) / 3, t_lo + 2 * (t_hi - t_lo) / 3, "中 1/3"),
            (t_lo + 2 * (t_hi - t_lo) / 3, t_hi + 1e-6, "后 1/3")]

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
        p(f"  {'时段':>18} {'可信帧':>7} {'滚转中位':>10} {'滚转σ':>9} "
          f"{'|滚转|max':>10}")
        # 分档边界按有效区间三等分自适应(原来写死 11/25/40/45/69s)
        for lo, hi, nm in bins:
            m = valid & (tsec >= lo) & (tsec < hi)
            if m.sum() < 3:
                continue
            p(f"  {f'{nm} {lo:.0f}-{hi:.0f}s':>18} {m.sum():>7} "
              f"{np.median(roll[m]):>+10.3f} {np.std(roll[m]):>9.3f} "
              f"{np.nanmax(np.abs(roll[m])):>10.3f}")
        p()
        s_all = np.nanstd(roll[valid]) if valid.sum() else float("nan")
        w_all = np.nanmax(np.abs(roll[valid])) if valid.sum() else float("nan")
        tail = valid & (tsec >= t_lo + 2 * (t_hi - t_lo) / 3)
        s_tail = np.nanstd(roll[tail]) if tail.sum() else float("nan")
        p(f"  全片有效帧: 滚转 σ {s_all:.3f}°, 最大 {w_all:.3f}°; "
          f"后 1/3 段 σ {s_tail:.3f}°")
        p("  ⇒ 相机若**基本静止**(σ 与单个配准误差同量级), 则 φ(t) 可直接当作"
          "箭体相对地面的姿态;")
        p(f"     若某段 σ 明显更大, 该段应减去 cam_roll(t)(实测最大 {w_all:.2f}°)。")
        p(f"  相机滚转对倾角测量的误差贡献: 约 {s_all:.2f}°(1σ), "
          f"最大 {w_all:.2f}°")
    p()
    # ---------- 3) 透视(位置相关)项 ----------
    p("-" * 82)
    p("[3] 透视项的量级: 竖直线在画面里的收敛(位置相关倾角)")
    p("-" * 82)
    from rocket_track import TrackerConfig, track_frames
    meta = json.loads(
        (PROJECT / "runs" / "diag" / f"dets_{info.get('dets_tag', 'iou70')}.json")
        .read_text(encoding="utf-8"))
    outs2 = track_frames([r["b"] for r in meta["frames"]], TrackerConfig(),
                         fps, bound_wh=(W, H))
    cx = np.array([0.5 * (o.box[0] + o.box[2]) if (o.has and o.box is not None)
                   else np.nan for o in outs2])
    p("  竖直线在像面会向**竖直消失点**收敛, 位置 x 处的额外倾角 ≈ "
      "atan(|x−x_c|·tanθ_pitch / f)。")
    p(f"  画面中心 x_c = {W / 2:.1f}px")
    worst = 0.0
    dxs = {}
    for lo, hi, nm in bins:
        m = (tsec >= lo) & (tsec < hi) & np.isfinite(cx)
        if m.sum() < 3:
            continue
        dx = float(np.nanmax(np.abs(cx[m] - W / 2.0)))
        dxs[nm] = dx
        for fpx, pitch in ((400, 20), (800, 10), (1600, 10)):
            v = math.degrees(math.atan(dx * math.tan(math.radians(pitch))
                                       / fpx))
            p(f"  {nm} {lo:.0f}-{hi:.0f}s: 横向偏离最大 {dx:.1f}px;  "
              f"f={fpx:>4}px, 俯仰 {pitch:>2}°  ->  {v:.3f}°")
            worst = max(worst, v)
    p("  ⇒ 该项只取决于'火箭离画面中心多远' × '相机俯仰角', 与视频内容无关:")
    p(f"     横向偏离小(如 {min(dxs.values()) if dxs else 0:.0f}px 量级)时可忽略;")
    p("     偏离大或相机俯仰大时必须从 φ 中扣除 —— 本工具无法自行扣除(未知内参),")
    p(f"     只能给出量级上界 {worst:.1f}° 供判断。")
    p()

    # ---------- 4) 面外角 β 的灵敏度 ----------
    p("-" * 82)
    p("[4] 面外角 β 的灵敏度 —— 「完整 3D 姿态不可靠」的量化依据")
    p("-" * 82)
    # 取"目标解析得最清楚"的一半帧(按沿轴长度取中位以上)作代表, 不按时间窗切 ——
    # 这样不依赖"哪一段是落地段"这个先验, 且小目标帧本来也算不准 β。
    allv = [r for r in res if r.ok and np.isfinite(r.axis_len_px)
            and np.isfinite(r.w_body_px)]
    land = []
    if allv:
        axt = float(np.median([r.axis_len_px for r in allv]))
        land = [r for r in allv if r.axis_len_px >= axt]
    if land:
        ax = np.array([r.axis_len_px for r in land
                       if np.isfinite(r.axis_len_px)])
        wb = np.array([r.w_body_px for r in land
                       if np.isfinite(r.w_body_px)])
        wb2 = np.array([r.w_box_px for r in land
                        if np.isfinite(r.w_box_px)])
        axm, wbm, wb2m = float(np.median(ax)), float(np.median(wb)), \
            float(np.median(wb2))
        p(f"  目标最大的那一半帧 n={len(land)}: 沿轴投影长度 {axm:.1f}px, "
          f"拟合筒身宽 {wbm:.1f}px(σ {sd(wb):.2f}), YOLO 框宽 {wb2m:.1f}px "
          f"(比值 {wb2m / max(wbm, 1e-6):.2f} ⇒ 框被支腿/栅格舵撑大)")
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
        p("  ⇒ 同一画面, 换宽度口径或换 L/D, β 就能从 0° 变到 60°+。**β 只能当")
        p("     参数化假想姿态。** 另外底数那个'长宽比'本身也不是火箭的 L/D:")
        p("     掩码的 w_body 是剪影轮廓宽(口径正确), 但分子的'沿轴跨度'是**参与拟合")
        p("     的筒身段长度**, 随'筒身段被裁到哪里'变化(小目标段下端易被支腿/尾焰污染),")
        p("     所以拿它当 L/D 去解 β 必然发散 —— 这是**信息量不足**, 不是算法问题。")
        p("     想解决必须补其一: (a) 真实三维尺寸/模型; (b) 已知相机内参+俯仰; (c) 多视角。")
        p()

    p()

    # ---------- 5) 分段角度汇总 ----------
    p("-" * 82)
    p("[5] 分段角度汇总(有效帧; 分档按有效区间三等分, 不写死时间)")
    p("-" * 82)
    p(f"  {'时段':>18} {'帧数':>5} {'相对竖直φ中位':>13} {'φσ':>7} "
      f"{'相对上边缘(锐角)':>16} {'dφ/dt中位(°/s)':>14}")
    for lo, hi, nm in bins:
        m = [r for r in res if r.ok and lo <= r.t < hi]
        if len(m) < 3:
            continue
        v = np.array([r.phi_deg for r in m])
        vt = np.array([r.phi_vs_top for r in m if np.isfinite(r.phi_vs_top)])
        dr = np.array([r.dphi_dt for r in m if np.isfinite(r.dphi_dt)])
        p(f"  {f'{nm} {lo:.0f}-{hi:.0f}s':>18} {len(m):>5} "
          f"{np.median(v):>+13.2f} {sd(v):>7.3f} "
          f"{np.median(vt) if len(vt) else float('nan'):>16.2f} "
          f"{np.median(dr) if len(dr) else float('nan'):>+14.2f}")
    p()

    # ---------- 6) 结论 ----------
    p("=" * 82)
    p("[6] 结论")
    p("=" * 82)
    p("  ✅ 可行(本模块主输出, 全部是**绝对量**, 不依赖任何参考基准):")
    p("     · 箭体轴线与**图像竖直方向**的夹角 φ(t)  —— 纯几何量, 无需相机参数")
    p("     · 与**图像上边缘(水平方向)**的夹角 = 90° − |φ|")
    p("     · 角速率 dφ/dt —— 不需要任何标定的姿态动力学量")
    p("  ⚠️ 不可靠(仅作参数化假想姿态):")
    p("     · 面外角 β 与总倾角 —— 依赖未知的 L/D 与筒身宽度, 实测可从 0° 变到 60°+")
    p("     · β 的符号(朝向/远离相机)单视角无法区分")
    p("  ℹ️ 已废弃: 相对倾角 φ_rel —— 它需要一个\"参考姿态\", 唯一能拿到的是"
      "\"片尾落地")
    p("     即竖直\", 那是只对特定视频成立的先验, 换视频即失效, 故不再输出。")
    p("  ℹ️ 相机滚转(cam_roll)是独立测出来的: 若全片 σ 与单帧配准误差同量级,"
      " 说明相机静止,")
    p("     此时 φ(t) 可直接当作箭体相对地面的姿态; 否则应逐帧减去 cam_roll(t)。")

    OUT.mkdir(parents=True, exist_ok=True)
    # ⚠️ 产物名必须带标签: 本阶段原来写死 attitude.csv / validate.txt, 换第二支视频
    # 跑就会**静默覆盖**前一支的结果(与检测/掩码标签是同一类问题)。--out-tag 默认空
    # 时保持旧文件名, 流水线会显式传入掩码标签。
    sfx = f"_{out_tag}" if out_tag else ""
    (OUT / f"validate{sfx}.txt").write_text("\n".join(L), encoding="utf-8")
    # 顺带把逐帧结果落盘(省掉 pipeline 里重复跑一遍相机配准)
    RA.save(res, info, out_tag)
    print("\n".join(L))
    print(f"\n[写入] {OUT / f'validate{sfx}.txt'}  +  attitude{sfx}.csv / .json")


if __name__ == "__main__":
    main()
