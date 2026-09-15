# -*- coding: utf-8 -*-
"""Rocket Body 检测的时序稳定模块(单目标 / 因果, 可用于实时)。

解决三件事:
  P1 漏检       —— 双阈值关联(高分建轨, 低分续轨) + 卡尔曼预测补位
  P2 嵌套框     —— 类内 IoS 抑制(普通 NMS 的 IoU 对"完全嵌套"的框失效)
  P3 框底边跳变 —— 因果中值 + 底边"模式状态机"(见下)

=============================== 关键实测结论 ===============================
在本项目视频(852x480 / 30fps / 朱雀3号再入回收)上逐帧量化后发现:

1) 框的**上边沿 y1 极稳**: 落地静止段(42-68s) σ=0.46px, 极差仅 2px。
   下边沿 y2 不稳: σ=3.47px, 且存在 9 帧 |Δy2|>30px 的**模式跳变**
   (帧 1160->1161 一次跳 +147px)。

2) 也就是说, 检测框"该不该把下半段(发动机/支腿/尾焰区)算进去"是**双稳**的,
   不是抖动。对倾角估计(第 3 步)而言, 下边沿几乎没有信息量 —— 箭体是直筒,
   方向只由上边沿与两侧轮廓决定。

    ⇒ 因此状态量取 **上边沿 ytop + 宽 w + 高 h**, 并把 h 交给一个
      "持续性门控的模式状态机", 连续 K 帧同向才认账, 从而:
        * 单帧孤立跳变被拒(不会污染速度)
        * 真实模式切换在 K 帧后干净跟上(不会两模式取平均)
    ⇒ 这一步替代了"检测到跳变就放大测量噪声 R"的做法。后者会让滤波器
      在两个模式之间取平均, 得到一个两边都不像的框(实测帧 1160-1170
      框底边漂到 y2=389, 而上边沿 y1 从 54 漂到 224, 完全跑偏)。

3) ROI 用**最近 K 帧输出框的并集**再外扩, 而不是单帧框:
   宁可多带背景也绝不裁掉箭体。边缘检测本身会剔除背景。

=============================== 与需求先验的对应 ===========================
  * 画面只有一个火箭  -> 不做多目标关联, 不需要匈牙利匹配, 逐帧 O(1)
  * 只依赖过去帧      -> 全程因果滤波, 无任何后向平滑, 可实时
  * 检测与决策解耦    -> 检测只跑一次低阈值(0.05), 所有阈值/逻辑在下游,
                        调参不必重新推理
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

RB, EF, SP = 1, 0, 2


# ==========================================================================
# 基础几何
# ==========================================================================
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def ios_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """交集 / 较小框面积。嵌套框的 IoU 会随面积比一起变小(本视频实测 0.643),
    所以判断"是否嵌套重复"必须用 IoS, 不能用 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    small = min(aa, ba)
    return float(inter / small) if small > 0 else 0.0


def union_box(boxes: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.stack(boxes)
    return np.array([arr[:, 0].min(), arr[:, 1].min(),
                     arr[:, 2].max(), arr[:, 3].max()])


@dataclass
class Det:
    cls: int
    score: float
    box: np.ndarray          # xyxy

    @staticmethod
    def from_dict(d: dict) -> "Det":
        return Det(int(d["c"]), float(d["s"]), np.asarray(d["x"], float))


# ==========================================================================
# P2: 类内嵌套框抑制
# ==========================================================================
def suppress_nested(dets: Sequence[Det], cls: int = RB, ios_thr: float = 0.85,
                    iou_cap: float = 0.95, keep: str = "score",
                    anchor: np.ndarray | None = None) -> list[Det]:
    """同类内, 若 IoS>=ios_thr 且 IoU<iou_cap, 判为"同一目标被重复框住"。

    实测: 本视频 47 帧出现嵌套对, IoS 中位 1.000(完全包含), IoU 只有 0.643-0.699
    —— 恰好卡在 ultralytics 默认 NMS 阈值 0.7 之下, 所以 NMS 抓不到它。
    把 iou 阈值降到 0.3 也能消掉全部 47 帧(已实测), 但那是全局改动, 会影响
    其它类别; 这里用类内 IoS 判定, 只针对"同一类被包含"的情形, 更可控。

    keep: score(分数高者, 原方案) / larger(面积大者) / anchor(与预测框更接近者)
    只对同类生效 —— Rocket Body 与 Engine Flames 是上下相邻的两个部件, 不该合并。
    """
    same = [d for d in dets if d.cls == cls]
    other = [d for d in dets if d.cls != cls]
    same = sorted(same, key=lambda d: -d.score)
    kept: list[Det] = []
    for d in same:
        dup = next((k for k, e in enumerate(kept)
                    if ios_xyxy(d.box, e.box) >= ios_thr
                    and iou_xyxy(d.box, e.box) < iou_cap), None)
        if dup is None:
            kept.append(d)
            continue
        e = kept[dup]
        if keep == "larger":
            ad = (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
            ae = (e.box[2] - e.box[0]) * (e.box[3] - e.box[1])
            if ad > ae:
                kept[dup] = d
        elif keep == "anchor" and anchor is not None:
            if iou_xyxy(d.box, anchor) > iou_xyxy(e.box, anchor):
                kept[dup] = d
        # keep == "score": same 已按分数降序, 直接丢弃 d
    return kept + other


# ==========================================================================
# 底边"模式状态机": 只认连续 K 帧一致的变化, 提升抗跳变能力
# ==========================================================================
class _HeightMode:
    """把"检测框高度"看成双稳量: 当前模式 h_mode, 另有待定模式 h_pend。

    |h_meas - h_mode| <= tol  -> 正常, 缓慢跟随
    否则                      -> 记为待定; 连续 persist 帧都指向同一个待定值才切换
    """

    def __init__(self, h0: float, tol: float = 0.22, tol_min: float = 3.0,
                 persist: int = 3, follow: float = 0.5):
        self.h_mode = float(h0)
        self.h_pend: float | None = None
        self.n_pend = 0
        self.tol, self.tol_min = tol, tol_min
        self.persist, self.follow = persist, follow
        self.switched = False

    def _tol(self) -> float:
        return max(self.tol * self.h_mode, self.tol_min)

    def step(self, h_meas: float) -> tuple[float, bool]:
        self.switched = False
        t = self._tol()
        if abs(h_meas - self.h_mode) <= t:
            self.h_mode += self.follow * (h_meas - self.h_mode)
            self.h_pend, self.n_pend = None, 0
        else:
            if self.h_pend is not None and abs(h_meas - self.h_pend) <= t:
                self.n_pend += 1
                self.h_pend += self.follow * (h_meas - self.h_pend)
            else:
                self.h_pend, self.n_pend = h_meas, 1
            if self.n_pend >= self.persist:
                self.h_mode, self.h_pend, self.n_pend = self.h_pend, None, 0
                self.switched = True
        return self.h_mode, self.switched


# ==========================================================================
# 卡尔曼滤波: 状态 [cx, ytop, w, h, vcx, vytop, vw, vh]
#   注意状态量是**上边沿 ytop** 而不是中心 cy —— 上边沿实测 σ=0.46px,
#   是整条链路唯一可靠的锚点; 用 cy 会让底边跳变直接污染位置。
# ==========================================================================
class _KF:
    __slots__ = ("x", "P", "F", "H", "Q", "R", "I")

    def __init__(self, box: np.ndarray, r_pos: float = 1.5, r_size: float = 2.5,
                 q_acc: float = 0.4):
        self.x = np.array([(box[0] + box[2]) / 2, box[1],
                           box[2] - box[0], box[3] - box[1],
                           0.0, 0.0, 0.0, 0.0])
        self.P = np.diag([r_pos * 4, r_pos * 4, r_size * 4, r_size * 4,
                          100.0, 100.0, 100.0, 100.0]).astype(float)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.hstack([np.eye(4), np.zeros((4, 4))])
        self.Q = np.zeros((8, 8))
        for i, s in enumerate((q_acc, q_acc, q_acc * 0.5, q_acc * 0.5)):
            self.Q[i, i] = 0.25 * s ** 2
            self.Q[i + 4, i + 4] = s ** 2
            self.Q[i, i + 4] = self.Q[i + 4, i] = 0.5 * s ** 2
        self.R = np.diag([r_pos ** 2, r_pos ** 2, r_size ** 2, r_size ** 2])
        self.I = np.eye(8)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    @staticmethod
    def meas(box: np.ndarray) -> np.ndarray:
        return np.array([(box[0] + box[2]) / 2, box[1],
                         box[2] - box[0], box[3] - box[1]])

    def update(self, box: np.ndarray, v_clamp=(6.0, 2.0),
               v_damp: float = 1.0) -> float:
        z = self.meas(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 0.0
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        for idx in (4, 5, 6, 7):
            self.x[idx] *= v_damp
        for idx, lim in ((4, v_clamp[0]), (5, v_clamp[0]),
                         (6, v_clamp[1]), (7, v_clamp[1])):
            self.x[idx] = float(np.clip(self.x[idx], -lim, lim))
        try:
            return float(np.sqrt(y @ np.linalg.inv(S) @ y))
        except np.linalg.LinAlgError:
            return 0.0

    def box(self) -> np.ndarray:
        cx, ytop, w, h = self.x[:4]
        w, h = max(w, 2.0), max(h, 2.0)
        return np.array([cx - w / 2, ytop, cx + w / 2, ytop + h])


# ==========================================================================
# 配置 / 输出
# ==========================================================================
@dataclass
class TrackerConfig:
    conf_high: float = 0.25        # 建轨 / 高质量更新
    conf_low: float = 0.10         # 续轨(关键: 实测漏检帧分数落在 0.105~0.273)
    min_side_track: float = 6.0    # 建轨最小短边(过滤噪点)
    gate_iou: float = 0.10         # 关联门限: 与预测框的 IoU
    gate_maha: float = 30.0        # 马氏距离门限(4 维, 约卡方 99.9%)
    max_coast: int = 20            # 最长连续预测帧数(20 帧 = 0.67s @30fps)
    median_win: int = 3            # 测量因果中值窗(去孤立跳变), 1=关闭
    # -- 底边模式状态机 --
    mode_tol: float = 0.22         # 相对容差
    mode_tol_min: float = 3.0      # 绝对容差下限(px)
    mode_persist: int = 3          # 连续几帧一致才切换模式
    # -- 测角门控 --
    min_side_angle: float = 20.0   # 最小短边(实测 11s 起稳定达标)
    min_slant_angle: float = 1.5   # 最小长宽比
    # -- ROI --
    roi_win: int = 5               # 取最近 K 帧输出框的并集
    roi_pad_ratio: float = 0.08
    roi_pad_min: float = 5.0
    # -- 嵌套抑制 --
    sup_ios: float = 0.85
    # 嵌套对里保留"更完整的那个"而不是"分数高的那个"。实测依据:
    #   keep=larger -> 底边模式切换 2 帧, |Δy2|max 27, ROI 漏裁 0
    #   keep=score  -> 底边模式切换 6 帧, |Δy2|max 57, ROI 漏裁 7
    # 且嵌套对的分差中位只有 0.12, 按分数排序本身就不稳。
    sup_keep: str = "larger"


@dataclass
class TrackOut:
    frame: int
    has: bool = False
    state: str = "LOST"              # TENTATIVE / OBSERVED / COASTED / LOST
    box: np.ndarray | None = None    # 稳定后的 RB 框(y2 = ytop + h_mode)
    raw_box: np.ndarray | None = None
    score: float = 0.0
    n_obs: int = 0
    coast_len: int = 0
    h_mode: float = 0.0
    mode_switched: bool = False      # 本帧发生了底边模式切换
    roi: np.ndarray | None = None    # 给第 3 步的边缘检测 ROI(已外扩, 不裁目标)
    angle_ok: bool = False
    reliability: float = 0.0
    n_dup_removed: int = 0
    n_cand: int = 0


# ==========================================================================
# 主跟踪器
# ==========================================================================
class RocketTracker:
    def __init__(self, cfg: TrackerConfig | None = None,
                 bound_wh: tuple[int, int] | None = None):
        self.cfg = cfg or TrackerConfig()
        self.bound_wh = bound_wh
        self.kf: _KF | None = None
        self.hm: _HeightMode | None = None
        self.coast = 0
        self.n_obs = 0
        self._med: list[np.ndarray] = []
        self._hist: deque[np.ndarray] = deque(maxlen=self.cfg.roi_win)

    # ---------- 工具 ----------
    def _median(self, box: np.ndarray) -> np.ndarray:
        self._med.append(box)
        if len(self._med) > self.cfg.median_win:
            self._med.pop(0)
        return box if len(self._med) == 1 else np.median(np.stack(self._med), axis=0)

    def _reset(self, box: np.ndarray) -> None:
        self._med = [box]

    def _clip(self, b: np.ndarray) -> np.ndarray:
        if self.bound_wh is None:
            return b
        W, H = self.bound_wh
        b = np.array([max(0.0, b[0]), max(0.0, b[1]),
                      min(float(W), b[2]), min(float(H), b[3])])
        if b[2] - b[0] < 2.0:
            b[2] = min(float(W), b[0] + 2.0)
        if b[3] - b[1] < 2.0:
            b[3] = min(float(H), b[1] + 2.0)
        return b

    # ---------- 主循环 ----------
    def step(self, frame: int, dets: Sequence[Det], fps: float = 30.0) -> TrackOut:
        c = self.cfg
        out = TrackOut(frame=frame)

        # 1) P2 嵌套框抑制
        pred = self.kf.box() if self.kf is not None else None
        n_before = sum(1 for d in dets if d.cls == RB)
        cands = suppress_nested(list(dets), RB, c.sup_ios, keep=c.sup_keep,
                                anchor=pred)
        rb = sorted([d for d in cands if d.cls == RB], key=lambda d: -d.score)
        out.n_dup_removed = n_before - len(rb)
        out.n_cand = len(rb)

        # 2) 建轨
        if self.kf is None:
            pick = next((d for d in rb
                         if d.score >= c.conf_high
                         and min(d.box[2] - d.box[0],
                                 d.box[3] - d.box[1]) >= c.min_side_track), None)
            if pick is None:
                return out
            self.kf = _KF(pick.box)
            self.hm = _HeightMode(pick.box[3] - pick.box[1], c.mode_tol,
                                  c.mode_tol_min, c.mode_persist)
            self.kf.predict()
            self.coast, self.n_obs = 0, 1
            self._reset(pick.box)
            self._hist.clear()
            self._hist.append(self._clip(pick.box))
            out.has, out.state, out.score = True, "TENTATIVE", pick.score
            out.raw_box = pick.box
            out.h_mode = self.hm.h_mode
            out.n_obs = self.n_obs
            out.box = self._clip(self.kf.box())
            self._finish(out)
            return out

        # 3) 双阈值关联
        self.kf.predict()
        pred = self.kf.box()
        matched, used = None, 0.0
        for thr in (c.conf_high, c.conf_low):
            best, best_m = None, None
            for d in rb:
                if d.score < thr or iou_xyxy(d.box, pred) < c.gate_iou:
                    continue
                m = self._maha(d.box)
                if best is None or m < best_m:
                    best, best_m = d, m
            if best is not None and best_m <= c.gate_maha:
                matched, used = best, best.score
                break

        if matched is not None:
            sm = self._median(matched.box)
            h_mode, sw = self.hm.step(max(sm[3] - sm[1], 2.0))
            out.h_mode, out.mode_switched = h_mode, sw
            # 用"上边沿 + 模式化高度"重构测量, 底边跳变不进滤波器
            syn = np.array([sm[0], sm[1], sm[2], sm[1] + h_mode])
            self.kf.update(syn)
            self.coast = 0
            self.n_obs += 1
            out.has, out.state, out.score = True, "OBSERVED", used
            out.raw_box = matched.box
            # ROI 历史里同时放入"跟踪框"和"本帧被接受的原始框":
            # 跟踪框可能因模式锁定而暂时小于原始框, 而 ROI 的原则是"宁可大不可裁"。
            self._hist.append(self._clip(matched.box))
        else:
            # 4) 预测补位
            self.coast += 1
            self._reset(self.kf.box())
            if self.coast > c.max_coast:
                self.kf = self.hm = None
                self.coast, self.n_obs = 0, 0
                self._hist.clear()
                out.state = "LOST"
                return out
            out.has, out.state = True, "COASTED"
            out.h_mode = self.hm.h_mode
        out.box = self._clip(self.kf.box())
        out.n_obs, out.coast_len = self.n_obs, self.coast
        self._hist.append(out.box)
        self._finish(out)
        return out

    def _maha(self, box: np.ndarray) -> float:
        y = self.kf.meas(box) - self.kf.H @ self.kf.x
        S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R
        try:
            return float(np.sqrt(y @ np.linalg.inv(S) @ y))
        except np.linalg.LinAlgError:
            return 1e9

    def _finish(self, out: TrackOut) -> None:
        """计算 ROI 与测角门控。ROI 用最近 K 帧输出框的并集, 只保证"不裁掉目标"。"""
        c = self.cfg
        roi = union_box(self._hist)
        pad_x = max(c.roi_pad_min, (roi[2] - roi[0]) * c.roi_pad_ratio)
        pad_y = max(c.roi_pad_min, (roi[3] - roi[1]) * c.roi_pad_ratio)
        roi = np.array([roi[0] - pad_x, roi[1] - pad_y,
                        roi[2] + pad_x, roi[3] + pad_y])
        if self.bound_wh is not None:
            W, H = self.bound_wh
            roi = np.array([max(0.0, roi[0]), max(0.0, roi[1]),
                            min(float(W), roi[2]), min(float(H), roi[3])])
        out.roi = roi
        b = out.box
        w, h = b[2] - b[0], b[3] - b[1]
        out.angle_ok = bool(min(w, h) >= c.min_side_angle
                            and h / max(w, 1e-6) >= c.min_slant_angle
                            and out.state == "OBSERVED")
        if out.state == "OBSERVED":
            r = min(1.0, out.score / 0.5)
        elif out.state == "TENTATIVE":
            r = 0.5 * min(1.0, out.score / 0.5)
        else:
            r = 0.5 * max(0.0, 1.0 - out.coast_len / max(c.max_coast, 1))
        out.reliability = float(r * (1.0 if out.angle_ok else 0.3))


# ==========================================================================
# 便捷封装
# ==========================================================================
def track_frames(frames_dets: Sequence[Sequence[dict]],
                 cfg: TrackerConfig | None = None, fps: float = 30.0,
                 bound_wh: tuple[int, int] | None = None) -> list[TrackOut]:
    """frames_dets: 每帧一个 list[dict(c,s,x)], 来自低阈值推理的存盘结果。"""
    trk = RocketTracker(cfg, bound_wh)
    return [trk.step(i, [Det.from_dict(d) for d in ds], fps)
            for i, ds in enumerate(frames_dets)]
