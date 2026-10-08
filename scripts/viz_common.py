# -*- coding: utf-8 -*-
"""可视化公共件: 中文字体回退链 + CSV/数值读取 + SAM 路线配色。

为什么单独拆出来: 这套东西原本长在 `compare_video.py`(对拍可视化)里, 结果
`sam_angle_viz.py`(成果件)反过来要 `from compare_video import ...` —— 成果件依赖
对拍件是方向错误的依赖。抽成公共模块后, 两个脚本各自独立。

配色是 BGR(cv2 顺序), 改名/改色值前先确认下游脚本的图例文字也跟着改 —— 
本项目踩过"图例写 orange 而 BGR 写成蓝多红少, 实际渲染成蓝色"的坑。
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

# ---- 字体: 中文必须走 PIL 渲染(cv2.putText 只支持 ASCII) ----
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

# ---- SAM 路线配色(BGR) ----
C_SIL = (255, 170, 60)      # 掩码剪影填充(淡蓝)
C_AXIS = (255, 0, 255)      # 拟合轴线(品红)
C_BODY = (0, 220, 255)      # 筒身段标记(黄)
C_BOX = (200, 200, 0)       # 稳定检出框(青)


def load_font(size: int):
    """按候选链找第一个可用的中文字体, 全失败则退回 PIL 内置(会显示方块)。"""
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def read_csv_map(path: Path) -> dict[int, dict]:
    """读逐帧 CSV → {frame: row dict}; 文件不存在返回空 dict(调用方按缺失处理)。"""
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8") as fh:
        return {int(r["frame"]): r for r in csv.DictReader(fh)}


def fnum(r: dict | None, k: str):
    """从 CSV 行里取浮点; 空串/"None"/NaN 一律返回 None(而不是 nan)。

    统一成 None 很重要: 下游大量用 `if v is None` 判断"这一帧没有值",
    返回 nan 会让判断失效。
    """
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
