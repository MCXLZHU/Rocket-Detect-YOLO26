# -*- coding: utf-8 -*-
"""视频路径解析 —— 让流程不依赖"视频必须放在项目根目录"。

背景(踩过的坑): detect 阶段原本把 `video.name`(只有文件名)写进 dets meta, 下游一律
`PROJECT / vm["video"]` 拼路径。于是换视频时 `--video D:\\videos\\r2.mp4` 能跑完 detect,
但 stabilize/seg/angle/attitude 全部 `FileNotFoundError` —— 文件名对, 目录在项目外。

约定:
  * meta 里 `video` 字段继续放**显示用**的文件名(兼容旧产物);
  * 新增 `video_path` 字段, 放"能直接打开的路径": 在项目内就存相对路径, 在项目外存绝对路径;
  * 下游统一用 `resolve_video(vm)`, 三种情况都能处理:
      1. 只有 video(旧产物)        -> PROJECT / video
      2. 有 video_path 相对路径     -> PROJECT / video_path
      3. 有 video_path 绝对路径     -> 原样使用
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

VIDEO_EXT = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def video_field(p: Path) -> str:
    """生成写进 meta 的 `video_path` 值: 项目内用相对路径, 项目外用绝对路径。

    项目内用相对路径是为了**仓库可复现**(把整棵工作区拷到别的机器上路径依然成立)。
    """
    p = Path(p).resolve()
    try:
        return str(p.relative_to(PROJECT))
    except ValueError:
        return str(p)


def resolve_video(vm: dict) -> Path:
    """从 dets meta 的 `meta` 字典里拿到视频路径(不存在则抛错并给出提示)。"""
    raw = vm.get("video_path") or vm.get("video")
    if not raw:
        raise KeyError("dets meta 里既没有 video_path 也没有 video 字段")
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT / p
    if not p.exists():
        # 旧产物只存了文件名, 而视频其实在别处 -> 给出可操作的报错, 别只抛路径
        raise FileNotFoundError(
            f"找不到视频 {p}\n"
            f"  提示: 旧产物只记了文件名, 若视频不在项目根目录, 请重跑 detect:\n"
            f"        pipeline.py --video <视频的完整路径> --tag <新标签>")
    return p


def video_key(vm_or_path, tag: str = "") -> str:
    """给"某个视频的某次运行"生成一个短标识, 用作帧缓存目录名。

    ⚠️ 为什么必须要有它: 帧缓存原本只按**帧号区间**命名(如 `c00000_00200`),
    换视频后帧号区间完全一样 ⇒ **直接复用了上一个视频的帧**。实测后果: 新视频
    640x360, 却拿 852x480 的老帧去做分割, 掩码长度 480 与画面高 360 对不上,
    直接 IndexError 崩掉; 更隐蔽的情况是尺寸恰好一致, 那就静默用错画面。
    key 里带路径 + 文件大小 + 分辨率 + 帧数, 任一变化都会换目录。
    """
    vm = vm_or_path if isinstance(vm_or_path, dict) else {}
    raw = (vm.get("video_path") or vm.get("video")) if vm else str(vm_or_path)
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT / p
    try:
        sig = (f"{p.resolve()}|{p.stat().st_size}|{vm.get('W')}x{vm.get('H')}|"
               f"{vm.get('frames') or vm.get('n_frame') or ''}")
    except OSError:                      # 文件不存在时也别崩, 退回只按路径
        sig = f"{p}|missing"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]
    sfx = f"_{tag}" if tag else ""
    return f"{p.stem[:24]}_{h}{sfx}"


def find_videos(root: Path | None = None) -> list[Path]:
    """列出候选视频(按文件大小降序)。默认扫项目根目录。"""
    root = root or PROJECT
    cands: list[Path] = []
    for ext in VIDEO_EXT:
        cands += list(root.glob(f"*{ext}"))
    return sorted(cands, key=lambda p: p.stat().st_size, reverse=True)


def pick_video(explicit: str | None = None, root: Path | None = None) -> Path:
    """显式给的路径优先; 否则取根目录下最大的一个; 都没有则报错。"""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"指定的视频不存在: {p}")
        return p
    cands = find_videos(root)
    if not cands:
        raise FileNotFoundError(
            f"在 {root or PROJECT} 下没找到视频"
            f"({'/'.join(e.lstrip('.') for e in VIDEO_EXT)}); "
            f"用 --video 指定完整路径")
    return cands[0]


def open_cap(vm: dict):
    """按 meta 打开视频, 返回 cv2.VideoCapture(已确认能打开)。"""
    import cv2
    p = resolve_video(vm)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 打不开视频: {p}")
    return cap
