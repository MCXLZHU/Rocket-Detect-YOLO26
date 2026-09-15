# -*- coding: utf-8 -*-
"""检查 train / valid 之间是否存在近重复图片(数据泄漏)。

动机
----
本数据集是 Roboflow 从火箭视频抽帧导出的, 文件名形如
    0001_png.rf.457fb41d52b3e36f467ac0ea63e96093.jpg
同一源帧会被导出成多个增强副本(不同 .rf.<hash>)。如果划分是"随机分帧",
那么 valid 里的某一帧, 在 train 里极可能存在肉眼几乎一致的相邻帧 ——
这种情况下验证集 mAP 会明显虚高, 不能代表真实泛化能力。

只用文件名猜测不可靠, 所以这里直接比像素: 把每张图压成 16x16 灰度做
感知哈希(aHash), 用 Hamming 距离找近重复。

用法:
    python scripts/check_leakage.py
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\RocketAttitudeEstimation\Rocket Detect.v37i.yolo26")
HASH_SIDE = 16                      # 16x16 -> 256 bit
NEAR_DUP_HAMMING = 10               # <=10 bit 差异(共256)视为近重复


def list_jpgs(split: str) -> list[Path]:
    d = ROOT / split / "images"
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".jpg")


def ahash(path: Path) -> np.ndarray | None:
    """返回 256 bit 的 aHash(已按 uint8 打包, 32 字节)。"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (HASH_SIDE, HASH_SIDE), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint8).ravel()   # 256 个 bit
    return np.packbits(bits)


def hash_all(files: list[Path], label: str) -> np.ndarray:
    out = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, h in enumerate(ex.map(ahash, files)):
            out.append(h if h is not None else np.zeros(HASH_SIDE * HASH_SIDE // 8, np.uint8))
            if (i + 1) % 5000 == 0:
                print(f"  {label}: {i + 1}/{len(files)}", flush=True)
    return np.stack(out)


def hamming_to_all(vh: np.ndarray, th: np.ndarray, chunk: int = 64) -> np.ndarray:
    """对每个 valid 哈希, 求它到所有 train 哈希的最小 Hamming 距离。"""
    best = np.full(len(vh), 999, dtype=np.int32)
    for s in range(0, len(vh), chunk):
        e = min(s + chunk, len(vh))
        # (c, 1, B) ^ (1, N, B) -> 统计 popcount
        xor = np.bitwise_xor(vh[s:e, None, :], th[None, :, :])
        bits = np.unpackbits(xor, axis=2)
        d = bits.sum(axis=2)
        best[s:e] = d.min(axis=1)
    return best


def main() -> None:
    tr_files = list_jpgs("train")
    va_files = list_jpgs("valid")
    print(f"train {len(tr_files)} 张, valid {len(va_files)} 张")
    print("计算感知哈希...", flush=True)
    th = hash_all(tr_files, "train")
    vh = hash_all(va_files, "valid")
    print("比对...", flush=True)
    dist = hamming_to_all(vh, th)

    total = len(dist)
    for thr in (0, 2, 4, 6, 8, 10):
        n = int((dist <= thr).sum())
        print(f"  Hamming <= {thr:>2} : {n:>5} / {total}  ({n / total * 100:5.1f}%)")

    near = int((dist <= NEAR_DUP_HAMMING).sum())
    print()
    print(f"结论: valid 中约 {near}/{total} ({near / total * 100:.1f}%) 的图像在 train 里"
          f"存在近重复(256bit 中差异 <= {NEAR_DUP_HAMMING})")
    print(f"      其中完全相同(Hamming=0)的有 {int((dist == 0).sum())} 张")
    print(f"      中位最小距离 = {int(np.median(dist))} bit")


if __name__ == "__main__":
    main()
