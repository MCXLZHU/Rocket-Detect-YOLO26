# -*- coding: utf-8 -*-
"""在被泄漏污染的数据集上, 估算"真实泛化能力"。

背景
----
`scripts/check_leakage.py` 实测发现: valid 中约 66% 的图在 train 里有近重复,
其中 19.6% 像素完全相同。也就是官方 valid 上的 mAP 是偏乐观的。

做法
----
1. 用同样的感知哈希, 从 valid 中挑出**在 train 里没有近重复**的图片(干净子集);
2. 用硬链接把这批图 + 对应标签组织成一个临时数据集(不复制、不删原文件);
3. 用 best.pt 在该子集上跑一次 val, 得到"未见过画面"上的真实指标;
4. 同时在原始 valid 上跑一次作为对照。

用法:
    python scripts/eval_clean_subset.py
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(r"E:\RocketAttitudeEstimation")
ROOT = PROJECT / "Rocket Detect.v37i.yolo26"
WEIGHTS = PROJECT / "runs" / "rocket_yolo26s" / "weights" / "best.pt"
WORK = PROJECT / ".cache" / "clean_eval"

HASH_SIDE = 16
NEAR_DUP_HAMMING = 10


def list_jpgs(split: str) -> list[Path]:
    return sorted(p for p in (ROOT / split / "images").iterdir() if p.suffix.lower() == ".jpg")


def ahash(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (HASH_SIDE, HASH_SIDE), interpolation=cv2.INTER_AREA)
    return np.packbits((small > small.mean()).astype(np.uint8).ravel())


def hash_all(files: list[Path]) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=8) as ex:
        hs = list(ex.map(ahash, files))
    n = HASH_SIDE * HASH_SIDE // 8
    return np.stack([h if h is not None else np.zeros(n, np.uint8) for h in hs])


def min_dist(vh: np.ndarray, th: np.ndarray, chunk: int = 64) -> np.ndarray:
    best = np.full(len(vh), 999, np.int32)
    for s in range(0, len(vh), chunk):
        e = min(s + chunk, len(vh))
        xor = np.bitwise_xor(vh[s:e, None, :], th[None, :, :])
        best[s:e] = np.unpackbits(xor, axis=2).sum(axis=2).min(axis=1)
    return best


def main() -> None:
    print("枚举文件...", flush=True)
    tr = list_jpgs("train")
    va = list_jpgs("valid")
    print(f"train {len(tr)}, valid {len(va)}")

    print("感知哈希...", flush=True)
    th, vh = hash_all(tr), hash_all(va)

    print("计算 valid 每张图到 train 的最小距离...", flush=True)
    dist = min_dist(vh, th)

    clean = [va[i] for i in range(len(va)) if dist[i] > NEAR_DUP_HAMMING]
    print(f"干净子集: {len(clean)} / {len(va)} 张 "
          f"({len(clean) / len(va) * 100:.1f}%) 在 train 里无近重复\n", flush=True)

    # --- 组织临时数据集(硬链接, 不复制原文件) ---
    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    img_dir = WORK / "images"
    lbl_dir = WORK / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_lbl = 0
    for p in clean:
        os.link(p, img_dir / p.name)
        lbl = ROOT / "valid" / "labels" / (p.stem + ".txt")
        if lbl.exists():
            os.link(lbl, lbl_dir / lbl.name)
            n_lbl += 1
    print(f"已硬链接 {len(clean)} 图 + {n_lbl} 标签 -> {WORK}")

    yaml_path = WORK / "clean.yaml"
    yaml_path.write_text(
        f"path: {WORK.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 3\n"
        f"names: ['Engine Flames', 'Rocket Body', 'Space']\n",
        encoding="utf-8",
    )

    # --- 跑 val ---
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".cache" / "yolo"))
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))

    print("\n" + "=" * 60)
    print("[A] 干净子集(无近重复)上的指标 —— 接近真实泛化")
    print("=" * 60)
    r_clean = model.val(data=str(yaml_path), split="val", imgsz=640, batch=16,
                        workers=4, plots=False, verbose=True)

    print("\n" + "=" * 60)
    print("[B] 官方 valid 全量上的指标 —— 含泄漏, 偏乐观")
    print("=" * 60)
    r_full = model.val(data=str(PROJECT / "configs" / "rocket.yaml"), split="val",
                       imgsz=640, batch=16, workers=4, plots=False, verbose=True)

    def show(tag, r):
        m = r.box
        print(f"\n--- {tag} ---")
        print(f"  all            P={m.mp:.4f}  R={m.mr:.4f}  mAP50={m.map50:.4f}  mAP50-95={m.map:.4f}")
        for i, c in enumerate(r.names.values()):
            print(f"  {c:<14} P={m.p[i]:.4f}  R={m.r[i]:.4f}  "
                  f"mAP50={m.ap50[i]:.4f}  mAP50-95={m.ap[i]:.4f}")

    show("干净子集(真实泛化)", r_clean)
    show("官方 valid(参考)", r_full)

    print("\n=== 差异 ===")
    print(f"  mAP50    : {r_full.box.map50:.4f} -> {r_clean.box.map50:.4f} "
          f"({(r_clean.box.map50 - r_full.box.map50) * 100:+.1f} 个点)")
    print(f"  mAP50-95 : {r_full.box.map:.4f} -> {r_clean.box.map:.4f} "
          f"({(r_clean.box.map - r_full.box.map) * 100:+.1f} 个点)")
    print(f"\n临时数据集(可删): {WORK}")


if __name__ == "__main__":
    main()
