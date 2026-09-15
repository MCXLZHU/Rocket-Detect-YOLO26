# -*- coding: utf-8 -*-
"""
预生成 ultralytics 的 disk 图片缓存（*.npy），供 `train.py --cache disk` 直接复用。

背景:
  本数据集是 2.4 万张 ~10KB 的小 JPEG，训练时每个样本(mosaic 要 4 张图)都要重新
  打开+解码 JPEG，是 DataLoader 的主要开销。cache='disk' 会把每张图解码后存成
  .npy（360x640x3 = 675KB），训练时直接 np.load，省掉反复解 JPEG。

行为:
  * 只在 <数据集>/<split>/images/ 下**新增** 同名 .npy 文件，不修改也不删除任何原始图片/标签
  * 已存在的 .npy 会跳过，可重复执行
  * 体积约 18 GB（train ~16.9GB + valid ~1.7GB），E 盘当前空闲 126 GB

用法:
    python scripts/build_cache.py                 # 缓存 train + valid
    python scripts/build_cache.py --splits train  # 只缓存 train
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# ---- 在 import ultralytics 之前，把缓存/临时目录重定向到工作区（与 train.py 一致）----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
for _sub in ("tmp", "pip", "torch", "mpl", "yolo"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMP"] = str(CACHE_DIR / "tmp")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "mpl")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "yolo")
os.environ.setdefault("PYTHONUTF8", "1")

from ultralytics.cfg import get_cfg                      # noqa: E402
from ultralytics.utils import DEFAULT_CFG_DICT           # noqa: E402
from ultralytics.data.utils import check_det_dataset     # noqa: E402
from ultralytics.data.build import build_yolo_dataset    # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "configs" / "rocket.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="预生成 disk 图片缓存(.npy)")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="数据集 yaml")
    ap.add_argument("--imgsz", type=int, default=640, help="占位参数，缓存与 imgsz 无关")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"],
                    help="要缓存的划分: train / valid / test")
    ap.add_argument("--batch", type=int, default=16, help="仅用于构造数据集的占位参数")
    a = ap.parse_args()

    args = get_cfg(DEFAULT_CFG_DICT)
    args.data = a.data
    args.imgsz = a.imgsz
    args.batch = a.batch
    args.cache = "disk"          # 关键
    args.task = "detect"
    args.fraction = 1.0
    args.single_cls = False
    args.classes = None
    args.rect = False

    data = check_det_dataset(a.data)
    print(f"数据集根目录: {data['path']}", flush=True)

    for split in a.splits:
        key = "val" if split in ("val", "valid") else split
        if key not in data:
            print(f"[跳过] 配置里没有 '{split}'", flush=True)
            continue
        mode = "train" if key == "train" else "val"
        print(f"\n=== 缓存 {split} -> {data[key]} ===", flush=True)
        build_yolo_dataset(args, data[key], args.batch, data, mode=mode)
        print(f"    {split} 缓存完成", flush=True)

    # ---- 统计结果 ----
    print("\n" + "=" * 60)
    for split in a.splits:
        key = "val" if split in ("val", "valid") else split
        if key not in data:
            continue
        img_dir = Path(data[key])
        npys = list(img_dir.glob("*.npy"))
        total = sum(f.stat().st_size for f in npys)
        print(f"  {split:<6} .npy {len(npys):>6} 个, 合计 {total / 1024 ** 3:.2f} GB  ({img_dir})")
    print("=" * 60)
    print("缓存已就绪。正式训练时加 --cache disk 即可自动复用。")


if __name__ == "__main__":
    main()
