# -*- coding: utf-8 -*-
"""
(可选工具) 构建"无泄漏"数据划分。

背景:
  Rocket Detect v37i 是 Roboflow 导出的增强数据集 —— 每张原图被随机旋转生成约 3 个版本,
  文件名形如  0001_png.rf.<hash>.jpg , 其中 "0001_png" 就是原图(源)标识。
  体检发现: 10018 个源图中, 有 472 个(4.7%)的不同增强版本被分到了不同划分里,
  即验证/测试集里存在与训练集"同源"的近似重复图 -> 指标会偏乐观。

本脚本:
  * 以"源图"为单位重新划分, 保证同一源图的所有增强版本全部落在同一个划分内;
  * 用硬链接(hardlink)方式生成新划分, 不复制数据、不额外占用磁盘;
  * 只读原始数据集, 不做任何修改或删除;
  * 输出到  <工作区>\data\leakfree\  并在 configs 下写出对应 yaml。

用法:
    python scripts/build_leakfree_split.py                 # 默认 8:1:1
    python scripts/build_leakfree_split.py --ratios 0.8 0.1 0.1 --seed 0
    python scripts/build_leakfree_split.py --copy          # 用复制代替硬链接
"""

from __future__ import annotations

import os
import sys
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(r"E:\RocketAttitudeEstimation")
SRC_ROOT = PROJECT_ROOT / "Rocket Detect.v37i.yolo26"
DST_ROOT = PROJECT_ROOT / "data" / "leakfree"
OUT_YAML = PROJECT_ROOT / "configs" / "rocket_leakfree.yaml"
SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def source_of(filename: str) -> str:
    """0001_png.rf.<hash>.jpg -> 0001_png"""
    return filename.split(".rf.")[0]


def collect_sources():
    """返回 {source_id: [(img_path, lbl_path_or_None), ...]} (只读扫描)"""
    sources = defaultdict(list)
    for split in SPLITS:
        img_dir = SRC_ROOT / split / "images"
        lbl_dir = SRC_ROOT / split / "labels"
        if not img_dir.is_dir():
            continue
        for f in os.listdir(img_dir):
            p = Path(f)
            if p.suffix.lower() not in IMG_EXT:
                continue
            lbl = lbl_dir / (p.stem + ".txt")
            sources[source_of(f)].append((img_dir / f, lbl if lbl.exists() else None))
    return sources


def link_or_copy(src: Path, dst: Path, use_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)          # 同盘硬链接: 几乎不占空间
    except OSError:
        shutil.copy2(src, dst)     # 跨盘/不支持硬链接时退化为复制


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="用复制代替硬链接")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不写文件")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit("比例之和必须为 1.0")

    sources = collect_sources()
    keys = sorted(sources.keys())
    random.Random(args.seed).shuffle(keys)

    n = len(keys)
    n_tr = int(n * args.ratios[0])
    n_va = int(n * args.ratios[1])
    buckets = {
        "train": keys[:n_tr],
        "valid": keys[n_tr:n_tr + n_va],
        "test":  keys[n_tr + n_va:],
    }

    print(f"源图总数: {n}")
    for k, v in buckets.items():
        imgs = sum(len(sources[s]) for s in v)
        print(f"  {k:<5}: {len(v):5d} 源  ->  {imgs:6d} 张图")
    if args.dry_run:
        print("(--dry-run, 未写入任何文件)")
        return

    total_written = 0
    for split, srcs in buckets.items():
        for s in srcs:
            for img, lbl in sources[s]:
                link_or_copy(img, DST_ROOT / split / "images" / img.name, args.copy)
                if lbl is not None:
                    link_or_copy(lbl, DST_ROOT / split / "labels" / lbl.name, args.copy)
                total_written += 1
        # 空标签也需要占位(背景图)
    print(f"已生成 {total_written} 张图 -> {DST_ROOT}")

    yaml_text = (
        "# 由 scripts/build_leakfree_split.py 自动生成 —— 以源图为单位重新划分, 无跨划分泄漏\n"
        f'path: "{DST_ROOT.as_posix()}"\n'
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 3\n"
        "names: ['Engine Flames', 'Rocket Body', 'Space']\n"
    )
    OUT_YAML.write_text(yaml_text, encoding="utf-8")
    print(f"数据配置已写入: {OUT_YAML}")
    print(f"\n训练时改用:  python train.py --data \"{OUT_YAML}\"")


if __name__ == "__main__":
    main()
