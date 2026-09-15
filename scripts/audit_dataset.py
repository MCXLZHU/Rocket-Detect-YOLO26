# -*- coding: utf-8 -*-
"""
Rocket Detect v37i (YOLO26 export) 数据集体检脚本
只读，不修改任何原始文件。
"""
import os
import sys
import json
from collections import Counter, defaultdict

from PIL import Image

ROOT = r"E:\RocketAttitudeEstimation\Rocket Detect.v37i.yolo26"
SPLITS = ["train", "valid", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

report = {}
problems = []


def collect(split):
    img_dir = os.path.join(ROOT, split, "images")
    lbl_dir = os.path.join(ROOT, split, "labels")
    imgs = {}
    lbls = {}
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXT:
                imgs[stem] = f
    if os.path.isdir(lbl_dir):
        for f in os.listdir(lbl_dir):
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".txt":
                lbls[stem] = f
    return imgs, lbls


def audit_split(split):
    print(f"\n{'='*70}\n[{split}]\n{'='*70}", flush=True)
    imgs, lbls = collect(split)
    n_img, n_lbl = len(imgs), len(lbls)

    only_img = sorted(set(imgs) - set(lbls))
    only_lbl = sorted(set(lbls) - set(imgs))

    cls_counter = Counter()
    empty_labels = 0
    bad_lines = []
    out_of_range = []
    degenerate = []
    lines_total = 0
    boxes_per_img = []

    sizes = Counter()
    unreadable = []

    for i, stem in enumerate(sorted(imgs)):
        img_path = os.path.join(ROOT, split, "images", imgs[stem])
        # 1) 图像可读性 + 尺寸
        try:
            with Image.open(img_path) as im:
                sizes[im.size] += 1
        except Exception as e:
            unreadable.append((imgs[stem], str(e)))

        # 2) 标签
        if stem not in lbls:
            boxes_per_img.append(0)
            continue
        lp = os.path.join(ROOT, split, "labels", lbls[stem])
        with open(lp, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
        if len(lines) == 0:
            empty_labels += 1
            boxes_per_img.append(0)
            continue
        boxes_per_img.append(len(lines))
        for ln in lines:
            lines_total += 1
            parts = ln.split()
            if len(parts) != 5:
                if len(bad_lines) < 20:
                    bad_lines.append((lbls[stem], ln, f"字段数={len(parts)}"))
                continue
            try:
                c = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:])
            except ValueError:
                if len(bad_lines) < 20:
                    bad_lines.append((lbls[stem], ln, "非数值"))
                continue
            cls_counter[c] += 1
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                if len(out_of_range) < 20:
                    out_of_range.append((lbls[stem], ln))
            if w <= 0 or h <= 0:
                if len(degenerate) < 20:
                    degenerate.append((lbls[stem], ln))
        if (i + 1) % 8000 == 0:
            print(f"  ... 已处理 {i+1}/{n_img}", flush=True)

    res = {
        "images": n_img,
        "labels": n_lbl,
        "image_only_no_label": len(only_img),
        "label_only_no_image": len(only_lbl),
        "empty_label_files": empty_labels,
        "total_boxes": lines_total,
        "class_distribution": dict(sorted(cls_counter.items())),
        "bad_lines_sample": bad_lines[:10],
        "out_of_range_sample": out_of_range[:10],
        "degenerate_sample": degenerate[:10],
        "unreadable_images": unreadable[:10],
        "image_sizes_top": [[list(k), v] for k, v in sizes.most_common(12)],
        "num_distinct_sizes": len(sizes),
        "imgs_with_boxes": sum(1 for b in boxes_per_img if b > 0),
        "max_boxes_per_img": max(boxes_per_img) if boxes_per_img else 0,
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res, {"only_img": only_img, "only_lbl": only_lbl, "unreadable": unreadable,
                 "bad": bad_lines, "oor": out_of_range, "deg": degenerate}


def main():
    allres = {}
    for s in SPLITS:
        if not os.path.isdir(os.path.join(ROOT, s)):
            print(f"!! 缺少划分目录: {s}")
            continue
        r, extra = audit_split(s)
        allres[s] = r
        if extra["only_img"]:
            problems.append(f"{s}: {len(extra['only_img'])} 张图无标签, e.g. {extra['only_img'][:3]}")
        if extra["only_lbl"]:
            problems.append(f"{s}: {len(extra['only_lbl'])} 个标签无对应图, e.g. {extra['only_lbl'][:3]}")
        if extra["unreadable"]:
            problems.append(f"{s}: {len(extra['unreadable'])} 张图无法读取, e.g. {extra['unreadable'][:3]}")
        if extra["bad"]:
            problems.append(f"{s}: 格式错误标签 {len(extra['bad'])} 例")
        if extra["oor"]:
            problems.append(f"{s}: 坐标越界 {len(extra['oor'])} 例")
        if extra["deg"]:
            problems.append(f"{s}: 退化框(w/h<=0) {len(extra['deg'])} 例")

    # 全局类别统计
    total_cls = Counter()
    for s, r in allres.items():
        for k, v in r["class_distribution"].items():
            total_cls[int(k)] += v

    summary = {
        "root": ROOT,
        "splits": allres,
        "global_class_distribution": dict(sorted(total_cls.items())),
        "problems": problems,
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_audit_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\n" + "#" * 70)
    print("全局类别分布 (id: 框数):", dict(sorted(total_cls.items())))
    print("发现问题:", problems if problems else "无")
    print("报告已写入:", out)


if __name__ == "__main__":
    main()
