# -*- coding: utf-8 -*-
"""读取 ultralytics 验证器内部的**原始**混淆矩阵(未做任何百分比归一化)。

ultralytics 画出的 confusion_matrix.png 做了归一化, 单看百分比容易误读。
这里直接拿 validator.metrics.confusion_matrix.matrix 的原始计数。

矩阵布局(ultralytics 约定):
    matrix[pred_class, true_class]
索引 0..nc-1 = 真实类别;  索引 nc = 背景
  * matrix[i, j]  (i<nc, j<nc) : 真值 j 被判成 i
  * matrix[nc, j]              : 真值 j 被漏检(判成背景)
  * matrix[i, nc]              : 背景被误报成 i
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT = Path(r"E:\RocketAttitudeEstimation")
WEIGHTS = PROJECT / "runs" / "rocket_yolo26s" / "weights" / "best.pt"


def main() -> None:
    os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".cache" / "yolo"))
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    r = model.val(data=str(PROJECT / "configs" / "rocket.yaml"), split="val",
                  imgsz=640, batch=16, workers=4, plots=True, verbose=False, conf=0.001, iou=0.7)

    cm = r.confusion_matrix
    m = cm.matrix
    names = list(r.names.values())

    print("=== 原始混淆矩阵(行=预测, 列=真值) ===")
    hdr = "".join(f"{n[:12]:>14}" for n in names)
    print(f"{'pred \\ true':<14}{hdr}{'背景(误报)':>14}")
    for i in range(m.shape[0]):
        rn = names[i] if i < len(names) else "背景(漏检)"
        row = "".join(f"{int(m[i, j]):>14}" for j in range(m.shape[1]))
        print(f"{rn:<14}{row}")

    print()
    print("=== 按真值列归一化(每个真实框的归宿) ===")
    print(f"{'pred \\ true':<14}{hdr}{'误报数':>10}")
    for j in range(len(names)):
        col = m[:, j].sum()
        cells = "".join(f"{m[i, j] / col * 100:>13.1f}%" for i in range(m.shape[0]))
        print(f"{names[j]:<14}{cells}{int(m[len(names), j]):>10}  (GT={int(col)})")

    print()
    print("=== 每类摘要 ===")
    K = len(names)
    for j in range(K):
        tot = m[:, j].sum()
        correct = m[j, j]
        missed = m[K, j]
        stolen = tot - correct - missed
        print(f"  {names[j]:<15} GT={int(tot):>5}  "
              f"正确={int(correct):>5} ({correct / tot * 100:5.1f}%)  "
              f"漏检={int(missed):>4} ({missed / tot * 100:4.1f}%)  "
              f"被其他类抢走={int(stolen):>4} ({stolen / tot * 100:4.1f}%)")
        for i in range(K):
            if i != j and m[i, j] > 0:
                print(f"        真值 {names[j]} 被误判为 {names[i]}: {int(m[i, j])}")
    print()
    print(f"  误报总数(背景被判成目标): {int(m[:K, K].sum())}")
    for i in range(K):
        print(f"    {names[i]}: {int(m[i, K])}")


if __name__ == "__main__":
    main()
