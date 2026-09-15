# -*- coding: utf-8 -*-
"""扫描 SAM 路线相关脚本的结构: 行数 / 顶层 def+class 及其行号 / 关键常量。

用途: 给"从零读代码"的人一份准确的索引(不是靠记忆写, 而是从磁盘实读)。
"""

import ast
import sys
from pathlib import Path

ROOT = Path(r"E:\RocketAttitudeEstimation")
FILES = [
    "scripts/rocket_seg.py",
    "scripts/rocket_mask_angle.py",
    "scripts/validate_mask_angle.py",
    "scripts/bench_sam2.py",
    "scripts/compare_video.py",
    "scripts/_seg_diag.py",
    "scripts/_seg_probe.py",
    "scripts/_seg_vs_grad.py",
    "scripts/rocket_angle.py",
    "scripts/rocket_track.py",
    "pipeline.py",
]
SAM2 = [
    "third_party/sam2/sam2/sam2_video_predictor.py",
    "third_party/sam2/sam2/modeling/sam2_base.py",
    "third_party/sam2/sam2/sam2_image_predictor.py",
    "third_party/sam2/sam2/build_sam.py",
]


def outline(p: Path, maxdef: int = 40):
    src = p.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return len(lines), [f"(解析失败: {e})"]
    items = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append((node.lineno, "def " + node.name))
        elif isinstance(node, ast.ClassDef):
            items.append((node.lineno, "class " + node.name))
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    items.append((sub.lineno, "    ." + sub.name))
    items.sort()
    return len(lines), [f"{ln:>5}  {s}" for ln, s in items[:maxdef]]


def main():
    out = []
    for rel in FILES + SAM2:
        p = ROOT / rel
        if not p.exists():
            out.append(f"\n{'=' * 70}\n{rel}   ** 不存在 **")
            continue
        n, items = outline(p)
        out.append(f"\n{'=' * 70}\n{rel}   ({n} 行)\n" + "\n".join(items))
    txt = "\n".join(out)
    (ROOT / ".cache" / "code_tour.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
