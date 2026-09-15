# -*- coding: utf-8 -*-
"""_bar_score 形状自检"""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))
import rocket_angle as RA

for ncol in (150, 192, 376, 400):
    patch = np.random.rand(20, ncol).astype(np.float32)
    for w in (10, 31, 57, 123, 150):
        try:
            S = RA._bar_score(patch, w)
            print(f"ncol={ncol:4d} w={w:4d} -> S.shape={S.shape} "
                  f"(期望列数 {ncol - w - 2 * max(2, int(round(0.55 * w)))} +1)")
        except Exception as e:
            print(f"ncol={ncol:4d} w={w:4d} -> ERROR {type(e).__name__}: {e}")
