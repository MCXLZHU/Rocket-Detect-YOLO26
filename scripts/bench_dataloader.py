# -*- coding: utf-8 -*-
"""
DataLoader 纯加载吞吐基准测试（不训练、不占 GPU）。

用途: 判断训练瓶颈是否在数据加载上，并据此确定合适的 --workers。
原理: 只迭代 DataLoader 取 batch（含 mosaic/letterbox 等全部增强），
      记录每秒能产出多少张图。若该吞吐远高于训练时的 img/s，说明瓶颈在 GPU；
      若接近或低于训练时的 img/s，说明瓶颈在数据加载。

用法:
    python scripts/bench_dataloader.py                    # 默认测 workers 8/16/24
    python scripts/bench_dataloader.py --workers 8 24 --batches 40
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
for _sub in ("tmp", "pip", "torch", "mpl", "yolo"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "mpl")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "yolo")
os.environ.setdefault("PYTHONUTF8", "1")

from ultralytics.cfg import get_cfg                      # noqa: E402
from ultralytics.utils import DEFAULT_CFG_DICT           # noqa: E402
from ultralytics.data.utils import check_det_dataset     # noqa: E402
from ultralytics.data.build import build_yolo_dataset, build_dataloader  # noqa: E402


def build_args(data_yaml: str, imgsz: int, batch: int, cache: str):
    args = get_cfg(DEFAULT_CFG_DICT)
    args.data = data_yaml
    args.imgsz = imgsz
    args.batch = batch
    args.cache = cache
    args.task = "detect"
    args.fraction = 1.0
    args.single_cls = False
    args.classes = None
    args.rect = False
    return args


def _force_jpeg_path(ds) -> None:
    """把 npy 路径指向不存在的文件，强制走原始 JPEG 读取。

    ultralytics 的 load_image() 只要发现同名 .npy 存在就会优先 np.load，
    与 cache 参数无关。所以要做"无缓存"对照，只能把 npy_files 指到不存在的路径。
    这是只读的内存内改写，不会动磁盘上的任何文件。
    """
    from pathlib import Path as _P
    targets = ds.datasets if hasattr(ds, "datasets") else [ds]
    for d in targets:
        d.npy_files = [_P("__bench_missing__/x.npy")] * len(d.im_files)


def run_once(data, data_yaml, imgsz, batch, cache, workers, n_batches, use_npy=True):
    args = build_args(data_yaml, imgsz, batch, cache)
    ds = build_yolo_dataset(args, data["train"], batch, data, mode="train")
    if not use_npy:
        _force_jpeg_path(ds)
    dl = build_dataloader(ds, batch, workers, shuffle=True)
    it = iter(dl)
    # 预热 3 个 batch，让 worker 进程完成初始化
    for _ in range(3):
        next(it)
    t0 = time.time()
    n_img = 0
    for _ in range(n_batches):
        b = next(it)
        n_img += b["img"].shape[0]
    dt = time.time() - t0
    del it, dl
    return n_img / dt, dt, n_img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "rocket.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", nargs="+", type=int, default=[8, 16, 24])
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()

    data = check_det_dataset(a.data)
    print(f"train 图数 = {len(list(Path(data['train']).glob('*.jpg')))}", flush=True)

    npys = list(Path(data["train"]).glob("*.npy"))
    print(f"disk 缓存: {len(npys)} 个 .npy "
          f"({sum(f.stat().st_size for f in npys) / 1024 ** 3:.2f} GB)\n", flush=True)

    print(f"{'workers':>8} {'图片源':>10} {'批次/s':>9} {'图片/s':>10} {'相对':>7}")
    base = None
    for w in a.workers:
        for use_npy, label in ((True, "npy缓存"), (False, "原始JPEG")):
            ips, dt, n = run_once(data, a.data, a.imgsz, a.batch, "disk" if use_npy else False,
                                  w, a.batches, use_npy=use_npy)
            if base is None:
                base = ips
            print(f"{w:>8} {label:>10} {a.batches / dt:>9.1f} {ips:>10.1f} {ips / base:>6.2f}x",
                  flush=True)

    print("\n判读方法:")
    print("  若这里的「图片/s」明显高于训练日志里的 img/s，瓶颈在 GPU，加 workers 无益；")
    print("  若两者接近，瓶颈在数据加载，加大 workers 能直接缩短每轮时间。")


if __name__ == "__main__":
    main()
