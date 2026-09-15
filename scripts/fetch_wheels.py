# -*- coding: utf-8 -*-
"""
多线程下载器 —— 用于从 PyTorch 镜像获取体积较大的 CUDA wheel。

背景: 本机工作网络经代理访问, download.pytorch.org 被拦截(502), 且单连接限速约 0.7 MB/s。
      国内镜像(mirrors.aliyun.com/pytorch-wheels)可访问, 通过多线程分块下载可显著提速。

用法:
    python scripts/fetch_wheels.py <URL> [<URL> ...] [-o 输出目录] [-t 线程数]
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import threading
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}


def remote_size(url: str) -> tuple[int, bool]:
    """返回 (大小, 是否支持 Range)"""
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        size = int(r.headers.get("Content-Length", 0))
        accept = (r.headers.get("Accept-Ranges") or "").lower()
    return size, ("bytes" in accept)


def download(url: str, out_dir: Path, threads: int = 8) -> Path:
    name = urllib.parse.unquote(url.rstrip("/").split("/")[-1])  # 还原 %2B -> +
    dst = out_dir / name
    size, ranges = remote_size(url)
    if size == 0:
        raise RuntimeError(f"无法获取文件大小: {url}")
    if dst.exists() and dst.stat().st_size == size:
        print(f"[跳过] 已存在且完整: {dst.name} ({size/1e9:.2f} GB)")
        return dst
    print(f"[下载] {name}  {size/1e9:.2f} GB   Range支持={ranges}   线程={threads}")

    if not ranges:
        threads = 1
    part = size // threads
    lock = threading.Lock()
    done = [0]
    errs: list[Exception] = []
    t0 = time.time()

    def worker(i: int) -> None:
        start = i * part
        end = size - 1 if i == threads - 1 else (start + part - 1)
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r, open(dst, "r+b") as f:
                    f.seek(start)
                    while True:
                        b = r.read(1024 * 256)
                        if not b:
                            break
                        f.write(b)
                        with lock:
                            done[0] += len(b)
                            if done[0] % (100 * 1024 * 1024) < 1024 * 256:
                                el = time.time() - t0
                                pct = done[0] / size * 100
                                print(f"   {pct:5.1f}%  {done[0]/1e9:.2f}/{size/1e9:.2f} GB"
                                      f"  {done[0]/1e6/el:.2f} MB/s", flush=True)
                return
            except Exception as e:  # 断点重试
                if attempt == 4:
                    with lock:
                        errs.append(e)
                else:
                    time.sleep(2 * (attempt + 1))

    # 预分配文件
    with open(dst, "wb") as f:
        f.truncate(size)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]

    if errs:
        raise RuntimeError(f"下载失败: {errs[0]}")

    got = dst.stat().st_size
    if got != size:
        raise RuntimeError(f"大小不符: {got} != {size}")
    el = time.time() - t0
    print(f"[完成] {dst.name}  {size/1e9:.2f} GB  用时 {el:.0f}s ({size/1e6/el:.2f} MB/s)")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("-o", "--out", default=r"E:\RocketAttitudeEstimation\.cache\wheels")
    ap.add_argument("-t", "--threads", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for u in a.urls:
        download(u, out, a.threads)


if __name__ == "__main__":
    main()
