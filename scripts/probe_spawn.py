# -*- coding: utf-8 -*-
"""探针: 测试 Windows spawn 管道能传输多大的对象。

背景: 训练时报
    OSError: [Errno 22] Invalid argument
    _pickle.UnpicklingError: pickle data was truncated
这是父进程向子进程的 spawn 管道写 pickle 失败。需要用二分法找出
本机上管道能承受的大小上限, 才能判断 YOLODataset(约 12 MB) 是否能通过。
"""

import multiprocessing as mp
import sys


def child(payload):
    n = len(payload) if payload is not None else -1
    print(f"  CHILD received {n} bytes", flush=True)


def main():
    ctx = mp.get_context("spawn")
    sizes_mb = [1, 2, 4, 6, 8, 10, 12, 16, 24]
    for mb in sizes_mb:
        payload = b"x" * (mb * 1024 * 1024)
        p = ctx.Process(target=child, args=(payload,))
        try:
            p.start()
            p.join(120)
            status = "OK" if p.exitcode == 0 else f"exitcode={p.exitcode}"
        except Exception as e:
            status = f"EXCEPTION {type(e).__name__}: {e}"
        print(f"{mb:>3} MB -> {status}", flush=True)


if __name__ == "__main__":
    main()
