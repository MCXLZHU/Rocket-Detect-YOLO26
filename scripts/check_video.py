# -*- coding: utf-8 -*-
"""在真实降落视频上快速验证检测效果。

产出三样东西(都放在 runs/video_check/ 下):
  1. <名字>_det.mp4      —— 带检测框的视频, 直接能看
  2. <名字>_sheet.jpg    —— 抽帧拼图, 一屏看完整个飞行过程(最快的方式)
  3. 控制台时间轴        —— 每秒的检测统计, 能看出哪个阶段检出了什么

用法:
    python scripts/check_video.py                       # 默认用视频目录下唯一的 mp4
    python scripts/check_video.py --conf 0.25           # 调置信度
    python scripts/check_video.py --stride 2            # 隔帧处理, 快一倍
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / ".cache" / "yolo"))

DEFAULT_WEIGHTS = PROJECT / "runs" / "rocket_yolo26s" / "weights" / "best.pt"
OUT_DIR = PROJECT / "runs" / "video_check"
CLASS_NAMES = ["Engine Flames", "Rocket Body", "Space"]


def find_video() -> Path:
    cands = [p for p in PROJECT.glob("*.mp4")] + [p for p in PROJECT.glob("*.mkv")]
    if not cands:
        print("[x] 工作区没找到视频文件")
        sys.exit(1)
    return max(cands, key=lambda p: p.stat().st_size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在视频上快速验证检测效果")
    p.add_argument("--video", type=str, default=None, help="视频路径, 默认自动找")
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--stride", type=int, default=1, help="隔几帧处理一次(1=每帧)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = 全部处理")
    p.add_argument("--sheet-cols", type=int, default=4)
    p.add_argument("--sheet-rows", type=int, default=5)
    p.add_argument("--no-video", action="store_true", help="不写视频, 只出拼图(更快)")
    p.add_argument("--out-name", type=str, default="result",
                   help="产物文件名前缀(原视频名含中文/标点时会让文件名过长)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video) if args.video else find_video()
    wpath = Path(args.weights)
    if not video.exists():
        print(f"[x] 找不到视频: {video}")
        sys.exit(1)
    if not wpath.exists():
        print(f"[x] 找不到权重: {wpath}")
        sys.exit(1)

    import cv2
    import numpy as np
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.out_name

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 74)
    print("视频检测验证")
    print("=" * 74)
    print(f"  视频  : {video.name}")
    print(f"  规格  : {W}x{H}  {fps:.1f} fps  {total} 帧  {total / fps:.1f} 秒")
    print(f"  权重  : {wpath}")
    print(f"  conf  : {args.conf}    imgsz: {args.imgsz}    stride: {args.stride}")
    print(f"  产物前缀: {args.out_name}   (输出到 runs/video_check/)")
    print()

    model = YOLO(str(wpath))

    writer = None
    out_video = OUT_DIR / f"{stem}_det.mp4"
    if not args.no_video:
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                fps, (W, H))

    expect_frames = (total // args.stride) if not args.max_frames else min(
        total // args.stride, args.max_frames)
    sheet_slots = args.sheet_cols * args.sheet_rows
    sheet_every = max(1, expect_frames // sheet_slots)
    sheet_frames: list[tuple[int, np.ndarray]] = []

    # 按秒统计
    per_sec_boxes: dict[int, list[int]] = {}
    frame_idx = 0
    done = 0
    t0 = time.time()
    hit_frames = 0
    conf_max: dict[int, float] = {}

    results = model.predict(source=str(video), stream=True, conf=args.conf,
                            imgsz=args.imgsz, vid_stride=args.stride,
                            verbose=False)

    for r in results:
        frame_idx += 1
        n_box = 0
        cls_count = [0, 0, 0]
        if r.boxes is not None and len(r.boxes):
            n_box = len(r.boxes)
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            cf_arr = r.boxes.conf.cpu().numpy()
            for c, cf in zip(cls_arr, cf_arr):
                if 0 <= c < 3:
                    cls_count[c] += 1
                    conf_max[c] = max(conf_max.get(c, 0.0), float(cf))
            hit_frames += 1
        sec = int(frame_idx * args.stride / fps)
        per_sec_boxes.setdefault(sec, [0, 0, 0])
        for i in range(3):
            per_sec_boxes[sec][i] += cls_count[i]

        annotated = r.plot(line_width=2, font_size=12)
        if writer is not None:
            writer.write(annotated)

        if len(sheet_frames) < sheet_slots and done >= len(sheet_frames) * sheet_every:
            thumb = cv2.resize(annotated, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            ts = frame_idx * args.stride / fps
            # 左上角写时间戳
            cv2.rectangle(thumb, (0, 0), (150, 24), (0, 0, 0), -1)
            cv2.putText(thumb, f"{ts:6.1f}s  f{frame_idx}", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            sheet_frames.append((frame_idx, thumb))

        done += 1
        if done % 200 == 0 or (args.max_frames and done >= args.max_frames):
            el = time.time() - t0
            f_ps = done / max(el, 1e-6)
            print(f"  已处理 {done:>5} 帧  {el:5.1f}s  ({f_ps:.1f} fps)  "
                  f"当前有目标的帧: {hit_frames}", flush=True)
        if args.max_frames and done >= args.max_frames:
            break

    if writer is not None:
        writer.release()

    el = time.time() - t0
    print()
    print(f"处理完成: {done} 帧, 用时 {el:.1f} 秒 ({done / max(el, 1e-6):.1f} fps)")
    print(f"有检测结果的帧: {hit_frames}/{done} ({hit_frames / max(done, 1) * 100:.1f}%)")

    # ---------- 拼图 ----------
    if sheet_frames:
        cols, rows = args.sheet_cols, args.sheet_rows
        th, tw = sheet_frames[0][1].shape[:2]
        sheet = np.zeros((rows * th + (rows - 1) * 4, cols * tw + (cols - 1) * 4, 3), np.uint8)
        for i, (_, img) in enumerate(sheet_frames):
            rr, cc = divmod(i, cols)
            y, x = rr * (th + 4), cc * (tw + 4)
            sheet[y:y + th, x:x + tw] = img
        sheet_path = OUT_DIR / f"{stem}_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"抽帧拼图: {sheet_path}")

    # ---------- 时间轴 ----------
    print()
    print("=" * 74)
    print("时间轴(每个时间点检出的目标数  EF=尾焰 RB=箭体 SP=光点)")
    print("=" * 74)
    print(f"{'时间':>7} {'EF':>5} {'RB':>5} {'SP':>5}   分布")
    for sec in sorted(per_sec_boxes):
        ef, rb, sp = per_sec_boxes[sec]
        bar = ("E" * min(ef, 20)) + ("R" * min(rb, 20)) + ("S" * min(sp, 20))
        print(f"{sec:>6}s {ef:>5} {rb:>5} {sp:>5}   {bar}")

    tot_ef = sum(v[0] for v in per_sec_boxes.values())
    tot_rb = sum(v[1] for v in per_sec_boxes.values())
    tot_sp = sum(v[2] for v in per_sec_boxes.values())
    print()
    print(f"累计检测框: Engine Flames {tot_ef} | Rocket Body {tot_rb} | Space {tot_sp}")
    print("各类最高置信度: " + " | ".join(
        f"{CLASS_NAMES[i]} {conf_max.get(i, 0.0):.2f}" for i in range(3)))
    print()
    print("产物目录: " + str(OUT_DIR))


if __name__ == "__main__":
    main()
