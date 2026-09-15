# -*- coding: utf-8 -*-
"""逐帧导出原始检测结果, 用于量化视频检测的三类问题(不写视频, 只出 JSON)。

目的:
  1. 漏检    —— 找出 Rocket Body 连续缺失的帧区间
  2. 重复框  —— 找出同一帧出现 >=2 个 RB 框的情况, 并算它们的 IoU / 包含度(IoS)
  3. 烟尘    —— 观察 RB 框高宽随时间的收缩趋势

之所以用很低的 conf(默认 0.02)跑一遍并把原始框全部存下来, 是为了后续
在不重新推理的前提下, 用不同阈值/不同 NMS 参数做离线对比。
推理一遍约 35 秒, 数据全部落在 runs/diag/ 下。

用法:
    python scripts/diag_video_detections.py
    python scripts/diag_video_detections.py --conf 0.02 --iou 0.7 --tag iou70
    python scripts/diag_video_detections.py --conf 0.02 --iou 0.3 --tag iou30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT / ".cache"
for _sub in ("tmp", "pip", "torch", "mpl", "yolo"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMP"] = str(CACHE_DIR / "tmp")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "mpl")
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "yolo")
os.environ.setdefault("PYTHONUTF8", "1")

DEFAULT_WEIGHTS = PROJECT / "runs" / "rocket_yolo26s" / "weights" / "best.pt"
OUT_DIR = PROJECT / "runs" / "diag"


def find_video() -> Path:
    cands = list(PROJECT.glob("*.mp4")) + list(PROJECT.glob("*.mkv"))
    if not cands:
        print("[x] 工作区没找到视频文件")
        sys.exit(1)
    return max(cands, key=lambda p: p.stat().st_size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="逐帧导出检测结果(诊断用)")
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS))
    p.add_argument("--conf", type=float, default=0.02, help="存盘用的极低阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--topk", type=int, default=12, help="每帧最多存几个框")
    p.add_argument("--tag", type=str, default="iou70", help="输出文件名后缀")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video = Path(args.video) if args.video else find_video()
    wpath = Path(args.weights)
    for pth, msg in ((video, "视频"), (wpath, "权重")):
        if not pth.exists():
            print(f"[x] 找不到{msg}: {pth}")
            sys.exit(1)

    import cv2
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("=" * 74)
    print("逐帧检测导出 (诊断)")
    print("=" * 74)
    print(f"  视频: {video.name}")
    print(f"  规格: {W}x{H}  {fps:.1f} fps  {total} 帧")
    print(f"  conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}  tag={args.tag}")
    print()

    model = YOLO(str(wpath))

    frames: list[dict] = []
    t0 = time.time()
    n = 0
    for r in model.predict(source=str(video), stream=True, conf=args.conf,
                           iou=args.iou, imgsz=args.imgsz, verbose=False,
                           max_det=50):
        n += 1
        rec: dict = {"f": n - 1, "t": round((n - 1) / fps, 4), "b": []}
        boxes = r.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            order = conf.argsort()[::-1][:args.topk]
            for i in order:
                x1, y1, x2, y2 = xyxy[i]
                rec["b"].append({
                    "c": int(cls[i]),
                    "s": round(float(conf[i]), 4),
                    "x": [round(float(x1), 2), round(float(y1), 2),
                          round(float(x2), 2), round(float(y2), 2)],
                })
        frames.append(rec)

        if n % 400 == 0:
            el = time.time() - t0
            print(f"  {n:>5}/{total} 帧  {el:5.1f}s  ({n / el:.1f} fps)", flush=True)
        if args.max_frames and n >= args.max_frames:
            break

    el = time.time() - t0
    out = OUT_DIR / f"dets_{args.tag}.json"
    meta = {
        "video": video.name, "fps": fps, "frames": len(frames), "W": W, "H": H,
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "weights": str(wpath), "elapsed_s": round(el, 1),
        "class_names": ["Engine Flames", "Rocket Body", "Space"],
    }
    out.write_text(json.dumps({"meta": meta, "frames": frames},
                              ensure_ascii=False), encoding="utf-8")
    print()
    print(f"完成: {len(frames)} 帧, {el:.1f} 秒 ({len(frames) / el:.1f} fps)")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
