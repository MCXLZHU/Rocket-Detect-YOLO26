# -*- coding: utf-8 -*-
"""
Rocket Detect  —  YOLO26s 检测模型训练脚本

用法示例(在 yolo26 环境中):
    python train.py                     # 用默认参数开始训练
    python train.py --check-only        # 只做配置/数据/模型自检, 不训练
    python train.py --epochs 200 --batch 8 --imgsz 640
    python train.py --imgsz 1280 --batch 6         # 针对 "Space" 小目标

设计要点:
  * 所有临时/缓存/输出目录都落在工作区内(E 盘), 不写 C 盘。
  * 训练产物写到 runs/ 下, 默认 runs/rocket_yolo26s。
  * 不覆盖原始数据集, 不修改任何原始文件。
  * **内存安全**: 见下。

================================ 历史故障与修正 ================================

故障 1: OSError [WinError 1455] 页面文件太小
    原因: --workers 默认给到 16, 而本机物理内存只有 15.7 GB。
          每个 DataLoader worker 实测占 0.63~0.65 GB, 16 个就是 10.1 GB,
          再加上 val 阶段还会另建一套 DataLoader, 直接耗尽提交内存。
    修正: 默认改成 workers=4, 并加入训练前内存估算与硬拦截。

故障 2: OSError [Errno 22] Invalid argument / pickle data was truncated
    原因: **不是内存问题**。父进程 pickle 数据集并写入 spawn 子进程的管道时
          被截断。真正原因是: ultralytics 在 spawn worker 时会将整个
          YOLODataset(含 24435 条 label, 序列化约 11.8 MB) 通过管道发给
          每一个 worker。size 本身能过, 但在**父进程同时开了 CUDA 上下文
          + 数十个 worker 争抢**时, 管道写入被资源压力打断。
    修正: --cache 默认改为 bounded, 并把 num_workers 严格控制在小值。

关于内存的实测数据(2026-09-12, RTX 4060 Laptop / 15.7 GB RAM):
  workers |  子进程内存  | 占物理内存 | 数据吞吐
  --------+-------------+-----------+---------
     4    |   2.6 GB    |    16%    |  213 张/s
     6    |   3.8 GB    |    24%    |  295 张/s
     8    |   5.1 GB    |    33%    |  320 张/s
    12    |   7.8 GB    |    49%    |  294 张/s
    16    |  10.1 GB    |    64%    |  131 张/s   <- 已不稳, 极易 OOM
  训练端实际只需约 57 张/s, 所以 4 个 worker 就完全够用。

关于图片缓存(三档):
  - bounded (默认): 只把一部分图片常驻内存(默认 4 GB), 训练期间复用。
  - False        : 完全不缓存, 每步从磁盘读(worker 内存最低)。
  - disk         : 使用数据集目录里已存在的 .npy(约 17.3 GB)。
                   注意 ultralytics 只要看到同名 .npy 就会优先读它, 与
                   --cache 取值无关, 要真正关掉磁盘缓存必须删除 .npy。
  - ram          : 全部常驻内存。本数据集 24435 张 × 675 KB ≈ 15.7 GB,
                   **超出物理内存, 不可用**(ultralytics 自己也会拒绝)。
"""

from __future__ import annotations

import os
import sys
import ctypes
import argparse
from pathlib import Path

# --------------------------------------------------------------------------
# 0) 在 import torch / ultralytics 之前, 先把各类缓存重定向到工作区(E 盘)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / ".cache"
for _sub in ("tmp", "pip", "torch", "mpl", "yolo"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(CACHE_DIR / "tmp")
os.environ["TMP"] = str(CACHE_DIR / "tmp")
os.environ["TMPDIR"] = str(CACHE_DIR / "tmp")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "mpl")          # matplotlib 缓存
os.environ["YOLO_CONFIG_DIR"] = str(CACHE_DIR / "yolo")      # ultralytics 会在其下建 Ultralytics/ 存放 settings.json
os.environ.setdefault("PYTHONUTF8", "1")

# 限制数值库线程数: 每个 worker 都开满 32 线程会严重放大内存与上下文切换开销
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# 默认值(可被命令行覆盖)
DEFAULT_MODEL = PROJECT_ROOT / "weights" / "yolo26s.pt"
DEFAULT_DATA = PROJECT_ROOT / "configs" / "rocket.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs"
REQUIRED_NOCOPY_ARGS = PROJECT_ROOT / "scripts" / "train_nocopy.py"

# 每个 DataLoader worker 的实测内存(GB), 用于训练前估算峰值。
BYTES_PER_WORKER_GB = 0.68
# 必须留出的安全余量(GB): 系统 + 显卡驱动 + val 阶段的第二套 DataLoader
SAFE_MARGIN_GB = 2.0



def available_ram_gb() -> float:
    """当前可用物理内存(GB)。失败时返回 -1。"""
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return -1.0


def total_ram_gb() -> float:
    """物理内存总量(GB)。失败时返回 -1。"""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        return -1.0


def peak_ram_estimate_gb(workers: int, batch: int, imgsz: int,
                         cache_gb: float = 0.0) -> float:
    """估算训练峰值内存(GB)。

    组成:
      * workers 个 DataLoader 子进程(每个含一份 torch/cv2/numpy 运行时) ≈ 0.68 GB
      * 再乘 2: ultralytics 的 val 会另建一套 DataLoader
      * batch * imgsz 的 pin_memory 缓冲(双缓冲)
      * 主进程预训练模型 + 优化器 + CUDA 主存 ≈ 2.0 GB
      * bounded 缓存占用的常驻内存
    """
    worker_mem = workers * BYTES_PER_WORKER_GB * 2
    pin_mem = 2 * batch * 3 * imgsz * imgsz / 1024 ** 3
    return worker_mem + pin_mem + 2.0 + max(cache_gb, 0.0)


def max_safe_workers(imgsz: int, batch: int, cache_gb: float = 0.0) -> int:
    """在留足安全余量的前提下, 最多能开多少个 worker。"""
    total = total_ram_gb()
    if total <= 0:
        return 4
    pin_mem = 2 * batch * 3 * imgsz * imgsz / 1024 ** 3
    budget = total - SAFE_MARGIN_GB - pin_mem - 2.0 - max(cache_gb, 0.0)
    return max(1, int(budget / (BYTES_PER_WORKER_GB * 2)))


def format_mem_note(workers: int, batch: int, imgsz: int,
                    cache_gb: float = 0.0) -> str:
    est = peak_ram_estimate_gb(workers, batch, imgsz, cache_gb)
    total = total_ram_gb()
    avail = available_ram_gb()
    lines = [f"估算训练峰值内存: 约 {est:.1f} GB"]
    if total > 0:
        lines.append(f"物理内存总量    : {total:.1f} GB  |  当前可用: {avail:.1f} GB")
        lines.append(f"安全上限 workers: {max_safe_workers(imgsz, batch, cache_gb)}")
        if est > total:
            lines.append(f"[!] 峰值 {est:.1f} GB > 物理内存 {total:.1f} GB，"
                         f"将依赖页面文件，极易出现 WinError 1455")
        elif est > total - 1.0:
            lines.append("[!] 峰值已逼近物理内存上限，建议把 --workers 或 --batch 再调小一档")
    return "\n".join("     " + x for x in lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO26s for rocket detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                   help="预训练权重路径(.pt)。默认使用工作区内下载好的官方 yolo26s.pt")
    p.add_argument("--data", type=str, default=str(DEFAULT_DATA), help="数据集 yaml")
    p.add_argument("--imgsz", type=str, default="640",
                   help="输入尺寸，只能给正方形整数，例如 640 / 960。"
                        "注意: train/val 不支持非正方形(如 640x360)，"
                        "ultralytics 会强制替换为 max(宽,高)；非正方形仅 predict/export 可用")
    p.add_argument("--epochs", type=int, default=100, help="训练轮数")
    p.add_argument("--batch", type=int, default=16,
                   help="批大小；设 -1 让 Ultralytics 按显存自动选择")
    p.add_argument("--device", type=str, default="0", help="'0' 用第一块 GPU, 'cpu' 用 CPU")
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader 进程数。默认 4 —— 每个 worker 实测约占 0.65 GB 内存，"
                        "16 个就会占满 10 GB 直接 OOM。训练端只需约 57 张/s，"
                        "而 4 个 worker 已达 213 张/s，完全够用")
    p.add_argument("--patience", type=int, default=30, help="早停耐心值(轮)")
    p.add_argument("--optimizer", type=str, default="auto", help="auto/SGD/AdamW/Adam/...")
    p.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    p.add_argument("--lrf", type=float, default=0.01, help="最终学习率系数")
    p.add_argument("--cos-lr", action="store_true", help="使用余弦退火学习率")
    p.add_argument("--cache", type=str, default="bounded",
                   choices=["bounded", "False", "ram", "disk"],
                   help="图片缓存方式。bounded(默认)只常驻一部分内存，最省内存且够用；"
                        "False 完全不缓存；ram 全部常驻(本数据集约 15.7 GB，放不下，会失败)；"
                        "disk 用数据集目录下已存在的 .npy(约 17.3 GB)")
    p.add_argument("--cache-gb", type=float, default=4.0,
                   help="--cache bounded 时允许占用的常驻内存上限(GB)")
    p.add_argument("--no-cache", action="store_true",
                   help="等价于 --cache False —— 完全禁用图片缓存，内存占用最低")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--project", type=str, default=str(DEFAULT_PROJECT), help="训练输出根目录")
    p.add_argument("--name", type=str, default="rocket_yolo26s", help="本次运行名称")
    p.add_argument("--resume", action="store_true", help="从上次中断处继续训练")
    p.add_argument("--check-only", action="store_true",
                   help="只做自检(检查数据/权重/环境), 不启动训练")
    p.add_argument("--skip-mem-check", action="store_true",
                   help="跳过训练前的内存估算检查(不建议)")
    return p.parse_args()


def parse_imgsz(s: str):
    """解析输入尺寸，返回 (正方形边长 int, 提示信息 str|None)。

    重要: ultralytics 的 trainer 用 check_imgsz(..., max_dim=1) 校验，
    非正方形输入会被强制替换成 max(高,宽)（仅 predict/export 支持 [高,宽] 列表）。
    本函数直接对齐该行为，避免出现"以为跑的是 640x360、实际跑的是 640"这种情况。
    """
    t = str(s).lower().replace(" ", "").replace("*", "x")
    if "x" in t:
        w, h = (int(v) for v in t.split("x"))
        img = max(h, w)
        note = (f"你给的是 {w}x{h}，但 train/val 只支持正方形整数输入。"
                f"ultralytics 会强制取 max(高,宽)={img} 来训练，"
                f"所以本次实际 imgsz={img}（非正方形仅 predict/export 可用）")
        return img, note
    return int(t), None


def imgsz_echo(imgsz: int) -> str:
    """把解析结果回显成人能核对的形式。"""
    return f"{imgsz}x{imgsz} (正方形) -> imgsz={imgsz}"


def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72, flush=True)


# --------------------------------------------------------------------------
# 图片缓存: bounded 模式的实现
# --------------------------------------------------------------------------
# bounded 缓存的容量(GB)。必须是模块级, 因为 spawn 出的子进程要能 import 它。
_BOUNDED_CACHE_GB = 4.0


class BoundedYOLODataset:
    """懒加载的包装类: 把 YOLODataset 换成"容量受限的常驻缓存"版本。

    为什么要这样写:
      Windows 下 DataLoader 用 spawn 起子进程, 数据集对象必须能被 pickle。
      在函数内部定义的类**无法被 pickle**(spawn 子进程 import 不到),
      所以这里用一个模块级类, 内部持有真正的 YOLODataset 实例。
    """

    def __init__(self, *args, **kwargs):
        from ultralytics.data.dataset import YOLODataset
        # 强制不走 ultralytics 原生的 ram / disk 全量缓存
        kwargs["cache"] = None
        self._ds = YOLODataset(*args, **kwargs)
        self._bcache = {}
        self._bcache_bytes = 0
        self._bcache_hits = 0
        self._bcache_miss = 0

    # 把属性访问透传给真正的数据集。
    # 关键: __getattr__ 只在常规查找失败时被调用。反序列化(spawn 子进程)时
    # 实例的 __dict__ 还是空的, 此时访问 _ds 会再次触发 __getattr__ 造成
    # 无限递归。所以必须对下划线开头的名字直接抛 AttributeError。
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_ds"], name)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        return self._ds[i]

    def load_image(self, i, *a, **kw):
        hit = self._bcache.get(i)
        if hit is not None:
            self._bcache_hits += 1
            return hit
        self._bcache_miss += 1
        out = self._ds.load_image(i, *a, **kw)
        nbytes = int(getattr(out[0], "nbytes", 0))
        cap = int(_BOUNDED_CACHE_GB * 1024 ** 3)
        if cap > 0 and nbytes <= cap:
            if self._bcache_bytes + nbytes > cap:
                # 容量满: 整体清空。简单, 且不留内存碎片
                self._bcache.clear()
                self._bcache_bytes = 0
            self._bcache[i] = out
            self._bcache_bytes += nbytes
        return out


def _install_bounded_cache(cache_gb: float) -> None:
    """把 bounded dataset 挂到 ultralytics 的数据构建入口上。

    注意: build_yolo_dataset 内部引用的是模块级名字 YOLODataset,
    所以必须替换 ultralytics.data.build 命名空间里的那一个。
    """
    global _BOUNDED_CACHE_GB
    _BOUNDED_CACHE_GB = float(cache_gb)
    import ultralytics.data.build as _build
    _build.YOLODataset = BoundedYOLODataset
    print(f"[ok] 已启用 bounded 图片缓存, 上限 {cache_gb:.1f} GB")
    print("     说明: 训练集走 bounded 缓存; 验证集仍按 ultralytics 默认逻辑加载")


def report_npy_state(data_yaml: Path) -> None:
    """报告数据集里已存在的 .npy 缓存情况。

    重要: ultralytics 的 load_image() 只要发现同名 .npy 存在就会优先 np.load，
    与 --cache 参数无关。所以 --cache False 并不能绕开磁盘上的 .npy。
    """
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        root = Path(cfg["path"])
    except Exception:
        return
    hits = []
    for split in ("train", "valid"):
        sub = root / cfg.get(split, "")
        if sub.exists():
            n = sum(1 for _ in sub.glob("*.npy"))
            if n:
                hits.append((split, sub, n))
    if hits:
        print("[!] 检测到数据集目录下已存在 .npy 磁盘缓存:")
        for split, sub, n in hits:
            print(f"    {split:<5}: {n} 个  ({sub})")
        print("    注意: ultralytics 只要看到同名 .npy 就会优先读它，与 --cache 无关。")
        print("    若要真正关闭磁盘缓存，需要删除这些 .npy 文件(约占 17.3 GB)。")


def preflight(args, imgsz, imgsz_note=None) -> float:
    """训练前自检: 权重、数据集、依赖、GPU、输入尺寸、内存。

    返回值: 本次训练实际使用的缓存内存预算(GB)。
    """
    banner("环境自检")

    # --- 权重 ---
    mp = Path(args.model)
    if not mp.exists():
        print(f"[x] 找不到权重文件: {mp}")
        print("    请先从 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt 下载")
        sys.exit(1)
    print(f"[ok] 权重: {mp}  ({mp.stat().st_size / 1e6:.2f} MB)")

    # --- 数据集 ---
    dp = Path(args.data)
    if not dp.exists():
        print(f"[x] 找不到数据配置: {dp}")
        sys.exit(1)
    import yaml
    cfg = yaml.safe_load(dp.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    print(f"[ok] 数据配置: {dp}")
    print(f"     根目录      : {root}  (存在={root.exists()})")
    for split in ("train", "val", "test"):
        sub = root / cfg[split]
        n = len(list(sub.glob("*"))) if sub.exists() else -1
        flag = "[ok]" if n > 0 else "[x] "
        print(f"     {flag} {split:<5}: {sub}  ({n} 个文件)")
    print(f"     类别数 nc   : {cfg['nc']}   names={cfg['names']}")

    # --- 依赖与 GPU ---
    import torch
    import ultralytics
    print(f"[ok] ultralytics {ultralytics.__version__}  |  torch {torch.__version__}")
    if args.device != "cpu":
        if torch.cuda.is_available():
            i = torch.cuda.current_device()
            name = torch.cuda.get_device_name(i)
            gpu_total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"[ok] GPU: {name}  显存 {gpu_total:.1f} GB  CUDA {torch.version.cuda}")
        else:
            print("[x] 未检测到可用 CUDA GPU, 请改用 --device cpu 或检查驱动")
            sys.exit(1)
    print(f"[ok] 缓存目录已重定向到: {CACHE_DIR}")
    print(f"[ok] 训练输出目录: {Path(args.project) / args.name}")

    # --- 输入尺寸核对 ---
    print(f"[ok] 输入尺寸: {imgsz_echo(imgsz)}")
    if imgsz_note:
        print(f"[!] 注意: {imgsz_note}")

    # --- 内存体检(重点: 直接决定会不会再报 WinError 1455) ---
    cache_gb = args.cache_gb if args.cache == "bounded" else 0.0
    cache_label = args.cache
    print(f"[内存体检] workers={args.workers}  batch={args.batch}  cache={cache_label}")
    print(format_mem_note(args.workers, args.batch, imgsz, cache_gb))

    est = peak_ram_estimate_gb(args.workers, args.batch, imgsz, cache_gb)
    total = total_ram_gb()
    if total > 0 and not args.skip_mem_check:
        if est > total - SAFE_MARGIN_GB:
            safe_w = max_safe_workers(imgsz, args.batch, cache_gb)
            print(f"\n[x] 内存不足: 估算峰值 {est:.1f} GB，物理内存仅 {total:.1f} GB，")
            print(f"    安全余量需 {SAFE_MARGIN_GB:.1f} GB。继续训练极可能再次报")
            print(f"    'OSError: [WinError 1455] 页面文件太小'。")
            print(f"\n    请改用(任选其一):")
            print(f"      python train.py --workers {safe_w} --batch {args.batch}")
            print(f"      python train.py --workers {safe_w} --batch 8")
            print(f"      python train.py --cache False       # 关掉缓存再试")
            print(f"    或先释放内存(关闭浏览器/其他 Python 进程)后重试。")
            print(f"    确认要强行继续: 加 --skip-mem-check\n")
            sys.exit(2)
        print(f"[ok] 内存充足({est:.1f} / {total:.1f} GB)，可以开始训练")

    # --- 数据集里已有的 .npy 磁盘缓存会覆盖 --cache 设置, 必须显式提醒 ---
    report_npy_state(dp)
    return cache_gb


def main() -> None:
    args = parse_args()
    imgsz, imgsz_note = parse_imgsz(args.imgsz)

    if args.no_cache:
        args.cache = "False"

    cache_gb = preflight(args, imgsz, imgsz_note)

    # 让 Ultralytics 的默认数据集/权重目录也落在工作区, 不写 C 盘
    try:
        from ultralytics import settings as ul_settings
        ul_settings.update({
            "datasets_dir": str(Path(args.data).resolve().parent.parent),
            "weights_dir": str(PROJECT_ROOT / "weights"),
            "runs_dir": str(Path(args.project).resolve()),
        })
        print(f"[ok] Ultralytics settings 已更新 -> {os.environ.get('YOLO_CONFIG_DIR')}")
    except Exception as e:  # 不同版本 API 略有差异, 失败不影响训练
        print(f"[warn] 更新 ultralytics settings 失败(可忽略): {e}")

    if args.check_only:
        banner("模型载入自检")
        from ultralytics import YOLO
        m = YOLO(args.model)
        n_p = sum(p.numel() for p in m.model.parameters()) / 1e6
        print(f"[ok] 模型载入成功: task={m.task}  参数量={n_p:.2f}M  "
              f"nc(预训练)={m.model.yaml.get('nc')}")
        print("\n自检通过。去掉 --check-only 即可开始训练。")
        return

    # ------------------------------------------------------------------
    # 正式训练
    # ------------------------------------------------------------------
    from ultralytics import YOLO

    if args.cache == "bounded":
        _install_bounded_cache(cache_gb)
        ul_cache = None          # 不走 ultralytics 原生 ram/disk 全量缓存
    elif args.cache == "False":
        ul_cache = False
    else:
        ul_cache = args.cache    # "ram" / "disk"

    banner("开始训练")
    print(f"  model   = {args.model}")
    print(f"  data    = {args.data}")
    print(f"  imgsz   = {imgsz}")
    print(f"  epochs  = {args.epochs}   batch = {args.batch}   device = {args.device}")
    print(f"  workers = {args.workers}   cache = {args.cache}")
    est = peak_ram_estimate_gb(args.workers, args.batch, imgsz, cache_gb)
    print(f"  预计峰值内存 ≈ {est:.1f} GB")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        cache=ul_cache,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        plots=True,
        val=True,
        amp=True,
        pretrained=True,
        # --- 针对本数据集的小目标特点, 略调增强 ---
        degrees=3.0,        # 与 Roboflow 导出的 ±3° 旋转一致
        translate=0.1,
        scale=0.5,
        fliplr=0.0,         # 火箭发射画面左右翻转会破坏物理合理性, 关闭
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,          # 混合样本对小光点类不友好, 关闭
        close_mosaic=10,
        verbose=True,
    )

    banner("训练完成")
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"最佳权重: {best}")


if __name__ == "__main__":
    main()
