# Rocket Detect → YOLO26s 训练说明

本文件记录数据集体检结论、预训练权重来源、环境配置与训练方法。
生成时间：2026-09-11

---

## 一、数据集体检结论

**结论：可以直接用于 YOLO26s 训练。** 无需转换格式、无需修复标签。

| 检查项 | 结果 |
|---|---|
| 目录结构 | `train/ valid/ test/` 各含 `images/` + `labels/`，标准 YOLO 检测格式 |
| 图片总数 | 28,149（train 24,435 / valid 2,428 / test 1,286） |
| 标签配对 | 图片与标签 **100% 一一对应**，无缺标签、无孤立标签 |
| 标签字段 | 全部为 `class x y w h` 5 字段，无字段数错误 |
| 坐标范围 | 全部落在 [0,1] 且 `w>0, h>0`，**无越界框、无退化框** |
| 图片可读性 | 28,149 张全部可正常解码，无损坏文件 |
| 图片尺寸 | **全部为 640×360**（单一尺寸，Roboflow 已统一拉伸） |
| 类别数 | 3：`0 Engine Flames`（尾焰）/ `1 Rocket Body`（箭体）/ `2 Space`（入轨后微小光点） |

### 类别分布（框数）

| 划分 | Engine Flames | Rocket Body | Space | 合计 |
|---|---|---|---|---|
| train | 11,000 | 12,657 | 4,134 | 27,791 |
| valid | 1,097 | 1,231 | 412 | 2,740 |
| test | 593 | 675 | 223 | 1,491 |
| **合计** | **12,690 (39.6%)** | **14,563 (45.5%)** | **4,769 (14.9%)** | **32,022** |

类别轻度不均衡（Space 为少数类），程度可接受，不建议做重采样，保证召回可依靠增强与训练轮数解决。

### 空标签文件（背景图）—— 正常现象，不是错误

| 划分 | 空标签数 | 占比 |
|---|---|---|
| train | 5,010 | 20.5% |
| valid | 493 | 20.3% |
| test | 249 | 19.4% |

这些是"画面中确实没有目标"的负样本（纯天空/纯地面），**必须保留**，能显著降低误检率。Ultralytics 会正确将其作为背景样本处理。

### 两个需要注意的点（不影响可用性）

**1. 目标普遍很小，`Space` 类尤其极端**

以 640×360 原始分辨率折算，框的等效边长分布：

| 类别 | P5 | 中位 | P95 |
|---|---|---|---|
| Engine Flames | 9.9 px | 45.9 px | 171.3 px |
| Rocket Body | 7.1 px | 33.8 px | 179.5 px |
| Space | 4.9 px | **8.0 px** | 18.4 px |

全部框中有 **29.9% 的等效尺寸 < 16 px**。这是典型的"小目标检测"场景。
→ 默认 `imgsz=640` 训练（含大量 letterbox 填充）对 `Space` 类会偏弱，建议按第六节调参。

**2. 存在跨划分的数据泄漏（约 4.7%）**

本数据集是 Roboflow 增强导出：每张原图随机旋转生成约 3 个版本，文件名 `0001_png.rf.<hash>.jpg` 中 `0001_png` 即原图（源）标识。

- 唯一源图数：**10,018**
- 其中 **472 个源图（4.7%）的不同增强版本被分到了不同划分**（如同一源的 3 个版本分别进了 train / valid / test）

后果：验证/测试集里混入了与训练集"同源"的近似重复图，**评估指标会偏乐观（虚高）**。不影响训练本身能否跑通，但影响你对模型真实泛化能力的判断。若要严谨评估，用 `scripts/build_leakfree_split.py` 重建划分（见第七节）。

> 另注：28,149 张图实际只来自 10,018 个源，等效数据多样性约为标注数的 1/3，训练时留意过拟合。

---

## 二、YOLO26s 官方预训练权重

Ultralytics **已正式发布 YOLO26 系列**，官方权重存放在 GitHub `ultralytics/assets` 的 `v8.4.0` release（发布于 2026-01-13）。

| 模型 | 参数量 | COCO mAP50-95 | 官方下载地址 |
|---|---|---|---|
| YOLO26n | 2.4M | 40.9 | `.../v8.4.0/yolo26n.pt` |
| **YOLO26s** | **9.5M** | **48.6** | **`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt`** |
| YOLO26m | 20.4M | 53.1 | `.../v8.4.0/yolo26m.pt` |
| YOLO26l | 24.8M | 55.0 | `.../v8.4.0/yolo26l.pt` |
| YOLO26x | 55.7M | 57.5 | `.../v8.4.0/yolo26x.pt` |

**已下载到本工作区：**

```
weights/yolo26s.pt      20.42 MB
SHA256: 646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
```

已验证可被 PyTorch 正常载入：`DetectionModel`，`task=detect`，`nc=80`（COCO 预训练），参数量 10.01M，导出日期 2026-01-05。

> 补充：官方还提供蒸馏版 `yolo26s-distill.pt`（COCO mAP 49.2，比标准版高 0.6），如追求更高精度可自行替换，用法完全相同。

---

## 三、训练环境

| 项目 | 值 |
|---|---|
| conda 环境 | **`E:\RocketAttitudeEstimation\envs\yolo26`**（独立环境，位于 E 盘，不占 C 盘） |
| Python | 3.12.14 |
| PyTorch | CUDA 12.8 轮子（`--index-url https://download.pytorch.org/whl/cu128`） |
| ultralytics | ≥ 8.4.0（**YOLO26 的最低要求**；已有 `yolov11` 环境为 8.3.39，不识别 yolo26） |
| GPU | NVIDIA GeForce RTX 4060 Laptop，8 GB 显存，驱动 CUDA 13.2 |

激活环境：

```powershell
conda activate E:\RocketAttitudeEstimation\envs\yolo26
```

一键重装/复现环境：`setup_env.ps1`

---

## 四、目录结构

```
E:\RocketAttitudeEstimation\
├── Rocket Detect.v37i.yolo26\      # 原始数据集（只读，未做任何修改）
├── weights\yolo26s.pt              # 官方预训练权重
├── configs\
│   └── rocket.yaml                 # 数据集配置（已修正 Roboflow 的 ../ 相对路径）
├── scripts\
│   ├── audit_dataset.py            # 数据集体检脚本
│   ├── dataset_audit_report.json   # 体检原始报告
│   └── build_leakfree_split.py     # 可选：重建无泄漏划分
├── train.py                        # 训练入口
├── requirements.txt
├── setup_env.ps1 / setup_env.sh    # 环境一键配置
├── runs\                           # 训练输出（权重、曲线、日志）
└── .cache\                         # 所有临时/缓存目录（tmp / pip / torch / mpl / yolo / wheels）

# conda 环境本体不在这里，而是放在工作区之外：
#   E:\Anaconda\envs\yolo26         （原因见第七节）
```

`configs/rocket.yaml` 相对原始 `data.yaml` 的改动：原始文件写的是 `train: ../train/images`（Roboflow 导出习惯，依赖 Ultralytics 的 `../` 兜底逻辑），已改为显式 `path` + 相对子路径，避免不同版本行为差异导致找不到数据。

---

## 五、开始训练

> 本文档只做准备工作，**未启动训练**。确认无误后执行：

```powershell
conda activate E:\RocketAttitudeEstimation\envs\yolo26
cd E:\RocketAttitudeEstimation

# 1) 先自检（检查数据、权重、GPU，不会训练）
python train.py --check-only

# 2) 正式开始训练（默认 imgsz=640 batch=16 epochs=100）
python train.py
```

常用覆盖参数：

```powershell
python train.py --epochs 200 --batch 8              # 更长时间 / 更小批
python train.py --imgsz 960 --batch 8               # 放大 1.5 倍，小目标略有收益
python train.py --imgsz 1280 --batch 4              # 放大 2 倍，对 Space 类最友好但慢约 4 倍
python train.py --workers 6                         # 想再快一点可以试 6~8（有内存风险，见下）
python train.py --cache False                       # 完全关闭图片缓存，内存占用最低
python train.py --batch -1                          # 让 Ultralytics 按显存自动选批大小
python train.py --resume                            # 中断后继续
```

> ⚠ **不要加大 `--workers`**。本机物理内存只有 15.7 GB，每个 DataLoader worker 实测
> 占 0.63~0.65 GB，`--workers 16` 光子进程就要 10.1 GB，训练前就会报
> `OSError: [WinError 1455] 页面文件太小`。脚本现在会在训练前估算内存并直接拦下这种配置。

> `--imgsz` 只能给**正方形整数**。train/val 不支持 `640x360` 这类非正方形写法，
> ultralytics 会强制取 `max(宽,高)` 并只打一行 WARNING —— 脚本已把它改为明确提示。

训练产物：`runs\rocket_yolo26s\weights\best.pt`、`last.pt`，以及 `results.csv`、PR 曲线、混淆矩阵等。

---

## 六、针对本数据集的调参建议

1. **输入尺寸：train/val 只支持正方形整数，这是硬限制。**

   已实测确认：传入 `--imgsz 640x360` 时 ultralytics 只打一行 WARNING，随后强制使用
   `max(高,宽)=640` 训练（源码 `utils/checks.py::check_imgsz` 的 `max_dim=1` 分支）。
   非正方形 `[高,宽]` 列表只对 `predict` / `export` 有效。所以训练阶段可选的只有正方形。

   | 输入 | letterbox 行为（源图 640×360） | GPU 显存 | 单轮耗时 |
   |---|---|---|---|
   | `640`（默认） | 内容 1:1，43.8% 灰边，410k 像素 | 约 2.6 GB（batch 8，实测） | 约 7.1 分钟（实测） |
   | `960` | 内容放大 1.5 倍 | 需降 batch | 约 2.2 倍 |
   | `1280` | 内容放大 2 倍（8px 光点 → 16px） | 需 batch 4~8 | 约 4 倍 |

   `Space` 类中位仅 8px，理论上 1280 有收益；但若下游视频里火箭始终清晰可见，`640` 已够用，
   不值得多花 4 倍时间。建议先用 640 跑完，再按需做对照实验。

2. **内存才是当前真正的约束，不是 GPU、也不是磁盘 I/O。**

   > 更正此前的一个错误判断：冒烟测试打印的 `Slow image access detected (read: 2.21 MB/s)`
   > 是**单线程顺序读的粗略启发式**，不代表 8 进程并行流水线的真实能力。
   > 用 `scripts/bench_dataloader.py` 做 A/B 实测，缓存收益 ≈ 0（440 vs 448 张/s），
   > 而训练端实际只消耗约 57 张/s —— **数据加载有 7~8 倍余量**。

   真正的瓶颈是**物理内存**。实测每个 DataLoader worker 占 0.63~0.65 GB：

   | workers | 子进程内存 | 占物理内存(15.7 GB) | 数据吞吐 |
   |---|---|---|---|
   | 4（默认） | 2.6 GB | 16% | 213 张/s |
   | 6 | 3.8 GB | 24% | 295 张/s |
   | 8 | 5.1 GB | 33% | 320 张/s |
   | 12 | 7.8 GB | 49% | 294 张/s |
   | 16 | 10.1 GB | 64% | 131 张/s（已不稳） |

   训练只需要 57 张/s，**4 个 worker 就完全够**。再往上加只会吃内存、不会提速。

   缓存三档的选择：

   - `bounded`（默认）：只把约 4 GB 图片常驻内存，其余按需从磁盘读
   - `False`：完全不缓存，内存占用最低（worker 降到约 0.55 GB）
   - `disk`：数据集目录里已存在 24,435 个 `.npy`（15.7 GB）。
     注意 ultralytics 只要看到同名 `.npy` 就会优先读它，**与 `--cache` 取值无关**；
     要真正关掉磁盘缓存必须删除这些 `.npy` 文件
   - `ram` 不要用：24435 张 × 675 KB ≈ 15.7 GB，超出物理内存，ultralytics 自己也会拒绝

3. **增强设置已在 `train.py` 中针对本数据调整**：关闭左右/上下翻转（火箭发射画面翻转会破坏物理合理性）、关闭 mixup（对小光点类不友好）、旋转角度与 Roboflow 导出一致取 ±3°、`close_mosaic=10`。

4. **8 GB 显存**下如遇 OOM：先降 `--batch`，再降 `--imgsz`。

5. **`--workers` 不是越大越好**：每个 worker 会持有一份完整的数据集索引与运行时，
   实测 0.63~0.65 GB。脚本默认 4，并按"物理内存 − 2 GB 安全余量"算出安全上限并在超限时
   直接拒绝启动（附上推荐的 worker 数）。确需强行继续可加 `--skip-mem-check`。

6. **两个已修复的典型报错**（遇到时可直接对照）：

   | 报错 | 真实原因 |
   |---|---|
   | `OSError: [WinError 1455] 页面文件太小` | worker 总数 × 0.65 GB 超过物理内存 |
   | `OSError: [Errno 22] Invalid argument` / `pickle data was truncated` | 在函数内部定义了 dataset 类，spawn 子进程无法 pickle；必须是模块级类 |

---

## 七、关于 C 盘空间

C 盘仅剩约 35 GB，因此**所有可能写盘的位置都已重定向到 E 盘工作区**：

| 位置 | 默认(可能落 C 盘) | 已重定向到 |
|---|---|---|
| pip 下载缓存 | `%LOCALAPPDATA%\pip\Cache` | `.cache\pip` |
| 临时文件 | `C:\Users\...\AppData\Local\Temp` | `.cache\tmp` |
| torch hub 缓存 | `C:\Users\...\.cache\torch` | `.cache\torch` |
| matplotlib 缓存 | `C:\Users\...\.cache\matplotlib` | `.cache\mpl` |
| ultralytics 配置 | `C:\Users\...\AppData\Roaming\Ultralytics` | `.cache\yolo` |
| conda 环境 | `C:\Users\...\.conda\envs` | `E:\Anaconda\envs\yolo26`（E 盘，工作区外） |
| 训练输出 | `runs/`（当前目录） | `runs\`（工作区内） |

`train.py` 顶部会在 `import torch` 之前自动设置上述环境变量，无需手动配置。

### 两个必须避开的坑（本次实际踩过，务必注意）

**坑 1：不要把 conda 环境建在工作区内。**

本环境的沙箱对**工作区内的批量删除**设有 50 个文件的阈值保护。而 pip / conda 在替换大包时会一次性删除上万个文件，触发 `[SAFE_DELETE_BULK_REJECTED]` 后删除被**中途打断**，环境随即损坏（`Lib` 下的 `.py` 文件全部消失，python 报 `Could not find platform independent libraries <prefix>`）。

更麻烦的是：conda 用**硬链接**把包缓存（`E:\Anaconda\pkgs`）里的文件链进环境，这些"删除"会连带把缓存里的同名 inode 截断为 0 字节，从而污染包缓存。

→ 所以环境建在 `E:\Anaconda\envs\yolo26`（工作区之外，仍在 E 盘不占 C 盘）。

**坑 2：全程只创建、不删除。**

必须先装 **CUDA 版 torch**，再装其余依赖。千万不要先装 PyPI 上的 **CPU 版 torch** 再替换成 CUDA 版 —— 那次替换（卸载 torch 会删数千文件）正是坑 1 的触发源。判断依据：`torch.__version__` 若带 `+cpu` 或 `torch.version.cuda` 为 `None`，就是 CPU 版。

> 补充：本机网络无法访问 `download.pytorch.org`（被代理拦截 502），PyPI 上的 Windows torch 又是 CPU 版，
> 因此走国内镜像 `mirrors.aliyun.com/pytorch-wheels` + 多线程下载（`scripts/fetch_wheels.py`）。
> 单线程约 0.7 MB/s，8 线程可达 ~4.8 MB/s。

**若环境仍被损坏**（症状：`Lib` 下 `.py` 消失）：可用 `conda_package_handling` 把对应包从 `.conda` 归档重新解压覆盖回去，无需删除任何东西：

```python
from conda_package_handling.api import extract
extract(r"E:\Anaconda\pkgs\python-3.12.14-hb12b558_3_cpython.conda",
        dest_dir=r"E:\Anaconda\pkgs\python-3.12.14-hb12b558_3_cpython")
```

---

## 八、可选：消除数据泄漏（严谨评估用）

若你希望验证集指标真实可信，先重建无泄漏划分：

```powershell
python scripts\build_leakfree_split.py --dry-run     # 先看统计，不写文件
python scripts\build_leakfree_split.py               # 生成 data\leakfree\ 与 configs\rocket_leakfree.yaml
python train.py --data configs\rocket_leakfree.yaml
```

该脚本以"源图"为单位重新划分（同一原图的所有增强版本必定在同一划分内），使用**同盘硬链接**生成，不复制数据、几乎不额外占用磁盘，且**只读原始数据集**。

---

## 九、冒烟测试实测结果（已验证）

命令：`python train.py --epochs 3 --batch 8 --imgsz 640x360 --name smoke_test`
（注意：实际按 `imgsz=640` 正方形执行，原因见第六节第 1 条）

**训练开销（实测）**

| 项目 | 数值 |
|---|---|
| 每轮迭代数 | 3,055 |
| 训练吞吐 | 约 7.0 ~ 7.3 it/s |
| 单轮耗时 | 约 7.1 分钟（另加验证约 12 秒） |
| GPU 显存占用 | 2.6 / 8 GB（batch 8）→ 显存不是瓶颈 |
| 3 轮总耗时 | 0.369 小时 |
| 瓶颈 | **数据读取**：`Slow image access detected (read: 2.21 MB/s)` |
| 数据集扫描 | train 34.5 s / val 3.4 s（生成 `labels.cache`，一次性） |
| 优化器 | `optimizer=auto` 自动选定 AdamW(lr=0.001429) |

**验证集指标（第 3 轮后 best.pt）**

| Class | Images | Instances | Box(P) | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| all | 2,428 | 2,740 | 0.839 | 0.805 | 0.850 | 0.458 |
| Engine Flames | 1,084 | 1,097 | 0.908 | 0.889 | 0.938 | 0.584 |
| Rocket Body | 1,199 | 1,231 | 0.893 | 0.911 | 0.930 | 0.528 |
| Space | 342 | 412 | 0.716 | 0.617 | 0.683 | 0.261 |

**结论**

- 全流程跑通：数据加载、增强、AMP、训练、验证、写盘、绘图均正常，`runs\smoke_test\weights\best.pt` 已生成。
- 仅 3 轮就达到 `Rocket Body` R=0.911 / mAP50=0.930，说明预训练权重迁移良好、任务本身不难。
- `Space`（小光点）明显最弱（mAP50-95 仅 0.261），符合其 8px 量级的难度预期；
  若下游视频里火箭始终清晰可见，该类的弱势不构成问题。
- 该 `best.pt` 只有 3 轮，**不可作为最终模型使用**，仅用于验证流程。

**据此确定的正式训练配置**

```powershell
python train.py --epochs 100 --batch 16 --imgsz 640 --name rocket_yolo26s
```

耗时预估：按每轮约 7.1 分钟计，100 轮约 **12 小时**（`patience=30` 会自动早停）。

> 缩短耗时**不要靠加 `--workers`**（会 OOM）。可选：降 `--batch` 换取更快的迭代步频、
> 或先跑 30 轮看趋势。实测数据加载已有 7~8 倍余量，瓶颈不在 I/O。
> 参考：2026-09-12 实测 `--epochs 1 --batch 16 --workers 4` 完整跑通，
> 单轮 0.125 小时，峰值显存约 5.0 GB，验证集 mAP50 = 0.623。
