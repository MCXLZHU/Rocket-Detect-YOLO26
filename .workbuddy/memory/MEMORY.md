# RocketAttitudeEstimation — 项目长期备忘（索引 + 不可推翻的结论）

## 一句话
YOLO26s 检测可回收火箭降落视频 → 检测稳定化 → 筒身倾角 → 相对姿态。
**四步链路已全部打通并验证**。入口：`README.md`（总览）、`pipeline.py`（一键入口）。

## 文档索引（细节在文档里，此处不重复）
| 文档 | 内容 |
|---|---|
| `TRAINING_REPORT.md` | 训练全过程总结（项目统一入口） |
| `DETECTION_STABILIZATION.md` | 第 1-2 步：检测稳定化 |
| `ANGLE_ESTIMATION.md` | 第 3 步：ROI 边缘检测 + 倾角 |
| `ATTITUDE_ESTIMATION.md` | 第 4 步：姿态反算 |
| `README.md` | 流水线用法 + 已知限制 |
| `README_TRAIN.md` | 训练环境与操作手册 |

## 环境（不要改动）
- conda：`E:\Anaconda\envs\yolo26`（Py 3.12.14），**必须放在工作区外**
- torch 2.14.0+cu130 / ultralytics 8.4.147 / RTX 4060 Laptop 8 GB / CUDA 13.0
- 缓存一律重定向到 `.cache`，**不写 C 盘**
- **物理内存仅 15.7 GB，是最紧的资源**；每 worker 0.63~0.65 GB ⇒ `--workers 4`
- GPU 默认功耗墙 55W，最大 140W

## 不可推翻的结论（已实测）
1. **训练**：100 轮 / 10.9 h → mAP50 **0.928** / mAP50-95 0.569。**后 70 轮白跑**
   （ep30 已达 0.5602），下次用 `--epochs 60 --patience 15`（省 4.4 h，指标几乎不变）。
   `--imgsz` 只能是正方形整数；`--cache` 收益 ≈ 0；数据加载有 7~8 倍余量不是瓶颈。
2. **数据集有泄漏**：valid 中 66% 在 train 有近重复、19.6% 像素完全相同（Roboflow 随机分帧）。
   无泄漏子集重测 mAP50 0.914 ⇒ **虚高仅 1.4 点，0.928 可信**。
3. **Space 类 mAP50-95 仅 0.339 是物理下限**（框中位 7×9 px，94.6% 短边≤16px），
   非训练不足；要提升只能上 `--imgsz 1280`（约 4 倍耗时）。
4. **检测稳定化**：漏检根因是**单个 conf 同时承担"建轨/续轨"** → 拆 conf_high 0.25 / conf_low 0.10。
   嵌套框 **IoS 中位 1.000 但 IoU 仅 0.64~0.70，卡在 NMS 阈值 0.7 之下，NMS 原理上抓不到**
   → 类内 IoS 抑制 0.85，且**保留"更大的框"优于"分更高的框"**。
   框**下边沿双稳**（≈250 / ≈400px，最猛 +147px/帧）而上边沿 σ=0.46px
   → 状态量用 **ytop 而非中心 cy** + 持续性门控模式状态机。
   效果：缺帧 34→0、连缺 11→0、嵌套框 0 残留、ROI 漏裁 0/1855，开销 <1ms/帧。
5. **测角**：**只看 σ 会选错方案** —— Otsu+minAreaRect 的 σ 更小(0.321°)但旋转增益仅 0.196，
   "稳定"是假象。本方案合成图增益 1.0088 / σ 0.024°，真实旋转增益 0.898，落地段 σ 0.264°。
   逐行 argmax 会跳变（φ 恒为 0、增益 0.000）**必须改分段行平均**；峰位要用**质心**（避平顶偏置）；
   必须"先定条带再精修"（否则左右边取自不同结构，增益掉到 0.71）。
   **旋转增益 0.90 ⇒ 绝对倾角有 ±10% 尺度误差**。
6. **完整 3D 姿态不可反算**（单视角 + L/D 未知 + 宽度口径不一致）：换口径/L/D，β 从 0° 变到 67°，
   且 cosβ_raw=1.06>1 几何不自洽。可交付：φ(t)、与上边缘夹角 90−|φ|、相对倾角 φ−φ_ref、dφ/dt。
7. **相机基本静止**（注入 0.25/0.5/1.0° 滚转，增益 1.000 ⇒ 是真结论不是不敏感）；
   但**不能逐帧累加滚转**（单帧 σ0.018° × 2202 帧会累出 6° 假漂移），须直接对参考帧配准。
   本视频**没有可用地平线**，只能拿"落地竖直"当基准。
8. **前 20s 倾角不可采信**（条带仅 7~10px，估计值随目标变大单调趋向 0，是尺度相关偏差）；
   可信区间从 ~25s 起，最可信 45-69s。落地段 Δφ(相对上边缘)=89.63°、σ=0.26°。

## 一键流水线
`python pipeline.py` → detect(GPU 36s) → stabilize(3s) → angle(33s) → attitude(66s)
**阶段可缓存**（产物存在即跳过，`--force` 强制重跑）；`--summary/--list/--only/--from/--to`。
日志 `runs/pipeline_log.txt`。评测数据源 `runs/diag/dets_iou70.json`（一次推理存盘，后续分析免 GPU）。

## 版本控制（2026-09-15 git init）
- 仓库在根目录，`main` 分支；`core.quotepath=false`。
- 远端 `origin` = `https://github.com/MCXLZHU/Rocket-Detect-YOLO26.git`。
- **入库 122 文件 / 31.68 MB**。排除：数据集（18 GB）、`.cache/`（2.2 GB）、
  除 `runs/rocket_yolo26s/weights/best.pt` 外的所有 `*.pt`、`runs/**/*.mp4`、
  `runs/smoke_*`、`runs/angle/debug*`、`*.log`、IDE 目录。
- 跟踪输入视频与全部分析产物（报告/CSV/JSON/图表/关键帧），保证可复现。
- **提交信息必须用「无 BOM 的 UTF-8 文件 + `git commit -F`」**，PowerShell 下 `-m "中文"` 会乱码。
- **`git push` 未打通**：本机无 GitHub 写凭据，GCM 能取到 token 却存不住（见当日日志）。
  **WorkBuddy 的 GitHub Connector 与 git push 是两条独立通道，重连 Connector 无效。**
- 注意：`git check-ignore -v` 对**否定规则**也打印并返回 0，别只看退出码。

## 本机环境坑
- **PowerShell 工具输出会被吞**（连 `Write-Output` 都拿不到）：必须
  `... 2>&1 | Out-File -FilePath x.log -Encoding utf8` 落文件再 Read。
  **不要用 `*>`**，会写成 UTF-16 读不了。
- **Bash 工具在本机不可用**（`ls`/`dirname` 缺失），一律用 PowerShell + 专用工具。
- Windows 控制台编码：重定向时 Python 中文按 GBK 输出而端点按 UTF-8 解释 ⇒ 乱码。
  **验证时读 `runs/pipeline_log.txt`，不要读 PowerShell 重定向的文件。**

## 用户偏好
- 临时文件、缓存、环境**一律不要放 C 盘**（C 盘仅剩约 31 GB）
- 喜欢先看清方案与原委再执行；改动前希望被告知影响
- 需要诚实纠正错误判断，而不是含糊带过

## 未清理（用户未授权删除）
`envs/`、`Ultralytics/`（目前都是空目录）
