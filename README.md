# Rocket Attitude Estimation —— 可回收火箭降落视频的姿态估计

从一段**网络下载的降落视频**出发，做四件事：检出火箭 → 稳定检测框 → 量出箭体倾角 → 给出姿态量。
**没有相机云台/内参参数**，所以姿态只做到"相对姿态 / 图像夹角"这一层（原因见第 4 步文档）。

> 测角用的是 **SAM 2.1 免训练分割（掩码剪影中心线）** 路线。
> 早期还实现过一条"ROI 梯度边缘"的传统方案，因**先验不成立 + 量错对象**（拟合条带落在
> 涂装条纹而非筒身轮廓）已下线，代码归档在 `legacy/`（详见 `legacy/README.md`）。

---

## 一键跑通

```powershell
# 默认用工作区最大的 mp4, 依次跑 detect -> stabilize -> seg -> angle_mask -> attitude
E:\Anaconda\envs\yolo26\python.exe pipeline.py

# 只打印汇总(不跑任何阶段), 看当前产物与关键指标
E:\Anaconda\envs\yolo26\python.exe pipeline.py --summary

# 查看阶段与产物
E:\Anaconda\envs\yolo26\python.exe pipeline.py --list
```

**阶段可缓存**：每阶段检查自己的产物文件，已存在就跳过；所以 `detect` 与 `seg`（两步需要 GPU）
只跑一次，后面 `angle_mask` / `attitude` 纯 CPU，反复调参只重跑对应阶段。
**显存/内存提示**：`seg` 会把每块帧张量常驻内存，默认 `--sam-chunk 200`；本机内存 15.7 GB 是最紧的资源。

```powershell
pipeline.py --force                        # 全部重跑
pipeline.py --only seg,angle_mask          # 只跑指定阶段
pipeline.py --from angle_mask              # 从某阶段往后
pipeline.py --to angle_mask                # 跑到某阶段为止
pipeline.py --video-out                    # 另外产出带标注的视频(较慢)
pipeline.py --l-over-d 17                  # 顺带算参数化的面外角 β
pipeline.py --sam-imgsz 1024               # 换 SAM 输入边长(1024 更准, 慢约 3 倍)
pipeline.py --sam-tag s1024                # 掩码路线产物标签(默认 s<imgsz>)
pipeline.py --video "D:\xx.mp4" --tag v1   # 换视频/换检测标签
```

全片（2203 帧 / 73.4s / 852×480）各阶段实测耗时（`SPEED_BENCH.md`）：
detect 46s · stabilize 0.2s · **seg 173s（含抽帧；帧已缓存时 110s）** · angle_mask 5s ·
attitude 66s（相机配准占大头）。日志写在 `runs/pipeline_log.txt`。

---

## 换一个新视频怎么跑

**代码里不含任何"只对某个视频成立"的先验**（片尾竖直段当真值、写死的秒数/帧号、
写死的画面尺寸、视频必须放项目根目录……这些都在 2026-10-08 的泛化改造里清掉了）。
交付量是**绝对倾角**：φ(t) = 箭体轴线相对**图像竖直**的夹角（纯几何量，不需要基准）。

```powershell
# 视频可以在任何位置(项目外也行); 两个标签分开: --tag 给检测, --sam-tag 给掩码链路
E:\Anaconda\envs\yolo26\python.exe pipeline.py `
    --video "D:\videos\r2.mp4" --tag r2 --sam-tag s512

# 也可以指定一个目录, 让它自己找里面最大的视频
E:\Anaconda\envs\yolo26\python.exe pipeline.py --video-dir "D:\videos" --tag r2

# 只看结果
E:\Anaconda\envs\yolo26\python.exe pipeline.py --summary --tag r2 --sam-tag s512
E:\Anaconda\envs\yolo26\python.exe scripts\sam_angle_viz.py --tag s512 --dets-tag r2
```

**新视频最容易卡在哪（按发生概率排序）**

| 现象 | 原因 | 怎么办 |
|---|---|---|
| `dets_r2.json` 里几乎没有框 → 后面全空 | **检测器不认识这种火箭**（模型只在这个数据集上训过）。这是最可能的失败点，且不是测角的问题 | 先看 `runs/diag/analysis.txt`；确认是检测问题就换权重 `--weights` 或微调模型 |
| 大量 `fit_failed` | 目标太小 / 掩码把尾焰或烟尘一起圈进来了 | 门限已按目标尺寸自适应，先看 `runs/seg/diag_s512.txt`（可用率诊断）与 `runs/sam/timeline_*.png` 上的"目标太小"阴影区 |
| 显存/内存不足 | 分辨率高（1080p+） | `--sam-chunk 100`（甚至 50）；像素门限已按分辨率自动缩放 |
| φ 曲线整体在漂 | **相机在动**（手持/摇镜） | 看 `runs/attitude/validate.txt` 的相机滚转段：φ 是相对**图像**的，相机在转时应逐帧减去 `cam_roll(t)` |
| 早期一段 φ 不可信 | 目标只有几像素宽，属尺度相关偏差 | 图里会**自动**阴影标出该段（判据：筒身宽首次稳定达到片尾中位宽的 60%）；这段本来就不该用 |

**已经实测过的泛化验证**：把原视频截成 15s 起、缩到 640×360、放到项目外的目录，
当成"另一个视频"跑全链路 → detect/seg/angle_mask 全部通过，测角有效帧 **1573/1753
（89.7%）**，落地稳定后 1/3 段 σ **0.038°**；与 480p 原视频同一时段的中位角差 0.08°。
产物见 `runs/sam/timeline_g360.png`、`runs/angle_mask/angles_g360.csv`。

---

## 四步结论速览

| 步骤 | 做什么 | 关键结果 |
|---|---|---|
| **1 检测** | YOLO26s 在 `Rocket Detect.v37i.yolo26` 上训练（100 轮 / 3 类） | 官方 valid mAP50 **0.928**；去泄漏重测 **0.914** ⇒ 分数可信 |
| **2 稳定化** | 类内 IoS 抑制嵌套框 + 双阈值续轨 + 底边模式状态机 + KF 补位 | 缺帧 **34 → 0**、嵌套重复框 **47 帧 → 0**、静止段平滑度无退化 |
| **3 倾角** | **SAM 2.1 掩码传播 → 剪影左右边界取中点 → 中心线鲁棒拟合** | 旋转增益 **0.963**（原方案 0.898）、帧内残差 σ **0.091°**（原 0.187）；落地段 σ **0.095°**；有效帧 **1730/2203** |
| **4 姿态** | 相机运动核查（+ 可选参数化面外角 β） | 相机实测静止（落地段滚转 +0.03°±0.06°）；全片有效帧 φ 中位 **+0.42°**（落地稳定段 σ **0.059°**） |

**可交付 / 不可交付**

- ✅ 箭体轴线与**图像竖直**的夹角 φ(t)、与**图像上边缘**的夹角 `90−|φ|`、角速率 `dφ/dt`
- ❌ 完整 3D 姿态（面外角 β / 总倾角）：单视角 + 长径比未知，实测 β 可从 0° 变到 60°+
  （掩码路线已经解决了"宽度口径"这一项 —— 旧方案量到的是涂装条纹；剩下的缺口是三维尺度与相机内参）
- ❌ **相对倾角 φ_rel**：它需要一个"参考姿态"，而唯一能拿到的是"片尾落地即竖直"——
  那是只对特定视频成立的先验，泛化改造中已删除。φ(t) 是绝对角，不需要它。

---

## 第 3 步：SAM 2.1 掩码测角（主线）

用第 2 步的稳定框提示 SAM 2.1，靠流式记忆传播出箭体掩码，再取**剪影的中心线**作为轴线。

```powershell
python scripts/bench_sam2.py --model tiny --image-size 512 --half    # 纯传播速度基准
python scripts/validate_mask_angle.py --tag s512                     # 旋转注入验证(准入门槛)
python scripts/sam_angle_viz.py --tag s512 --stills 6                # 成果件(汇报用)
python scripts/bench_pipeline.py --tag bench512 && python scripts/bench_report.py  # 整链路测速
```

**成果件**——`runs/sam/`：
- `timeline_s512.png` —— 上段 φ(t)（品红=平滑 / 灰=逐帧原值），下段 dφ/dt 角速率，
  顶部三条健康度微条带（测角有效 / 分割可信 / 重锚定），并标出 0-25s 尺度受限区与 45-69s 落地段
- `overlay_s512.mp4` —— 逐帧：掩码剪影（填充）+ 拟合轴线（品红）+ 筒身段范围（黄）+ 检出框（青）
  + 中文读数（φ / φ_rel / dφ/dt / 筒身宽 / 体检结论），底部同一曲线带带游标
- `stills_s512.png` / `summary_s512.txt` —— 抽帧拼图 / 关键数字摘要

**为什么不猜也能测角**：箭体是**回旋体**，其剪影左右边界的中点连线严格等于轴线在像面的投影。
已下线的原方案必须靠"左右边平行 + 间距恒定"两条先验去挑哪两条边属于同一根筒子；
掩码直接给出剪影，不需要猜。

### 与已下线方案的对拍结论（选型依据，详见 `legacy/README.md`）

| 指标 | 原方案（梯度边缘，已下线） | **SAM 2.1 @512** | **SAM 2.1 @1024** |
|---|---|---|---|
| 旋转增益（理想 1.0） | 0.898 | 0.963 | **0.975** |
| 帧内残差 σ | 0.187° | 0.091° | 0.091° |
| 落地静止段 46-66s σ | 0.223° | 0.095° | **0.059°** |
| 筒身宽度 `w_body` | ❌ 量到**涂装条纹**（≈0.5×框宽） | ✅ 剪影轮廓（≈0.85×框宽） | ✅ 同左 |
| 全片耗时（2026-09-21 复测） | 33 s（纯 CPU） | **110 s**（热抽帧）/ 173 s（含抽帧） | 315 s（热抽帧） |
| 有效覆盖 | 测角 907 帧 | 测角 1730 帧 | 测角 1741 帧 |

**512 与 1024 互拍**：Δφ 中位 **+0.026°**、均值 **−0.000°**，落地段彼此只差 0.037°
⇒ **512 已经够用**，1024 买到的是"增益更稳（帧间 σ 0.018 vs 0.061）+ 落地 σ 再降 1.6 倍"，代价 2.4 倍耗时。

**实时性**（RTX 4060 Laptop 8 GB，实测传播速度）：
`1024 fp32` 3.7 fps → `1024 fp16` 11.3 fps → **`512 fp16` 30.9 fps**；
2026-09-21 复测 **40.2 fps**（240 帧落地段，中位 23.6 ms/帧）—— 同一台机器不同轮次会差
20% 以上（功耗墙自适应浮动/温度/后台负载），**比速度必须同条件同时测**。
**配置选错会得出"不能实时"的错误结论**（官方标称 47.2 fps 是 A100 的数字）。

**整链路速度实测**（2026-09-21，`SPEED_BENCH.md`，含 GPU 采样）：
端到端 **74.7 ms/帧（13.4 fps，热抽帧）** / 103.0 ms/帧（含抽帧），
其中 `seg` 占 50.0 ms，`detect` 21.1 ms，其余 < 4 ms。
**瓶颈不在 GPU**：GPU 利用率仅 42%~57%、功耗 36~54 W（功耗墙 63~80 W、上限 140 W），
`seg` 阶段**一半时间花在 CPU 侧的 JPEG 帧载入**（25.9 ms 载入 vs 23.6 ms 传播 @512）。

**一个附带发现**：原方案拟合的"条带"相对掩码剪影是**倾斜**的（左右两侧等效角度差 −0.4°~−0.8°），
这解释了两条路线之间 **+0.76°** 的系统性偏移。物理原因是圆柱面上的一条纵向涂装条纹，
在斜视角下投影方向**并不平行于剪影边缘**——用表面特征带代表轴线会引入视角相关的角度偏差。
详见 `SAM2_MASK_ANGLE.md` §5。

---

## 目录结构

```
pipeline.py                  ← 一键入口(编排下面 4 步)
train.py                     ← 第 1 步: 训练入口(内存体检 + 硬拦截)

scripts/
  diag_video_detections.py   ← 第 1 步: 低阈值逐帧推理, 存盘原始检测框(后续分析不必重跑 GPU)
  rocket_track.py            ← 第 2 步: 时序稳定化核心模块
  validate_stabilize.py      ← 第 2 步: 前后对比 + 参数扫描
  track_video.py             ← 第 2 步: 稳定框标注视频 + 时间线图
  angle_core.py              ← 第 3 步: 共享内核(结果容器/IRLS 拟合/时序平滑/落盘)
  rocket_seg.py              ← 第 3 步: SAM 分块传播 + 体检 + 重锚定 + 逐行边界落盘
  rocket_mask_angle.py       ← 第 3 步: 掩码剪影 -> 倾角 φ(t)(含 --compare 与另一套配置互拍)
  validate_mask_angle.py     ← 第 3 步: 旋转注入测试(准入门槛)
  viz_common.py              ← 可视化公共件(字体链/CSV 读取/配色)
  sam_angle_viz.py           ← 第 3 步: **成果件**可视化(只看本路线, 含 φ(t)+dφ/dt 与健康度条带)
  bench_sam2.py              ← 测速: 纯传播速度(帧载入 vs 传播 拆分)
  bench_pipeline.py          ← 测速: 按真实流水线分阶段计时 + nvidia-smi 采样(功耗/显存/时钟)
  bench_report.py            ← 测速: 出速度分解图与表格(读 runs/bench/pipeline_speed.json)
  _seg_diag.py               ← 第 3 步: 可用率诊断
  _seg_probe.py              ← 第 3 步: 掩码圈住了什么(宽度口径探测)
  _phi_range_check.py        ← 第 3 步: φ 真实范围与落地段绝对值的复核
  _dphi_scale_check.py       ← 第 3 步: 绘图取轴规则的定量比较
  rocket_attitude.py         ← 第 4 步: 相机配准 + 夹角换算 + 参数化 β
  validate_attitude.py       ← 第 4 步: 六节验证报告(含相机估计器灵敏度校验)
  attitude_report.py         ← 第 4 步: 姿态时间线图 + 标注视频
  _probe_bg.py               ← 第 4 步前置探测: 有没有地平线 / 背景配准可不可用

runs/
  rocket_yolo26s/            ← 训练产物(best.pt 在用)
  diag/                      ← 检测存盘 JSON + 稳定化报告 + 曲线 + 标注视频
  seg/                       ← SAM 掩码逐行边界 npz + 逐帧体检 CSV + 运行配置 meta
  angle_mask/                ← 掩码路线的倾角 CSV/JSON + 旋转注入报告
  sam/                       ← **成果件**: overlay mp4 / 曲线图 / 抽帧图 / 数字摘要
  bench/                     ← 测速结果: pipeline_speed.json + speed_breakdown.png + 原始日志
  attitude/                  ← 姿态 CSV/JSON + 时间线图 + 标注视频 + 验证报告
  pipeline_log.txt           ← 流水线完整日志

legacy/                      ← 已下线的"ROI 梯度边缘"方案(归档留档, 不再维护)
                                对比报告/关键图表/原方案文档 + 如何用 git tag 取回代码

文档
  README.md                  ← 本文件(项目入口)
  TRAINING_REPORT.md         ← 第 1 步: 训练全过程总结
  DETECTION_STABILIZATION.md ← 第 2 步: 稳定化方案与踩坑
  SAM2_MASK_ANGLE.md         ← 第 3 步: SAM 2.1 掩码方案实现/实测/对拍(对拍章节已标为留档)
  SAM2_PROPOSAL.md           ← 第 3 步: 方案细化与路线对比(选型阶段的文档)
  SAM2_CODE_TOUR.md          ← 想从零读代码看这个: 分 5 层的阅读路线 + 必读函数索引
  SPEED_BENCH.md             ← 本机处理速度实测: 分阶段耗时/瓶颈定位/优化建议/实时性判定
  ATTITUDE_ESTIMATION.md     ← 第 4 步: 姿态可行性论证与结论(输入已切到掩码路线)
  README_TRAIN.md            ← 训练环境与操作手册
```

---

## 环境

| 项 | 值 |
|---|---|
| Python | `E:\Anaconda\envs\yolo26`（3.12.14）**位于工作区之外**，不要挪进来 |
| torch / ultralytics | 2.14.0+cu130 / 8.4.147 |
| GPU | RTX 4060 Laptop 8 GB（第 1 步 `detect` 与第 3 步 `seg` 需要） |
| 物理内存 | 15.7 GB —— 本项目最紧的资源；训练时 `--workers` ≤ 4，`seg` 必须分块（默认 200 帧） |

所有临时/缓存目录（TEMP/TORCH_HOME/MPLCONFIGDIR/YOLO_CONFIG_DIR）都由脚本重定向到
工作区的 `.cache/`，**不写 C 盘**。`angle_mask` / `attitude` 不需要 GPU，只要有
`runs/seg/bounds_*.npz` 就能重跑。

---

## 已知限制（重要）

1. **框的下边沿是双稳的**（≈250px / ≈400px 两档）——模型对"要不要把下半段算进框"不稳定。
   稳定化里已用"模式状态机"处理。注意这只是**检出框**的特性：测角走的是掩码剪影，
   已经不再受它影响（旧方案直接从框内找边缘，才会被带偏）。
2. **目标太小的前段不可采信**——那段箭体在画面里只有 21×53px，剪影条带仅 7~10px，
   估计值随目标变大单调趋向 0，是**尺度相关偏差**而非真实姿态。该区间现在由数据
   自动标出（本视频落在 ~20s 之前），绘图时不要用固定 ±8° 的纵轴（会把早期峰值削平）。
3. **完整 3D 姿态不可反算**。要解决需补：真实长径比/三维模型、相机内参与俯仰、或多视角。
   详见 `ATTITUDE_ESTIMATION.md` §6。（掩码路线已解决"可靠分割出筒身轮廓"这一项并给出
   正确的 `w_body` 口径；剩下的缺口是三维尺度与相机内参。）
4. 第 3 步的旋转增益是 **0.963（@512）/ 0.975（@1024）** ⇒ 绝对倾角仍有约 ±4% / ±2.5% 的
   尺度误差（旧方案是 0.90 ⇒ ±10%）。见 `SAM2_MASK_ANGLE.md`。

---

*每一步的关键数字都能在对应文档与 `runs/` 产物中复核；验证脚本可无 GPU 重跑。*
