# Rocket Attitude Estimation —— 可回收火箭降落视频的姿态估计

从一段**网络下载的降落视频**出发，做四件事：检出火箭 → 稳定检测框 → 量出箭体倾角 → 给出姿态量。
**没有相机云台/内参参数**，所以姿态只做到"相对姿态 / 图像夹角"这一层（原因见第 4 步文档）。

---

## 一键跑通

```powershell
# 默认用工作区最大的 mp4, 依次跑 detect -> stabilize -> angle -> attitude
E:\Anaconda\envs\yolo26\python.exe pipeline.py

# 只打印汇总(不跑任何阶段), 看当前产物与关键指标
E:\Anaconda\envs\yolo26\python.exe pipeline.py --summary

# 查看阶段与产物
E:\Anaconda\envs\yolo26\python.exe pipeline.py --list
```

**阶段可缓存**：每阶段检查自己的产物文件，已存在就跳过；所以 `detect`（唯一需要 GPU 的一步，约 36s）
只跑一次，后面几阶段纯 CPU，反复调参只重跑对应阶段。

```powershell
pipeline.py --force                        # 全部重跑
pipeline.py --only stabilize,angle         # 只跑指定阶段
pipeline.py --from angle                   # 从某阶段往后
pipeline.py --to angle                     # 跑到某阶段为止
pipeline.py --video-out                    # 另外产出带标注的视频(较慢)
pipeline.py --l-over-d 17                  # 顺带算参数化的面外角 β
pipeline.py --sam                          # 额外跑 SAM 2.1 掩码路线(见下)
pipeline.py --sam --sam-imgsz 512          # tiny@512 fp16, 传播可达 30 fps
pipeline.py --video "D:\xx.mp4" --tag v1   # 换视频/换标签
```

全片（2203 帧 / 73.4s / 852×480）各阶段实测耗时：detect 36s · stabilize 3s · angle 33s ·
attitude 66s（相机配准占大头）· 图表 ~10s。日志写在 `runs/pipeline_log.txt`。

---

## 四步结论速览

| 步骤 | 做什么 | 关键结果 |
|---|---|---|
| **1 检测** | YOLO26s 在 `Rocket Detect.v37i.yolo26` 上训练（100 轮 / 3 类） | 官方 valid mAP50 **0.928**；去泄漏重测 **0.914** ⇒ 分数可信 |
| **2 稳定化** | 类内 IoS 抑制嵌套框 + 双阈值续轨 + 底边模式状态机 + KF 补位 | 缺帧 **34 → 0**、嵌套重复框 **47 帧 → 0**、静止段平滑度无退化 |
| **3 倾角** | ROI 内"分段行平均 + 先定条带再精修 + 中心线鲁棒拟合" | 合成图增益 **1.0088**、残差 σ **0.024°**；真实图像落地段 σ **0.264°**、旋转增益 **0.90** |
| **4 姿态** | 相机静止性实测核查 + 相对倾角（+ 可选参数化面外角 β） | 相机实测静止（落地段滚转 +0.03°±0.06°）；落地段相对倾角 **+0.05°** |

**可交付 / 不可交付**

- ✅ 箭体轴线与**图像竖直**的夹角 φ(t)、与**图像上边缘**的夹角 `90−|φ|`、相对倾角、角速率
- ❌ 完整 3D 姿态（面外角 β / 总倾角）：单视角 + 长径比未知 + 宽度口径不一致，实测 β 可从 0° 变到 67°

---

## 另一条路线：SAM 2.1 掩码（可选，`feature/sam2-video-mask` 分支）

第 3 步除了"ROI 内梯度边缘检测"，还实现了一条**免训练分割**路线：
用第 2 步的稳定框提示 SAM 2.1，靠流式记忆传播出箭体掩码，再取**剪影的中心线**作为轴线。

```powershell
pipeline.py --sam --video-out        # = detect -> stabilize -> angle -> seg -> angle_mask -> attitude
python scripts/bench_sam2.py --model tiny --image-size 512 --half   # 实时性基准
python scripts/validate_mask_angle.py --tag iou70                   # 旋转注入验证
```

**为什么不猜也能测角**：箭体是**回旋体**，其剪影左右边界的中点连线严格等于轴线在像面的投影。
原方案必须靠"左右边平行 + 间距恒定"两条先验去挑哪两条边属于同一根筒子；掩码直接给出剪影，不需要猜。

| 指标 | 原方案（梯度边缘） | **SAM 2.1 掩码** |
|---|---|---|
| 旋转增益（理想 1.0） | 0.898 | **0.963** |
| 帧内残差 σ | 0.187° | **0.091°** |
| 落地静止段 46-66s σ | 0.223° | **0.095°** |
| 筒身宽度 `w_body` | ❌ 量到**涂装条纹**（≈0.5×框宽） | ✅ 剪影轮廓（≈0.85×框宽） |
| 全片耗时 | 33 s（纯 CPU，60 fps） | 126 s（GPU，tiny@512 fp16，端到端 15.5 fps） |
| 有效覆盖 | — | 1837/2203 帧有掩码，10-66s 可用率 92~100% |

**实时性**（RTX 4060 Laptop 8 GB，实测传播速度）：
`1024 fp32` 3.7 fps → `1024 fp16` 11.3 fps → **`512 fp16` 30.9 fps**。
**配置选错会得出"不能实时"的错误结论**（官方标称 47.2 fps 是 A100 的数字）。

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
  rocket_angle.py            ← 第 3 步: ROI 边缘检测 + 倾角核心模块
  validate_angle.py          ← 第 3 步: 三层验证(旋转注入/重复性/估计器对比)
  test_angle_synth.py        ← 第 3 步: 合成图单元测试(已知倾角)
  angle_report.py            ← 第 3 步: 倾角时间线图
  bench_sam2.py              ← SAM 路线: 本机实时性基准(回答"能不能实时")
  rocket_seg.py              ← SAM 路线: 分块传播 + 体检 + 重锚定 + 边界落盘
  rocket_mask_angle.py       ← SAM 路线: 掩码轮廓 -> 倾角 + 与原方案对拍
  validate_mask_angle.py     ← SAM 路线: 旋转注入测试(准入门槛)
  _seg_diag.py               ← SAM 路线: 可用率诊断
  _seg_vs_grad.py            ← SAM 路线: 掩码边界 vs 梯度拟合边线的逐帧对拍
  _seg_probe.py              ← SAM 路线: 掩码圈住了什么(宽度口径探测)
  rocket_attitude.py         ← 第 4 步: 相机配准 + 夹角换算 + 参数化 β
  validate_attitude.py       ← 第 4 步: 六节验证报告(含相机估计器灵敏度校验)
  attitude_report.py         ← 第 4 步: 姿态时间线图 + 标注视频
  _probe_bg.py               ← 前置探测: 有没有地平线 / 背景配准可不可用

runs/
  rocket_yolo26s/            ← 训练产物(best.pt 在用)
  diag/                      ← 检测存盘 JSON + 稳定化报告 + 曲线 + 标注视频
  angle/                     ← 倾角 CSV/JSON + 时间线图 + 标注视频 + 验证报告
  seg/                       ← SAM 掩码逐行边界 npz + 逐帧体检 CSV + 叠加视频
  angle_mask/                ← 掩码路线的倾角 CSV/JSON + 对拍报告 + 旋转注入报告
  attitude/                  ← 姿态 CSV/JSON + 时间线图 + 标注视频 + 验证报告
  pipeline_log.txt           ← 流水线完整日志

文档
  README.md                  ← 本文件(项目入口)
  TRAINING_REPORT.md         ← 第 1 步: 训练全过程总结
  DETECTION_STABILIZATION.md ← 第 2 步: 稳定化方案与踩坑
  ANGLE_ESTIMATION.md        ← 第 3 步: 测角算法与三层验证
  SAM2_MASK_ANGLE.md         ← 第 3 步(替代路线): SAM 2.1 掩码方案实现/实测/对拍
  ATTITUDE_ESTIMATION.md     ← 第 4 步: 姿态可行性论证与结论
  README_TRAIN.md            ← 训练环境与操作手册
```

---

## 环境

| 项 | 值 |
|---|---|
| Python | `E:\Anaconda\envs\yolo26`（3.12.14）**位于工作区之外**，不要挪进来 |
| torch / ultralytics | 2.14.0+cu130 / 8.4.147 |
| GPU | RTX 4060 Laptop 8 GB（只有第 1 步 `detect` 需要） |
| 物理内存 | 15.7 GB —— 本项目最紧的资源，训练时 `--workers` 必须 ≤ 4 |

所有临时/缓存目录（TEMP/TORCH_HOME/MPLCONFIGDIR/YOLO_CONFIG_DIR）都由脚本重定向到
工作区的 `.cache/`，**不写 C 盘**。第 2~4 步不需要 GPU，只要有 `runs/diag/dets_*.json` 就能重跑。

---

## 已知限制（重要）

1. **框的下边沿是双稳的**（≈250px / ≈400px 两档）——模型对"要不要把下半段算进框"不稳定。
   稳定化里已用"模式状态机"处理，但 `w_body` 因此可能量到涂装条纹而非筒身轮廓。
2. **前 20s 的倾角不可采信**——那段箭体在 640 输入下只有 21×53px，拟合出的条带仅 7~10px，
   估计值随目标变大单调趋向 0，是**尺度相关偏差**而非真实姿态。可信区间从 ~25s 起。
3. **完整 3D 姿态不可反算**。要解决需补：真实长径比/三维模型、可靠分割出筒身轮廓、
   相机内参与俯仰、或多视角。详见 `ATTITUDE_ESTIMATION.md` §6。
4. 第 3 步的旋转增益是 **0.90**（合成图 1.008）⇒ 绝对倾角存在约 ±10% 的尺度误差。

---

*每一步的关键数字都能在对应文档与 `runs/` 产物中复核；验证脚本可无 GPU 重跑。*
