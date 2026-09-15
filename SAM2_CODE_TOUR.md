# SAM 2.1 路线 —— 从零读代码指南

**目标**：把"YOLO 稳定框 → SAM 2.1 掩码 → 逐行边界 → 倾角 φ"这条链路自己读通。
**原则**：先看产物流向，再看单个函数。每个阶段都能用一条命令实际验证，不要干读。

所有行号来自当前分支（`sam2-video-mask`），可能随提交轻微漂移；以函数名定位更稳。

---

## 第 0 层：先跑一遍，建立"东西长什么样"的感觉（15 分钟）

在读任何代码之前，先把产物看一遍，否则后面看到 `bounds_*.npz`、`mask_stats_*.csv`
这些名字时不知道它们对应画面上什么。

```powershell
# 产物已经生成好了, 直接看:
#   runs/seg/compare_timeline_s512_vs_s1024.png   全片曲线对照(先不看视频)
#   runs/seg/compare_stills_s512_vs_s1024.png     6 帧抽帧拼图
#   runs/seg/compare_overlay_s1024.mp4            逐帧视频

# 想自己重跑一条最短链路(约 2 分钟):
python scripts\rocket_seg.py --image-size 512 --start 1400 --end 1600 --tag demo
python scripts\rocket_mask_angle.py --tag demo
```

**看的时候记住三件事**，后面一直会用到：

| 概念 | 它是什么 | 落盘在哪 |
|---|---|---|
| **逐行边界** `xl(y)/xr(y)` | 掩码剪影每一行的最左/最右，只存这个不存整张掩码 | `runs/seg/bounds_<tag>.npz` |
| **筒身段** | 剪影里"宽度恒定"的那一段（要剔掉头锥和支腿） | 现算，见 `body_rows` |
| **中心线** | 筒身段里 `(xl+xr)/2` 的拟合直线，斜率就是倾角 | `runs/angle_mask/angles_<tag>.csv` |

---

## 第 1 层：主流程 `scripts/rocket_seg.py`（655 行）★ 核心

**这是唯一必须精读的文件。** 按下面顺序读，不要从上往下硬啃。

### 1) 先看数据结构（5 分钟）

| 位置 | 看什么 |
|---|---|
| `SegConfig` L114 | 所有可调参数集中在这里。重点看两组：**体检门限**（`escape_px_min` / `escape_frac` / `area_lo` / `area_hi` / `center_max` / `bad_persist`）和**分块**（`chunk`）。每个字段上面都有注释说明为什么是这个值 |
| `FrameSeg` L145 | 单帧的全部输出字段。看懂 `ok` vs `has` 的区别：`has`=有掩码，`ok`=通过了体检 |

### 2) 再看三个"纯函数"，它们是整个算法的数学核心（20 分钟）

| 位置 | 作用 | 为什么这么写 |
|---|---|---|
| `mask_geom` L177 | 掩码 → 最大连通域 → 逐行左右边界 `xl/xr`、质心、行宽中位 | 取**最大连通域**是因为 SAM 偶尔会分出碎片；`w_row_med` 后面用来定筒身段 |
| `health` L219 | 判断"掩码有没有漂移出稳定框" → 返回 reason | **这个文件最值得细读的函数**。注释里写清了为什么不能用 IoU、也不能用固定比例的 contain，最终选择"越界像素数 escape" |
| `_mem_str` L86 | 一行内存快照 | 分块跑时用它确认"每块真的释放了" |

> `health` 的返回值里 `escape` / `esc_tol` / `contain` / `area_ratio` / `center_off`
> 都会落进 CSV，可以直接对着 `runs/seg/mask_stats_s512.csv` 一行行看，非常有助于理解。

### 3) 再看主流程（30 分钟）

`main` L644 → `run` L409 → `Sam2Segmenter.segment_chunk` L274

- **`run` L409** —— 分块循环。重点看三件事：
  1. 帧怎么来的（`extract_frames` 抽帧写 JPEG，缓存复用）
  2. 块怎么切、为什么切（**`init_state` 会把整块帧张量常驻显存**，全片 27.7 GB 放不下）
  3. 块结束后怎么释放（`del state` + `gc.collect()` + `empty_cache()`）
- **`segment_chunk` L274** —— 单块内的完整生命周期，是最该逐行读的一段：
  1. 选锚点帧（**必须是块内第一个"有稳定框"的帧**，否则 SAM 会抛异常）
  2. `add_new_points_or_box` 下提示
  3. `propagate_in_video` 逐帧产出
  4. **每帧跑 `health` 体检**，连续 `bad_persist` 帧不过就重锚定
  5. 重锚定的两个关键约束：**新锚点必须严格更靠后**（否则原地打转）、
     **额度用尽不能 break 掉整块**（要靠记忆把余下帧走完，如实标成不可信）
- **`save` L583** —— 落盘。注意它存的是**逐行边界 npz + 逐帧体检 CSV**，不是整张掩码，
  目的是让后续测角调参**不用再上 GPU**。

### 4) 最后看 `Sam2Segmenter.load` L265

很短，但有两个环境细节值得记：`sam2` 是用 `sys.path` 挂 `third_party/sam2` 直连源码
（没做 pip 安装），以及 `hydra_overrides_extra` 怎么把 `image_size` 传进去。

---

## 第 2 层：掩码 → 倾角 `scripts/rocket_mask_angle.py`（298 行）

比第 1 层轻松得多，因为**数学是复用原方案的**（同一套 IRLS 中心线拟合）。

| 位置 | 看什么 |
|---|---|
| `MaskAngleConfig` L57 | 筒身段判定与拟合的参数 |
| **`body_rows` L72** | **本文件最关键的函数**：用"行宽鲁棒中位"从剪影里切出真正的筒身段，剔掉头锥/支腿。这一段做不好，倾角就废了 |
| **`estimate` L97** | 逐行取中点 → IRLS 拟合中心线 → 换算成 φ；同时算 `w_body` |
| `run` L127 | 遍历所有帧，套用 `rocket_angle.smooth_and_reference` 做平滑与基准 |
| `compare` L190 | 与另一路线/另一配置逐帧对拍（`--vs` 参数） |

**怎么验证读懂了**：`runs/angle_mask/angles_s512.csv` 里有 `fit_rms` 和 `n_rows` 两列。
`fit_rms` 大 = 拟合不好，`n_rows` 小 = 筒身段没找对。翻几行极端值对应的画面就明白了。

---

## 第 3 层：验证（读懂"凭什么说它更准"）

| 文件 | 行数 | 读什么 |
|---|---|---|
| `scripts/validate_mask_angle.py` | 225 | **旋转注入**，唯一的准入门槛。`rot_box` L64 里有一句关键注释：`cv2.getRotationMatrix2D` 正角是逆时针，所以**理想增益是 −1 不是 +1**。主流程在 `main` L75，做的是"单帧序列 + 视频预测器"（不用 SAM2ImagePredictor，因为它在 torch 2.14 下有 bug） |
| `scripts/bench_sam2.py` | 360 | 实时性基准。`bench_video` L137 测传播速度，`bench_image` L203 测逐帧图像模式。结论：fp16 值 3 倍、降分辨率值 2.7 倍 |

**为什么必须做旋转注入**：项目踩过的坑——Otsu+minAreaRect 的 σ 比最终方案还小，
但旋转增益只有 0.196，**"稳定"是假象**。只看 σ 会选错方案。

---

## 第 4 层：可视化与诊断（用来对照理解，不必精读）

| 文件 | 行数 | 用途 |
|---|---|---|
| `scripts/compare_video.py` | 489 | 对照可视化。`build_timeline` L118 画双段曲线带（φ + Δφ），`draw_frame` L241 画单帧叠加。注意：**cv2.putText 只支持 ASCII，中文必须走 PIL** |
| `scripts/_seg_diag.py` | 100 | 可用率诊断。产出 `runs/seg/diag_*.txt`，调体检门限时必看 |
| `scripts/_seg_vs_grad.py` | 181 | 掩码边界 vs 原方案拟合边线的逐帧对拍。**"原方案量到涂装条纹"这个结论就是从这里出来的** |
| `scripts/_seg_probe.py` | 129 | 探测"掩码到底圈住了什么" |

---

## 第 5 层（可选）：SAM 2 本体

不用全读（1223 + 909 行），**只读 3 个函数**就够理解它在干什么：

| 文件 | 函数 | 位置 | 一句话 |
|---|---|---|---|
| `sam2_video_predictor.py` | `init_state` | L42 | 把一段视频预处理成常驻特征（**这就是必须分块的根因**） |
| | `add_new_points_or_box` | L161 | 下提示并**立刻**得到该帧掩码 |
| | `propagate_in_video` | L546 | 带着记忆逐帧往后推 |
| `sam2_base.py` | `_prepare_memory_conditioned_features` | L497 | 记忆注意力怎么和当前帧特征融合（想深入再看） |
| `build_sam.py` | `build_sam2_video_predictor` | L100 | 怎么用 hydra 组装模型 |

---

## 前置依赖（不是 SAM 的，但读不懂会卡住）

| 文件 | 只读这些 | 为什么 |
|---|---|---|
| `scripts/rocket_track.py` | `track_frames` L445、`RocketTracker.step` L322、`suppress_nested` L93 | 第 2 步的稳定框就是 SAM 的提示来源；`ios_xyxy` L61 解释了为什么嵌套框 NMS 抓不到 |
| `scripts/rocket_angle.py` | `fit_band_axis` L373、`_robust_line` L175、`smooth_and_reference` L770 | 中心线拟合与平滑是**复用**的，原方案的数学在这里 |
| `pipeline.py` | `_stages` L76 | 看 `seg` / `angle_mask` 两个可选阶段怎么接进流水线 |

---

## 完整文件清单

| 文件 | 行数 | 层次 | 必读函数 |
|---|---|---|---|
| `scripts/rocket_seg.py` | 655 | 1 核心 | `mask_geom` 177 · `health` 219 · `segment_chunk` 274 · `run` 409 · `save` 583 |
| `scripts/rocket_mask_angle.py` | 298 | 2 | `body_rows` 72 · `estimate` 97 · `compare` 190 |
| `scripts/validate_mask_angle.py` | 225 | 3 | `rot_box` 64 · `main` 75 |
| `scripts/bench_sam2.py` | 360 | 3 | `bench_video` 137 · `bench_image` 203 |
| `scripts/compare_video.py` | 489 | 4 | `build_timeline` 118 · `draw_frame` 241 |
| `scripts/_seg_diag.py` | 100 | 4 | `main` 35 |
| `scripts/_seg_vs_grad.py` | 181 | 4 | `_edge_line` 44 |
| `scripts/_seg_probe.py` | 129 | 4 | `main` 41 |
| `scripts/rocket_track.py` | 451 | 前置 | `track_frames` 445 · `RocketTracker.step` 322 |
| `scripts/rocket_angle.py` | 948 | 前置 | `fit_band_axis` 373 · `smooth_and_reference` 770 |
| `pipeline.py` | 455 | 前置 | `_stages` 76 |
| `third_party/sam2/sam2/sam2_video_predictor.py` | 1223 | 5 可选 | `init_state` 42 · `add_new_points_or_box` 161 · `propagate_in_video` 546 |
| `third_party/sam2/sam2/modeling/sam2_base.py` | 909 | 5 可选 | `_prepare_memory_conditioned_features` 497 |
| `third_party/sam2/sam2/build_sam.py` | 174 | 5 可选 | `build_sam2_video_predictor` 100 |

**背景文档**（读代码时随时翻）：`SAM2_MASK_ANGLE.md`（实现/实测/对拍，§3 讲实现）、
`SAM2_PROPOSAL.md`（方案对比与选型理由）、`ANGLE_ESTIMATION.md`（第 3 步原方案，
理解"复用了什么数学"）。

---

## 自检：答得出来就说明读懂了

1. 为什么必须**分块**？不分的代价是多少？（提示：算一下 1024² 全片常驻要多少内存）
2. 体检判据为什么**不能用 IoU、也不能用固定比例的 contain**？最终用了什么？
3. 重锚定为什么要**强制新锚点更靠后**？额度用尽后为什么**不能 break**？
4. `xl/xr` 存的是什么？为什么不直接存整张掩码？
5. 为什么"只看 σ"会选错方案？旋转注入测的是什么？
6. 旋转注入的理想增益是 **−1** 而不是 +1，为什么？
7. 原方案量到的 `w_body` 为什么偏小？它的拟合边相对剪影是**平行错开**还是**倾斜**？
8. 1024 相比 512，**帧内残差 σ 变了吗**？如果没变，那它买到的是什么？
