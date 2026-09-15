#!/usr/bin/env bash
# =============================================================================
#  Rocket Detect — YOLO26s 训练环境配置脚本 (Git Bash / WSL 风格)
#
#  等价于 setup_env.ps1, 便于在 bash 中一键复现环境。
#  用法:  bash setup_env.sh
#
#  ⚠ 两条必须遵守的规则(踩过坑, 照做可避免环境损坏):
#
#  1) 环境必须建在【工作区之外】, 这里用 E:\Anaconda\envs\yolo26。
#     原因: 沙箱对工作区(E:\RocketAttitudeEstimation)内的批量删除有 50 文件阈值保护,
#     pip/conda 替换大包时会一次删除上万个文件, 触发 [SAFE_DELETE_BULK_REJECTED]
#     导致删除中途被打断; 又因为 conda 用硬链接共享包缓存, 还会连带损坏
#     E:\Anaconda\pkgs 里的同名 inode(文件被截断为 0 字节)。
#     症状: python 报 "Could not find platform independent libraries <prefix>",
#           Lib/ 下 .py 全部消失。修复: 用 conda_package_handling 重新解压对应包。
#
#  2) 全程必须【只创建、不删除】。必须先装 CUDA 版 torch, 再装其余依赖;
#     千万不要先装 PyPI 上的 CPU 版 torch 再换成 CUDA 版 —— 那次替换就是上面第 1 条的
#     触发源(卸载 torch 会删除数千个文件)。
#
#  3) 安装过程务必在后台/独立终端跑完, 不要带超时运行。
#
#  已验证组合: Python 3.12.14 + torch 2.14.0+cu130 + torchvision 0.29.0+cu130
#              + ultralytics 8.4.147 + RTX 4060 (驱动 CUDA 13.2)
# =============================================================================
set -euo pipefail

WROOT='E:\RocketAttitudeEstimation'                 # Windows 形式(给 conda/-p 用)
BROOT='/e/RocketAttitudeEstimation'                 # Bash 形式
ENVP='E:\Anaconda\envs\yolo26'                      # 环境放在工作区之外(见上)
BENV='/e/Anaconda/envs/yolo26'
CONDA='/e/Anaconda/Scripts/conda.exe'
PY="$BENV/python.exe"
MIRROR='https://mirrors.aliyun.com/pytorch-wheels'
TORCH='2.14.0'; TV='0.29.0'; CUTAG='cu130'

# ---- 1. 缓存/临时目录全部重定向到 E 盘工作区 -------------------------------
for d in tmp pip torch mpl yolo wheels; do mkdir -p "$BROOT/.cache/$d"; done
export TEMP="E:\\RocketAttitudeEstimation\\.cache\\tmp"
export TMP="$TEMP"
export TMPDIR="$TEMP"
export PIP_CACHE_DIR="E:\\RocketAttitudeEstimation\\.cache\\pip"
export TORCH_HOME="E:\\RocketAttitudeEstimation\\.cache\\torch"
export MPLCONFIGDIR="E:\\RocketAttitudeEstimation\\.cache\\mpl"
export YOLO_CONFIG_DIR="E:\\RocketAttitudeEstimation\\.cache\\yolo"

# ---- 2. 创建 conda 环境 ----------------------------------------------------
echo "[1/4] 创建 conda 环境 $ENVP"
if [ -x "$PY" ] && "$PY" -c "import sys" 2>/dev/null; then
    echo "      已存在且可用, 跳过"
else
    "$CONDA" create -p "$ENVP" python=3.12 pip -y
fi

# ---- 3. 安装 CUDA 版 torch / torchvision -----------------------------------
echo "[2/4] 准备 torch $TORCH+$CUTAG / torchvision $TV+$CUTAG"
TW="$BROOT/.cache/wheels/torch-$TORCH+$CUTAG-cp312-cp312-win_amd64.whl"
VW="$BROOT/.cache/wheels/torchvision-$TV+$CUTAG-cp312-cp312-win_amd64.whl"
if [ ! -f "$TW" ] || [ ! -f "$VW" ]; then
    "$PY" "$BROOT/scripts/fetch_wheels.py" \
        "$MIRROR/$CUTAG/torch-$TORCH%2B$CUTAG-cp312-cp312-win_amd64.whl" \
        "$MIRROR/$CUTAG/torchvision-$TV%2B$CUTAG-cp312-cp312-win_amd64.whl" \
        -o "$BROOT/.cache/wheels" -t 8
fi

echo "[3/4] 安装 torch / torchvision"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "$TW" "$VW"

# ---- 4. 安装 ultralytics 及其余依赖 ---------------------------------------
echo "[4/4] 安装 ultralytics 及其余依赖"
"$PY" -m pip install -r "$BROOT/requirements.txt"

# ---- 验证 ------------------------------------------------------------------
echo
echo "=== 验证 ==="
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
"$PY" "$BROOT/train.py" --check-only
echo "全部完成。"
