# =============================================================================
#  Rocket Detect — YOLO26s 训练环境一键配置 (Windows PowerShell)
#
#  用法: 在 PowerShell 中执行
#          Set-ExecutionPolicy -Scope Process Bypass -Force
#          .\setup_env.ps1
#
#  设计原则:
#    * conda 环境装在 E 盘【工作区之外】(E:\Anaconda\envs\yolo26), 不占 C 盘
#      —— 工作区内的批量删除会被沙箱 50 文件阈值拦截, 会导致环境损坏, 详见 README_TRAIN.md
#    * pip 缓存 / 临时目录 / torch / matplotlib / ultralytics 配置全部重定向到工作区
#    * torch 从国内镜像装 CUDA 版(PyPI 上是 CPU 版, download.pytorch.org 在本网络不可达)
#    * 全程只创建不删除: 先装 CUDA torch, 再装其余依赖
#    * 可重复执行(幂等)
#
#  已验证组合: Python 3.12.14 + torch 2.14.0+cu130 + torchvision 0.29.0+cu130
#              + ultralytics 8.4.147 + RTX 4060 (驱动 CUDA 13.2)
# =============================================================================

$ErrorActionPreference = "Stop"

$Root      = "E:\RocketAttitudeEstimation"
$EnvPath   = "E:\Anaconda\envs\yolo26"      # 放在工作区之外
$Cache     = Join-Path $Root ".cache"
$Conda     = "E:\Anaconda\Scripts\conda.exe"
$Mirror    = "https://mirrors.aliyun.com/pytorch-wheels"
$TorchVer  = "2.14.0"
$TvVer     = "0.29.0"
$CudaTag   = "cu130"          # 需要驱动 >= 580; 老驱动可换成 cu128(但最高只有 torch 2.11.0)
$WheelDir  = Join-Path $Cache "wheels"

# ---- 1. 重定向缓存与临时目录(关键: 避免写 C 盘) ----------------------------
foreach ($d in @("tmp", "pip", "torch", "mpl", "yolo", "wheels")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Cache $d) | Out-Null
}
$env:TEMP            = Join-Path $Cache "tmp"
$env:TMP             = Join-Path $Cache "tmp"
$env:TMPDIR          = Join-Path $Cache "tmp"
$env:PIP_CACHE_DIR   = Join-Path $Cache "pip"
$env:TORCH_HOME      = Join-Path $Cache "torch"
$env:MPLCONFIGDIR    = Join-Path $Cache "mpl"
$env:YOLO_CONFIG_DIR = Join-Path $Cache "yolo"

# ---- 2. 创建 conda 环境 ----------------------------------------------------
Write-Host "[1/4] 创建/复用 conda 环境: $EnvPath" -ForegroundColor Cyan
if (Test-Path (Join-Path $EnvPath "python.exe")) {
    Write-Host "      已存在, 跳过"
} else {
    & $Conda create -p $EnvPath python=3.12 pip -y
    if ($LASTEXITCODE -ne 0) { throw "conda create 失败" }
}
$Py = Join-Path $EnvPath "python.exe"
& $Py -m pip install --upgrade pip

# ---- 3. 从国内镜像获取 CUDA 版 torch / torchvision -------------------------
Write-Host "[2/4] 下载 CUDA 版 torch($TorchVer+$CudaTag) ..." -ForegroundColor Cyan
$TorchWhl = Join-Path $WheelDir "torch-$TorchVer+$CudaTag-cp312-cp312-win_amd64.whl"
$TvWhl    = Join-Path $WheelDir "torchvision-$TvVer+$CudaTag-cp312-cp312-win_amd64.whl"
if (-not (Test-Path $TorchWhl) -or -not (Test-Path $TvWhl)) {
    & $Py (Join-Path $Root "scripts\fetch_wheels.py") `
        "$Mirror/$CudaTag/torch-$TorchVer%2B$CudaTag-cp312-cp312-win_amd64.whl" `
        "$Mirror/$CudaTag/torchvision-$TvVer%2B$CudaTag-cp312-cp312-win_amd64.whl" `
        -o $WheelDir -t 8
    if ($LASTEXITCODE -ne 0) { throw "wheel 下载失败" }
}

Write-Host "[3/4] 安装 torch / torchvision ..." -ForegroundColor Cyan
& $Py -m pip install $TorchWhl $TvWhl
if ($LASTEXITCODE -ne 0) { throw "torch 安装失败" }

# ---- 4. 安装 ultralytics 及其余依赖 ---------------------------------------
Write-Host "[4/4] 安装 ultralytics 及其余依赖 ..." -ForegroundColor Cyan
& $Py -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }

# ---- 验证 ------------------------------------------------------------------
Write-Host ""
Write-Host "验证 CUDA:" -ForegroundColor Green
& $Py -c "import torch; print(' torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host ""
Write-Host "配置完成。自检:" -ForegroundColor Green
& $Py (Join-Path $Root "train.py") --check-only
