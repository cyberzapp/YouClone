<#
.SYNOPSIS
    One-command setup, updater & launcher for YouClone on Windows 10/11 (NVIDIA CUDA / RTX 2050 4 GB+ or CPU).
.EXAMPLE
    .\setup_windows.ps1          # First-time setup (installs dependencies, downloads models, launches app)
    .\setup_windows.ps1 -Update  # Pull latest GitHub updates, rebuild UI, and launch (keeps existing .venv & models!)
    .\setup_windows.ps1 -Offline # Fast offline launch after setup
    .\setup_windows.ps1 -CpuOnly # Force CPU-only install (no NVIDIA GPU)
    .\setup_windows.ps1 -Clean   # Recreate Python .venv from scratch
#>
param(
    [switch]$Offline,
    [switch]$Update,
    [switch]$CpuOnly,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$env:HF_HOME = Join-Path $PSScriptRoot ".voxm5\huggingface"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:NEXT_TELEMETRY_DISABLED = "1"

function Require-Command($Name, $InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "Missing required tool: $Name" -ForegroundColor Red
        Write-Host "Install it first with: $InstallHint" -ForegroundColor Yellow
        exit 1
    }
}

Require-Command "node"   "winget install OpenJS.NodeJS.LTS"
Require-Command "ffmpeg" "winget install Gyan.FFmpeg"

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if ($Update) {
    Require-Command "git" "winget install Git.Git"
    Require-Command "npm" "winget install OpenJS.NodeJS.LTS"
    Write-Host "`n[1/4] Pulling latest YouClone updates from GitHub..." -ForegroundColor Cyan
    git pull origin main

    Write-Host "`n[2/4] Updating JavaScript packages..." -ForegroundColor Cyan
    if (Get-Command "pnpm" -ErrorAction SilentlyContinue) {
        pnpm install --frozen-lockfile
    } else {
        npm install
    }

    if (Test-Path $venvPy) {
        Write-Host "`n[3/4] Syncing Python packages..." -ForegroundColor Cyan
        if (Get-Command "uv" -ErrorAction SilentlyContinue) {
            uv pip install --python $venvPy -r requirements-windows.txt
        } else {
            & $venvPy -m pip install -r requirements-windows.txt
        }
    } else {
        Write-Host "Python .venv not found — running full initial setup..." -ForegroundColor Yellow
        $Update = $false
    }

    if ($Update) {
        Write-Host "`n[4/4] Rebuilding YouClone desktop UI..." -ForegroundColor Cyan
        npm run build
    }
}

if (-not $Offline -and -not $Update) {
    Require-Command "git" "winget install Git.Git"
    Require-Command "npm" "winget install OpenJS.NodeJS.LTS"
    if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv Python package manager..." -ForegroundColor Cyan
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
    }

    $hasNvidia = $false
    if (-not $CpuOnly -and (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
        $hasNvidia = $true
        Write-Host "NVIDIA GPU detected — configuring CUDA 12.4 acceleration (4 GB VRAM safe mode)." -ForegroundColor Green
    } else {
        Write-Host "No NVIDIA GPU detected (or -CpuOnly specified) — configuring CPU mode." -ForegroundColor Yellow
    }

    Write-Host "`n[1/5] Installing Node.js & Electron dependencies..." -ForegroundColor Cyan
    if (Get-Command "pnpm" -ErrorAction SilentlyContinue) {
        pnpm install --frozen-lockfile
    } else {
        npm install
    }

    Write-Host "`n[2/5] Setting up Python 3.12 backend (.venv)..." -ForegroundColor Cyan
    if ($Clean -or -not (Test-Path $venvPy)) {
        uv venv .venv --python 3.12 --clear
        if ($hasNvidia) {
            uv pip install --python $venvPy torch torchaudio --index-url https://download.pytorch.org/whl/cu124
            try {
                uv pip install --python $venvPy llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
            } catch {
                Write-Host "Pre-built cu124 llama-cpp-python wheel unavailable — installing standard wheel..." -ForegroundColor Yellow
                uv pip install --python $venvPy llama-cpp-python
            }
        } else {
            uv pip install --python $venvPy torch torchaudio --index-url https://download.pytorch.org/whl/cpu
            uv pip install --python $venvPy llama-cpp-python
        }
    }
    uv pip install --python $venvPy -r requirements-windows.txt

    Write-Host "`n[3/5] Setting up Seed-VC 44.1 kHz voice-cloning engine (.voxm5\seed-vc)..." -ForegroundColor Cyan
    $seedVcDir = Join-Path $PSScriptRoot ".voxm5\seed-vc"
    if (-not (Test-Path (Join-Path $seedVcDir "inference.py"))) {
        New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot ".voxm5") | Out-Null
        git clone --depth 1 https://github.com/Plachta/seed-vc.git $seedVcDir
    }
    $seedVenv = Join-Path $seedVcDir ".venv"
    $seedPy = Join-Path $seedVenv "Scripts\python.exe"
    if ($Clean -or -not (Test-Path $seedPy)) {
        uv venv $seedVenv --python 3.10
        if ($hasNvidia) {
            uv pip install --python $seedPy torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124
        } else {
            uv pip install --python $seedPy torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
        }
        uv pip install --python $seedPy -r (Join-Path $seedVcDir "requirements.txt")
    }

    Write-Host "`n[4/5] Downloading local AI models (~2.5 GB Stage 1 GGUF + SNAC + Whisper)..." -ForegroundColor Cyan
    & $venvPy -m backend.download

    Write-Host "`n[5/5] Building YouClone desktop UI..." -ForegroundColor Cyan
    npm run rebuild
    npm run build
}

if (-not (Test-Path $venvPy) -or -not (Test-Path "out\index.html") -or -not (Test-Path "node_modules\electron\cli.js")) {
    Write-Host "Setup incomplete. Run .\setup_windows.ps1 without -Offline first." -ForegroundColor Red
    exit 1
}

if ($Offline) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}

Write-Host "`nLaunching YouClone..." -ForegroundColor Green
& node node_modules\electron\cli.js .
