<#
.SYNOPSIS
    Installs PartLabeler on Windows: uv, Python, a .venv, PyTorch (CUDA or CPU), PartLabeler, the models.

.DESCRIPTION
    Double-click install_windows.bat, or run from the repository folder:

        powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [options]

        -Cpu           install the CPU build of PyTorch even if an NVIDIA GPU is present
        -NoModels      skip the model download (about 4 GB); get them later with
                       .venv\Scripts\python.exe -m engine.models
        -Dev           also install the developer tools (pytest, playwright, jupyterlab)
        -NoTeach       skip the Teach & Transfer extras (RF-DETR training)
        -Python 3.12   Python version for .venv (3.12 is what Colab uses)
        -Cuda cu130    pick the PyTorch build yourself: cu132, cu130, cu126 or cpu (default: auto)

    Safe to run again: it reuses .venv, keeps a working PyTorch and skips finished downloads.
    Written for Windows PowerShell 5.1 (no &&, no ternary, no ??).
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$NoModels,
    [switch]$Dev,
    [switch]$NoTeach,
    [string]$Python = "3.12",
    [ValidateSet("auto", "cu132", "cu130", "cu126", "cpu")]
    [string]$Cuda = "auto"
)

# Native tools (uv, python) print progress on stderr. In PowerShell 5.1 that must not count as an
# error, so failures are detected through $LASTEXITCODE after every call instead.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$Root = $PSScriptRoot
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$TotalSteps = 7
$script:StepNo = 0
$Clock = [System.Diagnostics.Stopwatch]::StartNew()

# ---------------------------------------------------------------------------------------------
# Which PyTorch build to install
#
# PyTorch 2.14.0 (the current release, checked 2026-09-25 on https://download.pytorch.org/whl/torch/
# and the pytorch.org "get started" data) ships these builds for Windows and Linux: cu126, cu130,
# cu132 and cpu. There is no cu128 build any more (the last one was torch 2.11).
#
# nvidia-smi prints the newest CUDA version the installed driver supports ("CUDA Version: 13.2").
# We pick the newest build that is not newer than that:
#
#   Driver branch (Windows / Linux minimum)           nvidia-smi "CUDA Version"   PyTorch build
#   R595 or newer                                     13.2 and up                  cu132
#   R580 - R594  (580.88 / 580.65.06)                 13.0 - 13.1                  cu130
#   R560 - R579  (560.76 / 560.28.03)                 12.6 - 12.9                  cu126
#   R525 - R559  (527.41 / 525.60.13)                 12.0 - 12.5                  cu126 (*)
#   older driver, or no NVIDIA GPU                    -                            cpu
#
#   (*) runs through CUDA minor-version compatibility; updating the driver is recommended.
#
# GPU generation matters too (compute capability, from nvidia-smi --query-gpu=compute_cap):
#   - CUDA 13 builds (cu130, cu132) need 7.5+ (Turing: T4, RTX 20xx, and newer). Older cards
#     (GTX 10xx Pascal 6.x, V100/Titan V Volta 7.0) get cu126.
#   - cu126 has no kernels for Blackwell (RTX 50xx, compute 10.x/12.x); those need driver R580+.
# ---------------------------------------------------------------------------------------------

function Write-Step([string]$Text) {
    $script:StepNo++
    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $script:StepNo, $TotalSteps, $Text) -ForegroundColor Cyan
}

function Write-Info([string]$Text) {
    Write-Host "      $Text"
}

function Write-Warn([string]$Text) {
    Write-Host "      WARNING: $Text" -ForegroundColor Yellow
}

function Stop-Install([string]$Text, [string]$Advice = "") {
    Write-Host ""
    Write-Host "ERROR: $Text" -ForegroundColor Red
    if ($Advice) {
        Write-Host $Advice -ForegroundColor Yellow
    }
    Write-Host ""
    exit 1
}

function Assert-LastExit([string]$Action, [string]$Advice = "") {
    if ($LASTEXITCODE -ne 0) {
        Stop-Install "$Action failed (exit code $LASTEXITCODE). The messages above say why." $Advice
    }
}

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    # Usual places when uv is installed but not on PATH (official installer, cargo, winget, pip --user).
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
    )
    foreach ($base in @((Join-Path $env:APPDATA "Python"), (Join-Path $env:LOCALAPPDATA "Programs\Python"))) {
        if (Test-Path -LiteralPath $base) {
            foreach ($dir in (Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue)) {
                $candidates += (Join-Path $dir.FullName "Scripts\uv.exe")
            }
        }
    }
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) {
            return $c
        }
    }
    return $null
}

function Find-NvidiaSmi {
    $cmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    foreach ($c in @((Join-Path $env:SystemRoot "System32\nvidia-smi.exe"),
                     (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe"))) {
        if (Test-Path -LiteralPath $c) {
            return $c
        }
    }
    return $null
}

function Get-TorchTag {
    # Build tag of the torch in .venv ("cu130", "cpu", ...); "" when torch is not installed.
    $v = & $VenvPy -c "import importlib.metadata as m; print(m.version('torch'))" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $v) {
        return ""
    }
    $v = "$v".Trim()
    $script:TorchInstalled = $v
    if ($v -match "\+(.+)$") {
        return $Matches[1]
    }
    return "cpu"   # PyPI wheels for Windows have no tag and are CPU-only
}

function ConvertTo-CudaVersion([string]$Tag) {
    # "cu132" -> 13.2, "cu126" -> 12.6; $null for anything else
    if ($Tag -match "^cu(\d+)(\d)$") {
        return [version]("{0}.{1}" -f $Matches[1], $Matches[2])
    }
    return $null
}

Write-Host ""
Write-Host "PartLabeler installer for Windows" -ForegroundColor Green
Write-Host "Folder: $Root"

if (-not (Test-Path -LiteralPath (Join-Path $Root "pyproject.toml"))) {
    Stop-Install "pyproject.toml not found next to install.ps1." "Run the installer from the PartLabeler folder (the one you cloned or unzipped)."
}
Set-Location -LiteralPath $Root

# About 15 GB are needed: PyTorch with CUDA (~5 GB), the other packages (~2 GB), the models (~4 GB),
# plus the download cache.
try {
    $drive = Get-PSDrive -Name ((Get-Item -LiteralPath $Root).PSDrive.Name) -ErrorAction Stop
    $freeGb = [math]::Round($drive.Free / 1GB)
    if ($freeGb -lt 15) {
        Write-Warn "only $freeGb GB free on drive $($drive.Name): the full install needs about 15 GB."
    }
} catch {
    # free space unknown (e.g. network drive); carry on
}

# ---- 1. uv ----------------------------------------------------------------------------------
Write-Step "uv (Python package manager)"
$uv = Find-Uv
if (-not $uv) {
    Write-Info "uv is not installed. Installing it with the official installer from https://astral.sh/uv ..."
    # A separate PowerShell process, so an 'exit' inside the downloaded script cannot end this one.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072; irm https://astral.sh/uv/install.ps1 | iex"
    Assert-LastExit "Installing uv" "Check your internet connection, or install uv yourself (https://docs.astral.sh/uv/getting-started/installation/) and run the installer again."
    # Pick up the PATH change the uv installer made, for this session only.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
    $uv = Find-Uv
    if (-not $uv) {
        Stop-Install "uv was installed but cannot be found." "Close this window, open a new one and run install_windows.bat again."
    }
}
$uvDir = Split-Path -Parent $uv
if (-not (($env:Path -split ";") -contains $uvDir)) {
    $env:Path = $uvDir + ";" + $env:Path
}
$uvVersion = & $uv --version
Write-Info "$uvVersion  ($uv)"

# ---- 2. Python + .venv ----------------------------------------------------------------------
Write-Step "Python $Python and the virtual environment (.venv)"
& $uv python install $Python
if ($LASTEXITCODE -ne 0) {
    # A damaged uv Python folder can make "install" fail although a usable Python is there.
    $foundPy = & $uv python find $Python --system --no-project 2>$null
    if ($LASTEXITCODE -eq 0 -and $foundPy) {
        Write-Warn "uv could not install Python $Python, but found one it can use: $foundPy"
    } else {
        Stop-Install "Installing Python $Python failed." "Check your internet connection and try again."
    }
}

$wantPy = (($Python -split "\.")[0..1]) -join "."
$makeVenv = $true
if (Test-Path -LiteralPath $VenvPy) {
    $havePy = & $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -eq 0 -and $havePy) {
        $makeVenv = $false
        Write-Info "Reusing .venv (Python $havePy)."
        if ("$havePy".Trim() -ne $wantPy) {
            Write-Warn ".venv has Python $havePy, not $wantPy. To switch, delete the .venv folder and run the installer again."
        }
    } else {
        Write-Warn ".venv exists but does not work (moved or copied from another PC?). Creating it again."
        & $uv venv --clear --python $Python .venv
        Assert-LastExit "Re-creating .venv"
        $makeVenv = $false
    }
}
if ($makeVenv) {
    & $uv venv --python $Python .venv
    Assert-LastExit "Creating .venv"
}

# ---- 3. GPU -> PyTorch build -----------------------------------------------------------------
Write-Step "Graphics card and PyTorch build"
$variant = "cpu"
$driverCuda = $null
$minCap = $null
$maxCap = $null
if ($Cpu) {
    Write-Info "-Cpu given: installing the CPU build."
} elseif ($Cuda -ne "auto") {
    $variant = $Cuda
    Write-Info "-Cuda $Cuda given: installing that build."
} else {
    $smi = Find-NvidiaSmi
    if (-not $smi) {
        Write-Info "No NVIDIA driver found (nvidia-smi is missing): installing the CPU build."
        Write-Info "Everything works on the CPU, only slower. With an NVIDIA GPU, install its driver and re-run."
    } else {
        $smiText = (& $smi 2>$null) -join "`n"
        if ($LASTEXITCODE -ne 0 -or -not ($smiText -match "CUDA Version:\s*(\d+)\.(\d+)")) {
            Write-Warn "nvidia-smi did not report a CUDA version (driver problem?). Installing the CPU build."
            Write-Info "Update the driver from https://www.nvidia.com/drivers and run the installer again."
        } else {
            $driverCuda = [version]("{0}.{1}" -f $Matches[1], $Matches[2])
            $gpus = @(& $smi "--query-gpu=name,driver_version,compute_cap" "--format=csv,noheader" 2>$null)
            if ($LASTEXITCODE -ne 0 -or $gpus.Count -eq 0) {
                # very old drivers cannot report compute_cap
                $gpus = @(& $smi "--query-gpu=name,driver_version" "--format=csv,noheader" 2>$null)
            }
            foreach ($line in $gpus) {
                $parts = @("$line" -split ",\s*")
                Write-Info ("GPU: {0}   driver {1}" -f $parts[0], $parts[1])
                if ($parts.Count -ge 3 -and $parts[2] -match "^\d+\.\d+$") {
                    $cap = [version]$parts[2]
                    if (-not $minCap -or $cap -lt $minCap) { $minCap = $cap }
                    if (-not $maxCap -or $cap -gt $maxCap) { $maxCap = $cap }
                }
            }
            Write-Info "The driver supports CUDA $driverCuda."
            if ($driverCuda -ge [version]"13.2") {
                $variant = "cu132"
            } elseif ($driverCuda -ge [version]"13.0") {
                $variant = "cu130"
            } elseif ($driverCuda -ge [version]"12.0") {
                $variant = "cu126"
                if ($driverCuda -lt [version]"12.6") {
                    Write-Warn "this driver is older than 560. PyTorch should still run, but updating the driver is recommended."
                }
            } else {
                Write-Warn "this driver is too old for current PyTorch (needs 527 or newer). Installing the CPU build."
                Write-Info "Update the driver from https://www.nvidia.com/drivers and run the installer again."
            }
            if ($minCap -and $minCap -lt [version]"7.5" -and $variant -ne "cpu" -and $variant -ne "cu126") {
                Write-Info "GPU compute capability $minCap is older than Turing: using cu126 (CUDA 13 builds dropped it)."
                $variant = "cu126"
            }
            if ($maxCap -and $maxCap -ge [version]"10.0" -and ($variant -eq "cu126" -or $variant -eq "cpu")) {
                Stop-Install "This RTX 50-series (Blackwell) GPU needs NVIDIA driver 580 or newer for PyTorch." "Update the driver from https://www.nvidia.com/drivers and run the installer again, or run it with -Cpu."
            }
        }
    }
}

# Keep a CUDA PyTorch that is already installed and works with this driver (no 2.5 GB re-download).
$script:TorchInstalled = ""
$haveTag = Get-TorchTag
$reinstall = $false
if ($haveTag) {
    if ($haveTag -eq $variant) {
        Write-Info "PyTorch $($script:TorchInstalled) is already installed."
    } else {
        $haveCuda = ConvertTo-CudaVersion $haveTag
        $keep = $false
        if ($Cuda -eq "auto" -and -not $Cpu -and $variant -ne "cpu" -and $haveCuda -and $driverCuda) {
            $fitsDriver = $haveCuda -le $driverCuda
            $fitsGpu = -not ($minCap -and $minCap -lt [version]"7.5" -and $haveCuda -ge [version]"13.0")
            $keep = $fitsDriver -and $fitsGpu
        }
        if ($keep) {
            Write-Info "Keeping the installed PyTorch $($script:TorchInstalled): it works with this driver ($variant would be the first choice for a new install)."
            $variant = $haveTag
        } else {
            Write-Info "Replacing PyTorch $($script:TorchInstalled) with the $variant build."
            $reinstall = $true
        }
    }
}
Write-Info "PyTorch build: $variant"

# ---- 4. PyTorch ------------------------------------------------------------------------------
Write-Step "PyTorch ($variant) - the first install downloads about 2.5 GB"
$indexUrl = "https://download.pytorch.org/whl/$variant"
$torchArgs = @("pip", "install", "--python", $VenvPy, "torch", "torchvision", "--index-url", $indexUrl)
if ($reinstall) {
    $torchArgs += @("--reinstall-package", "torch", "--reinstall-package", "torchvision")
}
& $uv @torchArgs
Assert-LastExit "Installing PyTorch" "Check your internet connection and run the installer again (finished downloads are kept). If it keeps failing, try -Cpu."

# ---- 5. PartLabeler --------------------------------------------------------------------------
$extras = @()
if (-not $NoTeach) { $extras += "teach" }
if ($Dev) { $extras += "dev" }
$target = "."
if ($extras.Count -gt 0) {
    $target = ".[" + ($extras -join ",") + "]"
}
Write-Step "PartLabeler and its packages ($target)"
# Pin the torch just installed, so no other package can swap it for a different (e.g. CPU-only) build.
$pins = & $VenvPy -c "import importlib.metadata as m; [print(p + '==' + m.version(p).split('+')[0]) for p in ('torch', 'torchvision')]"
Assert-LastExit "Reading the PyTorch version"
$pinFile = Join-Path $env:TEMP "partlabeler-torch-pins.txt"
Set-Content -LiteralPath $pinFile -Value $pins -Encoding Ascii
& $uv pip install --python $VenvPy -e $target --constraints $pinFile
Assert-LastExit "Installing PartLabeler" "Check your internet connection and run the installer again."
Remove-Item -LiteralPath $pinFile -ErrorAction SilentlyContinue

# ---- 6. Models -------------------------------------------------------------------------------
Write-Step "Models (SAM 3 and DINOv3, about 4 GB, checksum-verified)"
if ($NoModels) {
    Write-Info "Skipped (-NoModels). Download them later with:  .venv\Scripts\python.exe -m engine.models"
} else {
    & $VenvPy -m engine.models
    Assert-LastExit "Downloading the models" "Check your internet connection and run the installer again: finished files are kept."
}

# ---- 7. Self-check ---------------------------------------------------------------------------
Write-Step "Self-check"
$check = "import torch, engine.hw as hw, engine.project, transformers, anywidget, fastapi; " +
         "print('      torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); " +
         "print('      device:', hw.describe())"
$checkOut = & $VenvPy -c $check
$checkCode = $LASTEXITCODE
$checkOut | ForEach-Object { Write-Host $_ }
if ($checkCode -ne 0) {
    Stop-Install "The self-check failed: PartLabeler cannot be imported." "Run the installer again; if it still fails, send the messages above to the team."
}
if ($variant -ne "cpu" -and -not (($checkOut -join " ") -match "CUDA available: True")) {
    Write-Warn "PyTorch was installed for CUDA but cannot use the GPU. Restart the PC after a driver update,"
    Write-Warn "or run the installer with -Cpu. PartLabeler still works on the CPU, only slower."
}
if (-not $NoTeach) {
    & $VenvPy -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('rfdetr') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "rfdetr (Teach & Transfer) is not importable."
    }
}

$minutes = [math]::Round($Clock.Elapsed.TotalMinutes, 1)
Write-Host ""
Write-Host "PartLabeler is installed ($minutes min)." -ForegroundColor Green
Write-Host ""
Write-Host "Start it:   double-click run_windows.bat"
Write-Host "   or run:  .venv\Scripts\python.exe -m engine.cli app"
Write-Host "Check:      .venv\Scripts\python.exe -m engine.cli doctor"
if ($NoModels) {
    Write-Host "Models:     .venv\Scripts\python.exe -m engine.models   (needed before annotating)"
}
if ($Dev) {
    Write-Host "Tests:      .venv\Scripts\python.exe -m pytest tests"
    Write-Host "UI tests need a browser once:  .venv\Scripts\python.exe -m playwright install chromium"
}
Write-Host ""
exit 0
