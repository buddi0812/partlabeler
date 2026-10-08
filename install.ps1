<#
.SYNOPSIS
    Installs PartLabeler on Windows: uv, Python, a .venv, PyTorch (CUDA or CPU), PartLabeler, the models.

.DESCRIPTION
    Double-click install_windows.bat, or run from the repository folder:

        powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [options]

        -Cpu           install the CPU build of PyTorch even with an NVIDIA GPU present
        -NoModels      skip the model download (about 4 GB); get them later with
                       .venv\Scripts\python.exe -m engine.models
        -Dev           also install the developer tools (pytest, playwright, jupyterlab)
        -NoTeach       skip the Teach & Transfer extras (RF-DETR training)
        -Python 3.12   Python version for .venv (3.12 is what Colab uses)
        -Cuda cu130    pick the PyTorch build yourself: cu132, cu130, cu126 or cpu (default: auto)
        -Prefetch DIR  download the torch/torchvision wheels this machine would use into DIR and exit
                       (resumable; no .venv change); writes DIR\prefetch.json ({"variant", "torch", "files",
                       "complete"}, "complete" set only once every file is downloaded and checked)
        -Wheels DIR    install torch/torchvision from a prefetch DIR instead of downloading now (offline;
                       refuses unless DIR\prefetch.json is complete and its variant matches this machine's)
        -Plan          print the PyTorch decision as JSON and change nothing on disk; with -Prefetch DIR,
                       also reports whether DIR already has a complete, matching prefetch (still no download)
        -NoSelfCheck   skip step 7 (importing PartLabeler to check it); for a build machine that only
                       collects packages (e.g. for a USB stick) and never actually runs them

    Safe to run again: it reuses .venv, skips finished downloads and never switches an already-installed
    PyTorch build on its own (CPU stays CPU, a CUDA build stays that CUDA build) - only -Cpu or -Cuda <build>
    does that; a CUDA build runs fine with no GPU present (engine/hw.py picks the device at runtime), so there
    is nothing to switch even on a box with no GPU. -Plan makes a couple of small requests to the package
    index, to report the real wheels a fresh install (or an explicit -Cuda <build>) would fetch; it still
    changes nothing.
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
    [string]$Cuda = "auto",
    [string]$Prefetch = "",
    [string]$Wheels = "",
    [switch]$Plan,
    [switch]$NoSelfCheck
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
$script:PlanLog = @()      # -Plan: one line per Write-Step/Write-Info, for the JSON "steps" field
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
    $script:PlanLog += ("[{0}/{1}] {2}" -f $script:StepNo, $TotalSteps, $Text)
    if ($Plan) { return }
    Write-Host ""
    Write-Host ("[{0}/{1}] {2}" -f $script:StepNo, $TotalSteps, $Text) -ForegroundColor Cyan
}

function Write-Info([string]$Text) {
    $script:PlanLog += "      $Text"
    if ($Plan) { return }
    Write-Host "      $Text"
}

function Write-Warn([string]$Text) {
    $script:PlanLog += "      WARNING: $Text"
    if ($Plan) { return }
    Write-Host "      WARNING: $Text" -ForegroundColor Yellow
}

function Stop-Install([string]$Text, [string]$Advice = "") {
    if ($Plan) {
        # -Plan changes nothing: a problem that would stop a real run is reported as JSON, not as an exit.
        [ordered]@{ ok = $false; error = $Text; advice = $Advice; steps = $script:PlanLog } | ConvertTo-Json -Depth 6
        exit 1
    }
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

function Test-VcRuntime {
    # PyTorch's DLLs need the Microsoft Visual C++ 2015-2022 runtime (x64). Binaries built with Visual Studio 17.10+
    # crash with a runtime older than 14.40, so an older one counts as missing. A fresh Windows may have none.
    $sys = Join-Path $env:SystemRoot "System32"
    foreach ($dll in "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll") {
        if (-not (Test-Path -LiteralPath (Join-Path $sys $dll))) { return $false }
    }
    foreach ($key in "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                     "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64") {
        $rt = Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue
        if ($rt -and $null -ne $rt.Major) {
            return ([int]$rt.Major -gt 14) -or ([int]$rt.Major -eq 14 -and [int]$rt.Minor -ge 40)
        }
    }
    return $true                                    # DLLs present, version unknown (installed by another app)
}

function Install-VcRuntime {
    $url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    $exe = Join-Path $env:TEMP "vc_redist.x64.exe"
    Write-Info "Downloading it from Microsoft ($url, about 25 MB) ..."
    $oldProgress = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"         # Windows PowerShell's progress bar slows downloads a lot
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    } catch {
        Stop-Install "Downloading the Microsoft Visual C++ runtime failed: $($_.Exception.Message)" "Install it yourself from $url and run the installer again."
    } finally {
        $ProgressPreference = $oldProgress
    }
    $sig = Get-AuthenticodeSignature -LiteralPath $exe
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "O=Microsoft Corporation") {
        Remove-Item -LiteralPath $exe -ErrorAction SilentlyContinue
        Stop-Install "The downloaded Visual C++ runtime is not signed by Microsoft; it was not run." "Install it yourself from $url and run the installer again."
    }
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) {
        Write-Info "Installing it: Windows asks for administrator permission (click Yes) ..."
    } else {
        Write-Info "Installing it ..."
    }
    try {
        $run = @{ FilePath = $exe; ArgumentList = @("/install", "/quiet", "/norestart"); Wait = $true; PassThru = $true }
        if (-not $admin) { $run.Verb = "RunAs" }
        $proc = Start-Process @run
    } catch {
        Stop-Install "The Visual C++ runtime was not installed: $($_.Exception.Message)" "Run the installer again and click Yes when Windows asks, or install it yourself from $url."
    }
    Remove-Item -LiteralPath $exe -ErrorAction SilentlyContinue
    switch ($proc.ExitCode) {
        0 { Write-Info "Visual C++ runtime installed." }
        1638 { Write-Info "A newer Visual C++ runtime is already installed." }
        3010 { Write-Warn "Visual C++ runtime installed; Windows wants a restart. Restart the PC if PartLabeler does not start." }
        default { Stop-Install "Installing the Visual C++ runtime failed (exit code $($proc.ExitCode))." "Install it yourself from $url and run the installer again." }
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

function Get-NvidiaDriver {
    # The NVIDIA driver version Windows reports, for when nvidia-smi gives no CUDA version (some laptop drivers, or a
    # driver updated without a restart). Windows shows driver 591.86 as 32.0.15.9186: the last five digits.
    $card = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "NVIDIA" } | Select-Object -First 1
    $d = "$($card.DriverVersion)" -replace "\.", ""
    if (-not $card -or $d.Length -lt 5) { return $null }
    return @{ Name = $card.Name; Driver = [version]("{0}.{1}" -f $d.Substring($d.Length - 5, 3), $d.Substring($d.Length - 2)) }
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

function Get-WheelInfo([string]$Variant, [string]$Pkg, [string]$PyTag) {
    # The real wheel (name, url, sha256, bytes) a real install would fetch for $Pkg (torch or torchvision) at
    # $Variant, for this Python ($PyTag, e.g. "cp312"): the highest version listed, read from the package index
    # page (PyPI for cpu - it IS the Windows CPU build, see Get-TorchTag above; pytorch.org otherwise). One GET
    # (the page) and one HEAD (Content-Length) - no download. $null on any network problem (not fatal to -Plan).
    $indexUrl = if ($Variant -eq "cpu") { "https://pypi.org/simple/$Pkg/" } else { "https://download.pytorch.org/whl/$Variant/$Pkg/" }
    try {
        $html = (Invoke-WebRequest -Uri $indexUrl -UseBasicParsing -TimeoutSec 20).Content
    } catch {
        return $null
    }
    $suffix = if ($Variant -eq "cpu") { "-$PyTag-$PyTag-win_amd64\.whl" } else { "%2B$Variant-$PyTag-$PyTag-win_amd64\.whl" }   # "+" is %2B in the href
    $pattern = 'href="([^"]*' + [regex]::Escape($Pkg) + '-([\d.]+)' + $suffix + ')(?:#sha256=([0-9a-f]{64}))?"'
    $found = [regex]::Matches($html, $pattern) | ForEach-Object {
        [ordered]@{ Url = ($_.Groups[1].Value -replace "%2B", "+"); Version = [version]$_.Groups[2].Value; Sha256 = $_.Groups[3].Value }
    }
    if (-not $found) {
        return $null
    }
    $withHash = @($found | Where-Object { $_.Sha256 })
    $best = ($(if ($withHash.Count) { $withHash } else { $found }) | Sort-Object { $_.Version } | Select-Object -Last 1)
    $bytes = 0
    try {
        $bytes = [int64]((Invoke-WebRequest -Uri $best.Url -Method Head -UseBasicParsing -TimeoutSec 20).Headers["Content-Length"])
    } catch {
    }
    return [ordered]@{ name = ($best.Url -split "/")[-1]; url = $best.Url; sha256 = $best.Sha256; bytes = $bytes }
}

function Write-Utf8NoBom([string]$Path, [string]$Text) {
    # Set-Content -Encoding utf8 writes a BOM in PowerShell 5.1 (no utf8NoBOM here); a BOM breaks a plain
    # json.load() in Python, which prefetch.json must stay readable by (other tools that read it, this script's tests).
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Get-PrefetchStatus([string]$Dir, [string]$Variant) {
    # What -Prefetch left in $Dir, checked against the build $Variant a real install would use now.
    $f = Join-Path $Dir "prefetch.json"
    if (-not (Test-Path -LiteralPath $f)) {
        return [ordered]@{ exists = $false; complete = $false; variant_matches = $false; json = $null }
    }
    try {
        $j = Get-Content -LiteralPath $f -Raw | ConvertFrom-Json
    } catch {
        return [ordered]@{ exists = $true; complete = $false; variant_matches = $false; json = $null }
    }
    return [ordered]@{ exists = $true; complete = ($j.complete -eq $true); variant_matches = ($j.variant -eq $Variant); json = $j }
}

if (-not $Plan) {
    Write-Host ""
    Write-Host "PartLabeler installer for Windows" -ForegroundColor Green
    Write-Host "Folder: $Root"
}

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
    if ($Plan) {
        Write-Info "uv is not installed: a real run would download and install it (https://astral.sh/uv)."
    } else {
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
}
if ($uv) {
    $uvDir = Split-Path -Parent $uv
    if (-not (($env:Path -split ";") -contains $uvDir)) {
        $env:Path = $uvDir + ";" + $env:Path
    }
    $uvVersion = & $uv --version
    Write-Info "$uvVersion  ($uv)"
}

# ---- 2. Python + .venv ----------------------------------------------------------------------
Write-Step "Python $Python and the virtual environment (.venv)"
if ($Plan) {
    # Read-only: never installs a Python build or creates/clears .venv.
    if (Test-Path -LiteralPath $VenvPy) {
        $havePy = & $VenvPy -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        Write-Info $(if ($LASTEXITCODE -eq 0 -and $havePy) { "Reusing .venv (Python $havePy)." } else { ".venv exists but does not work; a real run would re-create it." })
    } else {
        Write-Info "No .venv yet: a real run would install Python $Python and create it."
    }
} else {
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
    $smiText = if ($smi) { (& $smi 2>$null) -join "`n" } else { "" }
    if ($smi -and $LASTEXITCODE -eq 0 -and $smiText -match "CUDA Version:\s*(\d+)\.(\d+)") {
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
    } elseif ($nv = Get-NvidiaDriver) {
        # nvidia-smi is missing or names no CUDA version (seen on a laptop with driver 616.92): the driver version
        # gives its CUDA level, as in the table above.
        Write-Info ("GPU: {0}   driver {1}" -f $nv.Name, $nv.Driver)
        Write-Info "nvidia-smi gave no CUDA version: using the driver version Windows reports."
        $driverCuda = [version]"0.0"
        foreach ($step in @(@("595.0", "13.2"), @("580.0", "13.0"), @("560.0", "12.6"), @("527.0", "12.0"))) {
            if ($nv.Driver -ge [version]$step[0]) { $driverCuda = [version]$step[1]; break }
        }
        if ($nv.Name -match "RTX 50\d\d") { $maxCap = [version]"12.0" }          # Blackwell: no cu126 kernels
    } else {
        Write-Info "No NVIDIA driver found: installing the CPU build."
        Write-Info "Everything works on the CPU, only slower. With an NVIDIA GPU, install its driver and re-run."
    }
    if ($driverCuda) {
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

$idealVariant = $variant                                # what a FRESH install would pick here, before "keep" below

# A re-run never switches PyTorch builds on its own - CPU stays CPU, a CUDA build stays that CUDA build - even
# if auto-detection would now pick a different one: an update must never start an unasked multi-GB download.
# "GPU always, CPU only as a fallback" is a downstream installer's own job: it explicitly asks for it
# (-Cuda <build>) on a box it knows has a capable GPU; install.ps1 itself stays conservative by default.
# -Cpu / -Cuda <build> are the explicit asks that can still change the installed build.
$script:TorchInstalled = ""
$haveTag = Get-TorchTag
$reinstall = $false
$torchAction = "install"                               # for -Plan: install (nothing there yet) | keep | reinstall
$torchReason = "no PyTorch installed yet"
$explicitAsk = $Cpu -or $Cuda -ne "auto"
if ($explicitAsk) { $variant = if ($Cpu) { "cpu" } else { $Cuda } }
if ($haveTag) {
    if ($haveTag -eq $variant) {
        Write-Info "PyTorch $($script:TorchInstalled) is already installed."
        $torchAction = "keep"; $torchReason = "already the build that would be chosen"
    } elseif ($explicitAsk) {
        $why = if ($Cpu) { "-Cpu given" } else { "-Cuda $Cuda given" }
        Write-Info "Replacing PyTorch $($script:TorchInstalled) with the $variant build ($why)."
        $reinstall = $true
        $torchAction = "reinstall"; $torchReason = $why
    } else {
        # auto, no explicit ask: keep whatever is installed, CPU or GPU, even if a different build would now
        # be the first choice for a fresh install. A CUDA build runs fine on CPU anyway (engine/hw.py picks
        # the device at runtime), so there is nothing to fix here even on a box with no GPU.
        Write-Info "Keeping the installed PyTorch $($script:TorchInstalled) ($variant would be the first choice for a new install; -Cuda $variant switches to it)."
        $variant = $haveTag
        $torchAction = "keep"; $torchReason = "a re-run keeps the installed build unless asked to switch"
    }
} elseif ($variant -eq "cpu") {
    $torchReason = "no capable NVIDIA GPU or driver found"
}
Write-Info "PyTorch build: $variant"

if ($Plan) {
    $prefetchInfo = $null
    if ($Prefetch) {
        $st = Get-PrefetchStatus $Prefetch $variant
        $prefetchInfo = [ordered]@{ dir = $Prefetch; exists = $st.exists; complete = $st.complete; variant_matches = $st.variant_matches }
    }
    # The chosen build's real wheels (name/url/sha256/bytes), so a caller (e.g. an explicit -Cuda <build> -Plan,
    # to learn what that build needs without installing it) can prefetch them with its own downloader.
    $pyTag = "cp" + (& $VenvPy -c "import sys; print('%d%d' % sys.version_info[:2])" 2>$null)
    # NOT $wheels: that's the same (case-insensitive) variable as the -Wheels parameter, which is typed [string]
    # and would silently stringify an array assigned to it.
    $torchWheels = @("torch", "torchvision") | ForEach-Object { Get-WheelInfo $variant $_ $pyTag } | Where-Object { $_ }
    [ordered]@{
        ok = $true
        torch = [ordered]@{
            installed_build = $(if ($script:TorchInstalled) { $haveTag } else { $null })
            installed_version = $(if ($script:TorchInstalled) { $script:TorchInstalled } else { $null })
            chosen_build = $variant
            ideal_build = $idealVariant
            action = $torchAction
            reason = $torchReason
            approx_bytes = $(if ($torchAction -eq "keep") { 0 } elseif ($variant -eq "cpu") { 220MB } else { 2.5GB })
            wheels = $torchWheels
        }
        prefetch = $prefetchInfo
        steps = $script:PlanLog
    } | ConvertTo-Json -Depth 6
    exit 0
}

# ---- 3b. -Prefetch: download torch/torchvision into a folder, .venv untouched, then exit --------------------
if ($Prefetch) {
    Write-Step "Prefetch: downloading PyTorch ($variant) into $Prefetch"
    New-Item -ItemType Directory -Force -Path $Prefetch | Out-Null
    $jsonPath = Join-Path $Prefetch "prefetch.json"
    Write-Utf8NoBom $jsonPath (([ordered]@{ variant = $variant; torch = ""; files = @(); complete = $false } | ConvertTo-Json))
    & $uv pip install pip --python $VenvPy --quiet               # uv has no "download" subcommand; pip does
    Assert-LastExit "Installing pip (needed to download, not to use)"
    $indexUrl = "https://download.pytorch.org/whl/$variant"
    $maxTries = 5
    for ($try = 1; $try -le $maxTries; $try++) {
        & $VenvPy -m pip download --no-deps --dest $Prefetch torch torchvision --index-url $indexUrl
        if ($LASTEXITCODE -eq 0) { break }
        if ($try -eq $maxTries) {
            Stop-Install "Downloading PyTorch ($variant) failed after $maxTries tries." "Check the internet connection and run -Prefetch again: whole files already downloaded are kept, only what's missing is retried."
        }
        Write-Warn "Download attempt $try of $maxTries failed; retrying (whole files already downloaded are kept)..."
    }
    $files = @(Get-ChildItem -LiteralPath $Prefetch -Filter "*.whl" | ForEach-Object {
        [ordered]@{ name = $_.Name; url = "$indexUrl/$($_.Name)"; bytes = $_.Length
                    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower() }
    })
    $torchFile = $files | Where-Object { $_.name -match "^torch-([\d.]+)\+" } | Select-Object -First 1
    $torchVersion = if ($torchFile) { ([regex]::Match($torchFile.name, "^torch-([\d.]+)\+")).Groups[1].Value } else { "" }
    Write-Utf8NoBom $jsonPath (([ordered]@{ variant = $variant; torch = $torchVersion; files = $files; complete = $true } | ConvertTo-Json -Depth 4))
    Write-Host ""
    Write-Host "Prefetch complete: $($files.Count) file(s) in $Prefetch." -ForegroundColor Green
    exit 0
}

# ---- 4. PyTorch ------------------------------------------------------------------------------
$useWheels = [bool]$Wheels -and $torchAction -ne "keep"
if ($useWheels) {
    $st = Get-PrefetchStatus $Wheels $variant
    if (-not $st.exists) {
        Stop-Install "$Wheels has no prefetch.json." "Run -Prefetch $Wheels first, or drop -Wheels to download now."
    } elseif (-not $st.complete) {
        Stop-Install "The prefetch in $Wheels is not complete yet." "Wait for -Prefetch $Wheels to finish, or drop -Wheels to download now."
    } elseif (-not $st.variant_matches) {
        Stop-Install "The prefetch in $Wheels is for $($st.json.variant), not $variant (this machine's build)." "Run -Prefetch $Wheels again for $variant, or drop -Wheels to download now."
    }
}
Write-Step $(if ($useWheels) { "PyTorch ($variant) - installing offline from $Wheels" } else { "PyTorch ($variant) - the first install downloads about 2.5 GB" })
if (Test-VcRuntime) {
    Write-Info "Microsoft Visual C++ runtime: present."
} else {
    Write-Info "PyTorch needs the Microsoft Visual C++ runtime (2015-2022, x64, 14.40 or newer); this PC has none or an older one."
    Install-VcRuntime
}
if ($useWheels) {
    $torchArgs = @("pip", "install", "--python", $VenvPy, "torch", "torchvision", "--no-index", "--find-links", $Wheels)
} else {
    $indexUrl = "https://download.pytorch.org/whl/$variant"
    $torchArgs = @("pip", "install", "--python", $VenvPy, "torch", "torchvision", "--index-url", $indexUrl)
}
if ($reinstall) {
    $torchArgs += @("--reinstall-package", "torch", "--reinstall-package", "torchvision")
}
& $uv @torchArgs
Assert-LastExit "Installing PyTorch" $(if ($useWheels) { "Check $Wheels has valid torch/torchvision wheels for $variant and run the installer again." } else { "Check your internet connection and run the installer again (finished downloads are kept). If it keeps failing, try -Cpu." })

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
if ($NoSelfCheck) {
    Write-Step "Self-check (skipped: -NoSelfCheck)"
} else {
    Write-Step "Self-check"
    # A CUDA build runs fine with no GPU present (engine/hw.py picks cpu at runtime; the UI shows which):
    # that is not a failure, so only an actual import error is checked here. 2>&1: the traceback of an import
    # error (e.g. a blocked DLL) is on stderr, and must be in $out for the pattern checks below to see it.
    $check = "import torch, engine.hw as hw, engine.project, transformers, anywidget, fastapi; " +
             "print('      torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); " +
             "print('      device:', hw.describe())"
    $out = & $VenvPy -c $check 2>&1
    $out | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) {
        $text = $out -join " "
        if ($text -match "WinError 4551|Application Control policy") {
            Stop-Install "Windows' app control (Smart App Control) on this PC blocks PyTorch's files; PartLabeler cannot run on this PC." "Turning Smart App Control off will not help: once a PC has it on, Windows cannot turn it back on without a full reset. Try another PC, or ask whoever manages this one about an exception for PartLabeler."
        }
        Stop-Install "The self-check failed: PartLabeler cannot be imported." "Run the installer again; if it still fails, send the messages above to the team."
    }
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
