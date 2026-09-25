#!/usr/bin/env bash
# PartLabeler installer for Linux, macOS and Google Colab.
#
#   bash install.sh [--cpu] [--no-models] [--dev] [--no-teach] [--python 3.12] [--cuda cu130]
#
#   --cpu          CPU build of PyTorch even if an NVIDIA GPU is present
#   --no-models    skip the model download (about 4 GB); later: python -m engine.models
#   --dev          also install developer tools (pytest, playwright, jupyterlab)
#   --no-teach     skip the Teach & Transfer extras (RF-DETR training)
#   --python X.Y   Python version for .venv (default 3.12, what Colab uses)
#   --cuda cuXYZ   pick the PyTorch build yourself: cu132, cu130, cu126 or cpu (default: auto)
#
# Linux/macOS: uv + a .venv in this folder, PyTorch from the matching PyTorch index.
# Colab: no venv; PartLabeler goes into Colab's own Python and Colab's CUDA PyTorch is kept.
# Safe to run again: it reuses .venv, keeps a working PyTorch and skips finished downloads.
#
# Which PyTorch build (same rules as install.ps1). PyTorch 2.14.0, the current release on
# 2026-09-25, ships cu126, cu130, cu132 and cpu (no cu128 any more). nvidia-smi reports the newest
# CUDA version the driver supports; we take the newest build that is not newer than that:
#
#   Driver branch (Linux / Windows minimum)          nvidia-smi "CUDA Version"   PyTorch build
#   R595 or newer                                    13.2 and up                  cu132
#   R580 - R594  (580.65.06 / 580.88)                13.0 - 13.1                  cu130
#   R560 - R579  (560.28.03 / 560.76)                12.6 - 12.9                  cu126
#   R525 - R559  (525.60.13 / 527.41)                12.0 - 12.5                  cu126 (minor-version compatibility)
#   older driver, or no NVIDIA GPU                   -                            cpu
#
# CUDA 13 builds need compute capability 7.5+ (Turing: T4, RTX 20xx and newer); older GPUs get
# cu126. cu126 has no Blackwell (RTX 50xx, compute 10.x/12.x) kernels: those need driver R580+.
# macOS: the regular PyPI build (CPU; Apple GPU via PARTLABELER_DEVICE=mps is untested).
set -euo pipefail

CPU=0; NO_MODELS=0; DEV=0; NO_TEACH=0; PY_VERSION="3.12"; CUDA_CHOICE="auto"
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) CPU=1 ;;
    --no-models) NO_MODELS=1 ;;
    --dev) DEV=1 ;;
    --no-teach) NO_TEACH=1 ;;
    --python) PY_VERSION="${2:?--python needs a version, e.g. 3.12}"; shift ;;
    --python=*) PY_VERSION="${1#*=}" ;;
    --cuda) CUDA_CHOICE="${2:?--cuda needs cu132, cu130, cu126 or cpu}"; shift ;;
    --cuda=*) CUDA_CHOICE="${1#*=}" ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see: bash install.sh --help)" >&2; exit 2 ;;
  esac
  shift
done
case "$CUDA_CHOICE" in auto|cu132|cu130|cu126|cpu) ;; *) echo "--cuda must be cu132, cu130, cu126 or cpu" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8
START=$(date +%s)
STEP=0
TOTAL=7

step() { STEP=$((STEP + 1)); printf '\n\033[1;36m[%d/%d] %s\033[0m\n' "$STEP" "$TOTAL" "$*"; }
info() { printf '      %s\n' "$*"; }
warn() { printf '\033[1;33m      WARNING: %s\033[0m\n' "$*"; }
DIED=0
die() {
  DIED=1
  printf '\n\033[1;31mERROR: %s\033[0m\n' "$1" >&2
  if [ -n "${2:-}" ]; then printf '\033[1;33m%s\033[0m\n' "$2" >&2; fi
  exit 1
}
# Anything that fails without calling die (set -e) still ends with a readable message.
trap 'rc=$?; if [ $rc -ne 0 ] && [ "$DIED" = 0 ]; then printf "\n\033[1;31mInstallation stopped at step %s (exit code %s). The messages above say why.\033[0m\n" "$STEP" "$rc" >&2; fi' EXIT

# "13.2" -> 1302, "8.6" -> 806: plain integers are easy to compare in bash
vnum() { local major="${1%%.*}" minor="${1#*.}"; [ "$minor" = "$1" ] && minor=0; echo $((10#$major * 100 + 10#${minor%%.*})); }

[ -f pyproject.toml ] || die "pyproject.toml not found next to install.sh." "Run the installer from the PartLabeler folder."

# Colab sets COLAB_RELEASE_TAG; a /content folder alone could exist elsewhere, so it also needs google.colab.
is_colab() {
  [ -n "${COLAB_RELEASE_TAG:-}" ] && return 0
  [ -d /content ] && python3 -c "import google.colab" >/dev/null 2>&1
}

EXTRAS=()
[ "$NO_TEACH" = 1 ] || EXTRAS+=(teach)
[ "$DEV" = 1 ] && EXTRAS+=(dev)
TARGET="."
if [ ${#EXTRAS[@]} -gt 0 ]; then TARGET=".[$(IFS=,; echo "${EXTRAS[*]}")]"; fi

# ============================================================================================
# Google Colab: Colab's Python and its CUDA PyTorch (fp16 on the T4 is chosen automatically)
# ============================================================================================
if is_colab; then
  TOTAL=4
  echo "PartLabeler installer - Google Colab detected"
  step "Colab's PyTorch (kept as it is)"
  python - <<'EOF'
import torch
print("      torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("      No GPU: Runtime > Change runtime type > T4 GPU is much faster (then run the install again).")
EOF

  step "PartLabeler and its packages ($TARGET) - takes a few minutes"
  # Pin Colab's torch family, so no package can replace Colab's CUDA build.
  PINS="$(mktemp)"
  python - >"$PINS" <<'EOF'
import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{p}=={m.version(p).split('+')[0]}")
    except m.PackageNotFoundError:
        pass
EOF
  python -m pip install -q --progress-bar off -c "$PINS" -e "$TARGET" \
    || die "Installing PartLabeler failed." "If pip says the pinned torch conflicts, send the messages above to the team."
  rm -f "$PINS"

  step "Models (SAM 3 and DINOv3, about 4 GB)"
  if [ "$NO_MODELS" = 1 ]; then
    info "Skipped (--no-models). Download them with:  python -m engine.models"
  else
    python -m engine.models || die "Downloading the models failed." "Run the cell again: finished files are kept."
  fi

  step "Self-check"
  python -c "import torch, engine.hw as hw, engine.project, transformers, anywidget; print('      torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('      device:', hw.describe())" \
    || die "The self-check failed: PartLabeler cannot be imported."
  printf '\n\033[1;32mPartLabeler is installed (%s s).\033[0m Continue with the next notebook step.\n' "$(( $(date +%s) - START ))"
  exit 0
fi

# ============================================================================================
# Linux / macOS: uv + .venv
# ============================================================================================
OS="$(uname -s)"
echo "PartLabeler installer for $OS"
echo "Folder: $ROOT"

step "uv (Python package manager)"
find_uv() {
  command -v uv 2>/dev/null && return 0
  local c
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
if ! UV="$(find_uv)"; then
  info "uv is not installed. Installing it with the official installer from https://astral.sh/uv ..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    die "Neither curl nor wget is available to download uv." "Install curl (e.g. sudo apt install curl) and run the installer again."
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  UV="$(find_uv)" || die "uv was installed but cannot be found." "Open a new terminal and run the installer again."
fi
info "$("$UV" --version)  ($UV)"

step "Python $PY_VERSION and the virtual environment (.venv)"
if ! "$UV" python install "$PY_VERSION"; then
  # A damaged uv Python folder can make "install" fail although a usable Python is there.
  FOUND_PY="$("$UV" python find "$PY_VERSION" --system --no-project 2>/dev/null)" \
    || die "Installing Python $PY_VERSION failed." "Check your internet connection and try again."
  warn "uv could not install Python $PY_VERSION, but found one it can use: $FOUND_PY"
fi
VENV_PY="$ROOT/.venv/bin/python"
WANT_PY="$(echo "$PY_VERSION" | cut -d. -f1-2)"
if [ -x "$VENV_PY" ] && HAVE_PY="$("$VENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"; then
  info "Reusing .venv (Python $HAVE_PY)."
  [ "$HAVE_PY" = "$WANT_PY" ] || warn ".venv has Python $HAVE_PY, not $WANT_PY. To switch, delete .venv and run the installer again."
elif [ -e .venv ]; then
  warn ".venv exists but does not work (moved or copied from another machine?). Creating it again."
  "$UV" venv --clear --python "$PY_VERSION" .venv || die "Re-creating .venv failed."
else
  "$UV" venv --python "$PY_VERSION" .venv || die "Creating .venv failed."
fi

step "Graphics card and PyTorch build"
VARIANT="cpu"; DRIVER_CUDA=""; MIN_CAP=""; MAX_CAP=""
if [ "$CPU" = 1 ]; then
  info "--cpu given: installing the CPU build."
elif [ "$CUDA_CHOICE" != auto ]; then
  VARIANT="$CUDA_CHOICE"; info "--cuda $CUDA_CHOICE given: installing that build."
elif [ "$OS" = Darwin ]; then
  VARIANT="pypi"; info "macOS: installing the regular PyPI build of PyTorch."
elif ! command -v nvidia-smi >/dev/null 2>&1; then
  info "No NVIDIA driver found (nvidia-smi is missing): installing the CPU build."
  info "Everything works on the CPU, only slower. With an NVIDIA GPU, install its driver and re-run."
else
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
  if [ -z "$DRIVER_CUDA" ]; then
    warn "nvidia-smi did not report a CUDA version (driver problem?). Installing the CPU build."
  else
    GPUS="$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || true)"
    # very old drivers cannot report compute_cap
    [ -n "$GPUS" ] || GPUS="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true)"
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      IFS=',' read -r name driver cap <<<"$line"
      info "GPU: ${name}   driver${driver}"
      cap="${cap// /}"
      if [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
        c=$(vnum "$cap")
        if [ -z "$MIN_CAP" ] || [ "$c" -lt "$MIN_CAP" ]; then MIN_CAP=$c; fi
        if [ -z "$MAX_CAP" ] || [ "$c" -gt "$MAX_CAP" ]; then MAX_CAP=$c; fi
      fi
    done <<<"$GPUS"
    info "The driver supports CUDA $DRIVER_CUDA."
    d=$(vnum "$DRIVER_CUDA")
    if [ "$d" -ge 1302 ]; then VARIANT="cu132"
    elif [ "$d" -ge 1300 ]; then VARIANT="cu130"
    elif [ "$d" -ge 1200 ]; then
      VARIANT="cu126"
      [ "$d" -ge 1206 ] || warn "this driver is older than 560. PyTorch should still run, but updating the driver is recommended."
    else
      warn "this driver is too old for current PyTorch (needs 525 or newer). Installing the CPU build."
    fi
    if [ -n "$MIN_CAP" ] && [ "$MIN_CAP" -lt 705 ] && { [ "$VARIANT" = cu130 ] || [ "$VARIANT" = cu132 ]; }; then
      info "GPU is older than Turing (compute capability < 7.5): using cu126 (CUDA 13 builds dropped it)."
      VARIANT="cu126"
    fi
    if [ -n "$MAX_CAP" ] && [ "$MAX_CAP" -ge 1000 ] && { [ "$VARIANT" = cu126 ] || [ "$VARIANT" = cpu ]; }; then
      die "This RTX 50-series (Blackwell) GPU needs NVIDIA driver 580 or newer for PyTorch." \
          "Update the driver and run the installer again, or use --cpu."
    fi
  fi
fi

# Keep a CUDA PyTorch that is already installed and works with this driver (no big re-download).
REINSTALL=()
HAVE_TORCH="$("$VENV_PY" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null || true)"
if [ -n "$HAVE_TORCH" ] && [ "$VARIANT" != pypi ]; then
  case "$HAVE_TORCH" in
    *+*) HAVE_TAG="${HAVE_TORCH#*+}" ;;
    *) if [ "$OS" = Linux ]; then HAVE_TAG="pypi"; else HAVE_TAG="cpu"; fi ;;
  esac
  if [ "$HAVE_TAG" = "$VARIANT" ]; then
    info "PyTorch $HAVE_TORCH is already installed."
  else
    KEEP=0
    if [ "$CUDA_CHOICE" = auto ] && [ "$CPU" = 0 ] && [ "$VARIANT" != cpu ] && [ -n "$DRIVER_CUDA" ] \
       && [[ "$HAVE_TAG" =~ ^cu([0-9]+)([0-9])$ ]]; then
      have=$(vnum "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}")
      if [ "$have" -le "$(vnum "$DRIVER_CUDA")" ] && ! { [ -n "$MIN_CAP" ] && [ "$MIN_CAP" -lt 705 ] && [ "$have" -ge 1300 ]; }; then
        KEEP=1
      fi
    fi
    if [ "$KEEP" = 1 ]; then
      info "Keeping the installed PyTorch $HAVE_TORCH: it works with this driver ($VARIANT would be the first choice for a new install)."
      VARIANT="$HAVE_TAG"
    else
      info "Replacing PyTorch $HAVE_TORCH with the $VARIANT build."
      REINSTALL=(--reinstall-package torch --reinstall-package torchvision)
    fi
  fi
fi
info "PyTorch build: $VARIANT"

step "PyTorch ($VARIANT) - the first install downloads 1-3 GB"
if [ "$VARIANT" = pypi ]; then
  "$UV" pip install --python "$VENV_PY" torch torchvision ${REINSTALL[@]+"${REINSTALL[@]}"}
else
  "$UV" pip install --python "$VENV_PY" torch torchvision --index-url "https://download.pytorch.org/whl/$VARIANT" \
    ${REINSTALL[@]+"${REINSTALL[@]}"}
fi || die "Installing PyTorch failed." "Check your internet connection and run the installer again (finished downloads are kept). If it keeps failing, try --cpu."

step "PartLabeler and its packages ($TARGET)"
# Pin the torch just installed, so no other package can swap it for a different build.
PINS="$(mktemp)"
"$VENV_PY" -c "import importlib.metadata as m; [print(p + '==' + m.version(p).split('+')[0]) for p in ('torch', 'torchvision')]" >"$PINS"
"$UV" pip install --python "$VENV_PY" -e "$TARGET" --constraints "$PINS" \
  || die "Installing PartLabeler failed." "Check your internet connection and run the installer again."
rm -f "$PINS"

step "Models (SAM 3 and DINOv3, about 4 GB, checksum-verified)"
if [ "$NO_MODELS" = 1 ]; then
  info "Skipped (--no-models). Download them later with:  .venv/bin/python -m engine.models"
else
  "$VENV_PY" -m engine.models || die "Downloading the models failed." "Run the installer again: finished files are kept."
fi

step "Self-check"
CHECK="$("$VENV_PY" -c "import torch, engine.hw as hw, engine.project, transformers, anywidget, fastapi; print('      torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('      device:', hw.describe())")" \
  || die "The self-check failed: PartLabeler cannot be imported." "Run the installer again; if it still fails, send the messages above to the team."
echo "$CHECK"
case "$VARIANT" in
  cu*) [[ "$CHECK" == *"CUDA available: True"* ]] || warn "PyTorch was installed for CUDA but cannot use the GPU (reboot after a driver update, or use --cpu). PartLabeler still works on the CPU, only slower." ;;
esac
if [ "$NO_TEACH" = 0 ]; then
  "$VENV_PY" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('rfdetr') else 1)" || warn "rfdetr (Teach & Transfer) is not importable."
fi

printf '\n\033[1;32mPartLabeler is installed (%s min).\033[0m\n\n' "$(( ($(date +%s) - START + 30) / 60 ))"
echo "Start it:   ./run.sh"
echo "   or run:  .venv/bin/python -m engine.cli app"
echo "Check:      .venv/bin/python -m engine.cli doctor"
[ "$NO_MODELS" = 1 ] && echo "Models:     .venv/bin/python -m engine.models   (needed before annotating)"
[ "$DEV" = 1 ] && echo "Tests:      .venv/bin/python -m pytest tests"
echo
exit 0
