#!/usr/bin/env bash
# PartLabeler launcher for Linux and macOS: opens the start screen in your browser.
# Options are passed to "partlabeler app", e.g.:  ./run.sh --port 8800 --home ~/labeling --no-browser
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -n "${COLAB_RELEASE_TAG:-}" ]; then
  echo "On Google Colab, use the notebook PartLabeler_Colab.ipynb: the annotator runs inside it." >&2
  echo "(Colab's free tier does not allow a separate web UI.)" >&2
  exit 1
fi

PY=".venv/bin/python"
[ -x "$PY" ] || PY=".venv/Scripts/python.exe"   # Git Bash on Windows
if [ ! -x "$PY" ]; then
  echo "PartLabeler is not installed yet. Run:  bash install.sh" >&2
  exit 1
fi

export PYTHONIOENCODING=utf-8
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1

ARGS=("$@")
# No display (e.g. over SSH): don't try to open a browser; print how to reach the page instead.
if [ "$(uname -s)" = Linux ] && [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [[ " $* " != *" --no-browser "* ]]; then
  ARGS+=(--no-browser)
  echo "No display found: open http://127.0.0.1:8765 (or your --port) in a browser on this machine,"
  echo "or from your PC through an SSH tunnel:  ssh -L 8765:127.0.0.1:8765 $(whoami)@$(hostname)"
fi

exec "$PY" -m engine.cli app ${ARGS[@]+"${ARGS[@]}"}
