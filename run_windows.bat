@echo off
rem PartLabeler launcher: double-click to open the start screen in your browser
rem (create a project from a video or an image folder, or reopen one).
rem Options are passed to "partlabeler app", e.g.:  run_windows.bat --port 8800 --home D:\labeling
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PartLabeler is not installed yet.
  echo Double-click install_windows.bat first, then start this file again.
  echo.
  pause
  exit /b 1
)
set "PYTHONIOENCODING=utf-8"
set "TRANSFORMERS_VERBOSITY=error"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
echo Starting PartLabeler. Keep this window open while you work; close it to stop.
rem One block: cmd reads it whole, so an update from the start screen may replace this file while it runs.
setlocal EnableDelayedExpansion
(
  ".venv\Scripts\python.exe" -m engine.cli app %*
  set "RC=!ERRORLEVEL!"
  if not "!RC!"=="0" (
    echo.
    echo PartLabeler stopped with an error ^(exit code !RC!^). The messages above say why.
    echo If a module is missing, run install_windows.bat again.
    pause
  )
  exit /b !RC!
)
