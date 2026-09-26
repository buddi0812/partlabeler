@echo off
rem PartLabeler updater: double-click to get the newest version from GitHub.
rem Your projects, labels, exports and models are kept; the labels are backed up first.
rem Close PartLabeler (its black window) before updating.
setlocal EnableDelayedExpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo PartLabeler is not installed yet. Double-click install_windows.bat first.
  pause
  exit /b 1
)
set "PYTHONIOENCODING=utf-8"
rem One block: cmd reads it whole, so the update may replace this file while it runs.
(
  ".venv\Scripts\python.exe" -m engine.cli update %*
  set "RC=!ERRORLEVEL!"
  echo.
  pause
  exit /b !RC!
)
