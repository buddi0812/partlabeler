@echo off
rem PartLabeler installer: double-click to install everything (uv, Python, PyTorch, models).
rem Options are passed to install.ps1, e.g.:  install_windows.bat -NoModels -Dev
rem   -Cpu  -NoModels  -Dev  -NoTeach  -Python 3.12  -Cuda cu130
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo Installation did not finish ^(exit code %RC%^). Read the messages above, fix the cause, and run this file again.
) else (
  echo Done. Double-click run_windows.bat to start PartLabeler.
)
pause
exit /b %RC%
