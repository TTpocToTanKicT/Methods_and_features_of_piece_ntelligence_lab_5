@echo off
REM First-time: pip install + run lab. Same Python discovery as lab.bat (no Store stub).
setlocal
set "SSLKEYLOGFILE="
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY%" goto havepy
set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%PY%" goto havepy
set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "%PY%" goto havepy
set "PY=%ProgramFiles%\Python312\python.exe"
if exist "%PY%" goto havepy

echo ERROR: Python not found. See LR5_Aproksymatsiya_Python_Student.pdf
exit /b 1

:havepy
echo Using: %PY%
"%PY%" --version
"%PY%" -m pip install -q -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo pip failed. Clear SSLKEYLOGFILE env var if set. See LR5_Aproksymatsiya_Python_Student.pdf
  exit /b 1
)
"%PY%" "%~dp0lab5_mlp_approximation.py" %*
exit /b %ERRORLEVEL%
