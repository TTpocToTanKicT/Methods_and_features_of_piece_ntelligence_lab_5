@echo off
REM Zapusk lab5 BEZ komandy "python" (obhid zahlushky Microsoft Store).
setlocal
set "SSLKEYLOGFILE="
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY%" goto run
set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if exist "%PY%" goto run
set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "%PY%" goto run
set "PY=%ProgramFiles%\Python312\python.exe"
if exist "%PY%" goto run

echo ERROR: Python not found in standard folders.
echo Install from https://www.python.org/ or run: run_lab5.cmd
echo See LR5_Aproksymatsiya_Python_Student.pdf (Troubleshooting section)
exit /b 1

:run
"%PY%" "%~dp0lab5_mlp_approximation.py" %*
exit /b %ERRORLEVEL%
