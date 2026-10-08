@echo off
setlocal
set "PYTHONW=%USERPROFILE%\.conda\envs\stock_quant\pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=pythonw.exe"
start "" "%PYTHONW%" "%~dp0launch_ui.py"
