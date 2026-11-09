@echo off
setlocal
cd /d "%~dp0\..\.."
set "PY=%USERPROFILE%\.conda\envs\stock_quant\python.exe"
if exist "%PY%" (
  "%PY%" -m strategies.formula_screener.app
) else (
  python -m strategies.formula_screener.app
)
