@echo off
cd /d %~dp0\..
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m jeena_sikho.run_repair
) else (
  python -m jeena_sikho.run_repair
)
