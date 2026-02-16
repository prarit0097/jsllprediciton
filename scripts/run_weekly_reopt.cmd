@echo off
setlocal
cd /d %~dp0\..
call .\.venv\Scripts\activate.bat
python -m jeena_sikho_tournament.run_weekly_reopt
endlocal
