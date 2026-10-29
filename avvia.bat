@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Ambiente non pronto. Esegui prima setup.bat
    pause
    exit /b 1
)
".venv\Scripts\python.exe" src\bot.py
echo.
pause
