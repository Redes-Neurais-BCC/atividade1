@echo off
echo =====================================
echo   MLP Application - Starting...
echo =====================================
echo.

cd /d "%~dp0"
python -m streamlit run src/interface/mlp_app.py

pause
