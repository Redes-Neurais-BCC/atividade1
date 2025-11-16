@echo off
echo =====================================
echo   Advanced Neural Network App
echo   Otimizacao com Optuna
echo =====================================
echo.

cd /d "%~dp0"
python -m streamlit run src/interface/advanced_app.py --server.port 8503

pause
