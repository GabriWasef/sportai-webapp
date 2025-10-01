@echo off
REM Verifica Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python non trovato.
    pause
    exit /b 1
)

REM Verifica file modelli
if not exist "pe_models.pkl" (
    echo File pe_models.pkl non trovato.
    pause
    exit /b 1
)

REM Installa dipendenze
pip install flask flask-cors pandas scikit-learn joblib numpy -q

REM Menu selezione versione
echo.
echo 1 - Avvia La Versione 1 (http://localhost:5001)
echo 0 - Esci
set /p choice="Scegli versione (1, 0 per uscire): "

if "%choice%"=="1" (
    start http://localhost:5001
    python app1.py
    exit /b 0
)
exit /b 0
