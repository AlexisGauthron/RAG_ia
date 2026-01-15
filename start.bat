@echo off
REM ============================================================
REM Script de demarrage du projet RAG-IA (Windows)
REM ============================================================

setlocal enabledelayedexpansion

REM Aller dans le dossier du script
cd /d "%~dp0"

echo ============================================
echo         RAG-IA - Demarrage du projet
echo ============================================
echo.

REM Verification de Python
echo [1/4] Verification de Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo   Veuillez installer Python 3.12 ou superieur.
    echo   https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   [OK] Python trouve: %PYTHON_VERSION%

REM Verification de Poetry
echo [2/4] Verification de Poetry...
where poetry >nul 2>&1
if %errorlevel% neq 0 (
    echo   Poetry non trouve. Installation...
    curl -sSL https://install.python-poetry.org | python -
    set PATH=%USERPROFILE%\.local\bin;%PATH%
    echo   [OK] Poetry installe.
) else (
    for /f "tokens=3" %%i in ('poetry --version 2^>^&1') do set POETRY_VERSION=%%i
    echo   [OK] Poetry trouve: !POETRY_VERSION!
)

REM Installation des dependances
echo [3/4] Installation des dependances...
if not exist ".venv" (
    echo   Creation de l'environnement virtuel...
    poetry install
    echo   [OK] Dependances installees.
) else (
    echo   [OK] Environnement virtuel existant detecte.
    echo   Mise a jour des dependances...
    poetry install --no-interaction
)

REM Verification d'Ollama
echo [4/4] Verification d'Ollama...
where ollama >nul 2>&1
if %errorlevel% neq 0 (
    echo   [INFO] Ollama non trouve.
    echo   Pour utiliser les LLM locaux, installez Ollama: https://ollama.ai
) else (
    echo   [OK] Ollama trouve.

    REM Verifier si Ollama est en cours d'execution
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
    if %errorlevel% neq 0 (
        echo   Demarrage d'Ollama en arriere-plan...
        start /B ollama serve >nul 2>&1
        timeout /t 2 /nobreak >nul
    )

    REM Verifier les modeles
    echo   Verification des modeles...
    ollama list | find "llama3.2" >nul 2>&1
    if %errorlevel% neq 0 (
        echo   Telechargement du modele llama3.2:3b...
        ollama pull llama3.2:3b
    )
    echo   [OK] Modeles prets.
)

echo.
echo ============================================
echo   Demarrage de l'application Streamlit...
echo ============================================
echo.

REM Lancement de l'application Streamlit avec ouverture automatique du navigateur
poetry run streamlit run src/front/app.py --server.headless false --browser.gatherUsageStats false

pause
