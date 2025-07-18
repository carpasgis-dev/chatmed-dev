@echo off
echo 🚀 Iniciando ChatMed v2.0 Flexible...
echo.

REM Cambiar al directorio del script
cd /d "%~dp0"

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no encontrado. Instalalo desde https://python.org
    pause
    exit /b 1
)

echo ✅ Python encontrado
echo 📂 Directorio: %CD%
echo.

REM Ejecutar el script de inicio
python start_chat.py

REM Pausa al final para ver errores
if errorlevel 1 (
    echo.
    echo ❌ El programa terminó con errores
    pause
) 