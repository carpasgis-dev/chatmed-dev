@echo off
echo Instalando ChatMed v2 Flexible...

:: Crear entorno virtual si no existe
if not exist "venv" (
    echo Creando entorno virtual...
    python -m venv venv
)

:: Activar entorno virtual
call venv\Scripts\activate

:: Instalar dependencias
echo Instalando dependencias...
pip install -r requirements.txt

:: Instalar el paquete en modo desarrollo
echo Instalando ChatMed v2 Flexible...
pip install -e .

echo Instalación completada!
pause 