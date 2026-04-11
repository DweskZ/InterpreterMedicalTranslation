@echo off
REM ============================================================
REM  Clinic Translate — Build GPU (CUDA 12 / RTX 4060)
REM  Correr desde la raiz del proyecto: scripts\build.bat
REM ============================================================
setlocal

set ROOT=%~dp0..
set VENV=%ROOT%\.venv\Scripts

echo.
echo ============================================================
echo  Clinic Translate BUILD
echo ============================================================
echo.

REM --- 1. Verificar que el venv exista ---
if not exist "%VENV%\python.exe" (
    echo [ERROR] No se encontro el venv en %ROOT%\.venv
    echo         Crea el venv y corre: pip install -r requirements.txt
    pause & exit /b 1
)

REM --- 2. Instalar PyInstaller si no esta ---
echo [1/4] Verificando PyInstaller...
"%VENV%\python.exe" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo       Instalando PyInstaller...
    "%VENV%\pip.exe" install pyinstaller
)
echo       OK

REM --- 3. Pre-descargar modelos si no existen ---
echo [2/4] Verificando assets (modelos + Argos)...
if not exist "%ROOT%\dist_assets\models" (
    echo       Descargando assets por primera vez...
    "%VENV%\python.exe" "%ROOT%\scripts\pre_download.py" base.en
    if errorlevel 1 ( echo [ERROR] Fallo la descarga & pause & exit /b 1 )
) else (
    echo       Assets ya presentes, omitiendo descarga.
    echo       (Para re-descargar borra dist_assets\ y vuelve a correr)
)

REM --- 4. Limpiar build anterior ---
echo [3/4] Limpiando builds anteriores...
if exist "%ROOT%\dist\ClinicTranslate" rmdir /s /q "%ROOT%\dist\ClinicTranslate"
if exist "%ROOT%\build\ClinicTranslate" rmdir /s /q "%ROOT%\build\ClinicTranslate"

REM --- 5. PyInstaller ---
echo [4/4] Compilando con PyInstaller...
cd /d "%ROOT%"
"%VENV%\pyinstaller.exe" clinic_translate.spec --noconfirm
if errorlevel 1 ( echo [ERROR] PyInstaller fallo & pause & exit /b 1 )

REM --- 6. Copiar modelos y paquetes Argos al dist ---
echo.
echo Copiando assets al directorio de distribucion...
xcopy /e /i /q "%ROOT%\dist_assets\models"          "%ROOT%\dist\ClinicTranslate\models"
xcopy /e /i /q "%ROOT%\dist_assets\argos-packages"  "%ROOT%\dist\ClinicTranslate\argos-packages"

REM --- 7. Crear launcher .bat ---
echo Creando launcher...
(
echo @echo off
echo REM Clinic Translate - Launcher
echo REM Configura rutas para que el exe encuentre modelos y paquetes
echo set "DIR=%%~dp0"
echo set "HF_HOME=%%DIR%%models"
echo set "ARGOS_PACKAGES_DIR=%%DIR%%argos-packages"
echo set "CT2_CUDA_ALLOW_FP16=1"
echo start "" "%%DIR%%ClinicTranslate.exe"
) > "%ROOT%\dist\ClinicTranslate\Abrir ClinicTranslate.bat"

REM --- Resumen ---
echo.
echo ============================================================
echo  BUILD COMPLETADO
echo ============================================================
echo.
echo  Distribucion lista en:
echo    %ROOT%\dist\ClinicTranslate\
echo.
echo  Para entregar al cliente:
echo    Comprimir dist\ClinicTranslate\ en un ZIP y compartir.
echo    El cliente hace doble clic en "Abrir ClinicTranslate.bat"
echo.
echo  Tamano aproximado:
for /f "tokens=3" %%a in ('dir /s /a "%ROOT%\dist\ClinicTranslate" ^| findstr "File(s)"') do set SIZE=%%a
echo    %SIZE% bytes
echo.
pause
