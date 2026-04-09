# Ejecutar desde la carpeta del proyecto (doble clic puede fallar; mejor PowerShell aquí):
#   .\run.ps1
#   .\run.ps1 --model small
#
# Opcional: token de Hugging Face (descargas más rápidas / menos avisos). Crea uno en https://huggingface.co/settings/tokens
#   $env:HF_TOKEN = "hf_..."
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# UTF-8 en consola para evitar UnicodeEncodeError con faster-whisper
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "No existe .venv. Ejecuta primero: python -m venv .venv" -ForegroundColor Red
    exit 1
}
& ".\.venv\Scripts\python.exe" -u ".\clinic_translate.py" @args
