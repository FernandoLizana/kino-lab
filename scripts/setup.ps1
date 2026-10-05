# Instalacion portable de Lotería Lab (Windows / PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host ">> Directorio: $Root"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.11+ no esta en PATH. Instala python.org y reabre la terminal."
}

python -m venv .venv
& "$Root\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Root\.venv\Scripts\python.exe" -m pip install -r "$Root\requirements.txt"

if (-not (Test-Path "$Root\.env")) {
    Copy-Item "$Root\.env.example" "$Root\.env"
    Write-Host ">> Se creo .env desde .env.example (editalo si quieres)."
}

New-Item -ItemType Directory -Force -Path "$Root\data", "$Root\uploads", "$Root\logs" | Out-Null

Write-Host ""
Write-Host "Listo. Siguiente paso:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python app.py"
Write-Host "Luego abre http://127.0.0.1:5000/inicio"
