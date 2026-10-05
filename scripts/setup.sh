#!/usr/bin/env bash
# Instalacion portable de Lotería Lab (macOS / Linux)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo ">> Directorio: $ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.11+ no esta en PATH." >&2
  exit 1
fi

python3 -m venv .venv
".venv/bin/python" -m pip install --upgrade pip
".venv/bin/python" -m pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo ">> Se creo .env desde .env.example (editalo si quieres)."
fi

mkdir -p data uploads logs

echo
echo "Listo. Siguiente paso:"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo "Luego abre http://127.0.0.1:5000/inicio"
