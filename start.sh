#!/usr/bin/env bash
set -e
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
  echo "✓ Loaded .env"
fi

if [ -z "$GEMINI_API_KEY" ]; then
  echo ""
  echo "⚠️  GEMINI_API_KEY not set."
  echo "   Create a .env file with: GEMINI_API_KEY=AIza-your-key-here"
  echo "   Get your key at: https://aistudio.google.com/app/apikey"
  echo ""
fi

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt -q
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  TalentSync AI Backend  (powered by Google Gemini)"
echo "═══════════════════════════════════════════════════════"
echo "  API:      http://localhost:8000"
echo "  Docs:     http://localhost:8000/docs"
echo "  Frontend: http://localhost:8000"
echo "═══════════════════════════════════════════════════════"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
