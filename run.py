"""Simple launcher script for the Multilingual Voice RAG Application."""

import sys
import uvicorn
from pathlib import Path

# Add root folder to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print("=" * 60)
    print("  Starting Voice-Enabled Multilingual Indic RAG Server")
    print("  Web Dashboard: http://127.0.0.1:8000")
    print("  API Docs:      http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
