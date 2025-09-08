import os
import sys
from pathlib import Path


# Ensure project root is on sys.path for 'import bot'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Provide dummy env so bot import doesn't fail
os.environ.setdefault("MISTRAL_API_KEY", "dummy")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
