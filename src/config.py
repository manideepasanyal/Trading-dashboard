"""Configuration and secrets.

Secrets are looked up in this order:
1. Environment variables (e.g. `export DATABASE_URL=...`)
2. `.streamlit/secrets.toml` (the same file Streamlit itself reads)

Nothing is hardcoded and nothing is committed: `.streamlit/secrets.toml` is in
.gitignore, and `.streamlit/secrets.toml.example` shows the expected format.
"""
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = ROOT / ".streamlit" / "secrets.toml"

SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA"]


def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if SECRETS_FILE.exists():
        with open(SECRETS_FILE, "rb") as f:
            data = tomllib.load(f)
        if name in data:
            return str(data[name])
    return default
