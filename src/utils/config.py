"""Settings for Echo, read from the .env file at the project root.

.env is gitignored — the password never leaves this machine.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Create your .env file first:\n"
            f"    cd {PROJECT_ROOT}\n"
            f"    cp .env.example .env\n"
            f"then open .env and fill in your PostgreSQL password."
        )
    return value


def db_settings() -> dict:
    """Connection settings for psycopg2."""
    return {
        "host": _require("DB_HOST"),
        "port": int(_require("DB_PORT")),
        "dbname": _require("DB_NAME"),
        "user": _require("DB_USER"),
        "password": _require("DB_PASSWORD"),
    }
