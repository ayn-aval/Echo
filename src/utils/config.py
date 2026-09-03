"""Settings for Echo: local .env on this machine, Streamlit secrets when deployed.

.env is gitignored — the password never leaves this machine.

Two sources, checked in this order:

1. **Environment / .env.** What the Mac uses. If DB_HOST and friends are set,
   they win outright.
2. **A connection URL** — NEON_POSTGRES_URL, NEON_URL or DATABASE_URL. What the
   deployed app uses, where there is no localhost to talk to.

Local credentials deliberately take priority. Both are present on the development
machine, and if the URL won, `python -m src.clustering.name_themes` would quietly
rewrite the production database instead of the local one. To point local tools at
Neon on purpose, comment out DB_HOST.

On Streamlit Community Cloud there is no .env at all; values come from the app's
Secrets panel, which this module reads through st.secrets. Streamlit is imported
lazily so that eval scripts and cron jobs never pay for it.
"""

import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

URL_KEYS = ("NEON_POSTGRES_URL", "NEON_URL", "DATABASE_URL")


def setting(name: str, default=None):
    """Environment first, then Streamlit secrets, then the default."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:                       # noqa: BLE001
        pass                                # not running under Streamlit, or no secrets
    return default


def _require(name: str) -> str:
    value = setting(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Create your .env file first:\n"
            f"    cd {PROJECT_ROOT}\n"
            f"    cp .env.example .env\n"
            f"then open .env and fill in your PostgreSQL password."
        )
    return value


def _from_url(url: str) -> dict:
    """psycopg2 kwargs from a postgresql:// connection string.

    Parsed rather than handed to psycopg2 as a DSN so that the rest of the project
    keeps one shape for connection settings, and so sslmode — which Neon requires
    and which lives in the query string — is not silently dropped.
    """
    u = urlparse(url)
    out = {
        "host": u.hostname,
        "port": u.port or 5432,
        "dbname": (u.path or "/").lstrip("/"),
        "user": unquote(u.username or ""),
        "password": unquote(u.password or ""),
    }
    for key, values in parse_qs(u.query).items():
        if key in ("sslmode", "connect_timeout", "options", "channel_binding"):
            out[key] = values[0]
    out.setdefault("sslmode", "require")
    return out


def db_settings() -> dict:
    """Connection settings for psycopg2. Local credentials win when present."""
    if setting("DB_HOST"):
        return {
            "host": _require("DB_HOST"),
            "port": int(_require("DB_PORT")),
            "dbname": _require("DB_NAME"),
            "user": _require("DB_USER"),
            "password": _require("DB_PASSWORD"),
        }
    for key in URL_KEYS:
        url = setting(key)
        if url:
            return _from_url(url.strip().strip('"').strip("'"))
    raise RuntimeError(
        "No database configured. Set DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD "
        f"in .env, or one of {', '.join(URL_KEYS)} for a hosted database."
    )


def describe_db() -> str:
    """Which database is live, without revealing the password."""
    s = db_settings()
    return f"{s['user']}@{s['host']}:{s['port']}/{s['dbname']}"
