"""Opening and closing PostgreSQL connections."""

from contextlib import contextmanager

import psycopg2

from src.utils.config import db_settings


def get_connection():
    """A plain psycopg2 connection. Caller is responsible for closing it."""
    return psycopg2.connect(**db_settings())


@contextmanager
def connection():
    """Connection that commits on success, rolls back on error, and always closes.

    Usage:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM reviews")
    """
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
