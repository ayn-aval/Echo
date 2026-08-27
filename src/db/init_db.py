"""Create the Echo tables. Safe to run as many times as you like."""

from pathlib import Path

from src.db.connection import connection

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def main() -> None:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_FILE.read_text())
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        tables = [row[0] for row in cur.fetchall()]

    print(f"Applied {SCHEMA_FILE.name}")
    print("Tables in the database:", ", ".join(tables) if tables else "(none)")


if __name__ == "__main__":
    main()
