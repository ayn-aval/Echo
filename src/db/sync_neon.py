"""Copy the tables the dashboard reads from local Postgres into Neon.

    python -m src.db.sync_neon --check     # connect to both, compare, change nothing
    python -m src.db.sync_neon             # dump locally, restore into Neon

Reads NEON_URL from .env. The deployed Streamlit app has no localhost to talk to,
so the six tables the dashboard actually queries are copied to a managed Postgres
and the app points at that instead.

Only those six move. The evaluation tables — eval_pool, eval_judgements,
eval_queries, theme_audit, scrape_checkpoints — stay local on purpose. They are
the working files of a labelling process, they are written to by app/label.py and
app/audit.py, and neither of those belongs on a public URL. They also total under
2 MB, so this is about keeping a public database to what it needs to hold, not
about saving space.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import psycopg2

from src.utils.config import PROJECT_ROOT, db_settings

# Everything app/main.py and its screens read, and nothing else.
TABLES = ["reviews", "review_themes", "themes", "theme_weekly", "theme_alerts",
          "saved_views"]

# The deployed app is public. Give it a role that cannot write, so a visitor
# cannot insert rows through the saved-views control.
READONLY_SQL = """
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'echo_readonly') THEN
        CREATE ROLE echo_readonly LOGIN PASSWORD %(pw)s;
    ELSE
        ALTER ROLE echo_readonly LOGIN PASSWORD %(pw)s;
    END IF;
END $$;
GRANT CONNECT ON DATABASE %(db)s TO echo_readonly;
GRANT USAGE ON SCHEMA public TO echo_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO echo_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO echo_readonly;
"""


# Either name works. Neon's console labels the variable differently depending on
# where you copy it from, and a rename is a pointless thing to make someone do.
URL_KEYS = ("NEON_POSTGRES_URL", "NEON_URL", "DATABASE_URL")


def neon_url() -> str:
    for key in URL_KEYS:
        url = os.getenv(key)
        if url:
            return url.strip().strip('"').strip("'")
    raise SystemExit(
        f"No Neon connection string found. Set one of {', '.join(URL_KEYS)} in .env:\n"
        "    NEON_POSTGRES_URL=postgresql://user:pass@ep-xxx.neon.tech/db?sslmode=require\n"
        "Copy it from the Neon console for your project's production branch.")


def direct(url: str) -> str:
    """The non-pooled endpoint for the same database.

    Neon's default connection string points at a pgbouncer pooler, which is the
    right choice for an app making many short connections and the wrong one for a
    bulk restore: the pooler runs in transaction mode, where session-level state a
    dump relies on does not survive between statements. The direct endpoint is the
    same hostname without the `-pooler` suffix, so it can be derived rather than
    asked for.
    """
    return url.replace("-pooler.", ".")


def server_version(dsn_or_kwargs) -> str:
    conn = (psycopg2.connect(dsn_or_kwargs) if isinstance(dsn_or_kwargs, str)
            else psycopg2.connect(**dsn_or_kwargs))
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW server_version")
            return cur.fetchone()[0]
    finally:
        conn.close()


def counts(dsn_or_kwargs) -> dict:
    conn = (psycopg2.connect(dsn_or_kwargs) if isinstance(dsn_or_kwargs, str)
            else psycopg2.connect(**dsn_or_kwargs))
    out = {}
    try:
        with conn.cursor() as cur:
            for t in TABLES:
                try:
                    cur.execute(f"SELECT count(*) FROM {t}")
                    out[t] = cur.fetchone()[0]
                except psycopg2.Error:
                    conn.rollback()
                    out[t] = None          # table not there yet
    finally:
        conn.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="compare both databases, change nothing")
    ap.add_argument("--readonly-password",
                    help="also create/refresh the echo_readonly role with this password")
    args = ap.parse_args()

    local, url = db_settings(), neon_url()
    lv, nv = server_version(local), server_version(url)
    print(f"local Postgres : {lv}")
    print(f"neon Postgres  : {nv}")
    if lv.split(".")[0] != nv.split(".")[0]:
        print(f"  note: major versions differ ({lv.split('.')[0]} vs {nv.split('.')[0]}). "
              f"pg_dump writes for the newer one, so the restore is the step to watch.")

    before_local, before_neon = counts(local), counts(url)
    print(f"\n{'table':16} {'local':>10} {'neon':>10}")
    for t in TABLES:
        n = before_neon[t]
        print(f"{t:16} {before_local[t]:>10,} {('—' if n is None else f'{n:,}'):>10}")

    if args.check:
        print("\ncheck only — nothing written")
        return

    env = {**os.environ, "PGPASSWORD": local["password"]}
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "echo.sql"
        cmd = ["pg_dump", "-h", local["host"], "-p", str(local["port"]),
               "-U", local["user"], "-d", local["dbname"],
               "--no-owner", "--no-privileges", "--clean", "--if-exists",
               "--format=plain", "-f", str(dump)]
        for t in TABLES:
            cmd += ["-t", t]
        print(f"\ndumping {len(TABLES)} tables ...")
        subprocess.run(cmd, env=env, check=True)
        print(f"  {dump.stat().st_size / 1e6:.1f} MB")

        print("restoring into Neon ... (a few minutes for 133 MB)")
        r = subprocess.run(["psql", direct(url), "-v", "ON_ERROR_STOP=1", "-q",
                            "-f", str(dump)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"restore failed:\n{r.stderr[-2000:]}")

    after = counts(url)
    print(f"\n{'table':16} {'local':>10} {'neon':>10}  match")
    ok = True
    for t in TABLES:
        same = before_local[t] == after[t]
        ok &= same
        print(f"{t:16} {before_local[t]:>10,} {after[t]:>10,}  {'yes' if same else 'NO'}")

    if args.readonly_password:
        conn = psycopg2.connect(url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(READONLY_SQL, {"pw": args.readonly_password,
                                       "db": psycopg2.extensions.AsIs(
                                           url.rsplit("/", 1)[-1].split("?")[0])})
        conn.close()
        print("\necho_readonly role created — use it for the deployed app")

    if not ok:
        sys.exit("row counts differ; the copy is not complete")
    print("\nall tables match")


if __name__ == "__main__":
    main()
