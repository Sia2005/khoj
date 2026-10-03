from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
LOCAL_DATABASE_URL = "postgresql://khoj:khoj@localhost:5432/khoj"


def extension_search_paths() -> list[Path]:
    configured = os.environ.get("KHOJ_BUILD_DIR")
    candidates = [ROOT / configured] if configured else [ROOT / "build"]
    return [path for path in candidates if path.is_dir()]


for directory in [*extension_search_paths(), ROOT]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    psycopg = pytest.importorskip("psycopg")
    from psycopg import sql

    server_url = os.environ.get("KHOJ_TEST_DATABASE_URL", LOCAL_DATABASE_URL)
    try:
        admin = psycopg.connect(server_url, autocommit=True)
    except psycopg.OperationalError as error:
        pytest.skip(f"PostgreSQL unreachable at {server_url}: {error}")

    schema = sql.Identifier(f"khoj_test_{uuid.uuid4().hex[:12]}")
    with admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        url = psycopg.conninfo.make_conninfo(
            server_url, options=f"-c search_path={schema.as_string(admin)}"
        )
        try:
            with psycopg.connect(url, autocommit=True) as connection:
                connection.execute(SCHEMA_PATH.read_text())
            yield url
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))


@pytest.fixture
def empty_database(database_url: str) -> str:
    import psycopg

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("TRUNCATE search_log, documents, owners RESTART IDENTITY CASCADE")
    return database_url
