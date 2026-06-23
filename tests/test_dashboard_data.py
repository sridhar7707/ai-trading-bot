import sqlite3
import pytest
from unittest.mock import patch
from dashboard.data import get_db_conn, safe_query, safe_execute


@pytest.fixture()
def tmp_db(tmp_path):
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, val TEXT)")
    conn.commit()
    conn.close()
    return db


def test_get_db_conn_yields_connection(tmp_db):
    with get_db_conn(tmp_db) as conn:
        assert isinstance(conn, sqlite3.Connection)


def test_get_db_conn_closes_on_exit(tmp_db):
    with get_db_conn(tmp_db) as conn:
        pass
    # Connection should be closed; further use raises ProgrammingError
    with pytest.raises(Exception):
        conn.execute("SELECT 1")


def test_safe_execute_insert_and_query(tmp_db):
    ok = safe_execute("INSERT INTO items (val) VALUES (?)", ("hello",), db_path=tmp_db)
    assert ok is True
    rows = safe_query("SELECT val FROM items", db_path=tmp_db, default=[])
    assert len(rows) == 1
    assert rows[0][0] == "hello"


def test_safe_execute_returns_false_on_bad_sql(tmp_db):
    result = safe_execute("NOT VALID SQL", db_path=tmp_db)
    assert result is False


def test_safe_query_returns_default_on_error(tmp_db):
    result = safe_query("SELECT * FROM nonexistent", db_path=tmp_db, default=[])
    assert result == []


def test_safe_query_empty_table(tmp_db):
    rows = safe_query("SELECT * FROM items", db_path=tmp_db, default=None)
    assert rows == []
