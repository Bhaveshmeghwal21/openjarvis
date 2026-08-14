import sqlite3

from jarvis.store import SCHEMA_VERSION, close_store, open_store


def _tables(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
    ).fetchall()
    return {r[0] for r in rows}


def test_open_store_creates_all_spec_tables(tmp_path):
    conn = open_store(tmp_path / "c.db")
    assert {"papers", "units", "embeddings", "cards", "claims",
            "verifications", "screen_log", "runs"} <= _tables(conn)
    close_store(conn)


def test_open_store_creates_fts5_index(tmp_path):
    conn = open_store(tmp_path / "c.db")
    assert "units_fts" in _tables(conn)
    close_store(conn)


def test_open_store_is_idempotent(tmp_path):
    path = tmp_path / "c.db"
    close_store(open_store(path))
    conn = open_store(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    close_store(conn)


def test_foreign_keys_are_enforced(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        conn.execute(
            "INSERT INTO units (unit_id, paper_id, type, page, section_path, "
            "verbatim_text, ordinal) VALUES ('u','missing','prose',1,'[]','x',0)"
        )
        raise AssertionError("expected FOREIGN KEY violation")
    except sqlite3.IntegrityError:
        pass
    finally:
        close_store(conn)


def test_rows_are_dict_accessible(tmp_path):
    conn = open_store(tmp_path / "c.db")
    conn.execute("INSERT INTO papers (paper_id, title) VALUES ('p1','T')")
    row = conn.execute("SELECT paper_id, title FROM papers").fetchone()
    assert row["title"] == "T"
    close_store(conn)


def test_store_works_in_memory():
    conn = open_store(":memory:")
    assert "papers" in _tables(conn)
    close_store(conn)
