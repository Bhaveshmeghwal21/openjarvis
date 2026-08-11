"""FTS5 keyword index — the BM25 half of hybrid retrieval (spec §7 Stage D).

SQLite's bm25() returns a negative score where more negative is better. Every score
leaving this module is sign-flipped so callers see "higher is better" consistently.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from jarvis.context import embedding_text
from jarvis.models import Unit

_TERM = re.compile(r"[A-Za-z0-9_]+")


def fts_escape(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every term is double-quoted, so FTS5 operators (AND/OR/NOT/NEAR/*/-/parens) in user
    text are treated as literals rather than syntax errors.
    """
    terms = _TERM.findall(query or "")
    return " ".join(f'"{t}"' for t in terms)


def index_units_fts(conn: sqlite3.Connection, units: Sequence[Unit]) -> int:
    """Index units for keyword search. Re-indexing a unit replaces its row."""
    if not units:
        return 0
    conn.executemany(
        "DELETE FROM units_fts WHERE unit_id = ?", [(u.unit_id,) for u in units]
    )
    conn.executemany(
        "INSERT INTO units_fts (unit_id, text) VALUES (?, ?)",
        [(u.unit_id, embedding_text(u)) for u in units],
    )
    conn.commit()
    return len(units)


def keyword_search(conn: sqlite3.Connection, query: str,
                   limit: int = 20) -> list[tuple[str, float]]:
    """BM25 search. Returns (unit_id, score) with higher scores first."""
    match = fts_escape(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            """
            SELECT unit_id, bm25(units_fts) AS score
            FROM units_fts WHERE units_fts MATCH ?
            ORDER BY score LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [(r["unit_id"], -float(r["score"])) for r in rows]
