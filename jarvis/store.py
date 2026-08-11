"""SQLite persistence — one file per project (spec §6).

FTS5 supplies BM25 natively. Embeddings are BLOBs, brute-forced in numpy at query time;
see the deviation note in the plan. papers / units / embeddings stay normalized so a
re-embed with a different model does not require re-parsing.

This is the only module that writes SQL.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id       TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    authors        TEXT NOT NULL DEFAULT '[]',   -- JSON array
    year           INTEGER,
    venue          TEXT NOT NULL DEFAULT '',
    doi            TEXT NOT NULL DEFAULT '',
    arxiv_id       TEXT NOT NULL DEFAULT '',
    s2_id          TEXT NOT NULL DEFAULT '',
    abstract       TEXT NOT NULL DEFAULT '',
    citation_count INTEGER NOT NULL DEFAULT 0,
    retracted      INTEGER NOT NULL DEFAULT 0,
    version        TEXT NOT NULL DEFAULT '',
    source_path    TEXT NOT NULL DEFAULT '',
    raw_text       TEXT NOT NULL DEFAULT '',     -- Layer 0, immutable
    depth          TEXT NOT NULL DEFAULT 'metadata'  -- metadata | abstract | deep
);

CREATE TABLE IF NOT EXISTS units (
    unit_id       TEXT PRIMARY KEY,
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    type          TEXT NOT NULL,
    page          INTEGER NOT NULL DEFAULT 1,
    section_path  TEXT NOT NULL DEFAULT '[]',    -- JSON array
    verbatim_text TEXT NOT NULL,
    ordinal       INTEGER NOT NULL DEFAULT 0,
    context_prefix TEXT NOT NULL DEFAULT '',
    parent_id     TEXT,
    label         TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_units_paper  ON units(paper_id);
CREATE INDEX IF NOT EXISTS idx_units_parent ON units(parent_id);

CREATE TABLE IF NOT EXISTS embeddings (
    unit_id TEXT NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    model   TEXT NOT NULL,
    dim     INTEGER NOT NULL,
    vector  BLOB NOT NULL,
    PRIMARY KEY (unit_id, model)
);

CREATE TABLE IF NOT EXISTS cards (
    paper_id TEXT PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,
    payload  TEXT NOT NULL                        -- JSON Card
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    text     TEXT NOT NULL,
    unit_id  TEXT NOT NULL REFERENCES units(unit_id) ON DELETE CASCADE,
    quote    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verifications (
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id) ON DELETE CASCADE,
    unit_id             TEXT NOT NULL,
    quote_found         INTEGER NOT NULL,
    verdict             TEXT NOT NULL,
    entailment_score    REAL NOT NULL DEFAULT 0.0,
    contradiction_score REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (claim_id, unit_id)
);

CREATE TABLE IF NOT EXISTS screen_log (
    paper_id  TEXT NOT NULL,
    run_id    TEXT NOT NULL DEFAULT '',
    decision  TEXT NOT NULL,                      -- read_deep | unsure | defer
    signals   TEXT NOT NULL DEFAULT '{}',         -- JSON per-signal scores
    PRIMARY KEY (paper_id, run_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id    TEXT PRIMARY KEY,
    question  TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL DEFAULT '',
    cost_usd  REAL NOT NULL DEFAULT 0.0
);

CREATE VIRTUAL TABLE IF NOT EXISTS units_fts USING fts5(
    unit_id UNINDEXED,
    text,
    tokenize = 'porter unicode61'
);
"""


def open_store(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) a corpus database and return a configured connection."""
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def close_store(conn: sqlite3.Connection) -> None:
    conn.commit()
    conn.close()
