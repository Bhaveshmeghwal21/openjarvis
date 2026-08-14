"""SQLite persistence — one file per project (spec §6).

FTS5 supplies BM25 natively. Embeddings are BLOBs, brute-forced in numpy at query time;
see the deviation note in the plan. papers / units / embeddings stay normalized so a
re-embed with a different model does not require re-parsing.

This is the only module that writes SQL.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from jarvis.models import Card, CardField, Paper, Unit, UnitType

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
    depth          TEXT NOT NULL DEFAULT 'metadata'  -- metadata | abstract | pending_deep | deep
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


def save_paper(conn: sqlite3.Connection, paper: Paper, raw_text: str = "",
               depth: str = "metadata") -> None:
    """Upsert a paper. Layer 0 `raw_text` is only ever written, never blanked."""
    conn.execute(
        """
        INSERT INTO papers (paper_id, title, authors, year, venue, doi, arxiv_id, s2_id,
                            abstract, citation_count, retracted, version, source_path,
                            raw_text, depth)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(paper_id) DO UPDATE SET
            title=excluded.title, authors=excluded.authors, year=excluded.year,
            venue=excluded.venue, doi=excluded.doi, arxiv_id=excluded.arxiv_id,
            s2_id=excluded.s2_id, abstract=excluded.abstract,
            citation_count=excluded.citation_count, retracted=excluded.retracted,
            version=excluded.version, source_path=excluded.source_path,
            depth=excluded.depth,
            raw_text=CASE WHEN excluded.raw_text != '' THEN excluded.raw_text
                          ELSE papers.raw_text END
        """,
        (paper.paper_id, paper.title, json.dumps(list(paper.authors)), paper.year,
         paper.venue, paper.doi, paper.arxiv_id, paper.s2_id, paper.abstract,
         paper.citation_count, int(paper.retracted), paper.version, paper.source_path,
         raw_text, depth),
    )
    conn.commit()


def _row_to_paper(row: sqlite3.Row) -> Paper:
    return Paper(
        paper_id=row["paper_id"], title=row["title"],
        authors=tuple(json.loads(row["authors"])), year=row["year"], venue=row["venue"],
        doi=row["doi"], arxiv_id=row["arxiv_id"], s2_id=row["s2_id"],
        abstract=row["abstract"], citation_count=row["citation_count"],
        retracted=bool(row["retracted"]), version=row["version"],
        source_path=row["source_path"],
    )


def get_paper(conn: sqlite3.Connection, paper_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return _row_to_paper(row) if row else None


def get_raw_text(conn: sqlite3.Connection, paper_id: str) -> str:
    row = conn.execute("SELECT raw_text FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return row["raw_text"] if row else ""


def save_units(conn: sqlite3.Connection, units: list[Unit]) -> None:
    """Upsert units. `unit_id` embeds paper_id/type/page/ordinal (see `Unit.key()`), so
    those can never silently go stale on a same-id re-save; section_path is not part of
    the id and must be refreshed explicitly, or a re-chunk with different section
    boundaries would leave the old path in storage."""
    conn.executemany(
        """
        INSERT INTO units (unit_id, paper_id, type, page, section_path, verbatim_text,
                           ordinal, context_prefix, parent_id, label)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(unit_id) DO UPDATE SET
            section_path=excluded.section_path,
            verbatim_text=excluded.verbatim_text,
            context_prefix=excluded.context_prefix,
            parent_id=excluded.parent_id, label=excluded.label
        """,
        [(u.unit_id, u.paper_id, u.type.value, u.page, json.dumps(list(u.section_path)),
          u.verbatim_text, u.ordinal, u.context_prefix, u.parent_id, u.label)
         for u in units],
    )
    conn.commit()


def _row_to_unit(row: sqlite3.Row) -> Unit:
    return Unit(
        unit_id=row["unit_id"], paper_id=row["paper_id"], type=UnitType(row["type"]),
        page=row["page"], section_path=tuple(json.loads(row["section_path"])),
        verbatim_text=row["verbatim_text"], ordinal=row["ordinal"],
        context_prefix=row["context_prefix"], parent_id=row["parent_id"],
        label=row["label"],
    )


def get_units(conn: sqlite3.Connection, paper_id: str) -> list[Unit]:
    rows = conn.execute(
        "SELECT * FROM units WHERE paper_id = ? ORDER BY ordinal", (paper_id,)
    ).fetchall()
    return [_row_to_unit(r) for r in rows]


def get_unit(conn: sqlite3.Connection, unit_id: str) -> Unit | None:
    row = conn.execute("SELECT * FROM units WHERE unit_id = ?", (unit_id,)).fetchone()
    return _row_to_unit(row) if row else None



# --- screening log (spec §7B: every decision auditable and re-runnable) -------------


def save_screen_decision(conn: sqlite3.Connection, paper_id: str, decision: str,
                         signals: dict, run_id: str = "") -> None:
    """Record one gate decision with the per-signal scores that produced it.

    The scores are the whole point: a decision without them cannot be audited, and a
    threshold cannot be recalibrated without re-fetching every abstract.
    """
    conn.execute(
        """
        INSERT INTO screen_log (paper_id, run_id, decision, signals) VALUES (?,?,?,?)
        ON CONFLICT(paper_id, run_id) DO UPDATE SET
            decision=excluded.decision, signals=excluded.signals
        """,
        (paper_id, run_id, decision, json.dumps(signals or {})),
    )
    conn.commit()


def get_screen_decisions(conn: sqlite3.Connection, run_id: str = "") -> dict[str, str]:
    rows = conn.execute(
        "SELECT paper_id, decision FROM screen_log WHERE run_id = ?", (run_id,)
    ).fetchall()
    return {r["paper_id"]: r["decision"] for r in rows}


def get_screen_signals(conn: sqlite3.Connection, run_id: str = "") -> dict[str, dict]:
    rows = conn.execute(
        "SELECT paper_id, signals FROM screen_log WHERE run_id = ?", (run_id,)
    ).fetchall()
    return {r["paper_id"]: json.loads(r["signals"]) for r in rows}


# --- Layer 2 cards -----------------------------------------------------------------


def _field_to_dict(field: CardField | None) -> dict | None:
    if field is None:
        return None
    return {"value": field.value, "unit_id": field.unit_id, "quote": field.quote,
            "binding_verified": field.binding_verified}


def _field_from_dict(data: dict | None) -> CardField | None:
    if not data:
        return None
    return CardField(value=data.get("value", ""), unit_id=data.get("unit_id", ""),
                     quote=data.get("quote", ""),
                     binding_verified=bool(data.get("binding_verified", False)))


_CARD_TUPLES = ("datasets", "metrics", "claims", "limitations")
_CARD_SINGLES = ("problem", "method")


def save_card(conn: sqlite3.Connection, card: Card) -> None:
    payload = {name: _field_to_dict(getattr(card, name)) for name in _CARD_SINGLES}
    payload.update(
        {name: [_field_to_dict(f) for f in getattr(card, name)] for name in _CARD_TUPLES}
    )
    conn.execute(
        """
        INSERT INTO cards (paper_id, payload) VALUES (?,?)
        ON CONFLICT(paper_id) DO UPDATE SET payload=excluded.payload
        """,
        (card.paper_id, json.dumps(payload)),
    )
    conn.commit()


def get_card(conn: sqlite3.Connection, paper_id: str) -> Card | None:
    row = conn.execute("SELECT payload FROM cards WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is None:
        return None
    data = json.loads(row["payload"])
    kwargs = {name: _field_from_dict(data.get(name)) for name in _CARD_SINGLES}
    kwargs.update(
        {name: tuple(f for f in (_field_from_dict(d) for d in data.get(name, [])) if f)
         for name in _CARD_TUPLES}
    )
    return Card(paper_id=paper_id, **kwargs)


# --- runs, depth, corpus-wide reads -------------------------------------------------


def save_run(conn: sqlite3.Connection, run_id: str, question: str = "",
             started_at: str = "", cost_usd: float = 0.0) -> None:
    conn.execute(
        """
        INSERT INTO runs (run_id, question, started_at, cost_usd) VALUES (?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
            question=excluded.question, started_at=excluded.started_at,
            cost_usd=excluded.cost_usd
        """,
        (run_id, question, started_at, cost_usd),
    )
    conn.commit()


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def set_depth(conn: sqlite3.Connection, paper_id: str, depth: str) -> None:
    """Promote or demote a paper's ingest depth. Never deletes; `defer` is demotion.

    Raises if `paper_id` has no row yet, rather than silently updating zero rows. A
    caller that screens a candidate before saving it (e.g. citation-expanded candidates
    from `expand_citations` persisted separately from search hits) would otherwise get no
    error and no depth change, while `screen_log` still records a decision for a paper
    that the corpus proper has no record of at all.
    """
    cursor = conn.execute("UPDATE papers SET depth = ? WHERE paper_id = ?", (depth, paper_id))
    if cursor.rowcount == 0:
        raise ValueError(
            f"set_depth: no paper row for paper_id={paper_id!r} — save it first "
            "(e.g. via save_paper/save_candidates) before screening or ingesting it"
        )
    conn.commit()


def get_papers_by_depth(conn: sqlite3.Connection, depth: str) -> list[Paper]:
    rows = conn.execute(
        "SELECT * FROM papers WHERE depth = ? ORDER BY paper_id", (depth,)
    ).fetchall()
    return [_row_to_paper(r) for r in rows]


def all_units(conn: sqlite3.Connection, exclude_paper_id: str | None = None) -> list[Unit]:
    """Every unit in the corpus, optionally excluding one paper's own units.

    The exclusion is what makes cross-paper contradiction search possible without a
    paper contradicting itself.
    """
    if exclude_paper_id is None:
        rows = conn.execute("SELECT * FROM units ORDER BY paper_id, ordinal").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM units WHERE paper_id != ? ORDER BY paper_id, ordinal",
            (exclude_paper_id,),
        ).fetchall()
    return [_row_to_unit(r) for r in rows]
