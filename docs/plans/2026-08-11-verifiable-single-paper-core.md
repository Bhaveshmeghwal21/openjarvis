# Verifiable Single-Paper Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the slice that can prove itself correct on a single paper — parse it, turn it into typed evidence units, index and retrieve them, and mechanically verify that a claim is grounded in a real verbatim span — before any paper-gathering exists.

**Architecture:** Three layers over one immutable artifact (spec §5). Layer 0 is the parsed paper, never mutated. Layer 1 is typed evidence units (`prose`/`table`/`figure`/`equation`) in a parent-child structure — the retrieval surface and the citation target. Layer 2 is the paper card, a coverage ledger whose every field points at a unit and a verbatim quote. Verification is two mechanical stages: deterministic string match, then NLI entailment. Every external model (parser, embedder, reranker, NLI) sits behind a Protocol with an offline fake, so the whole suite runs with no network, no keys, and no model downloads.

**Tech Stack:** Python 3.10+, stdlib `sqlite3` with FTS5, `numpy` for brute-force vector search, `pytest`. Real adapters (`docling`, `sentence-transformers`, `transformers`) are lazily imported and live behind optional extras.

## Global Constraints

- Python **>= 3.10**. Use `X | None`, not `Optional[X]`.
- **Never read `.env`.** Configuration is environment variables or `$JARVIS_CONFIG` JSON only.
- **Every test is offline.** No network, no API keys, no model downloads. Heavy dependencies are lazily imported inside functions, never at module top level.
- All external models are consumed through a `typing.Protocol` with a deterministic fake used in tests.
- Line length **100**. Target `py310`. Run `ruff check .` before every commit.
- Layer 0 is **immutable** — once a paper is parsed and stored, its `raw_text` and blocks are never rewritten.
- No LLM may be routed to a verification task (enforced by `test_verification_is_not_routed_to_an_llm`, already passing).
- Frozen dataclasses for all domain types; tuples not lists in frozen types.
- Commit after every task with a `feat:`/`test:`/`fix:` prefix.

## Deviation from spec §6, recorded deliberately

The spec names **sqlite-vec** for vector search. This plan stores embeddings as `BLOB`s and brute-forces cosine in numpy instead, behind a `VectorIndex` protocol.

Reason: a 500-paper corpus yields ~100k units; 100k × 384 floats is a few tens of milliseconds per query in numpy. sqlite-vec is a native extension whose loading is fragile on Windows, and it buys nothing at this scale. The protocol keeps it a drop-in later if a corpus ever outgrows brute force.

## File Structure

| File | Responsibility |
|---|---|
| `jarvis/models.py` | Frozen domain types for Layers 0/1/2 and verification. No behaviour. |
| `jarvis/text.py` | Text normalization and span-finding. Shared by unit building and verification. |
| `jarvis/store.py` | SQLite schema, connection, and CRUD. The only module that writes SQL. |
| `jarvis/parse.py` | `Parser` protocol, `FakeParser`, lazy Docling adapter. Produces Layer 0. |
| `jarvis/units.py` | Layer 0 → Layer 1. Chunking, unit typing, binding rules, parent-child. |
| `jarvis/context.py` | Contextual prefix generation for child units. |
| `jarvis/embed.py` | `Embedder` protocol, `FakeEmbedder`, lazy BGE adapter, `VectorIndex`. |
| `jarvis/index.py` | FTS5 keyword index: build, query, escape. |
| `jarvis/retrieve.py` | RRF fusion, hybrid search, parent expansion, reranking. |
| `jarvis/verify.py` | `NLIModel` protocol, `FakeNLI`, quote matcher, three-label verdicts. |
| `jarvis/evaluate.py` | Metrics from spec §10. |

Tests mirror module names: `tests/test_models.py`, `tests/test_text.py`, and so on.

---

### Task 1: Domain types

**Files:**
- Create: `jarvis/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing
- Produces: `UnitType`, `Verdict`, `Block`, `ParsedPaper`, `Paper`, `Unit`, `CardField`, `Card`, `Claim`, `Verification` — all frozen dataclasses; `Unit.key()` returns the deterministic id string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import dataclasses

import pytest

from jarvis.models import (
    Block, Card, CardField, Claim, Paper, ParsedPaper, Unit, UnitType, Verdict, Verification,
)


def test_unit_types_cover_the_four_evidence_kinds():
    assert {t.value for t in UnitType} == {"prose", "table", "figure", "equation"}


def test_verdicts_include_quote_not_found():
    assert {v.value for v in Verdict} == {
        "supported", "contradicted", "neutral", "quote_not_found",
    }


def test_domain_types_are_frozen():
    for cls in (Block, ParsedPaper, Paper, Unit, CardField, Card, Claim, Verification):
        assert dataclasses.fields(cls) is not None
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


def test_paper_requires_only_id_and_title():
    p = Paper(paper_id="p1", title="T")
    assert p.year is None
    assert p.retracted is False
    assert p.authors == ()


def test_unit_key_is_deterministic_and_position_scoped():
    u = Unit(unit_id="", paper_id="p1", type=UnitType.PROSE, page=3,
             section_path=("Methods",), verbatim_text="x", ordinal=7)
    assert u.key() == "p1:prose:3:7"


def test_unit_is_immutable():
    u = Unit(unit_id="u", paper_id="p1", type=UnitType.PROSE, page=1,
             section_path=(), verbatim_text="x", ordinal=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.verbatim_text = "y"


def test_card_field_defaults_to_unverified_binding():
    f = CardField(value="94.2", unit_id="u1", quote="94.2% on KITTI")
    assert f.binding_verified is False


def test_card_holds_tuples_not_lists():
    c = Card(paper_id="p1", metrics=(CardField("94.2", "u1", "q"),))
    assert isinstance(c.metrics, tuple)
    assert c.datasets == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.models'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/models.py
"""Frozen domain types for Layers 0/1/2 and verification (spec §5).

No behaviour beyond identity helpers. Everything is frozen and uses tuples so instances
are hashable and safe to pass between subagents.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UnitType(str, Enum):
    """Kinds of evidence unit (spec §5, Layer 1)."""
    PROSE = "prose"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"


class Verdict(str, Enum):
    """Outcome of verifying one claim against one cited unit (spec §8)."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    QUOTE_NOT_FOUND = "quote_not_found"


@dataclass(frozen=True)
class Block:
    """One element emitted by a parser. Layer 0 is a sequence of these."""
    kind: str                       # heading | paragraph | table | figure | equation | caption
    text: str
    page: int = 1
    section_path: tuple[str, ...] = ()
    label: str = ""                 # "Table 3", "Figure 1" when the parser knows it


@dataclass(frozen=True)
class ParsedPaper:
    """Layer 0 — immutable. The only source of truth for any claim."""
    paper_id: str
    blocks: tuple[Block, ...] = ()
    raw_text: str = ""


@dataclass(frozen=True)
class Paper:
    """Paper-level record and provenance metadata (spec §5)."""
    paper_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    s2_id: str = ""
    abstract: str = ""
    citation_count: int = 0
    retracted: bool = False
    version: str = ""
    source_path: str = ""


@dataclass(frozen=True)
class Unit:
    """Layer 1 — a typed evidence unit. The retrieval surface and the citation target."""
    unit_id: str
    paper_id: str
    type: UnitType
    page: int
    section_path: tuple[str, ...]
    verbatim_text: str
    ordinal: int = 0
    context_prefix: str = ""
    parent_id: str | None = None
    label: str = ""

    def key(self) -> str:
        """Deterministic id: stable across re-ingest of the same parse."""
        return f"{self.paper_id}:{self.type.value}:{self.page}:{self.ordinal}"


@dataclass(frozen=True)
class CardField:
    """One card field, anchored to a unit and a verbatim quote (spec §5, Layer 2)."""
    value: str
    unit_id: str
    quote: str
    binding_verified: bool = False


@dataclass(frozen=True)
class Card:
    """Layer 2 — coverage ledger and comparison index. Never the ground for a claim."""
    paper_id: str
    problem: CardField | None = None
    method: CardField | None = None
    datasets: tuple[CardField, ...] = ()
    metrics: tuple[CardField, ...] = ()
    claims: tuple[CardField, ...] = ()
    limitations: tuple[CardField, ...] = ()


@dataclass(frozen=True)
class Claim:
    """A statement asserted by the system, with the unit and quote it rests on."""
    claim_id: str
    text: str
    unit_id: str
    quote: str


@dataclass(frozen=True)
class Verification:
    """Result of the two-stage verification pass (spec §8)."""
    claim_id: str
    unit_id: str
    quote_found: bool
    verdict: Verdict
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v && ruff check jarvis/models.py`
Expected: 8 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/models.py tests/test_models.py
git commit -m "feat: domain types for layers 0/1/2 and verification"
```

---

### Task 2: Text normalization and span finding

**Files:**
- Create: `jarvis/text.py`
- Test: `tests/test_text.py`

**Why this comes before parsing:** PDF text carries ligatures, hyphens at line breaks, smart quotes, and collapsed whitespace. A naive `quote in raw_text` check fails constantly on real papers, which would make the 100% quote-fidelity target in spec §10 unreachable for reasons that have nothing to do with hallucination. Normalization is the crux of the whole verification stage, so it gets built and tested first.

**Interfaces:**
- Consumes: nothing
- Produces: `normalize(text: str) -> str`, `find_span(needle: str, haystack: str) -> tuple[int, int] | None`, `approx_tokens(text: str) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_text.py
from jarvis.text import approx_tokens, find_span, normalize


def test_normalize_collapses_whitespace():
    assert normalize("a   b\n\nc\t d") == "a b c d"


def test_normalize_repairs_hyphenation_across_line_breaks():
    assert normalize("distur-\nbance rejection") == "disturbance rejection"


def test_normalize_keeps_real_hyphens():
    assert normalize("state-of-the-art method") == "state-of-the-art method"


def test_normalize_folds_ligatures():
    assert normalize("ﬁne-tuned classiﬁer") == "fine-tuned classifier"


def test_normalize_folds_smart_quotes_and_dashes():
    assert normalize("“wind” — the agent’s") == '"wind" - the agent\'s'


def test_normalize_is_idempotent():
    once = normalize("distur-\nbance   “x”")
    assert normalize(once) == once


def test_find_span_locates_quote_despite_pdf_artifacts():
    haystack = "We show that distur-\nbance   rejection  improves."
    span = find_span("disturbance rejection", haystack)
    assert span is not None
    start, end = span
    assert normalize(haystack)[start:end] == "disturbance rejection"


def test_find_span_returns_none_for_absent_quote():
    assert find_span("never written", "some other text") is None


def test_find_span_is_case_and_punctuation_sensitive_enough_to_matter():
    # Fabrication must not pass: a different number is a different quote.
    assert find_span("94.2% on KITTI", "we report 91.7% on KITTI") is None


def test_find_span_rejects_a_partial_number_match_inside_a_longer_one():
    # "2.5% error" is a literal tail-substring of "12.5% error" but is a different,
    # fabricated number. An unanchored substring search would wrongly accept this.
    assert find_span("2.5% error", "we measure 12.5% error on the test set") is None


def test_find_span_rejects_a_partial_word_match():
    assert find_span("cat", "the results concatenate nicely") is None


def test_find_span_of_empty_needle_is_none():
    assert find_span("", "anything") is None


def test_approx_tokens_scales_with_length():
    assert approx_tokens("") == 0
    assert approx_tokens("one two three four") == 5  # 4 words * 1.3, rounded
    assert approx_tokens("a " * 100) > approx_tokens("a " * 50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.text'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/text.py
"""Text normalization and span finding.

Shared by unit building (what gets stored as verbatim_text) and verification (whether a
quote genuinely appears in Layer 0). Both sides MUST normalize identically, or the quote
matcher reports fabrication for text that is actually present.

PDF extraction introduces: ligatures, hyphenation at line breaks, smart quotes and dashes,
non-breaking spaces, and irregular whitespace. None of these are semantic.
"""
from __future__ import annotations

import re
import unicodedata

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ", " ": " ", " ": " ", " ": " ",
}

# A hyphen followed by a line break, joining a word split across lines.
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form for storage and matching. Idempotent."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        out = out.replace(src, dst)
    for src, dst in _PUNCT.items():
        out = out.replace(src, dst)
    out = _HYPHEN_BREAK.sub(r"\1\2", out)
    out = _WHITESPACE.sub(" ", out)
    return out.strip()


def find_span(needle: str, haystack: str) -> tuple[int, int] | None:
    """Locate `needle` in `haystack` after normalizing both.

    Returns (start, end) offsets into `normalize(haystack)`, or None when absent.
    Matching stays exact after normalization: a changed number or word is not a match.

    Boundary-anchored: a plain substring search would let "2.5% error" match inside
    "12.5% error" — a materially different, fabricated number reported as present. A
    genuine verbatim quote always starts and ends on a token boundary in the source
    text, so a match is only accepted when the character immediately before and after
    it (if any) is not alphanumeric.
    """
    n = normalize(needle)
    if not n:
        return None
    h = normalize(haystack)
    idx = h.find(n)
    while idx >= 0:
        before_ok = idx == 0 or not h[idx - 1].isalnum()
        after_ok = (idx + len(n) == len(h)) or not h[idx + len(n)].isalnum()
        if before_ok and after_ok:
            return (idx, idx + len(n))
        idx = h.find(n, idx + 1)
    return None


def approx_tokens(text: str) -> int:
    """Rough token count without a tokenizer dependency.

    English prose averages ~1.3 tokens per whitespace word for BPE tokenizers. Used only
    for chunk sizing, where being off by 10% costs nothing.
    """
    words = len(normalize(text).split())
    return round(words * 1.3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_text.py -v && ruff check jarvis/text.py`
Expected: 11 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/text.py tests/test_text.py
git commit -m "feat: text normalization and span finding for quote verification"
```

---

### Task 3: SQLite schema and connection

**Files:**
- Create: `jarvis/store.py`
- Test: `tests/test_store_schema.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SCHEMA_VERSION: int`, `open_store(path: str | Path) -> sqlite3.Connection`, `close_store(conn) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_schema.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/store.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_schema.py -v && ruff check jarvis/store.py`
Expected: 6 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/store.py tests/test_store_schema.py
git commit -m "feat: sqlite schema for corpus store"
```

---

### Task 4: Paper and unit persistence

**Files:**
- Modify: `jarvis/store.py` (append; do not touch `_SCHEMA`)
- Test: `tests/test_store_crud.py`

**Interfaces:**
- Consumes: `open_store`, `jarvis.models.{Paper, Unit, UnitType}`
- Produces: `save_paper(conn, paper, raw_text="", depth="metadata") -> None`, `get_paper(conn, paper_id) -> Paper | None`, `get_raw_text(conn, paper_id) -> str`, `save_units(conn, units) -> None`, `get_units(conn, paper_id) -> list[Unit]`, `get_unit(conn, unit_id) -> Unit | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_crud.py
import pytest

from jarvis.models import Paper, Unit, UnitType
from jarvis.store import (
    close_store, get_paper, get_raw_text, get_unit, get_units, open_store,
    save_paper, save_units,
)


@pytest.fixture
def conn():
    c = open_store(":memory:")
    yield c
    close_store(c)


def _unit(uid="u1", paper="p1", ordinal=0, parent=None):
    return Unit(unit_id=uid, paper_id=paper, type=UnitType.PROSE, page=2,
                section_path=("Methods", "Setup"), verbatim_text="text body",
                ordinal=ordinal, parent_id=parent)


def test_save_and_get_paper_roundtrips_all_fields(conn):
    p = Paper(paper_id="p1", title="T", authors=("A", "B"), year=2025, venue="V",
              doi="10.1/x", arxiv_id="2501.1", abstract="abs", citation_count=9,
              retracted=True, version="v2")
    save_paper(conn, p)
    got = get_paper(conn, "p1")
    assert got == p


def test_get_paper_returns_none_when_absent(conn):
    assert get_paper(conn, "nope") is None


def test_raw_text_is_stored_and_readable(conn):
    save_paper(conn, Paper("p1", "T"), raw_text="LAYER ZERO")
    assert get_raw_text(conn, "p1") == "LAYER ZERO"


def test_saving_paper_again_does_not_erase_raw_text(conn):
    """Layer 0 is immutable: re-saving metadata must not blank the parsed text."""
    save_paper(conn, Paper("p1", "T"), raw_text="LAYER ZERO")
    save_paper(conn, Paper("p1", "T updated"))
    assert get_raw_text(conn, "p1") == "LAYER ZERO"
    assert get_paper(conn, "p1").title == "T updated"


def test_save_and_get_units_roundtrips(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    got = get_units(conn, "p1")
    assert got == [_unit()]
    assert got[0].section_path == ("Methods", "Setup")


def test_units_are_returned_in_ordinal_order(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit("u2", ordinal=2), _unit("u0", ordinal=0), _unit("u1", ordinal=1)])
    assert [u.unit_id for u in get_units(conn, "p1")] == ["u0", "u1", "u2"]


def test_save_units_is_idempotent(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    save_units(conn, [_unit()])
    assert len(get_units(conn, "p1")) == 1


def test_get_unit_by_id(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    assert get_unit(conn, "u1").verbatim_text == "text body"
    assert get_unit(conn, "missing") is None


def test_deleting_paper_cascades_to_units(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    conn.execute("DELETE FROM papers WHERE paper_id='p1'")
    assert get_units(conn, "p1") == []


def test_parent_id_survives_roundtrip(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit("parent"), _unit("child", ordinal=1, parent="parent")])
    child = get_unit(conn, "child")
    assert child.parent_id == "parent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store_crud.py -v`
Expected: FAIL with `ImportError: cannot import name 'save_paper'`

- [ ] **Step 3: Write minimal implementation**

Append to `jarvis/store.py`:

```python
import json

from jarvis.models import Paper, Unit, UnitType


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
    conn.executemany(
        """
        INSERT INTO units (unit_id, paper_id, type, page, section_path, verbatim_text,
                           ordinal, context_prefix, parent_id, label)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(unit_id) DO UPDATE SET
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_crud.py -v && ruff check jarvis/store.py`
Expected: 10 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/store.py tests/test_store_crud.py
git commit -m "feat: paper and unit persistence"
```

---

### Task 5: Parser protocol, fake, and Docling adapter

**Files:**
- Create: `jarvis/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: `jarvis.models.{Block, ParsedPaper}`, `jarvis.text.normalize`
- Produces: `Parser` protocol with `parse(path: str, paper_id: str) -> ParsedPaper`; `FakeParser(blocks: list[Block])`; `DoclingParser()`; `blocks_to_raw_text(blocks) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parse.py
from jarvis.models import Block
from jarvis.parse import FakeParser, blocks_to_raw_text


def _blocks():
    return [
        Block(kind="heading", text="Methods", page=2, section_path=("Methods",)),
        Block(kind="paragraph", text="We train on KITTI.", page=2, section_path=("Methods",)),
        Block(kind="table", text="| m | v |\n|---|---|\n| acc | 94.2 |", page=3,
              section_path=("Results",), label="Table 3"),
        Block(kind="caption", text="Table 3: Accuracy by method.", page=3,
              section_path=("Results",), label="Table 3"),
    ]


def test_fake_parser_returns_given_blocks():
    parsed = FakeParser(_blocks()).parse("ignored.pdf", "p1")
    assert parsed.paper_id == "p1"
    assert len(parsed.blocks) == 4
    assert parsed.blocks[0].kind == "heading"


def test_fake_parser_blocks_are_a_tuple():
    assert isinstance(FakeParser(_blocks()).parse("x", "p1").blocks, tuple)


def test_raw_text_contains_every_block_normalized():
    raw = blocks_to_raw_text(_blocks())
    assert "We train on KITTI." in raw
    assert "Table 3: Accuracy by method." in raw
    assert "94.2" in raw


def test_parsed_paper_raw_text_is_populated():
    parsed = FakeParser(_blocks()).parse("x", "p1")
    assert "We train on KITTI." in parsed.raw_text


def test_raw_text_normalizes_pdf_artifacts():
    blocks = [Block(kind="paragraph", text="distur-\nbance   rejection")]
    assert blocks_to_raw_text(blocks) == "disturbance rejection"


def test_empty_blocks_give_empty_raw_text():
    assert blocks_to_raw_text([]) == ""


def test_docling_parser_import_is_lazy():
    """Importing jarvis.parse must not require docling to be installed."""
    import jarvis.parse as p
    assert hasattr(p, "DoclingParser")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.parse'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/parse.py
"""Layer 0 production — PDF to an immutable block sequence (spec §5).

Docling is the default (MIT, structured lossless output, CPU-capable). MinerU is the later
escalation path for formula-dense papers. Both sit behind the `Parser` protocol so tests
run with `FakeParser` and never touch a model.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from jarvis.models import Block, ParsedPaper
from jarvis.text import normalize


@runtime_checkable
class Parser(Protocol):
    def parse(self, path: str, paper_id: str) -> ParsedPaper: ...


def blocks_to_raw_text(blocks: Sequence[Block]) -> str:
    """Concatenate blocks into the normalized Layer 0 text that quotes are matched against."""
    return normalize("\n".join(b.text for b in blocks))


class FakeParser:
    """Deterministic parser for tests. Returns the blocks it was constructed with."""

    def __init__(self, blocks: Sequence[Block]) -> None:
        self._blocks = tuple(blocks)

    def parse(self, path: str, paper_id: str) -> ParsedPaper:
        return ParsedPaper(paper_id=paper_id, blocks=self._blocks,
                           raw_text=blocks_to_raw_text(self._blocks))


_DOCLING_KINDS = {
    "section_header": "heading", "title": "heading", "paragraph": "paragraph",
    "text": "paragraph", "table": "table", "picture": "figure", "caption": "caption",
    "formula": "equation",
}


class DoclingParser:
    """Real adapter. `docling` is imported lazily so this module loads without it."""

    def parse(self, path: str, paper_id: str) -> ParsedPaper:
        from docling.document_converter import DocumentConverter

        doc = DocumentConverter().convert(path).document
        blocks: list[Block] = []
        section: list[str] = []
        for item, _level in doc.iterate_items():
            kind = _DOCLING_KINDS.get(getattr(item, "label", ""), "paragraph")
            text = getattr(item, "text", "") or ""
            if kind == "table" and hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown()
            if not text.strip():
                continue
            if kind == "heading":
                section = [text.strip()]
            page = getattr(getattr(item, "prov", [None])[0], "page_no", 1) or 1
            blocks.append(Block(kind=kind, text=text, page=page,
                                section_path=tuple(section),
                                label=getattr(item, "label", "") or ""))
        return ParsedPaper(paper_id=paper_id, blocks=tuple(blocks),
                           raw_text=blocks_to_raw_text(blocks))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_parse.py -v && ruff check jarvis/parse.py`
Expected: 7 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/parse.py tests/test_parse.py
git commit -m "feat: parser protocol with fake and lazy docling adapter"
```

---

### Task 6: Prose chunking

**Files:**
- Create: `jarvis/units.py`
- Test: `tests/test_units_prose.py`

**Interfaces:**
- Consumes: `jarvis.models.{Block, ParsedPaper, Unit, UnitType}`, `jarvis.text.{approx_tokens, normalize}`
- Produces: `build_prose_units(parsed: ParsedPaper, max_tokens: int = 512) -> list[Unit]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_units_prose.py
from jarvis.models import Block, ParsedPaper, UnitType
from jarvis.text import approx_tokens
from jarvis.units import build_prose_units


def _parsed(blocks):
    return ParsedPaper(paper_id="p1", blocks=tuple(blocks))


def test_short_section_becomes_one_unit():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="A short paragraph.", page=1, section_path=("Intro",)),
    ]))
    assert len(units) == 1
    assert units[0].type == UnitType.PROSE
    assert units[0].section_path == ("Intro",)


def test_units_never_span_two_sections():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="alpha " * 10, page=1, section_path=("Intro",)),
        Block(kind="paragraph", text="beta " * 10, page=2, section_path=("Methods",)),
    ]))
    sections = {u.section_path for u in units}
    assert sections == {("Intro",), ("Methods",)}
    for u in units:
        assert not ("alpha" in u.verbatim_text and "beta" in u.verbatim_text)


def test_long_section_is_split_under_the_token_budget():
    long_text = "word " * 2000
    units = build_prose_units(
        _parsed([Block(kind="paragraph", text=long_text, page=1, section_path=("Methods",))]),
        max_tokens=512,
    )
    assert len(units) > 1
    for u in units:
        assert approx_tokens(u.verbatim_text) <= 512


def test_split_preserves_all_content():
    text = " ".join(f"w{i}" for i in range(3000))
    units = build_prose_units(
        _parsed([Block(kind="paragraph", text=text, page=1, section_path=("M",))]),
        max_tokens=512,
    )
    joined = " ".join(u.verbatim_text for u in units)
    assert "w0" in joined and "w2999" in joined


def test_non_prose_blocks_are_ignored_here():
    units = build_prose_units(_parsed([
        Block(kind="table", text="| a |", page=1),
        Block(kind="figure", text="", page=1),
        Block(kind="heading", text="Methods", page=1),
    ]))
    assert units == []


def test_units_get_deterministic_ids_and_ordinals():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="one", page=1, section_path=("A",)),
        Block(kind="paragraph", text="two", page=2, section_path=("B",)),
    ]))
    assert [u.ordinal for u in units] == [0, 1]
    assert units[0].unit_id == "p1:prose:1:0"
    assert len({u.unit_id for u in units}) == 2


def test_rebuilding_the_same_parse_yields_identical_ids():
    parsed = _parsed([Block(kind="paragraph", text="x", page=1, section_path=("A",))])
    assert [u.unit_id for u in build_prose_units(parsed)] == \
           [u.unit_id for u in build_prose_units(parsed)]


def test_verbatim_text_is_normalized():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="distur-\nbance   rejection", page=1),
    ]))
    assert units[0].verbatim_text == "disturbance rejection"


def test_empty_paper_yields_no_units():
    assert build_prose_units(_parsed([])) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_units_prose.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.units'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/units.py
"""Layer 0 -> Layer 1: typed evidence units (spec §5).

Chunking is recursive ~512 tokens on section boundaries. Not semantic chunking: measured
69% vs 54% on academic papers, and 14x slower.
"""
from __future__ import annotations

from jarvis.models import ParsedPaper, Unit, UnitType
from jarvis.text import approx_tokens, normalize

PROSE_KINDS = {"paragraph", "text"}
DEFAULT_MAX_TOKENS = 512


def _split_to_budget(text: str, max_tokens: int) -> list[str]:
    """Split on word boundaries, flushing a piece as soon as it reaches the token budget."""
    if approx_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if approx_tokens(" ".join(current)) >= max_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def build_prose_units(parsed: ParsedPaper, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Unit]:
    """Group contiguous prose blocks by section, then split each group to the token budget."""
    groups: list[tuple[tuple[str, ...], int, list[str]]] = []
    for block in parsed.blocks:
        if block.kind not in PROSE_KINDS or not block.text.strip():
            continue
        if groups and groups[-1][0] == block.section_path and groups[-1][1] == block.page:
            groups[-1][2].append(block.text)
        else:
            groups.append((block.section_path, block.page, [block.text]))

    units: list[Unit] = []
    ordinal = 0
    for section_path, page, texts in groups:
        for piece in _split_to_budget(normalize(" ".join(texts)), max_tokens):
            unit = Unit(unit_id="", paper_id=parsed.paper_id, type=UnitType.PROSE,
                        page=page, section_path=section_path, verbatim_text=piece,
                        ordinal=ordinal)
            units.append(_with_id(unit))
            ordinal += 1
    return units


def _with_id(unit: Unit) -> Unit:
    from dataclasses import replace
    return replace(unit, unit_id=unit.key())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_units_prose.py -v && ruff check jarvis/units.py`
Expected: 9 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/units.py tests/test_units_prose.py
git commit -m "feat: recursive prose chunking on section boundaries"
```

---

### Task 7: Table, figure, and equation units with the binding rule

**Files:**
- Modify: `jarvis/units.py` (append)
- Test: `tests/test_units_artifacts.py`

**The binding rule (spec §5, non-negotiable):** a table without its headers is misleading, a figure without its caption is useless, and "as shown in Figure 3" must stay linked to Figure 3. A table/figure unit is the artifact **plus** its caption **plus** the prose that references it — one indivisible unit.

**Interfaces:**
- Consumes: everything from Task 6
- Produces: `build_artifact_units(parsed: ParsedPaper, start_ordinal: int = 0) -> list[Unit]`, `find_references(blocks, label) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_units_artifacts.py
from jarvis.models import Block, ParsedPaper, UnitType
from jarvis.units import build_artifact_units, find_references


def _parsed(blocks):
    return ParsedPaper(paper_id="p1", blocks=tuple(blocks))


TABLE = Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
              page=3, section_path=("Results",), label="Table 3")
CAPTION = Block(kind="caption", text="Table 3: Accuracy on KITTI.", page=3,
                section_path=("Results",), label="Table 3")
REFERRING = Block(kind="paragraph", text="As shown in Table 3, ours reaches 94.2%.",
                  page=2, section_path=("Results",))
UNRELATED = Block(kind="paragraph", text="Weather was mild.", page=1)


def test_table_unit_includes_markdown_caption_and_referring_text():
    units = build_artifact_units(_parsed([REFERRING, TABLE, CAPTION, UNRELATED]))
    table = next(u for u in units if u.type == UnitType.TABLE)
    assert "94.2" in table.verbatim_text            # the artifact
    assert "Accuracy on KITTI" in table.verbatim_text  # the caption
    assert "As shown in Table 3" in table.verbatim_text  # the referring prose
    assert "Weather was mild" not in table.verbatim_text


def test_table_unit_carries_label_and_page():
    units = build_artifact_units(_parsed([TABLE, CAPTION]))
    table = next(u for u in units if u.type == UnitType.TABLE)
    assert table.label == "Table 3"
    assert table.page == 3


def test_figure_unit_includes_caption_and_referring_text():
    fig = Block(kind="figure", text="", page=4, label="Figure 1")
    cap = Block(kind="caption", text="Figure 1: Architecture.", page=4, label="Figure 1")
    ref = Block(kind="paragraph", text="Figure 1 shows the encoder.", page=4)
    units = build_artifact_units(_parsed([fig, cap, ref]))
    figure = next(u for u in units if u.type == UnitType.FIGURE)
    assert "Architecture" in figure.verbatim_text
    assert "shows the encoder" in figure.verbatim_text


def test_equation_unit_includes_surrounding_prose():
    eq = Block(kind="equation", text="E = mc^2", page=5, section_path=("Theory",))
    before = Block(kind="paragraph", text="Energy is given by", page=5,
                   section_path=("Theory",))
    units = build_artifact_units(_parsed([before, eq]))
    equation = next(u for u in units if u.type == UnitType.EQUATION)
    assert "E = mc^2" in equation.verbatim_text
    assert "Energy is given by" in equation.verbatim_text


def test_artifact_without_caption_still_becomes_a_unit():
    units = build_artifact_units(_parsed([TABLE]))
    assert len(units) == 1
    assert "94.2" in units[0].verbatim_text


def test_find_references_matches_label_variants():
    blocks = [
        Block(kind="paragraph", text="see Table 3 for details"),
        Block(kind="paragraph", text="Tab. 3 confirms this"),
        Block(kind="paragraph", text="Table 30 is different"),
        Block(kind="paragraph", text="nothing here"),
    ]
    found = find_references(blocks, "Table 3")
    assert len(found) == 2
    assert all("30" not in f for f in found)


def test_find_references_with_empty_label_returns_nothing():
    assert find_references([Block(kind="paragraph", text="x")], "") == []


def test_ordinals_continue_from_start_ordinal():
    units = build_artifact_units(_parsed([TABLE, CAPTION]), start_ordinal=10)
    assert units[0].ordinal == 10
    assert units[0].unit_id == "p1:table:3:10"


def test_captions_are_not_emitted_as_standalone_units():
    units = build_artifact_units(_parsed([TABLE, CAPTION]))
    assert len(units) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_units_artifacts.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_artifact_units'`

- [ ] **Step 3: Write minimal implementation**

Append to `jarvis/units.py`:

```python
import re
from typing import Sequence

from jarvis.models import Block

ARTIFACT_KINDS = {"table": UnitType.TABLE, "figure": UnitType.FIGURE,
                  "equation": UnitType.EQUATION}


def _label_pattern(label: str) -> re.Pattern[str] | None:
    """Match 'Table 3' / 'Tab. 3' / 'Fig 3' but never 'Table 30'."""
    m = re.match(r"([A-Za-z]+)\.?\s*(\d+)", label.strip())
    if not m:
        return None
    word, number = m.group(1), m.group(2)
    stem = re.escape(word[:3])
    return re.compile(rf"\b{stem}[a-z]*\.?\s*{re.escape(number)}\b(?!\d)", re.IGNORECASE)


def find_references(blocks: Sequence[Block], label: str) -> list[str]:
    """Prose blocks that mention `label`. This is what keeps 'as shown in Figure 3' bound."""
    pattern = _label_pattern(label)
    if pattern is None:
        return []
    return [b.text for b in blocks
            if b.kind in PROSE_KINDS and pattern.search(b.text)]


def build_artifact_units(parsed: ParsedPaper, start_ordinal: int = 0) -> list[Unit]:
    """One unit per table/figure/equation: artifact + caption + referring prose, indivisible."""
    blocks = list(parsed.blocks)
    units: list[Unit] = []
    ordinal = start_ordinal

    for i, block in enumerate(blocks):
        unit_type = ARTIFACT_KINDS.get(block.kind)
        if unit_type is None:
            continue

        parts: list[str] = []
        if block.text.strip():
            parts.append(block.text)

        if block.label:
            parts += [b.text for b in blocks
                      if b.kind == "caption" and b.label == block.label]
            parts += find_references(blocks, block.label)
        else:
            # Unlabelled artifact (common for equations): take the preceding prose block.
            for previous in reversed(blocks[:i]):
                if previous.kind in PROSE_KINDS:
                    parts.insert(0, previous.text)
                    break

        if not parts:
            continue

        unit = Unit(unit_id="", paper_id=parsed.paper_id, type=unit_type, page=block.page,
                    section_path=block.section_path,
                    verbatim_text=normalize(" ".join(parts)), ordinal=ordinal,
                    label=block.label)
        units.append(_with_id(unit))
        ordinal += 1
    return units
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_units_artifacts.py -v && ruff check jarvis/units.py`
Expected: 9 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/units.py tests/test_units_artifacts.py
git commit -m "feat: table/figure/equation units with caption and reference binding"
```

---

### Task 8: Parent-child structure and the full unit builder

**Files:**
- Modify: `jarvis/units.py` (append)
- Test: `tests/test_units_build.py`

**Interfaces:**
- Consumes: `build_prose_units`, `build_artifact_units`
- Produces: `build_units(parsed: ParsedPaper, max_tokens: int = 512) -> list[Unit]` — children plus one `prose` parent per section, children carrying `parent_id`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_units_build.py
from jarvis.models import Block, ParsedPaper, UnitType
from jarvis.units import build_units


def _parsed(blocks):
    return ParsedPaper(paper_id="p1", blocks=tuple(blocks))


def _long_section(name, word, page):
    return Block(kind="paragraph", text=f"{word} " * 2000, page=page, section_path=(name,))


def test_every_child_has_a_parent():
    units = build_units(_parsed([_long_section("Methods", "alpha", 1)]))
    children = [u for u in units if u.parent_id is not None]
    parents = {u.unit_id for u in units if u.parent_id is None}
    assert children
    assert all(c.parent_id in parents for c in children)


def test_parent_holds_the_whole_section_text():
    units = build_units(_parsed([_long_section("Methods", "alpha", 1)]))
    parent = next(u for u in units if u.parent_id is None)
    child = next(u for u in units if u.parent_id is not None)
    assert len(parent.verbatim_text) > len(child.verbatim_text)
    assert child.verbatim_text.split()[0] in parent.verbatim_text


def test_children_of_different_sections_have_different_parents():
    units = build_units(_parsed([
        _long_section("Intro", "alpha", 1),
        _long_section("Methods", "beta", 2),
    ]))
    parents = {u.parent_id for u in units if u.parent_id is not None}
    assert len(parents) == 2


def test_artifact_units_are_included_and_are_never_split():
    table = Block(kind="table", text="| m | v |\n| acc | 94.2 |", page=3, label="Table 1")
    caption = Block(kind="caption", text="Table 1: Results.", page=3, label="Table 1")
    units = build_units(_parsed([table, caption]))
    tables = [u for u in units if u.type == UnitType.TABLE]
    assert len(tables) == 1
    assert "94.2" in tables[0].verbatim_text


def test_all_unit_ids_are_unique():
    units = build_units(_parsed([
        _long_section("Intro", "alpha", 1),
        Block(kind="table", text="| a |", page=2, label="Table 1"),
        _long_section("Methods", "beta", 3),
    ]))
    assert len({u.unit_id for u in units}) == len(units)


def test_build_is_deterministic():
    parsed = _parsed([_long_section("Methods", "alpha", 1)])
    assert [u.unit_id for u in build_units(parsed)] == [u.unit_id for u in build_units(parsed)]


def test_short_section_needs_no_parent():
    units = build_units(_parsed([
        Block(kind="paragraph", text="Just one line.", page=1, section_path=("Intro",)),
    ]))
    assert len(units) == 1
    assert units[0].parent_id is None


def test_empty_paper_builds_nothing():
    assert build_units(_parsed([])) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_units_build.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_units'`

- [ ] **Step 3: Write minimal implementation**

Append to `jarvis/units.py`:

```python
from dataclasses import replace as _replace


def build_units(parsed: ParsedPaper, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Unit]:
    """Full Layer 1 build: prose children with section parents, plus artifact units.

    Children are what get embedded and matched; parents are what get sent to the model at
    answer time (spec §5, parent/child structure).
    """
    children = build_prose_units(parsed, max_tokens=max_tokens)
    artifacts = build_artifact_units(parsed, start_ordinal=len(children))

    by_section: dict[tuple[tuple[str, ...], int], list[Unit]] = {}
    for child in children:
        by_section.setdefault((child.section_path, child.page), []).append(child)

    out: list[Unit] = []
    parent_ordinal = len(children) + len(artifacts)
    for (section_path, page), group in by_section.items():
        if len(group) < 2:
            out.extend(group)          # a section that fits in one unit needs no parent
            continue
        parent = Unit(unit_id="", paper_id=parsed.paper_id, type=UnitType.PROSE, page=page,
                      section_path=section_path,
                      verbatim_text=" ".join(u.verbatim_text for u in group),
                      ordinal=parent_ordinal)
        parent = _with_id(parent)
        parent_ordinal += 1
        out.append(parent)
        out.extend(_replace(u, parent_id=parent.unit_id) for u in group)

    out.extend(artifacts)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_units_build.py -v && ruff check jarvis/units.py`
Expected: 8 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/units.py tests/test_units_build.py
git commit -m "feat: parent-child unit structure and full layer 1 builder"
```

---

### Task 9: Contextual prefixes

**Files:**
- Create: `jarvis/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `jarvis.models.{Paper, Unit}`
- Produces: `PrefixGenerator` protocol with `describe(paper, unit) -> str`; `TemplatePrefix()`; `LLMPrefix(router, chat_fn=None)`; `apply_prefixes(units, paper, generator) -> list[Unit]`; `embedding_text(unit) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context.py
from jarvis.context import LLMPrefix, TemplatePrefix, apply_prefixes, embedding_text
from jarvis.models import Paper, Unit, UnitType

PAPER = Paper(paper_id="p1", title="Wind Rejection for Quadrotors", year=2025)
UNIT = Unit(unit_id="u1", paper_id="p1", type=UnitType.TABLE, page=3,
            section_path=("Results",), verbatim_text="| acc | 94.2 |", label="Table 3")


def test_template_prefix_names_paper_section_and_artifact():
    prefix = TemplatePrefix().describe(PAPER, UNIT)
    assert "Wind Rejection for Quadrotors" in prefix
    assert "Results" in prefix
    assert "Table 3" in prefix


def test_template_prefix_handles_missing_section_and_label():
    bare = Unit(unit_id="u", paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=(), verbatim_text="x")
    assert TemplatePrefix().describe(PAPER, bare)


def test_apply_prefixes_sets_prefix_on_every_unit():
    out = apply_prefixes([UNIT], PAPER, TemplatePrefix())
    assert out[0].context_prefix
    assert out[0].unit_id == UNIT.unit_id


def test_apply_prefixes_does_not_mutate_verbatim_text():
    out = apply_prefixes([UNIT], PAPER, TemplatePrefix())
    assert out[0].verbatim_text == "| acc | 94.2 |"


def test_embedding_text_is_prefix_then_verbatim():
    unit = apply_prefixes([UNIT], PAPER, TemplatePrefix())[0]
    text = embedding_text(unit)
    assert text.startswith(unit.context_prefix)
    assert text.endswith(unit.verbatim_text)


def test_embedding_text_without_prefix_is_just_verbatim():
    assert embedding_text(UNIT) == "| acc | 94.2 |"


def test_llm_prefix_uses_injected_chat_and_routes_to_cheap_task():
    calls = []

    def fake_chat(router, task, prompt, **kwargs):
        calls.append((task, prompt))
        return "This table reports accuracy in the Results section."

    prefix = LLMPrefix(router=None, chat_fn=fake_chat).describe(PAPER, UNIT)
    assert prefix == "This table reports accuracy in the Results section."
    assert calls[0][0] == "contextual_prefix"
    assert "Wind Rejection for Quadrotors" in calls[0][1]


def test_llm_prefix_falls_back_to_template_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    prefix = LLMPrefix(router=None, chat_fn=boom).describe(PAPER, UNIT)
    assert "Wind Rejection for Quadrotors" in prefix
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.context'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/context.py
"""Contextual prefixes for child units (spec §5).

Anthropic's Contextual Retrieval: a 50-100 token description of where a chunk sits in its
document, prepended before embedding. Measured 35% fewer retrieval failures alone, 49%
with BM25, 67% with reranking.

The prefix is embedded but never stored as part of verbatim_text — a claim must always
resolve to text that genuinely appears in Layer 0.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol, Sequence, runtime_checkable

from jarvis.models import Paper, Unit

_PROMPT = (
    "Write one sentence of 50-100 tokens locating this excerpt within its paper, so it can "
    "be understood on its own. State the paper, the section, and what the excerpt reports. "
    "Do not add facts that are not present.\n\n"
    "Paper: {title} ({year})\nSection: {section}\nArtifact: {label}\n\nExcerpt:\n{text}"
)


@runtime_checkable
class PrefixGenerator(Protocol):
    def describe(self, paper: Paper, unit: Unit) -> str: ...


class TemplatePrefix:
    """Deterministic, free, no model. The fallback and the test default."""

    def describe(self, paper: Paper, unit: Unit) -> str:
        section = " > ".join(unit.section_path) if unit.section_path else "body"
        artifact = f" ({unit.label})" if unit.label else ""
        year = f", {paper.year}" if paper.year else ""
        return (f"From \"{paper.title}\"{year}, {unit.type.value} in section {section}"
                f"{artifact}, page {unit.page}.")


class LLMPrefix:
    """LLM-written prefix, routed to the cheap tier. Falls back to the template on failure."""

    def __init__(self, router, chat_fn: Callable[..., str] | None = None) -> None:
        self._router = router
        self._chat = chat_fn
        self._fallback = TemplatePrefix()

    def _chat_fn(self) -> Callable[..., str]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def describe(self, paper: Paper, unit: Unit) -> str:
        prompt = _PROMPT.format(
            title=paper.title, year=paper.year or "n.d.",
            section=" > ".join(unit.section_path) or "body",
            label=unit.label or "none", text=unit.verbatim_text[:2000],
        )
        try:
            out = self._chat_fn()(self._router, "contextual_prefix", prompt)
            return (out or "").strip() or self._fallback.describe(paper, unit)
        except Exception:
            return self._fallback.describe(paper, unit)


def apply_prefixes(units: Sequence[Unit], paper: Paper,
                   generator: PrefixGenerator) -> list[Unit]:
    """Return copies of `units` with `context_prefix` filled in. verbatim_text is untouched."""
    return [replace(u, context_prefix=generator.describe(paper, u)) for u in units]


def embedding_text(unit: Unit) -> str:
    """What actually gets embedded: prefix then verbatim text."""
    if not unit.context_prefix:
        return unit.verbatim_text
    return f"{unit.context_prefix}\n\n{unit.verbatim_text}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_context.py -v && ruff check jarvis/context.py`
Expected: 8 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/context.py tests/test_context.py
git commit -m "feat: contextual prefixes for child units"
```

---

### Task 10: Embedder protocol and vector index

**Files:**
- Create: `jarvis/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `jarvis.store.open_store`, `jarvis.models.Unit`, `jarvis.context.embedding_text`
- Produces: `Embedder` protocol (`.name`, `.dim`, `encode(texts) -> list[list[float]]`); `FakeEmbedder(dim=8)`; `BGEEmbedder(model_name=...)`; `index_units(conn, units, embedder) -> int`; `vector_search(conn, query_vector, model, limit=20) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed.py
import pytest

from jarvis.embed import FakeEmbedder, index_units, vector_search
from jarvis.models import Paper, Unit, UnitType
from jarvis.store import close_store, open_store, save_paper, save_units


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"))
    yield c
    close_store(c)


def _unit(uid, text, ordinal=0):
    return Unit(unit_id=uid, paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=("A",), verbatim_text=text, ordinal=ordinal)


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=8)
    assert e.encode(["hello"]) == e.encode(["hello"])


def test_fake_embedder_respects_dim_and_name():
    e = FakeEmbedder(dim=16)
    assert e.dim == 16
    assert len(e.encode(["x"])[0]) == 16
    assert e.name


def test_fake_embedder_gives_different_vectors_for_different_text():
    e = FakeEmbedder()
    assert e.encode(["alpha"]) != e.encode(["beta"])


def test_index_units_stores_one_row_per_unit(conn):
    e = FakeEmbedder()
    units = [_unit("u1", "alpha"), _unit("u2", "beta", 1)]
    save_units(conn, units)
    assert index_units(conn, units, e) == 2
    count = conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
    assert count == 2


def test_index_units_is_idempotent(conn):
    e = FakeEmbedder()
    units = [_unit("u1", "alpha")]
    save_units(conn, units)
    index_units(conn, units, e)
    index_units(conn, units, e)
    assert conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"] == 1


def test_vector_search_ranks_the_matching_unit_first(conn):
    e = FakeEmbedder()
    units = [_unit("u_alpha", "alpha alpha alpha"), _unit("u_beta", "beta beta beta", 1)]
    save_units(conn, units)
    index_units(conn, units, e)
    hits = vector_search(conn, e.encode(["alpha alpha alpha"])[0], e.name, limit=2)
    assert hits[0][0] == "u_alpha"


def test_vector_search_respects_limit(conn):
    e = FakeEmbedder()
    units = [_unit(f"u{i}", f"text {i}", i) for i in range(5)]
    save_units(conn, units)
    index_units(conn, units, e)
    assert len(vector_search(conn, e.encode(["text 1"])[0], e.name, limit=3)) == 3


def test_vector_search_ignores_other_models(conn):
    e = FakeEmbedder()
    units = [_unit("u1", "alpha")]
    save_units(conn, units)
    index_units(conn, units, e)
    assert vector_search(conn, e.encode(["alpha"])[0], "some-other-model") == []


def test_vector_search_on_empty_index_returns_empty(conn):
    assert vector_search(conn, [0.1] * 64, "fake-64") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_embed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.embed'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/embed.py
"""Unit-level embedding and vector search.

Spec §7 Stage D: unit-level retrieval uses a strong *general* embedder (BGE-class), not a
scientific one — a medical retrieval study found BGE beat every domain-specific model.
SPECTER2 is for paper-level work and is not used here.

Vectors are stored as float32 BLOBs and searched by brute-force cosine in numpy. At ~100k
units that is tens of milliseconds; see the deviation note in the plan.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from typing import Protocol, Sequence, runtime_checkable

from jarvis.context import embedding_text
from jarvis.models import Unit


@runtime_checkable
class Embedder(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def dim(self) -> int: ...
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic hash-based embedder for tests. No model, no network.

    Each token contributes to two independent buckets, so the chance two distinct tokens
    produce identical vectors is ~1/dim**2 rather than 1/dim. At the default dim that is
    ~1 in 4096, which keeps ranking assertions in tests from flaking.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fake-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode()).digest()
                vec[digest[0] % self._dim] += 1.0
                vec[digest[1] % self._dim] += 0.5
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


class BGEEmbedder:
    """Real adapter. `sentence_transformers` is imported lazily."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vecs = self._load().encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", blob))


def index_units(conn: sqlite3.Connection, units: Sequence[Unit], embedder: Embedder) -> int:
    """Embed each unit's `embedding_text` and store it. Returns the number indexed."""
    if not units:
        return 0
    vectors = embedder.encode([embedding_text(u) for u in units])
    conn.executemany(
        """
        INSERT INTO embeddings (unit_id, model, dim, vector) VALUES (?,?,?,?)
        ON CONFLICT(unit_id, model) DO UPDATE SET
            dim=excluded.dim, vector=excluded.vector
        """,
        [(u.unit_id, embedder.name, len(v), _pack(v)) for u, v in zip(units, vectors)],
    )
    conn.commit()
    return len(units)


def vector_search(conn: sqlite3.Connection, query_vector: Sequence[float], model: str,
                  limit: int = 20) -> list[tuple[str, float]]:
    """Brute-force cosine over stored vectors. Returns (unit_id, similarity), best first."""
    rows = conn.execute(
        "SELECT unit_id, dim, vector FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    if not rows:
        return []

    import numpy as np

    query = np.asarray(query_vector, dtype="float32")
    qn = float(np.linalg.norm(query)) or 1.0
    ids = [r["unit_id"] for r in rows]
    matrix = np.asarray([_unpack(r["vector"], r["dim"]) for r in rows], dtype="float32")
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    scores = (matrix @ query) / (norms * qn)

    order = np.argsort(-scores)[:limit]
    return [(ids[i], float(scores[i])) for i in order]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_embed.py -v && ruff check jarvis/embed.py`
Expected: 9 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/embed.py tests/test_embed.py
git commit -m "feat: embedder protocol and brute-force vector search"
```

---

### Task 11: FTS5 keyword index

**Files:**
- Create: `jarvis/index.py`
- Test: `tests/test_index.py`

**Note on BM25 sign:** SQLite's `bm25()` returns a **negative** score where more negative is a better match. This module flips the sign so callers always see "higher is better".

**Interfaces:**
- Consumes: `jarvis.store.open_store`, `jarvis.models.Unit`, `jarvis.context.embedding_text`
- Produces: `fts_escape(query: str) -> str`, `index_units_fts(conn, units) -> int`, `keyword_search(conn, query, limit=20) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index.py
import pytest

from jarvis.index import fts_escape, index_units_fts, keyword_search
from jarvis.models import Paper, Unit, UnitType
from jarvis.store import close_store, open_store, save_paper, save_units


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"))
    yield c
    close_store(c)


def _unit(uid, text, ordinal=0):
    return Unit(unit_id=uid, paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=("A",), verbatim_text=text, ordinal=ordinal)


def _seed(conn, pairs):
    units = [_unit(uid, text, i) for i, (uid, text) in enumerate(pairs)]
    save_units(conn, units)
    index_units_fts(conn, units)
    return units


def test_fts_escape_quotes_terms():
    assert fts_escape("wind rejection") == '"wind" "rejection"'


def test_fts_escape_neutralises_operators():
    escaped = fts_escape('NEAR(a b) OR "x" -y*')
    assert "NEAR" not in escaped or escaped.count('"') % 2 == 0
    assert escaped.startswith('"')


def test_fts_escape_of_empty_query_is_empty():
    assert fts_escape("   ") == ""


def test_index_returns_count(conn):
    assert index_units_fts(conn, _seed(conn, [("u1", "alpha")])) == 1


def test_keyword_search_finds_the_matching_unit(conn):
    _seed(conn, [("u1", "quadrotor wind rejection"), ("u2", "cake recipes")])
    hits = keyword_search(conn, "wind rejection")
    assert hits[0][0] == "u1"


def test_keyword_search_scores_are_positive_higher_is_better(conn):
    _seed(conn, [("u1", "wind wind wind"), ("u2", "wind once")])
    hits = keyword_search(conn, "wind")
    assert all(score > 0 for _, score in hits)
    assert hits[0][1] >= hits[-1][1]


def test_keyword_search_respects_limit(conn):
    _seed(conn, [(f"u{i}", "wind") for i in range(5)])
    assert len(keyword_search(conn, "wind", limit=2)) == 2


def test_keyword_search_with_no_match_returns_empty(conn):
    _seed(conn, [("u1", "alpha")])
    assert keyword_search(conn, "zebra") == []


def test_keyword_search_does_not_crash_on_operator_input(conn):
    _seed(conn, [("u1", "alpha")])
    assert keyword_search(conn, 'AND OR NOT "(' ) == []


def test_reindexing_does_not_duplicate_rows(conn):
    units = _seed(conn, [("u1", "wind")])
    index_units_fts(conn, units)
    assert len(keyword_search(conn, "wind")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.index'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/index.py
"""FTS5 keyword index — the BM25 half of hybrid retrieval (spec §7 Stage D).

SQLite's bm25() returns a negative score where more negative is better. Every score
leaving this module is sign-flipped so callers see "higher is better" consistently.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Sequence

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_index.py -v && ruff check jarvis/index.py`
Expected: 10 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/index.py tests/test_index.py
git commit -m "feat: fts5 keyword index with safe query escaping"
```

---

### Task 12: Hybrid retrieval — RRF, reranking, parent expansion

**Files:**
- Create: `jarvis/retrieve.py`
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `keyword_search`, `vector_search`, `Embedder`, `jarvis.store.{get_unit, get_units}`
- Produces: `rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]`; `Reranker` protocol; `FakeReranker(order)`; `CrossEncoderReranker(model_name=...)`; `search(conn, query, embedder, limit=10, reranker=None, expand_parents=True) -> list[Unit]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retrieve.py
import pytest

from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Paper, ParsedPaper, Unit, UnitType
from jarvis.retrieve import FakeReranker, rrf, search
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.units import build_units


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"))
    yield c
    close_store(c)


def _seed(conn, texts):
    units = [
        Unit(unit_id=f"u{i}", paper_id="p1", type=UnitType.PROSE, page=1,
             section_path=("A",), verbatim_text=t, ordinal=i)
        for i, t in enumerate(texts)
    ]
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    return units


# --- RRF ---------------------------------------------------------------------

def test_rrf_rewards_agreement_across_rankings():
    fused = dict(rrf([["a", "b", "c"], ["a", "c", "b"]]))
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_rrf_includes_items_present_in_only_one_ranking():
    ids = [uid for uid, _ in rrf([["a"], ["b"]])]
    assert set(ids) == {"a", "b"}


def test_rrf_uses_k_60_by_default():
    fused = dict(rrf([["a"]]))
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_of_nothing_is_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_output_is_sorted_descending():
    scores = [s for _, s in rrf([["a", "b", "c"], ["a", "b", "c"]])]
    assert scores == sorted(scores, reverse=True)


# --- search ------------------------------------------------------------------

def test_search_returns_units_not_ids(conn):
    _seed(conn, ["quadrotor wind rejection", "cake"])
    results = search(conn, "wind rejection", FakeEmbedder(), limit=1)
    assert isinstance(results[0], Unit)


def test_search_finds_the_relevant_unit(conn):
    _seed(conn, ["quadrotor wind rejection under gusts", "sourdough baking"])
    results = search(conn, "wind rejection", FakeEmbedder(), limit=1)
    assert "wind rejection" in results[0].verbatim_text


def test_search_respects_limit(conn):
    _seed(conn, [f"wind {i}" for i in range(5)])
    assert len(search(conn, "wind", FakeEmbedder(), limit=2)) == 2


def test_search_on_empty_corpus_returns_empty(conn):
    assert search(conn, "anything", FakeEmbedder()) == []


def test_reranker_reorders_results(conn):
    _seed(conn, ["wind alpha", "wind beta"])
    reranked = search(conn, "wind", FakeEmbedder(), limit=2,
                      reranker=FakeReranker(order=["u1", "u0"]))
    assert [u.unit_id for u in reranked] == ["u1", "u0"]


def test_search_expands_children_to_parents(conn):
    parsed = ParsedPaper(paper_id="p1", blocks=(
        Block(kind="paragraph", text="wind " * 2000, page=1, section_path=("Methods",)),
    ))
    units = build_units(parsed)
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    results = search(conn, "wind", FakeEmbedder(), limit=3, expand_parents=True)
    assert all(u.parent_id is None for u in results), "children should be swapped for parents"


def test_parent_expansion_dedupes_siblings(conn):
    parsed = ParsedPaper(paper_id="p1", blocks=(
        Block(kind="paragraph", text="wind " * 2000, page=1, section_path=("Methods",)),
    ))
    units = build_units(parsed)
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    results = search(conn, "wind", FakeEmbedder(), limit=5, expand_parents=True)
    assert len({u.unit_id for u in results}) == len(results)


def test_expand_parents_off_returns_children(conn):
    parsed = ParsedPaper(paper_id="p1", blocks=(
        Block(kind="paragraph", text="wind " * 2000, page=1, section_path=("Methods",)),
    ))
    units = build_units(parsed)
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    results = search(conn, "wind", FakeEmbedder(), limit=3, expand_parents=False)
    assert any(u.parent_id is not None for u in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.retrieve'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/retrieve.py
"""Hybrid retrieval: BM25 + vector, fused by RRF, reranked, expanded to parents.

RRF rather than score normalization because cosine and BM25 live on incompatible scales
and RRF needs no tuning (spec §7 Stage D). Children do the matching; parents do the
generating, so hits are swapped for their parents before the text reaches a model.
"""
from __future__ import annotations

import sqlite3
from typing import Protocol, Sequence, runtime_checkable

from jarvis.embed import Embedder, vector_search
from jarvis.index import keyword_search
from jarvis.models import Unit
from jarvis.store import get_unit

RRF_K = 60
CANDIDATE_MULTIPLIER = 5


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. score(d) = sum over lists of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]: ...


class FakeReranker:
    """Test reranker: returns units in the given unit_id order, unknown ids last."""

    def __init__(self, order: Sequence[str]) -> None:
        self._order = list(order)

    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]:
        def position(unit: Unit) -> int:
            return self._order.index(unit.unit_id) if unit.unit_id in self._order else 10**6
        return sorted(units, key=position)


class CrossEncoderReranker:
    """Real adapter. `sentence_transformers` is imported lazily."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, units: Sequence[Unit]) -> list[Unit]:
        if not units:
            return []
        scores = self._load().predict([(query, u.verbatim_text) for u in units])
        return [u for _, u in sorted(zip(scores, units), key=lambda p: -p[0])]


def _expand_to_parents(conn: sqlite3.Connection, units: Sequence[Unit]) -> list[Unit]:
    """Swap each child for its parent, preserving order and dropping duplicate siblings."""
    out: list[Unit] = []
    seen: set[str] = set()
    for unit in units:
        target = unit
        if unit.parent_id:
            parent = get_unit(conn, unit.parent_id)
            if parent is not None:
                target = parent
        if target.unit_id in seen:
            continue
        seen.add(target.unit_id)
        out.append(target)
    return out


def search(conn: sqlite3.Connection, query: str, embedder: Embedder, limit: int = 10,
           reranker: Reranker | None = None, expand_parents: bool = True) -> list[Unit]:
    """One retrieval pass. Callers are expected to run this repeatedly with refined queries."""
    candidates = max(limit * CANDIDATE_MULTIPLIER, limit)

    keyword_ids = [uid for uid, _ in keyword_search(conn, query, limit=candidates)]
    query_vec = embedder.encode([query])[0]
    vector_ids = [uid for uid, _ in vector_search(conn, query_vec, embedder.name,
                                                  limit=candidates)]

    fused = rrf([keyword_ids, vector_ids])
    units = [u for u in (get_unit(conn, uid) for uid, _ in fused) if u is not None]

    if reranker is not None:
        units = reranker.rerank(query, units)
    if expand_parents:
        units = _expand_to_parents(conn, units)
    return units[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_retrieve.py -v && ruff check jarvis/retrieve.py`
Expected: 14 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/retrieve.py tests/test_retrieve.py
git commit -m "feat: hybrid retrieval with rrf fusion and parent expansion"
```

---

### Task 13: Verification — quote matcher and NLI entailment

**Files:**
- Create: `jarvis/verify.py`
- Test: `tests/test_verify.py`

**This is the task the whole plan exists to reach.** Two failure modes, handled separately (spec §8):
- *Citation hallucination* — quote absent from Layer 0. Caught by deterministic string match. Free, exact, no model.
- *Statement hallucination* — quote real, claim not supported by it. Caught by NLI entailment. **Never LLM-as-judge**: 0.101 Pearson vs 0.638 for NLI.

**Interfaces:**
- Consumes: `jarvis.text.find_span`, `jarvis.store.{get_unit, get_raw_text}`, `jarvis.models.{Claim, Verdict, Verification}`
- Produces: `NLIModel` protocol with `predict(premise, hypothesis) -> dict[str, float]`; `FakeNLI(mapping, default=...)`; `HFNLI(model_name=...)`; `quote_is_grounded(conn, claim) -> bool`; `verify_claim(conn, claim, nli, threshold=0.5) -> Verification`; `find_contradictions(conn, claim, units, nli, threshold=0.5) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verify.py
import pytest

from jarvis.models import Claim, Paper, Unit, UnitType, Verdict
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.verify import FakeNLI, find_contradictions, quote_is_grounded, verify_claim

QUOTE = "our method reaches 94.2% on KITTI"
RAW = f"In Section 4 we show that {QUOTE} without extra supervision."


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"), raw_text=RAW)
    save_units(c, [Unit(unit_id="u1", paper_id="p1", type=UnitType.PROSE, page=1,
                        section_path=("Results",), verbatim_text=RAW, ordinal=0)])
    yield c
    close_store(c)


def _claim(text="The method reaches 94.2% on KITTI.", quote=QUOTE):
    return Claim(claim_id="c1", text=text, unit_id="u1", quote=quote)


# --- stage 1: quote grounding ------------------------------------------------

def test_real_quote_is_grounded(conn):
    assert quote_is_grounded(conn, _claim()) is True


def test_fabricated_quote_is_not_grounded(conn):
    assert quote_is_grounded(conn, _claim(quote="reaches 99.9% on KITTI")) is False


def test_quote_grounded_despite_pdf_artifacts(conn):
    save_paper(conn, Paper("p2", "T"), raw_text="distur-\nbance   rejection works")
    save_units(conn, [Unit(unit_id="u2", paper_id="p2", type=UnitType.PROSE, page=1,
                           section_path=(), verbatim_text="x", ordinal=0)])
    claim = Claim(claim_id="c", text="t", unit_id="u2", quote="disturbance rejection")
    assert quote_is_grounded(conn, claim) is True


def test_quote_against_missing_unit_is_not_grounded(conn):
    assert quote_is_grounded(conn, Claim("c", "t", "nope", QUOTE)) is False


def test_empty_quote_is_not_grounded(conn):
    assert quote_is_grounded(conn, _claim(quote="")) is False


# --- stage 2: entailment -----------------------------------------------------

def test_supported_claim_gets_supported_verdict(conn):
    nli = FakeNLI({(QUOTE, _claim().text): {"entailment": 0.95, "neutral": 0.03,
                                            "contradiction": 0.02}})
    result = verify_claim(conn, _claim(), nli)
    assert result.verdict == Verdict.SUPPORTED
    assert result.quote_found is True
    assert result.entailment_score == pytest.approx(0.95)


def test_unsupported_claim_gets_neutral_verdict(conn):
    nli = FakeNLI(default={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})
    assert verify_claim(conn, _claim(), nli).verdict == Verdict.NEUTRAL


def test_contradicted_claim_gets_contradicted_verdict(conn):
    nli = FakeNLI(default={"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85})
    result = verify_claim(conn, _claim(), nli)
    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradiction_score == pytest.approx(0.85)


def test_fabricated_quote_short_circuits_before_nli(conn):
    """A missing quote blocks the claim; the NLI model must never be consulted."""
    calls = []

    class SpyNLI:
        def predict(self, premise, hypothesis):
            calls.append((premise, hypothesis))
            return {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}

    result = verify_claim(conn, _claim(quote="reaches 99.9%"), SpyNLI())
    assert result.verdict == Verdict.QUOTE_NOT_FOUND
    assert result.quote_found is False
    assert calls == []


def test_verification_carries_claim_and_unit_ids(conn):
    nli = FakeNLI(default={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
    result = verify_claim(conn, _claim(), nli)
    assert result.claim_id == "c1"
    assert result.unit_id == "u1"


def test_threshold_is_respected(conn):
    nli = FakeNLI(default={"entailment": 0.60, "neutral": 0.35, "contradiction": 0.05})
    assert verify_claim(conn, _claim(), nli, threshold=0.5).verdict == Verdict.SUPPORTED
    assert verify_claim(conn, _claim(), nli, threshold=0.9).verdict == Verdict.NEUTRAL


# --- contradiction detection (spec §8, same pass) ----------------------------

def test_find_contradictions_returns_conflicting_units(conn):
    other = Unit(unit_id="u9", paper_id="p1", type=UnitType.PROSE, page=2,
                 section_path=("Results",), verbatim_text="the method reaches 71% on KITTI",
                 ordinal=1)
    save_units(conn, [other])
    nli = FakeNLI(default={"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85})
    found = find_contradictions(conn, _claim(), [other], nli)
    assert found[0][0] == "u9"
    assert found[0][1] == pytest.approx(0.85)


def test_find_contradictions_ignores_agreeing_units(conn):
    other = Unit(unit_id="u9", paper_id="p1", type=UnitType.PROSE, page=2,
                 section_path=(), verbatim_text="also 94.2%", ordinal=1)
    save_units(conn, [other])
    nli = FakeNLI(default={"entailment": 0.90, "neutral": 0.05, "contradiction": 0.05})
    assert find_contradictions(conn, _claim(), [other], nli) == []


def test_find_contradictions_sorts_by_confidence(conn):
    a = Unit(unit_id="ua", paper_id="p1", type=UnitType.PROSE, page=2,
             section_path=(), verbatim_text="A", ordinal=1)
    b = Unit(unit_id="ub", paper_id="p1", type=UnitType.PROSE, page=3,
             section_path=(), verbatim_text="B", ordinal=2)
    save_units(conn, [a, b])
    nli = FakeNLI({
        ("A", _claim().text): {"entailment": 0.0, "neutral": 0.3, "contradiction": 0.7},
        ("B", _claim().text): {"entailment": 0.0, "neutral": 0.1, "contradiction": 0.9},
    })
    found = find_contradictions(conn, _claim(), [a, b], nli)
    assert [uid for uid, _ in found] == ["ub", "ua"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.verify'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/verify.py
"""Two-stage mechanical verification (spec §8).

Stage 1 — quote grounding. Deterministic string match against Layer 0. Free, exact, no
model. A claim whose quote is absent is blocked and never reaches stage 2.

Stage 2 — entailment. An NLI model over (quote -> claim). NOT LLM-as-judge: measured
Pearson correlation with human judgement is 0.101 for GPT-3.5-as-judge versus 0.638 for
AutoAIS/NLI. Contradiction detection is the same pass reading NLI's third label.
"""
from __future__ import annotations

import sqlite3
from typing import Mapping, Protocol, Sequence, runtime_checkable

from jarvis.models import Claim, Unit, Verdict, Verification
from jarvis.store import get_raw_text, get_unit
from jarvis.text import find_span

LABELS = ("entailment", "neutral", "contradiction")


@runtime_checkable
class NLIModel(Protocol):
    def predict(self, premise: str, hypothesis: str) -> dict[str, float]: ...


class FakeNLI:
    """Deterministic NLI for tests. Looks up (premise, hypothesis), else returns `default`."""

    def __init__(self, mapping: Mapping[tuple[str, str], dict[str, float]] | None = None,
                 default: dict[str, float] | None = None) -> None:
        self._mapping = dict(mapping or {})
        self._default = default or {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self._mapping.get((premise, hypothesis), self._default)


class HFNLI:
    """Real adapter. `transformers` is imported lazily; runs locally, no API cost."""

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli") -> None:
        self._model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self._model_name,
                                  top_k=None)
        return self._pipe

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        raw = self._load()(f"{premise}</s></s>{hypothesis}")[0]
        scores = {r["label"].lower(): float(r["score"]) for r in raw}
        return {label: scores.get(label, 0.0) for label in LABELS}


def quote_is_grounded(conn: sqlite3.Connection, claim: Claim) -> bool:
    """Stage 1. True only when the quote appears verbatim in the unit or its paper's Layer 0."""
    if not claim.quote.strip():
        return False
    unit = get_unit(conn, claim.unit_id)
    if unit is None:
        return False
    if find_span(claim.quote, unit.verbatim_text) is not None:
        return True
    return find_span(claim.quote, get_raw_text(conn, unit.paper_id)) is not None


def verify_claim(conn: sqlite3.Connection, claim: Claim, nli: NLIModel,
                 threshold: float = 0.5) -> Verification:
    """Run both stages. Stage 2 is never reached when stage 1 fails."""
    if not quote_is_grounded(conn, claim):
        return Verification(claim_id=claim.claim_id, unit_id=claim.unit_id,
                            quote_found=False, verdict=Verdict.QUOTE_NOT_FOUND)

    scores = nli.predict(claim.quote, claim.text)
    entail = float(scores.get("entailment", 0.0))
    contra = float(scores.get("contradiction", 0.0))

    if contra >= threshold and contra > entail:
        verdict = Verdict.CONTRADICTED
    elif entail >= threshold:
        verdict = Verdict.SUPPORTED
    else:
        verdict = Verdict.NEUTRAL

    return Verification(claim_id=claim.claim_id, unit_id=claim.unit_id, quote_found=True,
                        verdict=verdict, entailment_score=entail,
                        contradiction_score=contra)


def find_contradictions(conn: sqlite3.Connection, claim: Claim, units: Sequence[Unit],
                        nli: NLIModel, threshold: float = 0.5) -> list[tuple[str, float]]:
    """Cross-corpus conflicts, free from the same NLI pass (spec §8).

    Returns (unit_id, contradiction_score) above threshold, most confident first. These are
    ranked candidates for human review, never assertions.
    """
    found: list[tuple[str, float]] = []
    for unit in units:
        if unit.unit_id == claim.unit_id:
            continue
        score = float(nli.predict(unit.verbatim_text, claim.text).get("contradiction", 0.0))
        if score >= threshold:
            found.append((unit.unit_id, score))
    return sorted(found, key=lambda pair: pair[1], reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_verify.py -v && ruff check jarvis/verify.py`
Expected: 15 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/verify.py tests/test_verify.py
git commit -m "feat: two-stage verification with quote matching and nli entailment"
```

---

### Task 14: Evaluation metrics

**Files:**
- Create: `jarvis/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `jarvis.models.{Verdict, Verification}`
- Produces: `quote_fidelity(verifications) -> float`; `statement_support(verifications) -> float`; `gate_recall(decisions, labels) -> float`; `coverage(cited_unit_ids, corpus_unit_ids) -> float`; `EvalReport` dataclass; `report(verifications, decisions=None, labels=None, cited=None, corpus=None) -> EvalReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import pytest

from jarvis.evaluate import (
    EvalReport, coverage, gate_recall, quote_fidelity, report, statement_support,
)
from jarvis.models import Verdict, Verification


def _v(claim_id, verdict, quote_found=True):
    return Verification(claim_id=claim_id, unit_id="u1", quote_found=quote_found,
                        verdict=verdict)


def test_quote_fidelity_is_one_when_every_quote_grounded():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.NEUTRAL)]
    assert quote_fidelity(vs) == 1.0


def test_quote_fidelity_drops_with_a_fabricated_quote():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.QUOTE_NOT_FOUND, quote_found=False)]
    assert quote_fidelity(vs) == 0.5


def test_quote_fidelity_of_nothing_is_one():
    assert quote_fidelity([]) == 1.0


def test_statement_support_counts_only_supported():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.NEUTRAL),
          _v("c", Verdict.CONTRADICTED), _v("d", Verdict.SUPPORTED)]
    assert statement_support(vs) == 0.5


def test_statement_support_of_nothing_is_zero():
    assert statement_support([]) == 0.0


def test_gate_recall_counts_relevant_papers_kept():
    decisions = {"p1": "read_deep", "p2": "unsure", "p3": "defer", "p4": "defer"}
    labels = {"p1": True, "p2": True, "p3": True, "p4": False}
    # 3 relevant; read_deep and unsure both count as kept -> 2/3
    assert gate_recall(decisions, labels) == pytest.approx(2 / 3)


def test_gate_recall_treats_unsure_as_kept():
    assert gate_recall({"p1": "unsure"}, {"p1": True}) == 1.0


def test_gate_recall_with_no_relevant_papers_is_one():
    assert gate_recall({"p1": "defer"}, {"p1": False}) == 1.0


def test_gate_recall_ignores_papers_without_labels():
    assert gate_recall({"p1": "defer", "p2": "read_deep"}, {"p2": True}) == 1.0


def test_coverage_is_fraction_of_corpus_cited():
    assert coverage({"u1", "u2"}, {"u1", "u2", "u3", "u4"}) == 0.5


def test_coverage_of_empty_corpus_is_zero():
    assert coverage(set(), set()) == 0.0


def test_coverage_ignores_citations_outside_the_corpus():
    assert coverage({"u1", "ghost"}, {"u1", "u2"}) == 0.5


def test_report_bundles_metrics_and_flags_targets():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.QUOTE_NOT_FOUND, quote_found=False)]
    r = report(vs, decisions={"p1": "read_deep"}, labels={"p1": True})
    assert isinstance(r, EvalReport)
    assert r.quote_fidelity == 0.5
    assert r.gate_recall == 1.0
    assert r.meets_quote_target is False   # target is 1.0
    assert r.meets_gate_target is True     # target is 0.95


def test_report_without_gate_data_leaves_gate_recall_none():
    r = report([_v("a", Verdict.SUPPORTED)])
    assert r.gate_recall is None
    assert r.meets_gate_target is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis/evaluate.py
"""Evaluation metrics (spec §10).

Built before capability expansion, deliberately: without these, "the corpus is good" is
unfalsifiable and every later build step rests on an unmeasured foundation.

Targets: quote fidelity 1.0 (any failure is fabrication), gate recall >= 0.95 (field
standard for screening tools), statement support >= 0.90.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from jarvis.models import Verdict, Verification

QUOTE_FIDELITY_TARGET = 1.0
GATE_RECALL_TARGET = 0.95
STATEMENT_SUPPORT_TARGET = 0.90

KEPT_DECISIONS = {"read_deep", "unsure"}


def quote_fidelity(verifications: Sequence[Verification]) -> float:
    """Fraction of claims whose quote was found verbatim in Layer 0. Target 1.0."""
    if not verifications:
        return 1.0
    return sum(1 for v in verifications if v.quote_found) / len(verifications)


def statement_support(verifications: Sequence[Verification]) -> float:
    """Fraction of claims entailed by their cited quote. Target >= 0.90."""
    if not verifications:
        return 0.0
    return sum(1 for v in verifications if v.verdict is Verdict.SUPPORTED) / len(verifications)


def gate_recall(decisions: Mapping[str, str], labels: Mapping[str, bool]) -> float:
    """Fraction of hand-labelled relevant papers the gate kept. Target >= 0.95.

    `unsure` counts as kept — spec §7B escalates it to deep read.
    """
    relevant = [pid for pid, is_relevant in labels.items() if is_relevant]
    if not relevant:
        return 1.0
    kept = sum(1 for pid in relevant if decisions.get(pid) in KEPT_DECISIONS)
    return kept / len(relevant)


def coverage(cited_unit_ids: Iterable[str], corpus_unit_ids: Iterable[str]) -> float:
    """Fraction of the deep-read corpus actually cited. Tracked, not targeted."""
    corpus = set(corpus_unit_ids)
    if not corpus:
        return 0.0
    return len(set(cited_unit_ids) & corpus) / len(corpus)


@dataclass(frozen=True)
class EvalReport:
    quote_fidelity: float
    statement_support: float
    gate_recall: float | None = None
    coverage: float | None = None

    @property
    def meets_quote_target(self) -> bool:
        return self.quote_fidelity >= QUOTE_FIDELITY_TARGET

    @property
    def meets_support_target(self) -> bool:
        return self.statement_support >= STATEMENT_SUPPORT_TARGET

    @property
    def meets_gate_target(self) -> bool | None:
        if self.gate_recall is None:
            return None
        return self.gate_recall >= GATE_RECALL_TARGET


def report(verifications: Sequence[Verification],
           decisions: Mapping[str, str] | None = None,
           labels: Mapping[str, bool] | None = None,
           cited: Iterable[str] | None = None,
           corpus: Iterable[str] | None = None) -> EvalReport:
    """Bundle the spec §10 metrics that the available data supports."""
    return EvalReport(
        quote_fidelity=quote_fidelity(verifications),
        statement_support=statement_support(verifications),
        gate_recall=(gate_recall(decisions, labels)
                     if decisions is not None and labels is not None else None),
        coverage=(coverage(cited, corpus)
                  if cited is not None and corpus is not None else None),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate.py -v && ruff check jarvis/evaluate.py`
Expected: 14 passed, ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation metrics for quote fidelity, support, gate recall, coverage"
```

---

### Task 15: End-to-end single-paper proof

**Files:**
- Create: `tests/test_end_to_end.py`
- Modify: `jarvis/__init__.py` (extend `__all__` with the new public surface)

**This is the deliverable of the whole plan:** one test that takes a paper from blocks to a verified claim, and proves a fabricated claim is caught.

**Interfaces:**
- Consumes: everything built in Tasks 1–14
- Produces: no new API; `jarvis/__init__.py` re-exports the modules built here

- [ ] **Step 1: Write the failing test**

```python
# tests/test_end_to_end.py
"""The proof the plan exists to produce: one paper, ingested, retrieved, verified."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.evaluate import report
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.parse import FakeParser
from jarvis.retrieve import search
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI, verify_claim

BLOCKS = [
    Block(kind="heading", text="Results", page=3, section_path=("Results",)),
    Block(kind="paragraph", text="As shown in Table 3, our controller reaches 94.2% "
                                 "tracking accuracy under gust distur-\nbance.",
          page=3, section_path=("Results",)),
    Block(kind="table", text="| method | accuracy |\n|---|---|\n| ours | 94.2 |",
          page=3, section_path=("Results",), label="Table 3"),
    Block(kind="caption", text="Table 3: Tracking accuracy under wind.", page=3,
          section_path=("Results",), label="Table 3"),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Quadrotor Control", year=2025)


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    parsed = FakeParser(BLOCKS).parse("paper.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")

    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    yield conn
    close_store(conn)


def test_ingest_produces_prose_and_table_units(corpus):
    kinds = {u.type.value for u in get_units(corpus, "p1")}
    assert "prose" in kinds
    assert "table" in kinds


def test_the_table_unit_carries_its_caption_and_referring_text(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    assert "94.2" in table.verbatim_text
    assert "Tracking accuracy under wind" in table.verbatim_text
    assert "As shown in Table 3" in table.verbatim_text


def test_retrieval_finds_the_evidence(corpus):
    hits = search(corpus, "tracking accuracy under wind", FakeEmbedder(), limit=3)
    assert any("94.2" in u.verbatim_text for u in hits)


def test_a_grounded_claim_verifies_as_supported(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    claim = Claim(claim_id="c1",
                  text="The controller reaches 94.2% tracking accuracy.",
                  unit_id=table.unit_id, quote="| ours | 94.2 |")
    nli = FakeNLI(default={"entailment": 0.93, "neutral": 0.05, "contradiction": 0.02})
    assert verify_claim(corpus, claim, nli).verdict == Verdict.SUPPORTED


def test_a_fabricated_number_is_caught_without_consulting_the_model(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    claim = Claim(claim_id="c2", text="The controller reaches 99.9% accuracy.",
                  unit_id=table.unit_id, quote="| ours | 99.9 |")
    nli = FakeNLI(default={"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0})
    result = verify_claim(corpus, claim, nli)
    assert result.verdict == Verdict.QUOTE_NOT_FOUND
    assert result.quote_found is False


def test_quote_survives_hyphenation_across_a_line_break(corpus):
    prose = next(u for u in get_units(corpus, "p1")
                 if u.type.value == "prose" and "gust" in u.verbatim_text)
    claim = Claim(claim_id="c3", text="Gusts disturb the controller.",
                  unit_id=prose.unit_id, quote="under gust disturbance")
    nli = FakeNLI(default={"entailment": 0.91, "neutral": 0.06, "contradiction": 0.03})
    assert verify_claim(corpus, claim, nli).quote_found is True


def test_eval_report_flags_the_fabricated_claim(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    nli = FakeNLI(default={"entailment": 0.93, "neutral": 0.05, "contradiction": 0.02})
    good = verify_claim(corpus, Claim("c1", "t", table.unit_id, "| ours | 94.2 |"), nli)
    bad = verify_claim(corpus, Claim("c2", "t", table.unit_id, "| ours | 99.9 |"), nli)

    r = report([good, bad])
    assert r.quote_fidelity == 0.5
    assert r.meets_quote_target is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_end_to_end.py -v`
Expected: FAIL — the fixture cannot build until every prior task is complete. If prior tasks are done, this should surface any real integration gap.

- [ ] **Step 3: Extend the package exports**

Replace the import and `__all__` blocks in `jarvis/__init__.py` with:

```python
from jarvis.citation_graph import CitationWalker, make_s2_neighbors, paper_id
from jarvis.config import Config
from jarvis.context import TemplatePrefix, apply_prefixes, embedding_text
from jarvis.embed import BGEEmbedder, FakeEmbedder, index_units, vector_search
from jarvis.evaluate import EvalReport, report
from jarvis.index import index_units_fts, keyword_search
from jarvis.models import (
    Block, Card, CardField, Claim, Paper, ParsedPaper, Unit, UnitType, Verdict, Verification,
)
from jarvis.parse import DoclingParser, FakeParser
from jarvis.retrieve import CrossEncoderReranker, rrf, search
from jarvis.router import CostTracker, ModelRouter
from jarvis.scoring import citation_weight, cosine, make_cosine_scorer, paper_text, recency
from jarvis.sources import (
    combine_sources, dedup_papers, make_core_search, make_crossref_search,
    make_unpaywall_pdf, normalize_crossref,
)
from jarvis.store import close_store, get_paper, get_units, open_store, save_paper, save_units
from jarvis.text import approx_tokens, find_span, normalize
from jarvis.units import build_units
from jarvis.verify import FakeNLI, HFNLI, verify_claim

__all__ = [
    "BGEEmbedder", "Block", "Card", "CardField", "CitationWalker", "Claim", "Config",
    "CostTracker", "CrossEncoderReranker", "DoclingParser", "EvalReport", "FakeEmbedder",
    "FakeNLI", "FakeParser", "HFNLI", "ModelRouter", "Paper", "ParsedPaper",
    "TemplatePrefix", "Unit", "UnitType", "Verdict", "Verification", "apply_prefixes",
    "approx_tokens", "build_units", "citation_weight", "close_store", "combine_sources",
    "cosine", "dedup_papers", "embedding_text", "find_span", "get_paper", "get_units",
    "index_units", "index_units_fts", "keyword_search", "make_core_search",
    "make_cosine_scorer", "make_crossref_search", "make_s2_neighbors", "make_unpaywall_pdf",
    "normalize", "normalize_crossref", "open_store", "paper_id", "paper_text", "recency",
    "report", "rrf", "save_paper", "save_units", "search", "vector_search", "verify_claim",
]
```

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass (23 ported + ~120 new), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/__init__.py tests/test_end_to_end.py
git commit -m "test: end-to-end single-paper ingest, retrieve, and verify"
```

---

## Definition of done

- `python -m pytest` passes with zero network access, no API keys, no model downloads.
- `test_a_fabricated_number_is_caught_without_consulting_the_model` passes — the system catches fabrication mechanically, not by asking a model to be honest.
- `jarvis.evaluate.report` returns real numbers for quote fidelity and statement support.
- `ruff check .` is clean.

## What this plan deliberately does not build

Spec build steps 6–10, each of which gets its own plan once this core measures correctly:

| Step | Why it waits |
|---|---|
| 6. Gather + gate | The gate is the highest-risk component (spec §7B). It needs the eval harness from Task 14 to calibrate against, which does not exist until this plan is done. |
| 7. Compile — Q&A | Needs retrieval and verification proven first. |
| 8. MCP server | Pure surface over a core that must work first. |
| 9. Contradiction detection | `find_contradictions` exists (Task 13) but is unused until there is a multi-paper corpus to run it over. |
| 10. Long-form reports | Furthest from the verifiable core. |

Also unbuilt here, and tracked: the Layer 2 **card extractor**. `Card`/`CardField` types exist (Task 1) and `binding_verified` is wired, but LLM card extraction belongs with the gather plan, because a card over a single hand-picked paper proves nothing about relational-binding accuracy at corpus scale.
