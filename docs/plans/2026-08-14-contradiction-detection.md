# Cross-Paper Contradiction Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface places where one paper in the corpus contradicts another, as ranked candidates for a human to review — never as assertions — reusing the NLI pass verification already runs.

**Architecture:** Spec §8, "Contradiction detection — free from the same pass." NLI emits three labels and verification already computes all of them; the third one has simply been unused. For each claim, retrieve evidence from *other papers* that is topically close to it, run the same NLI model over (their evidence → this claim), and keep the ones scoring high on `contradiction`. Retrieval-first, not corpus-wide: a 500-paper corpus is ~100k units, and running NLI over every (claim, unit) pair is quadratic and unaffordable. Candidates are persisted with their scores, rendered as a review sheet, and scored for precision against human judgment.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`/`json`, `pytest`. The NLI model is the existing `NLIModel` protocol with `FakeNLI`; retrieval is the existing hybrid stack.

**Prerequisite:** the verifiable single-paper core (`docs/plans/2026-08-11-verifiable-single-paper-core.md`, merged at `d7f8672`), which already contains `jarvis.verify.find_contradictions` — written during that plan and unused ever since, because there was no multi-paper corpus to run it over. This plan is buildable and testable against a hand-built two-paper corpus, but it is only *useful* once `docs/plans/2026-08-14-gather-and-gate.md` has landed and there is a real corpus.

## Global Constraints

- Python **>= 3.10**. Use `X | None`, not `Optional[X]`.
- **Never read `.env`.** Configuration is environment variables or `$JARVIS_CONFIG` JSON only.
- **Every test is offline.** No network, no API keys, no model downloads.
- All external models are consumed through a `typing.Protocol` with a deterministic fake used in tests.
- Line length **100**. Target `py310`. Run `ruff check .` against **both** the module and its test file before every commit.
- **`jarvis/store.py` is the only module that writes SQL.** The new `contradictions` table and every query against it go there.
- **Output is candidates, never assertions.** No function in this plan may return, render, or persist a contradiction as an established fact. Spec §8 is explicit, and arXiv 2504.00180 found multi-document contradiction detection is hard for LLMs *and* for humans.
- **Cross-paper only.** A claim is never checked against evidence from its own paper — that is the paper disagreeing with itself, which is a parsing or claim-extraction bug, not a finding.
- **No LLM may judge contradiction.** This is the same NLI pass verification uses. The `contradiction_review` route in `jarvis/router.py` exists for *summarizing candidates for a human*, never for deciding whether one exists.
- Frozen dataclasses for all new types; tuples not lists in frozen types.
- Commit after every task with a `feat:`/`test:`/`fix:` prefix.
- Repo-wide `ruff check .` baseline is **11 pre-existing violations** in `citation_graph.py` (2), `config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1). Do not fix them; do not add to them.

## Why this is the feature no single-shot system can ship

Spec §8 makes the argument in one line: this is *"structurally impossible for any system that does not retain a corpus."* A system that re-researches from zero per query has nothing to compare against. One that keeps 300 papers has 300 papers' worth of claims to cross-check, and the check costs nothing extra because the NLI model is already loaded and already emitting the label.

The calibration target comes from the only published measurement of it: PaperQA2's ContraCrow found **2.34 ± 1.99 contradictions per paper** across 93 biology papers, with **70% validated by human experts**. Spec §10 sets contradiction precision at **≥70%** — ContraCrow parity — and Task 4 is how that number gets measured rather than assumed.

The caveat is equally load-bearing. Multi-document contradiction detection is hard for LLMs (arXiv 2504.00180) and for humans. A 70%-precision detector presented as an oracle is a machine for generating confident nonsense about the literature. Presented as a ranked queue for a researcher to skim, it is a genuinely superhuman-shaped tool: *one versus many* per claim, *many versus many* across a corpus.

## File Structure

| File | Responsibility |
|---|---|
| `jarvis/store.py` | **Modify.** Add the `contradictions` table and its CRUD. |
| `jarvis/contradict.py` | Create. Cross-paper candidate retrieval, the corpus scan, ranking, review round-trip. |
| `jarvis/evaluate.py` | **Modify.** Add `contradiction_precision` against human review. |
| `jarvis/__init__.py` | **Modify.** Export the new surface. |

Tests: `tests/test_store_contradictions.py`, `tests/test_contradict.py`, `tests/test_contradict_end_to_end.py`.

---

### Task 1: Storage for contradiction candidates

**Files:**
- Modify: `jarvis/store.py` (add to `_SCHEMA`, then append functions)
- Test: `tests/test_store_contradictions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: the `contradictions` table; `save_contradictions(conn, rows, run_id="")`, `get_contradictions(conn, run_id="") -> list[dict]`, `set_contradiction_review(conn, claim_id, unit_id, verdict, run_id="")`, `get_contradiction_reviews(conn, run_id="") -> dict[tuple[str, str], bool]`.

The `reviewed` column holds `''` (unreviewed), `'valid'`, or `'invalid'` — a three-state field, because "not yet looked at" and "looked at and rejected" are different facts and collapsing them would make the precision metric silently wrong.

Adding to `_SCHEMA` is safe for existing databases: `open_store` runs `executescript` on every open and every statement is `CREATE TABLE IF NOT EXISTS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_contradictions.py
"""Persistence for contradiction candidates and their human review."""
import pytest

from jarvis.models import Paper
from jarvis.store import (
    close_store,
    get_contradiction_reviews,
    get_contradictions,
    open_store,
    save_contradictions,
    save_paper,
    set_contradiction_review,
)

ROWS = [
    {"claim_id": "c1", "unit_id": "p2:prose:1:0", "score": 0.91},
    {"claim_id": "c1", "unit_id": "p3:prose:2:1", "score": 0.72},
]


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / "c.db")
    yield c
    close_store(c)


def test_candidates_round_trip_with_their_scores(conn):
    assert save_contradictions(conn, ROWS, run_id="r1") == 2
    rows = get_contradictions(conn, "r1")
    assert {r["unit_id"] for r in rows} == {"p2:prose:1:0", "p3:prose:2:1"}
    assert rows[0]["score"] == pytest.approx(0.91)


def test_candidates_come_back_most_confident_first(conn):
    save_contradictions(conn, list(reversed(ROWS)), run_id="r1")
    scores = [r["score"] for r in get_contradictions(conn, "r1")]
    assert scores == sorted(scores, reverse=True)


def test_candidates_start_unreviewed(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    assert all(r["reviewed"] == "" for r in get_contradictions(conn, "r1"))
    assert get_contradiction_reviews(conn, "r1") == {}


def test_a_review_is_recorded_and_readable(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    set_contradiction_review(conn, "c1", "p2:prose:1:0", "valid", run_id="r1")
    set_contradiction_review(conn, "c1", "p3:prose:2:1", "invalid", run_id="r1")

    reviews = get_contradiction_reviews(conn, "r1")
    assert reviews == {("c1", "p2:prose:1:0"): True, ("c1", "p3:prose:2:1"): False}


def test_unreviewed_candidates_are_absent_from_reviews_not_false(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    set_contradiction_review(conn, "c1", "p2:prose:1:0", "valid", run_id="r1")
    reviews = get_contradiction_reviews(conn, "r1")
    assert len(reviews) == 1, "not-yet-looked-at is not the same fact as rejected"


def test_rescanning_the_same_pair_updates_its_score_and_keeps_the_review(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    set_contradiction_review(conn, "c1", "p2:prose:1:0", "valid", run_id="r1")
    save_contradictions(conn, [{"claim_id": "c1", "unit_id": "p2:prose:1:0", "score": 0.55}],
                        run_id="r1")

    row = next(r for r in get_contradictions(conn, "r1") if r["unit_id"] == "p2:prose:1:0")
    assert row["score"] == pytest.approx(0.55)
    assert row["reviewed"] == "valid", "human judgment survives a rescan"


def test_two_runs_are_independent(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    save_contradictions(conn, ROWS[:1], run_id="r2")
    assert len(get_contradictions(conn, "r1")) == 2
    assert len(get_contradictions(conn, "r2")) == 1


def test_an_invalid_review_verdict_is_rejected(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    with pytest.raises(ValueError):
        set_contradiction_review(conn, "c1", "p2:prose:1:0", "maybe", run_id="r1")


def test_saving_nothing_is_zero_not_an_error(conn):
    assert save_contradictions(conn, [], run_id="r1") == 0
    assert get_contradictions(conn, "r1") == []


def test_candidates_do_not_require_a_papers_row(conn):
    """A candidate references units, not papers — no foreign key should block a scan."""
    save_paper(conn, Paper(paper_id="p1", title="T"))
    assert save_contradictions(conn, ROWS, run_id="r1") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store_contradictions.py -v`
Expected: FAIL with `ImportError: cannot import name 'save_contradictions' from 'jarvis.store'`

- [ ] **Step 3: Write the implementation**

Add to the end of the `_SCHEMA` string in `jarvis/store.py`, before the closing `"""`:

```sql
CREATE TABLE IF NOT EXISTS contradictions (
    claim_id  TEXT NOT NULL,
    unit_id   TEXT NOT NULL,
    run_id    TEXT NOT NULL DEFAULT '',
    score     REAL NOT NULL DEFAULT 0.0,
    reviewed  TEXT NOT NULL DEFAULT '',     -- '' | valid | invalid
    PRIMARY KEY (claim_id, unit_id, run_id)
);
CREATE INDEX IF NOT EXISTS idx_contradictions_run ON contradictions(run_id, score DESC);
```

Then append these functions:

```python
# --- contradiction candidates (spec §8) ---------------------------------------------

REVIEW_VERDICTS = ("valid", "invalid")


def save_contradictions(conn: sqlite3.Connection, rows, run_id: str = "") -> int:
    """Persist candidates. A rescan updates scores but never clears a human review."""
    rows = list(rows)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO contradictions (claim_id, unit_id, run_id, score) VALUES (?,?,?,?)
        ON CONFLICT(claim_id, unit_id, run_id) DO UPDATE SET score=excluded.score
        """,
        [(r["claim_id"], r["unit_id"], run_id, float(r.get("score", 0.0))) for r in rows],
    )
    conn.commit()
    return len(rows)


def get_contradictions(conn: sqlite3.Connection, run_id: str = "") -> list[dict]:
    """Candidates for one run, most confident first."""
    rows = conn.execute(
        "SELECT * FROM contradictions WHERE run_id = ? ORDER BY score DESC, claim_id, unit_id",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def set_contradiction_review(conn: sqlite3.Connection, claim_id: str, unit_id: str,
                             verdict: str, run_id: str = "") -> None:
    """Record a human's judgment on one candidate."""
    if verdict not in REVIEW_VERDICTS:
        raise ValueError(f"verdict must be one of {REVIEW_VERDICTS}, got {verdict!r}")
    conn.execute(
        "UPDATE contradictions SET reviewed = ? WHERE claim_id = ? AND unit_id = ? "
        "AND run_id = ?",
        (verdict, claim_id, unit_id, run_id),
    )
    conn.commit()


def get_contradiction_reviews(conn: sqlite3.Connection,
                              run_id: str = "") -> dict[tuple[str, str], bool]:
    """Reviewed candidates only. Unreviewed ones are absent, never False."""
    rows = conn.execute(
        "SELECT claim_id, unit_id, reviewed FROM contradictions "
        "WHERE run_id = ? AND reviewed != ''",
        (run_id,),
    ).fetchall()
    return {(r["claim_id"], r["unit_id"]): r["reviewed"] == "valid" for r in rows}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_contradictions.py tests/test_store_schema.py tests/test_store_crud.py -v && ruff check jarvis/store.py tests/test_store_contradictions.py`
Expected: PASS (10 new + all existing store tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/store.py tests/test_store_contradictions.py
git commit -m "feat: contradiction candidate storage with human review state"
```

---

### Task 2: Cross-paper candidate retrieval

**Files:**
- Create: `jarvis/contradict.py`
- Test: `tests/test_contradict.py`

**Interfaces:**
- Consumes: `search` and `Reranker` from `jarvis.retrieve`; `Embedder` from `jarvis.embed`; `Claim`, `Unit` from `jarvis.models`; `get_unit` from `jarvis.store`.
- Produces: `opposing_units(conn, claim, embedder, *, limit=20, reranker=None) -> list[Unit]`.

Two exclusions, and both matter:

1. **The claim's own unit.** `jarvis.verify.find_contradictions` already handles this.
2. **Every unit from the claim's own paper.** `find_contradictions` does **not** handle this — it only compares `unit_id`. A paper's Results section "contradicting" its own Limitations section is a claim-extraction artifact, not a finding about the literature, and a scan that reports it will bury the real signal in self-conflict noise.

Why retrieval-first rather than scanning everything: a 500-paper corpus is roughly 100k units. Cross-checking 20 claims per paper against all of them is ~200M NLI calls. Retrieval cuts each claim's comparison set to a couple of dozen topically-close units, which is where a genuine contradiction would live anyway — two papers that never discuss the same thing cannot disagree.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contradict.py
"""Cross-paper contradiction candidates (spec §8)."""
import pytest

from jarvis.contradict import opposing_units
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units

AGREES = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
          page=1, section_path=("Results",)),
    Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
    Block(kind="paragraph", text="Tracking accuracy degrades above 12 m/s gusts.",
          page=2, section_path=("Limitations",)),
]
DISAGREES = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Reproducing this controller, tracking accuracy never exceeded 61% under "
               "gust disturbance.", page=1, section_path=("Results",)),
]


def _ingest(conn, paper_id, title, blocks):
    paper = Paper(paper_id=paper_id, title=title, year=2025)
    parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    return units


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    _ingest(conn, "p1", "Gust-Robust Control", AGREES)
    _ingest(conn, "p2", "A Reproduction Study", DISAGREES)
    yield conn
    close_store(conn)


def _claim(conn):
    unit = next(u for u in get_units(conn, "p1") if "94.2" in u.verbatim_text)
    return Claim(claim_id="c1", text="The controller reaches 94.2% tracking accuracy.",
                 unit_id=unit.unit_id, quote="94.2% tracking accuracy")


def test_opposing_units_finds_the_other_papers_evidence(corpus):
    units = opposing_units(corpus, _claim(corpus), FakeEmbedder())
    assert any("61%" in u.verbatim_text for u in units)


def test_no_unit_from_the_claims_own_paper_is_returned(corpus):
    units = opposing_units(corpus, _claim(corpus), FakeEmbedder())
    assert all(u.paper_id != "p1" for u in units), \
        "a paper disagreeing with itself is a claim-extraction bug, not a finding"


def test_the_claims_own_unit_is_never_returned(corpus):
    claim = _claim(corpus)
    assert claim.unit_id not in {u.unit_id for u in opposing_units(corpus, claim,
                                                                  FakeEmbedder())}


def test_a_claim_whose_unit_is_unknown_yields_nothing(corpus):
    claim = Claim(claim_id="c9", text="anything", unit_id="ghost", quote="q")
    assert opposing_units(corpus, claim, FakeEmbedder()) == []


def test_a_single_paper_corpus_yields_no_candidates(tmp_path):
    conn = open_store(tmp_path / "solo.db")
    try:
        _ingest(conn, "p1", "Alone", AGREES)
        assert opposing_units(conn, _claim(conn), FakeEmbedder()) == []
    finally:
        close_store(conn)


def test_the_limit_is_respected(corpus):
    assert len(opposing_units(corpus, _claim(corpus), FakeEmbedder(), limit=1)) <= 1


def test_results_are_deduplicated(corpus):
    units = opposing_units(corpus, _claim(corpus), FakeEmbedder(), limit=20)
    assert len({u.unit_id for u in units}) == len(units)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contradict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.contradict'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/contradict.py
"""Cross-paper contradiction detection (spec §8).

Free from the verification pass: NLI emits entailment, neutral, and contradiction, and the
verifier has been computing all three and reading two. Running claims from one paper
against evidence from others surfaces cross-corpus conflicts at no additional model cost.

This is structurally impossible for any system that does not retain a corpus, and the
shape genuinely favours a machine: one-versus-many per claim, many-versus-many across a
corpus. But it is hard for LLMs and for humans alike (arXiv 2504.00180), and ContraCrow's
own precision against expert review was 70%. Output is therefore RANKED CANDIDATES FOR
HUMAN REVIEW, never assertions. Nothing in this module returns a contradiction as a fact.

Retrieval-first, not corpus-wide: 500 papers is ~100k units, and cross-checking every
claim against all of them is quadratic and unaffordable. Two papers that never discuss the
same thing cannot disagree, so retrieval loses nothing real.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from jarvis.embed import Embedder
from jarvis.models import Claim, Unit
from jarvis.retrieve import Reranker, search
from jarvis.store import get_unit


def opposing_units(conn: sqlite3.Connection, claim: Claim, embedder: Embedder, *,
                   limit: int = 20, reranker: Reranker | None = None) -> list[Unit]:
    """Evidence from OTHER papers that is topically close to this claim.

    Excludes the claim's own paper entirely, not merely its own unit: a paper's Results
    section "contradicting" its own Limitations is a claim-extraction artifact, and a scan
    that reports those will bury the real signal in self-conflict noise.
    """
    own = get_unit(conn, claim.unit_id)
    if own is None:
        return []

    # Over-fetch: the claim's own paper is usually the best match for its own claim text,
    # so a tight limit would return nothing but self-hits before filtering.
    hits = search(conn, claim.text, embedder, limit=max(limit * 3, limit),
                  reranker=reranker, expand_parents=False)

    out: list[Unit] = []
    seen: set[str] = set()
    for unit in hits:
        if unit.paper_id == own.paper_id or unit.unit_id in seen:
            continue
        seen.add(unit.unit_id)
        out.append(unit)
        if len(out) >= limit:
            break
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contradict.py -v && ruff check jarvis/contradict.py tests/test_contradict.py`
Expected: PASS (7 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/contradict.py tests/test_contradict.py
git commit -m "feat: cross-paper opposing-evidence retrieval"
```

---

### Task 3: The corpus scan

**Files:**
- Modify: `jarvis/contradict.py` (append)
- Test: `tests/test_contradict.py` (append)

**Interfaces:**
- Consumes: `find_contradictions` and `NLIModel` from `jarvis.verify`; `get_unit`, `get_paper`, `save_contradictions` from `jarvis.store`; `opposing_units` (Task 2).
- Produces: `Conflict` (frozen: `claim_id`, `claim_text`, `claim_paper_id`, `unit_id`, `paper_id`, `score`, `evidence`), `scan_claim(conn, claim, nli, embedder, *, limit=20, threshold=0.5, reranker=None) -> list[Conflict]`, `scan_corpus(conn, claims, nli, embedder, *, limit=20, threshold=0.5, budget=500, run_id="", reranker=None) -> list[Conflict]`, `rank(conflicts) -> list[Conflict]`.

`scan_claim` reuses `jarvis.verify.find_contradictions` verbatim rather than reimplementing the NLI loop — it is already written, already tested, and already returns `(unit_id, score)` sorted most-confident-first. This module's job is choosing *what* to compare, not *how*.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contradict.py`:

```python
from jarvis.contradict import Conflict, rank, scan_claim, scan_corpus
from jarvis.store import get_contradictions
from jarvis.verify import FakeNLI

CONTRADICTS = FakeNLI(default={"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90})
AGREES_NLI = FakeNLI(default={"entailment": 0.90, "neutral": 0.08, "contradiction": 0.02})


def test_a_scan_finds_the_disagreeing_paper(corpus):
    conflicts = scan_claim(corpus, _claim(corpus), CONTRADICTS, FakeEmbedder())
    assert conflicts
    assert all(c.paper_id == "p2" for c in conflicts)
    assert conflicts[0].score == pytest.approx(0.90)


def test_a_conflict_carries_enough_context_to_review_it(corpus):
    conflict = scan_claim(corpus, _claim(corpus), CONTRADICTS, FakeEmbedder())[0]
    assert conflict.claim_text.startswith("The controller reaches")
    assert conflict.claim_paper_id == "p1"
    assert conflict.paper_id == "p2"
    assert conflict.evidence


def test_agreement_produces_no_candidates(corpus):
    assert scan_claim(corpus, _claim(corpus), AGREES_NLI, FakeEmbedder()) == []


def test_the_threshold_gates_what_is_reported(corpus):
    weak = FakeNLI(default={"entailment": 0.1, "neutral": 0.5, "contradiction": 0.40})
    assert scan_claim(corpus, _claim(corpus), weak, FakeEmbedder(), threshold=0.5) == []
    assert scan_claim(corpus, _claim(corpus), weak, FakeEmbedder(), threshold=0.3) != []


def test_conflicts_come_back_most_confident_first():
    conflicts = [
        Conflict("c1", "t", "p1", "u1", "p2", 0.6, "e"),
        Conflict("c1", "t", "p1", "u2", "p3", 0.9, "e"),
        Conflict("c1", "t", "p1", "u3", "p4", 0.7, "e"),
    ]
    assert [c.score for c in rank(conflicts)] == [0.9, 0.7, 0.6]


def test_ranking_deduplicates_a_repeated_pair_keeping_the_higher_score():
    conflicts = [Conflict("c1", "t", "p1", "u1", "p2", 0.6, "e"),
                 Conflict("c1", "t", "p1", "u1", "p2", 0.9, "e")]
    ranked = rank(conflicts)
    assert len(ranked) == 1
    assert ranked[0].score == pytest.approx(0.9)


def test_a_corpus_scan_covers_every_claim(corpus):
    unit1 = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    unit2 = next(u for u in get_units(corpus, "p1") if "12 m/s" in u.verbatim_text)
    claims = [
        Claim("c1", "It reaches 94.2% accuracy.", unit1.unit_id, "94.2% tracking accuracy"),
        Claim("c2", "It degrades above 12 m/s.", unit2.unit_id, "above 12 m/s"),
    ]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder())
    assert {c.claim_id for c in conflicts} == {"c1", "c2"}


def test_a_corpus_scan_persists_its_candidates(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder(), run_id="scan1")

    stored = get_contradictions(corpus, "scan1")
    assert stored
    assert stored[0]["claim_id"] == "c1"
    assert stored[0]["reviewed"] == ""


def test_a_scan_without_a_run_id_does_not_persist(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder())
    assert conflicts
    assert get_contradictions(corpus, "") == []


def test_the_budget_caps_the_scan(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim(f"c{i}", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")
              for i in range(20)]
    assert len(scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder(), budget=3)) <= 3


def test_one_bad_claim_does_not_abort_the_scan(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("bad", "x", "ghost-unit", "q"),
              Claim("good", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder())
    assert {c.claim_id for c in conflicts} == {"good"}


def test_conflict_is_frozen():
    with pytest.raises(Exception):
        Conflict("c1", "t", "p1", "u1", "p2", 0.6, "e").score = 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contradict.py -v`
Expected: FAIL with `ImportError: cannot import name 'Conflict' from 'jarvis.contradict'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/contradict.py`, adding `from dataclasses import dataclass`, `from jarvis.store import save_contradictions` and `from jarvis.verify import NLIModel, find_contradictions` to the imports:

```python
EVIDENCE_PREVIEW = 500


@dataclass(frozen=True)
class Conflict:
    """One candidate disagreement. A prompt for a human to look, never a finding."""
    claim_id: str
    claim_text: str
    claim_paper_id: str
    unit_id: str
    paper_id: str
    score: float
    evidence: str = ""


def scan_claim(conn: sqlite3.Connection, claim: Claim, nli: NLIModel, embedder: Embedder, *,
               limit: int = 20, threshold: float = 0.5,
               reranker: Reranker | None = None) -> list[Conflict]:
    """Candidates for one claim. Reuses `verify.find_contradictions` for the NLI pass."""
    own = get_unit(conn, claim.unit_id)
    if own is None:
        return []

    units = opposing_units(conn, claim, embedder, limit=limit, reranker=reranker)
    if not units:
        return []

    by_id = {u.unit_id: u for u in units}
    out: list[Conflict] = []
    for unit_id, score in find_contradictions(conn, claim, units, nli, threshold=threshold):
        unit = by_id.get(unit_id)
        if unit is None:
            continue
        out.append(Conflict(
            claim_id=claim.claim_id, claim_text=claim.text, claim_paper_id=own.paper_id,
            unit_id=unit.unit_id, paper_id=unit.paper_id, score=score,
            evidence=unit.verbatim_text[:EVIDENCE_PREVIEW],
        ))
    return out


def rank(conflicts: Sequence[Conflict]) -> list[Conflict]:
    """Most confident first, one row per (claim, unit) pair."""
    best: dict[tuple[str, str], Conflict] = {}
    for conflict in conflicts:
        key = (conflict.claim_id, conflict.unit_id)
        if key not in best or conflict.score > best[key].score:
            best[key] = conflict
    return sorted(best.values(), key=lambda c: (-c.score, c.claim_id, c.unit_id))


def scan_corpus(conn: sqlite3.Connection, claims: Sequence[Claim], nli: NLIModel,
                embedder: Embedder, *, limit: int = 20, threshold: float = 0.5,
                budget: int = 500, run_id: str = "",
                reranker: Reranker | None = None) -> list[Conflict]:
    """Scan every claim against the rest of the corpus.

    One unscannable claim never aborts the run — a scan over 300 papers that dies on claim
    40 is worth less than one that reports 299 papers' worth of candidates.

    Persists only when `run_id` is given, so an exploratory scan costs nothing permanent.
    """
    found: list[Conflict] = []
    for claim in claims:
        if len(found) >= budget:
            break
        try:
            found += scan_claim(conn, claim, nli, embedder, limit=limit,
                                threshold=threshold, reranker=reranker)
        except Exception:  # noqa: BLE001 - one claim's failure is not the scan's failure
            continue

    ranked = rank(found)[:budget]
    if run_id:
        save_contradictions(
            conn,
            [{"claim_id": c.claim_id, "unit_id": c.unit_id, "score": c.score}
             for c in ranked],
            run_id=run_id,
        )
    return ranked
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contradict.py -v && ruff check jarvis/contradict.py tests/test_contradict.py`
Expected: PASS (19 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/contradict.py tests/test_contradict.py
git commit -m "feat: corpus-wide contradiction scan with ranking and budget"
```

---

### Task 4: Human review and contradiction precision

**Files:**
- Modify: `jarvis/contradict.py` (append), `jarvis/evaluate.py`
- Test: `tests/test_contradict.py` (append)

**Interfaces:**
- Consumes: `Conflict` (Task 3); `get_contradictions`, `set_contradiction_review`, `get_contradiction_reviews` from `jarvis.store`.
- Produces: `write_review_sheet(path, conflicts) -> int`, `read_reviews(path) -> dict[tuple[str, str], bool]`, `apply_reviews(conn, reviews, run_id="") -> int`, `render_conflicts(conflicts, top_n=20) -> str`; in `jarvis/evaluate.py`: `CONTRADICTION_PRECISION_TARGET = 0.70`, `contradiction_precision(reviews) -> float`, and `EvalReport.contradiction_precision` / `meets_contradiction_target`.

The review sheet is the same JSONL shape as the gate's label sheet — one candidate per line, `verdict: null` until a human edits it — for the same reasons: diffable, resumable, editable anywhere, no UI.

Spec §10 sets contradiction precision at **≥70%**, ContraCrow parity. It is measured only over *reviewed* candidates; unreviewed ones are excluded, not counted as failures.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_contradict.py`:

```python
import json

from jarvis.contradict import (
    apply_reviews,
    read_reviews,
    render_conflicts,
    write_review_sheet,
)
from jarvis.evaluate import CONTRADICTION_PRECISION_TARGET, contradiction_precision, report
from jarvis.store import get_contradiction_reviews

CONFLICTS = [
    Conflict("c1", "It reaches 94.2%.", "p1", "u9", "p2", 0.91, "never exceeded 61%"),
    Conflict("c1", "It reaches 94.2%.", "p1", "u8", "p3", 0.72, "we measured 90%"),
]


def test_the_review_sheet_is_one_candidate_per_line_awaiting_a_verdict(tmp_path):
    path = tmp_path / "review.jsonl"
    assert write_review_sheet(path, CONFLICTS) == 2

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["claim_id"] == "c1"
    assert rows[0]["unit_id"] == "u9"
    assert rows[0]["verdict"] is None
    assert "61%" in rows[0]["evidence"]
    assert rows[0]["claim_text"].startswith("It reaches")


def test_a_fresh_sheet_has_no_verdicts(tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_sheet(path, CONFLICTS)
    assert read_reviews(path) == {}


def test_verdicts_round_trip(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text('{"claim_id": "c1", "unit_id": "u9", "verdict": "valid"}\n'
                    '{"claim_id": "c1", "unit_id": "u8", "verdict": "invalid"}\n'
                    '{"claim_id": "c1", "unit_id": "u7", "verdict": null}\n',
                    encoding="utf-8")
    assert read_reviews(path) == {("c1", "u9"): True, ("c1", "u8"): False}


def test_common_hand_typed_verdicts_are_accepted(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text('{"claim_id": "c1", "unit_id": "a", "verdict": "yes"}\n'
                    '{"claim_id": "c1", "unit_id": "b", "verdict": "no"}\n'
                    '{"claim_id": "c1", "unit_id": "c", "verdict": true}\n',
                    encoding="utf-8")
    assert read_reviews(path) == {("c1", "a"): True, ("c1", "b"): False, ("c1", "c"): True}


def test_a_malformed_review_line_is_skipped(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text('not json\n{"claim_id": "c1", "unit_id": "a", "verdict": "valid"}\n'
                    '{"verdict": "valid"}\n', encoding="utf-8")
    assert read_reviews(path) == {("c1", "a"): True}


def test_reviews_can_be_applied_back_into_the_store(corpus, tmp_path):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder(), run_id="scan1")

    reviews = {(conflicts[0].claim_id, conflicts[0].unit_id): True}
    assert apply_reviews(corpus, reviews, run_id="scan1") == 1
    assert get_contradiction_reviews(corpus, "scan1") == reviews


def test_precision_is_measured_over_reviewed_candidates_only():
    assert contradiction_precision({("c1", "u1"): True, ("c1", "u2"): True,
                                    ("c1", "u3"): False}) == pytest.approx(2 / 3)


def test_precision_with_nothing_reviewed_is_zero():
    assert contradiction_precision({}) == 0.0


def test_the_target_is_contracrow_parity():
    assert CONTRADICTION_PRECISION_TARGET == 0.70


def test_the_report_flags_whether_the_target_is_met():
    good = report([], contradiction_reviews={("c1", "u1"): True, ("c1", "u2"): True,
                                             ("c1", "u3"): False})
    bad = report([], contradiction_reviews={("c1", "u1"): True, ("c1", "u2"): False,
                                            ("c1", "u3"): False})
    assert good.meets_contradiction_target is True
    assert bad.meets_contradiction_target is False


def test_the_report_omits_the_metric_when_nothing_was_reviewed():
    r = report([])
    assert r.contradiction_precision is None
    assert r.meets_contradiction_target is None


def test_rendering_presents_candidates_as_questions_not_findings():
    text = render_conflicts(CONFLICTS)
    lowered = text.lower()
    assert "candidate" in lowered or "review" in lowered
    assert "0.91" in text
    assert "61%" in text


def test_rendering_shows_only_the_top_n():
    many = [Conflict("c1", "t", "p1", f"u{i}", "p2", 0.9 - i / 100, "e") for i in range(50)]
    assert render_conflicts(many, top_n=5).count("candidate") <= 6


def test_rendering_nothing_says_so_plainly():
    assert "no " in render_conflicts([]).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contradict.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_reviews' from 'jarvis.contradict'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/contradict.py`, adding `import json`, `from collections.abc import Mapping`, `from pathlib import Path`, and `from jarvis.store import get_contradictions, set_contradiction_review` to the imports:

```python
_TRUE = {"true", "valid", "yes", "y", "1"}
_FALSE = {"false", "invalid", "no", "n", "0"}


def write_review_sheet(path: str | Path, conflicts: Sequence[Conflict]) -> int:
    """Write candidates as JSONL for a human to adjudicate. Returns rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "claim_id": c.claim_id, "claim_text": c.claim_text,
            "claim_paper_id": c.claim_paper_id, "unit_id": c.unit_id,
            "paper_id": c.paper_id, "score": round(c.score, 4), "evidence": c.evidence,
            "verdict": None,
        }, ensure_ascii=False)
        for c in conflicts
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def read_reviews(path: str | Path) -> dict[tuple[str, str], bool]:
    """Read adjudicated verdicts. Unreviewed and unparseable rows are absent, not False."""
    target = Path(path)
    if not target.is_file():
        return {}
    out: dict[tuple[str, str], bool] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not row.get("claim_id") or not row.get("unit_id"):
            continue
        verdict = _as_bool(row.get("verdict"))
        if verdict is not None:
            out[(str(row["claim_id"]), str(row["unit_id"]))] = verdict
    return out


def apply_reviews(conn: sqlite3.Connection, reviews: Mapping[tuple[str, str], bool],
                  run_id: str = "") -> int:
    """Write adjudicated verdicts back into the store. Returns rows updated."""
    for (claim_id, unit_id), valid in reviews.items():
        set_contradiction_review(conn, claim_id, unit_id,
                                 "valid" if valid else "invalid", run_id=run_id)
    return len(reviews)


def render_conflicts(conflicts: Sequence[Conflict], top_n: int = 20) -> str:
    """Human-readable queue. Every line is a question, never a finding (spec §8)."""
    ranked = rank(conflicts)[:top_n]
    if not ranked:
        return "No contradiction candidates found in this corpus."

    lines = [f"{len(ranked)} contradiction candidate(s) for review — these are prompts to "
             f"look, not findings:", ""]
    for index, conflict in enumerate(ranked, start=1):
        lines += [
            f"{index}. candidate (contradiction score {conflict.score:.2f})",
            f"   claim  [{conflict.claim_paper_id}]: {conflict.claim_text}",
            f"   versus [{conflict.paper_id}] {conflict.unit_id}: {conflict.evidence}",
            "",
        ]
    return "\n".join(lines)
```

In `jarvis/evaluate.py`, add the target constant next to the others, the metric after `citation_recall`, the `EvalReport` field, its property, and the `report()` keyword:

```python
CONTRADICTION_PRECISION_TARGET = 0.70   # ContraCrow parity (spec §10)


def contradiction_precision(reviews: Mapping[tuple[str, str], bool]) -> float:
    """Fraction of human-reviewed contradiction candidates that were genuine. Target >= 0.70.

    Measured over reviewed candidates only. An unreviewed candidate is not a failure — it
    is an unanswered question, and counting it as either would be a lie about the number.
    """
    if not reviews:
        return 0.0
    return sum(1 for valid in reviews.values() if valid) / len(reviews)
```

```python
@dataclass(frozen=True)
class EvalReport:
    quote_fidelity: float
    statement_support: float
    gate_recall: float | None = None
    coverage: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    contradiction_precision: float | None = None

    ...

    @property
    def meets_contradiction_target(self) -> bool | None:
        if self.contradiction_precision is None:
            return None
        return self.contradiction_precision >= CONTRADICTION_PRECISION_TARGET
```

Extend `report()` with a `contradiction_reviews: Mapping[tuple[str, str], bool] | None = None` keyword argument and pass:

```python
        contradiction_precision=(contradiction_precision(contradiction_reviews)
                                 if contradiction_reviews else None),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_contradict.py tests/test_evaluate.py tests/test_evaluate_citations.py -v && ruff check jarvis/contradict.py jarvis/evaluate.py tests/test_contradict.py`
Expected: PASS (33 contradiction tests + every existing evaluate test still green), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/contradict.py jarvis/evaluate.py tests/test_contradict.py
git commit -m "feat: contradiction review round-trip and precision metric"
```

---

### Task 5: End to end — two papers that disagree

**Files:**
- Create: `tests/test_contradict_end_to_end.py`
- Modify: `jarvis/__init__.py`

**Interfaces:**
- Consumes: everything this plan built.
- Produces: the extended public surface.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contradict_end_to_end.py
"""Three papers, one disagreement, one review pass, one measured precision number."""
import pytest

from jarvis.contradict import apply_reviews, read_reviews, scan_corpus, write_review_sheet
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.evaluate import report
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper
from jarvis.parse import FakeParser
from jarvis.store import (
    close_store,
    get_contradiction_reviews,
    get_contradictions,
    get_units,
    open_store,
    save_paper,
    save_units,
)
from jarvis.units import build_units
from jarvis.verify import FakeNLI

ORIGINAL = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
          page=1, section_path=("Results",)),
]
REPRODUCTION = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Reproducing the controller, tracking accuracy never exceeded 61% under "
               "gust disturbance.", page=1, section_path=("Results",)),
]
UNRELATED = [
    Block(kind="heading", text="Method", page=1, section_path=("Method",)),
    Block(kind="paragraph", text="We fold proteins with a transformer.", page=1,
          section_path=("Method",)),
]


def _ingest(conn, paper_id, title, blocks):
    paper = Paper(paper_id=paper_id, title=title, year=2025)
    parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    _ingest(conn, "p1", "Gust-Robust Control", ORIGINAL)
    _ingest(conn, "p2", "A Reproduction Study", REPRODUCTION)
    _ingest(conn, "p3", "Protein Folding", UNRELATED)
    yield conn
    close_store(conn)


@pytest.fixture
def claim(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    return Claim("c1", "The controller reaches 94.2% tracking accuracy under gusts.",
                 unit.unit_id, "94.2% tracking accuracy")


def _nli_that_only_disputes_the_reproduction(corpus):
    reproduction = next(u for u in get_units(corpus, "p2") if "61%" in u.verbatim_text)
    return FakeNLI(
        mapping={(reproduction.verbatim_text,
                  "The controller reaches 94.2% tracking accuracy under gusts."):
                 {"entailment": 0.02, "neutral": 0.06, "contradiction": 0.92}},
        default={"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05},
    )


def test_the_scan_finds_the_reproduction_and_ignores_the_unrelated_paper(corpus, claim):
    conflicts = scan_corpus(corpus, [claim], _nli_that_only_disputes_the_reproduction(corpus),
                            FakeEmbedder(), run_id="scan1")
    assert [c.paper_id for c in conflicts] == ["p2"]
    assert conflicts[0].score == pytest.approx(0.92)


def test_the_scan_never_reports_the_paper_disagreeing_with_itself(corpus, claim):
    always = FakeNLI(default={"entailment": 0.0, "neutral": 0.0, "contradiction": 1.0})
    conflicts = scan_corpus(corpus, [claim], always, FakeEmbedder())
    assert all(c.paper_id != "p1" for c in conflicts)


def test_the_review_loop_produces_a_precision_number(corpus, claim, tmp_path):
    conflicts = scan_corpus(corpus, [claim], _nli_that_only_disputes_the_reproduction(corpus),
                            FakeEmbedder(), run_id="scan1")

    sheet = tmp_path / "review.jsonl"
    write_review_sheet(sheet, conflicts)

    # A human reads the sheet and marks the one candidate as a genuine disagreement.
    lines = sheet.read_text(encoding="utf-8").replace('"verdict": null',
                                                      '"verdict": "valid"')
    sheet.write_text(lines, encoding="utf-8")

    reviews = read_reviews(sheet)
    apply_reviews(corpus, reviews, run_id="scan1")

    assert get_contradiction_reviews(corpus, "scan1") == reviews
    r = report([], contradiction_reviews=reviews)
    assert r.contradiction_precision == 1.0
    assert r.meets_contradiction_target is True


def test_a_scan_is_rerunnable_without_losing_earlier_judgments(corpus, claim):
    nli = _nli_that_only_disputes_the_reproduction(corpus)
    conflicts = scan_corpus(corpus, [claim], nli, FakeEmbedder(), run_id="scan1")
    apply_reviews(corpus, {(conflicts[0].claim_id, conflicts[0].unit_id): True},
                  run_id="scan1")

    scan_corpus(corpus, [claim], nli, FakeEmbedder(), run_id="scan1")
    assert get_contradiction_reviews(corpus, "scan1") != {}


def test_candidates_are_never_presented_as_facts(corpus, claim):
    from jarvis.contradict import render_conflicts
    conflicts = scan_corpus(corpus, [claim], _nli_that_only_disputes_the_reproduction(corpus),
                            FakeEmbedder())
    rendered = render_conflicts(conflicts).lower()
    assert "candidate" in rendered
    assert "contradicts" not in rendered, "spec §8: ranked candidates, never assertions"


def test_a_scan_over_a_corpus_with_no_disagreement_finds_nothing(corpus, claim):
    agreeable = FakeNLI(default={"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02})
    assert scan_corpus(corpus, [claim], agreeable, FakeEmbedder(), run_id="quiet") == []
    assert get_contradictions(corpus, "quiet") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_contradict_end_to_end.py -v`
Expected: FAIL on import until the exports land. If an assertion fails, fix the *module*.

- [ ] **Step 3: Extend the package exports**

Add to `jarvis/__init__.py` (keeping imports and `__all__` sorted):

```python
from jarvis.contradict import (
    Conflict,
    apply_reviews,
    opposing_units,
    rank,
    read_reviews,
    render_conflicts,
    scan_claim,
    scan_corpus,
    write_review_sheet,
)
from jarvis.evaluate import contradiction_precision
from jarvis.store import (
    get_contradiction_reviews,
    get_contradictions,
    save_contradictions,
    set_contradiction_review,
)
```

Note `read_reviews` and `write_review_sheet` sit alongside `jarvis.label`'s `read_labels` / `write_label_sheet` — different files, different shapes, deliberately parallel names. Do not merge them; a gate label is one boolean about a paper and a contradiction verdict is one boolean about a pair.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass. `ruff check .` reports exactly the **11 pre-existing** violations.

- [ ] **Step 5: Commit**

```bash
git add jarvis/__init__.py tests/test_contradict_end_to_end.py
git commit -m "test: end-to-end contradiction scan, review, and precision"
```

---

## Definition of done

- `python -m pytest` passes with zero network access, no API keys, no model downloads.
- `test_the_scan_never_reports_the_paper_disagreeing_with_itself` passes — cross-paper only, enforced by test rather than by convention.
- `test_candidates_are_never_presented_as_facts` passes — the rendered output is a review queue, not a set of findings.
- `test_a_scan_is_rerunnable_without_losing_earlier_judgments` passes — human review survives a rescan.
- `jarvis.evaluate.report(..., contradiction_reviews=...)` returns a real precision number and says whether it clears the 70% ContraCrow-parity target.
- `ruff check .` reports exactly the 11 pre-existing violations.

## Where this stops, and the one number to watch

This plan produces candidates and the machinery to measure them. It does not decide anything.

The number that determines whether this feature ships to anyone else: **contradiction precision on the first real corpus.** Spec §10 targets ≥70%. If the first hundred reviewed candidates come back at 30%, the honest response is to raise the threshold, tighten `opposing_units`, or shelve the feature — not to ship a queue of noise and let a researcher's time absorb the cost. Record the measured number in the ledger when it exists.

Two extensions deliberately not built here:

- **An LLM summary of each candidate for review.** `jarvis/router.py` already routes `contradiction_review` to the frontier tier for exactly this, and it would make skimming a 200-item queue much faster. It is not built yet because a summary layer over a detector whose precision is unmeasured makes the noise more persuasive, not less. Build it after the precision number exists.
- **An MCP `find_contradictions` tool.** Worth adding to `docs/plans/2026-08-14-mcp-server.md`'s registry once this lands and the precision holds.
