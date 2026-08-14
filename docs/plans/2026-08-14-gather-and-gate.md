# Gather + Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a research question into a few hundred gathered papers, decide which are worth reading with a calibrated union gate that cannot silently drop relevant work, and deep-read the survivors into the corpus the single-paper core already knows how to retrieve and verify.

**Architecture:** Spec §7 Stages A–C. Stage A gathers for **recall**: a search plan fans one question into many queries, several literature APIs answer them, and the citation graph is walked outward from the best hits. Stage B is the gate — a **union** of four cheap, independently-computed signals (embedding similarity, citation-graph proximity, BM25 overlap, one LLM vote), calibrated per project against a hand-labeled seed to ≥95% recall, emitting `read_deep` / `unsure` / `defer` with every per-signal score written to `screen_log`. There is no `exclude`. Stage C deep-reads the survivors through the pipeline built in the first plan (parse → typed units → contextual prefixes → embed → index) and extracts a Layer 2 card whose every field is quote-verified against Layer 0.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`/`json`/`hashlib`/`xml.etree`, `httpx` for the live API adapters (lazily imported), `pytest`. Everything that touches a network or a model sits behind a `typing.Protocol` or an injected callable with a deterministic offline double.

**Prerequisite:** the verifiable single-paper core (`docs/plans/2026-08-11-verifiable-single-paper-core.md`, spec build steps 1–5), merged into `main` at `d7f8672`. Every module it produced is consumed here and none of them is modified except `store.py`, `sources.py`, and `__init__.py`, which are extended.

## Global Constraints

- Python **>= 3.10**. Use `X | None`, not `Optional[X]`.
- **Never read `.env`.** Configuration is environment variables or `$JARVIS_CONFIG` JSON only.
- **Every test is offline.** No network, no API keys, no model downloads. `httpx`, `openai`, `docling`, `sentence-transformers` and `transformers` are imported *inside* the function that needs them, never at module top level.
- All external models and all network calls are consumed through a `typing.Protocol` or an injected callable, with a deterministic fake used in tests.
- Line length **100**. Target `py310`. Run `ruff check .` against **both** the module and its test file before every commit.
- **`jarvis/store.py` is the only module that writes SQL.** Every new table and every new query in this plan goes there.
- Layer 0 is **immutable** — `save_paper` already refuses to blank `raw_text`; nothing here may work around that.
- No LLM may be routed to a verification task (enforced by `test_verification_is_not_routed_to_an_llm`, already passing).
- **The gate has no `exclude` outcome.** `defer` is demotion to metadata depth, never deletion. Any code path that removes a paper from the corpus is a defect.
- **The gate is a union, never an intersection.** A paper kept by any single signal is kept.
- Frozen dataclasses for all new domain types; tuples not lists in frozen types.
- Commit after every task with a `feat:`/`test:`/`fix:` prefix.
- Repo-wide `ruff check .` has a **pre-existing baseline of 11 violations** in `citation_graph.py` (2), `config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1). Do not "fix" them as part of this plan and do not add to them. Any new violation in a file this plan touches is a defect.

## The one risk this plan exists to control

Spec §7B is blunt about it: SESR-Eval benchmarked 9 LLMs on title-abstract screening across 24 systematic reviews and 34,528 records **in software engineering**, the closest published domain to ours, and the best model reached **0.66 recall**. Claude 3.7 Sonnet reached 0.46. The verdict was *"no LLM managed a high recall with reasonable precision."* Medical and environmental domains report >95% for the identical task — this is domain-dependent and our domain is the bad one.

A single-LLM gate would silently discard a third to half of the relevant literature while the system went on producing confident, well-cited answers over what was left. That failure is invisible from the output. Everything in Task 7 through Task 10 — four independent signals, union not intersection, three outcomes with `unsure` escalating, demotion not deletion, per-signal scores logged, thresholds fitted to a hand-labeled seed — exists to make that failure both unlikely and measurable.

The human baseline for calibration: single-reviewer screening misses **13%** of relevant studies. The field standard for a screening tool is **≥95% recall, explicitly accepting poor precision.**

## File Structure

| File | Responsibility |
|---|---|
| `jarvis/store.py` | **Modify.** Add `screen_log`, `cards`, `runs` CRUD and corpus-wide reads. Still the only module that writes SQL. |
| `jarvis/sources.py` | **Modify.** Add arXiv / Semantic Scholar / OpenAlex normalizers and search adapters, plus the Crossref retraction check. |
| `jarvis/gather.py` | Create. Search plan, multi-source fan-out, citation expansion, dedup, `Candidate`. Stage A. |
| `jarvis/gate.py` | Create. The four signals, the union decision, and threshold calibration. Stage B. |
| `jarvis/label.py` | Create. Seed-set sampling and the hand-label round-trip that calibration and `gate_recall` both need. |
| `jarvis/ingest.py` | Create. Stage C: deep read one paper into the corpus, and drive it over a decision set. |
| `jarvis/card.py` | Create. Layer 2 extraction and per-field quote verification. |
| `jarvis/__init__.py` | **Modify.** Export the new public surface. |

Tests mirror module names: `tests/test_gather.py`, `tests/test_gate.py`, `tests/test_label.py`, `tests/test_ingest.py`, `tests/test_card.py`, `tests/test_store_screen.py`, `tests/test_sources_adapters.py`, and `tests/test_gather_end_to_end.py`.

---

### Task 1: Store — screening log, cards, runs, and corpus-wide reads

**Files:**
- Modify: `jarvis/store.py` (append to the schema string and add functions at the end)
- Test: `tests/test_store_screen.py`

**Interfaces:**
- Consumes: `Paper`, `Unit`, `Card`, `CardField`, `UnitType` from `jarvis.models`; the existing `_row_to_paper` and `_row_to_unit` helpers in `jarvis/store.py`.
- Produces: `save_screen_decision(conn, paper_id, decision, signals, run_id="")`, `get_screen_decisions(conn, run_id="") -> dict[str, str]`, `get_screen_signals(conn, run_id="") -> dict[str, dict]`, `save_card(conn, card)`, `get_card(conn, paper_id) -> Card | None`, `save_run(conn, run_id, question="", started_at="", cost_usd=0.0)`, `get_run(conn, run_id) -> dict | None`, `set_depth(conn, paper_id, depth)`, `get_papers_by_depth(conn, depth) -> list[Paper]`, `all_units(conn, exclude_paper_id=None) -> list[Unit]`.

The `screen_log`, `cards`, and `runs` tables already exist in `_SCHEMA` — this task adds no DDL, only the CRUD the rest of the plan needs.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store_screen.py
"""Screening log, card, and run persistence — the bookkeeping the gate is audited from."""
import pytest

from jarvis.models import Card, CardField, Paper, Unit, UnitType
from jarvis.store import (
    all_units,
    close_store,
    get_card,
    get_papers_by_depth,
    get_run,
    get_screen_decisions,
    get_screen_signals,
    open_store,
    save_card,
    save_paper,
    save_run,
    save_screen_decision,
    save_units,
    set_depth,
)


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / "corpus.db")
    yield c
    close_store(c)


def _unit(paper_id: str, ordinal: int) -> Unit:
    u = Unit(unit_id="", paper_id=paper_id, type=UnitType.PROSE, page=1,
             section_path=(), verbatim_text=f"text {ordinal}", ordinal=ordinal)
    return Unit(unit_id=u.key(), paper_id=u.paper_id, type=u.type, page=u.page,
                section_path=u.section_path, verbatim_text=u.verbatim_text, ordinal=u.ordinal)


def test_a_decision_round_trips_with_its_per_signal_scores(conn):
    save_screen_decision(conn, "p1", "read_deep",
                         {"embedding": 0.71, "graph": 0.0, "keyword": 0.4, "llm_vote": 1.0},
                         run_id="r1")
    assert get_screen_decisions(conn, "r1") == {"p1": "read_deep"}
    assert get_screen_signals(conn, "r1")["p1"]["embedding"] == pytest.approx(0.71)


def test_rescreening_the_same_paper_in_the_same_run_overwrites(conn):
    save_screen_decision(conn, "p1", "defer", {"embedding": 0.1}, run_id="r1")
    save_screen_decision(conn, "p1", "unsure", {"embedding": 0.4}, run_id="r1")
    assert get_screen_decisions(conn, "r1") == {"p1": "unsure"}


def test_the_same_paper_can_be_screened_differently_in_two_runs(conn):
    save_screen_decision(conn, "p1", "defer", {}, run_id="r1")
    save_screen_decision(conn, "p1", "read_deep", {}, run_id="r2")
    assert get_screen_decisions(conn, "r1") == {"p1": "defer"}
    assert get_screen_decisions(conn, "r2") == {"p1": "read_deep"}


def test_decisions_without_a_run_id_are_readable(conn):
    save_screen_decision(conn, "p1", "read_deep", {})
    assert get_screen_decisions(conn) == {"p1": "read_deep"}


def test_a_card_round_trips_with_every_field_and_flag(conn):
    save_paper(conn, Paper(paper_id="p1", title="T"))
    card = Card(
        paper_id="p1",
        problem=CardField(value="gust rejection", unit_id="u1", quote="gusts"),
        metrics=(CardField(value="94.2", unit_id="u2", quote="94.2", binding_verified=True),),
        datasets=(CardField(value="KITTI", unit_id="u3", quote="KITTI"),),
    )
    save_card(conn, card)

    got = get_card(conn, "p1")
    assert got.problem.value == "gust rejection"
    assert got.metrics[0].binding_verified is True
    assert got.datasets[0].unit_id == "u3"
    assert got.method is None
    assert got.claims == ()


def test_saving_a_card_twice_replaces_it(conn):
    save_paper(conn, Paper(paper_id="p1", title="T"))
    save_card(conn, Card(paper_id="p1", problem=CardField("a", "u1", "a")))
    save_card(conn, Card(paper_id="p1", problem=CardField("b", "u1", "b")))
    assert get_card(conn, "p1").problem.value == "b"


def test_a_missing_card_is_none(conn):
    assert get_card(conn, "nope") is None


def test_depth_can_be_promoted_without_touching_layer_zero(conn):
    save_paper(conn, Paper(paper_id="p1", title="T"), raw_text="ORIGINAL", depth="metadata")
    set_depth(conn, "p1", "deep")
    assert [p.paper_id for p in get_papers_by_depth(conn, "deep")] == ["p1"]
    assert get_papers_by_depth(conn, "metadata") == []


def test_all_units_can_exclude_one_paper(conn):
    save_paper(conn, Paper(paper_id="p1", title="A"))
    save_paper(conn, Paper(paper_id="p2", title="B"))
    save_units(conn, [_unit("p1", 0), _unit("p2", 0), _unit("p2", 1)])

    assert len(all_units(conn)) == 3
    assert {u.paper_id for u in all_units(conn, exclude_paper_id="p2")} == {"p1"}


def test_a_run_records_its_question_and_cost(conn):
    save_run(conn, "r1", question="how do quadrotors reject gusts?", cost_usd=1.25)
    assert get_run(conn, "r1")["question"] == "how do quadrotors reject gusts?"
    assert get_run(conn, "r1")["cost_usd"] == pytest.approx(1.25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_store_screen.py -v`
Expected: FAIL with `ImportError: cannot import name 'save_screen_decision' from 'jarvis.store'`

- [ ] **Step 3: Write the implementation**

Append to the end of `jarvis/store.py`. Add `Card` and `CardField` to the existing `from jarvis.models import ...` line at the top (it becomes `from jarvis.models import Card, CardField, Paper, Unit, UnitType`).

```python
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
    """Promote or demote a paper's ingest depth. Never deletes; `defer` is demotion."""
    conn.execute("UPDATE papers SET depth = ? WHERE paper_id = ?", (depth, paper_id))
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store_screen.py tests/test_store_crud.py tests/test_store_schema.py -v && ruff check jarvis/store.py tests/test_store_screen.py`
Expected: PASS (11 new + all existing store tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/store.py tests/test_store_screen.py
git commit -m "feat: screening log, card, and run persistence"
```

---

### Task 2: The search plan — one question into many queries

**Files:**
- Create: `jarvis/gather.py`
- Test: `tests/test_gather.py`

**Interfaces:**
- Consumes: nothing from this plan. `jarvis.llm.chat` and `jarvis.router.ModelRouter` are used through injection only.
- Produces: `SearchPlan` (frozen: `question: str`, `sub_questions: tuple[str, ...]`, `queries: tuple[str, ...]`), `Planner` protocol with `plan(question: str) -> SearchPlan`, `TemplatePlanner`, `LLMPlanner(router, chat_fn=None, max_sub=4, per_sub=3)`.

This mirrors the `PrefixGenerator` / `TemplatePrefix` / `LLMPrefix` shape already established in `jarvis/context.py`: a protocol, a deterministic no-model implementation that is both the test default and the runtime fallback, and an LLM implementation that falls back to it on any failure.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gather.py
"""Stage A — recall-optimized gathering (spec §7A)."""
import pytest

from jarvis.gather import LLMPlanner, Planner, SearchPlan, TemplatePlanner


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_template_planner_satisfies_the_protocol():
    assert isinstance(TemplatePlanner(), Planner)


def test_template_planner_keeps_the_question_verbatim_as_a_query():
    plan = TemplatePlanner().plan("how do quadrotors reject gusts?")
    assert plan.question == "how do quadrotors reject gusts?"
    assert "how do quadrotors reject gusts?" in plan.queries


def test_template_planner_fans_out_to_several_distinct_queries():
    plan = TemplatePlanner().plan("gust rejection")
    assert len(plan.queries) >= 4
    assert len(set(plan.queries)) == len(plan.queries)


def test_template_planner_collapses_whitespace():
    assert TemplatePlanner().plan("  a   b\n").question == "a b"


def test_template_planner_is_deterministic():
    assert TemplatePlanner().plan("x") == TemplatePlanner().plan("x")


def test_llm_planner_uses_the_models_sub_questions_and_queries():
    def fake_chat(router, task, prompt, **kwargs):
        assert task == "query_expansion"
        assert kwargs.get("json_mode") is True
        return {"sub_questions": ["what disturbs a quadrotor?"],
                "queries": ["quadrotor wind disturbance", "gust rejection control"]}

    plan = LLMPlanner(_Router(), chat_fn=fake_chat).plan("gust rejection")
    assert plan.sub_questions == ("what disturbs a quadrotor?",)
    assert "gust rejection control" in plan.queries
    assert "gust rejection" in plan.queries, "the raw question is always searched too"


def test_llm_planner_falls_back_to_the_template_when_the_model_raises():
    def boom(router, task, prompt, **kwargs):
        raise RuntimeError("no api key")

    plan = LLMPlanner(_Router(), chat_fn=boom).plan("gust rejection")
    assert plan == TemplatePlanner().plan("gust rejection")


def test_llm_planner_falls_back_when_the_model_returns_junk():
    for junk in (None, {}, {"queries": []}, "not a dict", {"queries": ["", "  "]}):
        plan = LLMPlanner(_Router(), chat_fn=lambda *a, **k: junk).plan("q")
        assert plan == TemplatePlanner().plan("q")


def test_llm_planner_caps_the_fan_out():
    many = {"sub_questions": [f"s{i}" for i in range(50)],
            "queries": [f"q{i}" for i in range(50)]}
    plan = LLMPlanner(_Router(), chat_fn=lambda *a, **k: many, max_sub=4, per_sub=3).plan("q")
    assert len(plan.sub_questions) == 4
    assert len(plan.queries) <= 4 * 3 + 1


def test_search_plan_is_frozen():
    plan = SearchPlan(question="q")
    with pytest.raises(Exception):
        plan.question = "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gather.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.gather'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/gather.py
"""Stage A — recall-optimized gathering (spec §7A).

No answer is being written here, so nothing is filtered for precision. One question fans
out into many queries, several APIs answer them, and the citation graph is walked outward
from the best hits. Cost is paid once and amortized across every future query against the
corpus (spec §4).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_PLAN_PROMPT = (
    "Decompose this research question for a literature search.\n"
    "Return JSON: {{\"sub_questions\": [...], \"queries\": [...]}}.\n"
    "Give at most {max_sub} sub-questions and at most {per_sub} search queries per "
    "sub-question. Queries are keyword phrases for an academic search API, not sentences. "
    "Vary the vocabulary: the same concept is named differently across communities.\n\n"
    "Question: {question}"
)


@dataclass(frozen=True)
class SearchPlan:
    """One question, decomposed. `queries` is what actually gets sent to the APIs."""
    question: str
    sub_questions: tuple[str, ...] = ()
    queries: tuple[str, ...] = ()


@runtime_checkable
class Planner(Protocol):
    def plan(self, question: str) -> SearchPlan: ...


class TemplatePlanner:
    """Deterministic, free, no model. The fallback and the test default.

    The facets are the ones that reliably surface different papers for the same topic: a
    survey names the field, a benchmark names the numbers, a method names the technique,
    and limitations surface the critical literature that positive-framing queries miss.
    """

    FACETS = ("survey", "benchmark", "method", "limitations")

    def plan(self, question: str) -> SearchPlan:
        q = " ".join((question or "").split())
        subs = tuple(f"{q} {facet}" for facet in self.FACETS)
        return SearchPlan(question=q, sub_questions=subs, queries=(q,) + subs)


def _clean(items, limit: int) -> tuple[str, ...]:
    """Strings only, stripped, de-duplicated in order, capped."""
    out: list[str] = []
    for item in items or []:
        text = " ".join(str(item).split())
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return tuple(out)


class LLMPlanner:
    """Model-written decomposition, routed to the cheap tier. Falls back to the template."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None,
                 max_sub: int = 4, per_sub: int = 3) -> None:
        self._router = router
        self._chat = chat_fn
        self._max_sub = max_sub
        self._per_sub = per_sub
        self._fallback = TemplatePlanner()

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def plan(self, question: str) -> SearchPlan:
        fallback = self._fallback.plan(question)
        prompt = _PLAN_PROMPT.format(max_sub=self._max_sub, per_sub=self._per_sub,
                                     question=fallback.question)
        try:
            raw = self._chat_fn()(self._router, "query_expansion", prompt, json_mode=True)
        except Exception:  # noqa: BLE001 - any failure means fall back, never crash a run
            return fallback
        if not isinstance(raw, dict):
            return fallback

        subs = _clean(raw.get("sub_questions"), self._max_sub)
        queries = _clean(raw.get("queries"), self._max_sub * self._per_sub)
        if not queries:
            return fallback
        # The raw question is always searched: a decomposition that drops it is a
        # recall regression, and this stage optimizes for recall only.
        if fallback.question not in queries:
            queries = (fallback.question,) + queries
        return SearchPlan(question=fallback.question, sub_questions=subs, queries=queries)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gather.py -v && ruff check jarvis/gather.py tests/test_gather.py`
Expected: PASS (10 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/gather.py tests/test_gather.py
git commit -m "feat: search plan generation for gather-stage fan-out"
```

---

### Task 3: Literature source adapters — arXiv, Semantic Scholar, OpenAlex

**Files:**
- Modify: `jarvis/sources.py` (append; do not touch the existing Crossref/CORE/Unpaywall code)
- Test: `tests/test_sources_adapters.py`

**Interfaces:**
- Consumes: the existing `PAPER_FIELDS` contract in `jarvis/sources.py` — every normalizer returns a dict with exactly those keys.
- Produces: `normalize_s2(item) -> dict`, `normalize_openalex(item) -> dict`, `normalize_arxiv_entry(entry) -> dict`, `openalex_abstract(inverted) -> str`, `make_s2_search(limit=20)`, `make_openalex_search(limit=20)`, `make_arxiv_search(limit=20)`.

Only the normalizers are tested. The `make_*_search` functions are thin `httpx` wrappers around them, import `httpx` lazily, and swallow failures to `[]` exactly like the existing `make_crossref_search` — `combine_sources` already degrades gracefully when one source is down.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources_adapters.py
"""Normalizers for the three sources spec §7A names alongside Crossref."""
from jarvis.sources import (
    PAPER_FIELDS,
    normalize_arxiv_entry,
    normalize_openalex,
    normalize_s2,
    openalex_abstract,
)

S2_ITEM = {
    "paperId": "abc123",
    "title": "Gust-Robust Quadrotor Control",
    "abstract": "We reject gusts.",
    "year": 2025,
    "venue": "ICRA",
    "citationCount": 42,
    "externalIds": {"ArXiv": "2501.00001", "DOI": "10.1/xyz"},
    "openAccessPdf": {"url": "https://example.org/p.pdf"},
    "fieldsOfStudy": ["Engineering"],
}

OPENALEX_ITEM = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/xyz",
    "title": "Gust-Robust Quadrotor Control",
    "publication_year": 2025,
    "cited_by_count": 42,
    "host_venue": {"display_name": "ICRA"},
    "authorships": [{"author": {"display_name": "A. Researcher"}}],
    "abstract_inverted_index": {"We": [0], "reject": [1], "gusts": [2]},
    "open_access": {"oa_url": "https://example.org/p.pdf"},
    "concepts": [{"display_name": "Control theory"}],
}

ARXIV_ENTRY = {
    "id": "http://arxiv.org/abs/2501.00001v2",
    "title": "Gust-Robust\n  Quadrotor Control",
    "summary": "We reject\n  gusts.",
    "published": "2025-01-03T00:00:00Z",
    "authors": ["A. Researcher", "B. Engineer"],
    "categories": ["cs.RO", "eess.SY"],
    "doi": "10.1/xyz",
}


def _has_contract(paper: dict) -> bool:
    return set(paper) == set(PAPER_FIELDS)


def test_every_normalizer_returns_exactly_the_common_contract():
    assert _has_contract(normalize_s2(S2_ITEM))
    assert _has_contract(normalize_openalex(OPENALEX_ITEM))
    assert _has_contract(normalize_arxiv_entry(ARXIV_ENTRY))


def test_s2_lifts_arxiv_and_doi_out_of_external_ids():
    p = normalize_s2(S2_ITEM)
    assert p["arxiv_id"] == "2501.00001"
    assert p["doi"] == "10.1/xyz"
    assert p["s2_id"] == "abc123"
    assert p["citation_count"] == 42


def test_s2_survives_a_record_with_nothing_but_a_title():
    p = normalize_s2({"title": "Bare"})
    assert p["title"] == "Bare"
    assert p["arxiv_id"] == ""
    assert p["year"] is None
    assert p["citation_count"] == 0


def test_s2_falls_back_to_an_arxiv_pdf_url_when_there_is_no_oa_pdf():
    p = normalize_s2({"title": "T", "externalIds": {"ArXiv": "2501.00001"}})
    assert p["pdf_url"] == "https://arxiv.org/pdf/2501.00001"


def test_openalex_abstract_is_rebuilt_from_the_inverted_index():
    assert openalex_abstract({"We": [0], "reject": [1], "gusts": [2]}) == "We reject gusts"
    assert openalex_abstract({"a": [0, 2], "b": [1]}) == "a b a"
    assert openalex_abstract(None) == ""


def test_openalex_strips_the_doi_url_prefix():
    assert normalize_openalex(OPENALEX_ITEM)["doi"] == "10.1/xyz"


def test_openalex_reads_authors_and_venue():
    p = normalize_openalex(OPENALEX_ITEM)
    assert p["authors"] == ["A. Researcher"]
    assert p["venue"] == "ICRA"
    assert p["abstract"] == "We reject gusts"


def test_arxiv_strips_the_version_suffix_and_collapses_wrapped_text():
    p = normalize_arxiv_entry(ARXIV_ENTRY)
    assert p["arxiv_id"] == "2501.00001"
    assert p["title"] == "Gust-Robust Quadrotor Control"
    assert p["abstract"] == "We reject gusts."
    assert p["year"] == 2025
    assert p["categories"] == ["cs.RO", "eess.SY"]
    assert p["pdf_url"] == "https://arxiv.org/pdf/2501.00001"


def test_normalized_records_dedup_against_each_other():
    from jarvis.sources import dedup_papers
    merged = dedup_papers([normalize_s2(S2_ITEM), normalize_arxiv_entry(ARXIV_ENTRY),
                           normalize_openalex(OPENALEX_ITEM)])
    assert len(merged) == 1, "the same paper from three APIs is one paper"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sources_adapters.py -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_s2' from 'jarvis.sources'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/sources.py`:

```python
def _arxiv_pdf(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""


def normalize_s2(item: dict) -> dict:
    """Semantic Scholar graph record -> the common paper dict."""
    ext = item.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv", "") or ""
    pdf = item.get("openAccessPdf") or {}
    return {
        "doi": ext.get("DOI", "") or "",
        "arxiv_id": arxiv_id,
        "s2_id": item.get("paperId", "") or "",
        "title": item.get("title", "") or "",
        "authors": [a.get("name", "") for a in (item.get("authors") or [])],
        "year": item.get("year"),
        "venue": item.get("venue", "") or "",
        "abstract": item.get("abstract", "") or "",
        "citation_count": item.get("citationCount", 0) or 0,
        "url": item.get("url", "") or "",
        "pdf_url": pdf.get("url", "") or _arxiv_pdf(arxiv_id),
        "categories": list(item.get("fieldsOfStudy") or []),
    }


def openalex_abstract(inverted: dict | None) -> str:
    """Rebuild prose from OpenAlex's inverted index ({word: [positions]})."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, where in inverted.items():
        positions += [(int(i), word) for i in (where or [])]
    return " ".join(word for _, word in sorted(positions))


def normalize_openalex(item: dict) -> dict:
    """OpenAlex work -> the common paper dict."""
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    venue = (item.get("host_venue") or {}).get("display_name", "") or ""
    oa_url = (item.get("open_access") or {}).get("oa_url", "") or ""
    return {
        "doi": doi,
        "arxiv_id": "",
        "s2_id": "",
        "title": item.get("title", "") or "",
        "authors": [(a.get("author") or {}).get("display_name", "")
                    for a in (item.get("authorships") or [])],
        "year": item.get("publication_year"),
        "venue": venue,
        "abstract": openalex_abstract(item.get("abstract_inverted_index")),
        "citation_count": item.get("cited_by_count", 0) or 0,
        "url": item.get("id", "") or "",
        "pdf_url": oa_url,
        "categories": [c.get("display_name", "") for c in (item.get("concepts") or [])],
    }


_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")


def normalize_arxiv_entry(entry: dict) -> dict:
    """A pre-extracted arXiv Atom entry -> the common paper dict.

    Takes a plain dict rather than an XML node so the mapping is testable without a feed.
    """
    match = _ARXIV_ID.search(entry.get("id", "") or "")
    arxiv_id = match.group(1) if match else ""
    published = entry.get("published", "") or ""
    year = int(published[:4]) if published[:4].isdigit() else None
    return {
        "doi": entry.get("doi", "") or "",
        "arxiv_id": arxiv_id,
        "s2_id": "",
        "title": " ".join((entry.get("title") or "").split()),
        "authors": list(entry.get("authors") or []),
        "year": year,
        "venue": "arXiv",
        "abstract": " ".join((entry.get("summary") or "").split()),
        "citation_count": 0,
        "url": entry.get("id", "") or "",
        "pdf_url": _arxiv_pdf(arxiv_id),
        "categories": list(entry.get("categories") or []),
    }


def make_s2_search(limit: int = 20) -> Callable[[str], list[dict]]:
    """Live Semantic Scholar keyword search. Uses $S2_API_KEY when present."""
    import os

    import httpx

    fields = ("paperId,title,abstract,year,venue,citationCount,externalIds,"
              "openAccessPdf,authors,fieldsOfStudy,url")

    def search(topic: str) -> list[dict]:
        key = os.environ.get("S2_API_KEY", "")
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": topic, "limit": limit, "fields": fields},
                    headers={"x-api-key": key} if key else {},
                )
                resp.raise_for_status()
                items = resp.json().get("data", [])
        except Exception:
            return []
        return [normalize_s2(i) for i in items]

    return search


def make_openalex_search(limit: int = 20, mailto: str = "") -> Callable[[str], list[dict]]:
    """Live OpenAlex search. `mailto` gets you into the polite pool."""
    import httpx

    def search(topic: str) -> list[dict]:
        params = {"search": topic, "per-page": limit}
        if mailto:
            params["mailto"] = mailto
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get("https://api.openalex.org/works", params=params)
                resp.raise_for_status()
                items = resp.json().get("results", [])
        except Exception:
            return []
        return [normalize_openalex(i) for i in items]

    return search


def make_arxiv_search(limit: int = 20) -> Callable[[str], list[dict]]:
    """Live arXiv Atom search, parsed with the stdlib XML parser."""
    import xml.etree.ElementTree as ET

    import httpx

    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def search(topic: str) -> list[dict]:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get("http://export.arxiv.org/api/query",
                                  params={"search_query": f"all:{topic}",
                                          "max_results": limit})
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
        except Exception:
            return []
        out: list[dict] = []
        for node in root.findall("a:entry", ns):
            doi_node = node.find("arxiv:doi", ns)
            out.append(normalize_arxiv_entry({
                "id": (node.findtext("a:id", "", ns) or ""),
                "title": (node.findtext("a:title", "", ns) or ""),
                "summary": (node.findtext("a:summary", "", ns) or ""),
                "published": (node.findtext("a:published", "", ns) or ""),
                "authors": [n.findtext("a:name", "", ns) or ""
                            for n in node.findall("a:author", ns)],
                "categories": [c.get("term", "") for c in node.findall("a:category", ns)],
                "doi": (doi_node.text if doi_node is not None else "") or "",
            }))
        return out

    return search
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sources_adapters.py tests/test_ported.py -v && ruff check jarvis/sources.py tests/test_sources_adapters.py`
Expected: PASS (10 new + existing ported tests). `ruff check jarvis/sources.py` still reports exactly its **6 pre-existing** violations and no more.

- [ ] **Step 5: Commit**

```bash
git add jarvis/sources.py tests/test_sources_adapters.py
git commit -m "feat: arxiv, semantic scholar, and openalex source adapters"
```

---

### Task 4: Retraction and provenance enrichment

**Files:**
- Modify: `jarvis/sources.py` (append)
- Test: `tests/test_sources_adapters.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `is_retracted_record(work) -> bool`, `make_retraction_check() -> Callable[[str], bool]`, `enrich_provenance(paper: dict, retraction_check=None) -> dict`.

Spec §14 names citing retracted work as the worst failure this system can produce, and a cheap one to prevent. The check runs at ingest **and** again at compile time, so it is a pure function over a Crossref work record plus a lazily-built live lookup.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sources_adapters.py`:

```python
from jarvis.sources import enrich_provenance, is_retracted_record


def test_a_retraction_notice_is_detected_by_type():
    assert is_retracted_record({"type": "retraction"}) is True


def test_a_work_with_a_retraction_update_is_detected():
    assert is_retracted_record({"update-to": [{"type": "retraction",
                                               "DOI": "10.1/original"}]}) is True


def test_a_work_flagged_by_subtype_is_detected():
    assert is_retracted_record({"subtype": "retracted-article"}) is True


def test_an_ordinary_article_is_not_retracted():
    assert is_retracted_record({"type": "journal-article",
                                "update-to": [{"type": "correction"}]}) is False
    assert is_retracted_record({}) is False
    assert is_retracted_record(None) is False


def test_enrich_marks_a_retracted_paper_and_leaves_the_rest_alone():
    paper = {"doi": "10.1/bad", "title": "T", "year": 2025}
    out = enrich_provenance(paper, retraction_check=lambda doi: doi == "10.1/bad")
    assert out["retracted"] is True
    assert out["title"] == "T"
    assert paper.get("retracted") is None, "enrich must not mutate its input"


def test_enrich_defaults_to_not_retracted_without_a_checker():
    assert enrich_provenance({"doi": "10.1/x"})["retracted"] is False


def test_enrich_skips_the_lookup_when_there_is_no_doi():
    calls = []

    def check(doi):
        calls.append(doi)
        return True

    assert enrich_provenance({"doi": ""}, retraction_check=check)["retracted"] is False
    assert calls == []


def test_enrich_treats_a_failing_lookup_as_unknown_not_retracted():
    def boom(doi):
        raise RuntimeError("crossref down")

    assert enrich_provenance({"doi": "10.1/x"}, retraction_check=boom)["retracted"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sources_adapters.py -v`
Expected: FAIL with `ImportError: cannot import name 'enrich_provenance' from 'jarvis.sources'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/sources.py`:

```python
_RETRACTION_MARKERS = ("retraction", "retracted")


def is_retracted_record(work: dict | None) -> bool:
    """True when a Crossref work is a retraction notice or is flagged as retracted."""
    if not work:
        return False
    for key in ("type", "subtype"):
        if any(m in str(work.get(key, "")).lower() for m in _RETRACTION_MARKERS):
            return True
    return any("retraction" in str(u.get("type", "")).lower()
               for u in (work.get("update-to") or []))


def make_retraction_check() -> Callable[[str], bool]:
    """Live Crossref retraction lookup by DOI. Spec §14: the cheapest failure to prevent."""
    import httpx

    def check(doi: str) -> bool:
        if not doi:
            return False
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"https://api.crossref.org/works/{doi}")
                resp.raise_for_status()
                work = resp.json().get("message", {})
        except Exception:
            return False
        return is_retracted_record(work)

    return check


def enrich_provenance(paper: dict,
                      retraction_check: Callable[[str], bool] | None = None) -> dict:
    """Return a copy of `paper` with a `retracted` flag resolved.

    A failed or absent lookup means *unknown*, which is recorded as not-retracted rather
    than blocking ingest — a paper is never dropped by this system, only flagged.
    """
    out = dict(paper)
    doi = (out.get("doi") or "").strip()
    retracted = False
    if doi and retraction_check is not None:
        try:
            retracted = bool(retraction_check(doi))
        except Exception:  # noqa: BLE001 - a source outage is not a retraction
            retracted = False
    out["retracted"] = retracted
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_sources_adapters.py -v && ruff check jarvis/sources.py tests/test_sources_adapters.py`
Expected: PASS (18 tests total in the file), `jarvis/sources.py` still at its 6 pre-existing violations

- [ ] **Step 5: Commit**

```bash
git add jarvis/sources.py tests/test_sources_adapters.py
git commit -m "feat: crossref retraction check and provenance enrichment"
```

---

### Task 5: Candidates — multi-source fan-out with dedup

**Files:**
- Modify: `jarvis/gather.py` (append)
- Test: `tests/test_gather.py` (append)

**Interfaces:**
- Consumes: `SearchPlan`, `Planner` (Task 2); `dedup_papers` from `jarvis.sources`; `paper_id` from `jarvis.citation_graph`; `Paper` from `jarvis.models`; `save_paper` from `jarvis.store`.
- Produces: `Candidate` (frozen: `paper: dict`, `origin: str = "search"`, `graph_depth: int = 0`, `queries: tuple[str, ...] = ()`, property `pid`), `run_searches(plan, search_fn) -> list[Candidate]`, `to_paper(candidate) -> Paper`, `save_candidates(conn, candidates) -> int`.

`Candidate` is the gather-stage working type: a raw source dict plus how it was found. The gate reads `graph_depth` and `queries` as signals; nothing downstream of ingest sees a `Candidate`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gather.py`:

```python
from jarvis.gather import Candidate, run_searches, save_candidates, to_paper
from jarvis.store import close_store, get_paper, get_papers_by_depth, open_store

CORPUS = {
    "gust rejection": [
        {"arxiv_id": "2501.00001", "title": "Gust-Robust Control", "abstract": "a",
         "year": 2025, "citation_count": 42, "doi": "10.1/a"},
    ],
    "gust rejection survey": [
        {"arxiv_id": "2501.00001", "title": "Gust-Robust Control", "abstract": "a"},
        {"arxiv_id": "2501.00002", "title": "A Survey of Wind Rejection", "abstract": "b"},
    ],
}


def fake_search(query: str) -> list[dict]:
    return [dict(p) for p in CORPUS.get(query, [])]


def test_run_searches_visits_every_query_in_the_plan():
    seen = []

    def spy(query):
        seen.append(query)
        return []

    plan = TemplatePlanner().plan("gust rejection")
    run_searches(plan, spy)
    assert seen == list(plan.queries)


def test_a_paper_found_by_two_queries_appears_once_carrying_both():
    plan = SearchPlan(question="gust rejection",
                      queries=("gust rejection", "gust rejection survey"))
    cands = run_searches(plan, fake_search)
    assert len(cands) == 2
    first = next(c for c in cands if c.pid == "2501.00001")
    assert first.queries == ("gust rejection", "gust rejection survey")


def test_search_candidates_are_at_graph_depth_zero():
    cands = run_searches(SearchPlan(question="q", queries=("gust rejection",)), fake_search)
    assert cands[0].origin == "search"
    assert cands[0].graph_depth == 0


def test_a_failing_source_does_not_abort_the_fan_out():
    def flaky(query):
        if query == "gust rejection":
            raise RuntimeError("rate limited")
        return fake_search(query)

    plan = SearchPlan(question="q", queries=("gust rejection", "gust rejection survey"))
    assert len(run_searches(plan, flaky)) == 2


def test_records_without_any_identity_are_dropped():
    plan = SearchPlan(question="q", queries=("x",))
    assert run_searches(plan, lambda q: [{"title": "", "abstract": "orphan"}]) == []


def test_to_paper_maps_the_source_dict_onto_the_domain_type():
    p = to_paper(Candidate(paper=CORPUS["gust rejection"][0]))
    assert p.paper_id == "2501.00001"
    assert p.title == "Gust-Robust Control"
    assert p.year == 2025
    assert p.citation_count == 42
    assert p.doi == "10.1/a"
    assert p.retracted is False


def test_to_paper_carries_the_retracted_flag_through():
    c = Candidate(paper={"arxiv_id": "x1", "title": "T", "retracted": True})
    assert to_paper(c).retracted is True


def test_candidates_are_saved_at_metadata_depth(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = run_searches(SearchPlan(question="q", queries=("gust rejection survey",)),
                             fake_search)
        assert save_candidates(conn, cands) == 2
        assert len(get_papers_by_depth(conn, "metadata")) == 2
        assert get_paper(conn, "2501.00002").title == "A Survey of Wind Rejection"
    finally:
        close_store(conn)


def test_saving_candidates_twice_does_not_duplicate_them(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = run_searches(SearchPlan(question="q", queries=("gust rejection survey",)),
                             fake_search)
        save_candidates(conn, cands)
        save_candidates(conn, cands)
        assert len(get_papers_by_depth(conn, "metadata")) == 2
    finally:
        close_store(conn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gather.py -v`
Expected: FAIL with `ImportError: cannot import name 'Candidate' from 'jarvis.gather'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/gather.py`. Add these imports at the top of the file:

```python
import sqlite3

from jarvis.citation_graph import paper_id
from jarvis.models import Paper
from jarvis.sources import dedup_papers
from jarvis.store import save_paper
```

```python
@dataclass(frozen=True)
class Candidate:
    """A gathered paper plus how it was found. The gate reads `graph_depth` as a signal."""
    paper: dict
    origin: str = "search"          # search | citation
    graph_depth: int = 0
    queries: tuple[str, ...] = ()

    @property
    def pid(self) -> str:
        return paper_id(self.paper)


def run_searches(plan: SearchPlan,
                 search_fn: Callable[[str], list[dict]]) -> list[Candidate]:
    """Run every query in the plan, dedup across them, and record which queries hit.

    A source that raises is skipped, never fatal: one API being rate-limited must not
    cost the whole gather run its recall.
    """
    by_pid: dict[str, list[str]] = {}
    papers: list[dict] = []
    for query in plan.queries:
        try:
            found = search_fn(query) or []
        except Exception:  # noqa: BLE001 - a dead source is a smaller loss than no gather
            continue
        for paper in found:
            pid = paper_id(paper)
            if not pid:
                continue
            if pid in by_pid:
                if query not in by_pid[pid]:
                    by_pid[pid].append(query)
                continue
            by_pid[pid] = [query]
            papers.append(paper)

    return [Candidate(paper=p, origin="search", graph_depth=0,
                      queries=tuple(by_pid[paper_id(p)]))
            for p in dedup_papers(papers) if paper_id(p)]


def to_paper(candidate: Candidate) -> Paper:
    """Source dict -> the frozen domain type. Unknown fields become their defaults."""
    p = candidate.paper
    year = p.get("year")
    return Paper(
        paper_id=candidate.pid,
        title=p.get("title", "") or "",
        authors=tuple(p.get("authors") or ()),
        year=int(year) if year else None,
        venue=p.get("venue", "") or "",
        doi=p.get("doi", "") or "",
        arxiv_id=p.get("arxiv_id", "") or "",
        s2_id=p.get("s2_id", "") or "",
        abstract=p.get("abstract", "") or "",
        citation_count=int(p.get("citation_count") or 0),
        retracted=bool(p.get("retracted", False)),
        source_path=p.get("pdf_url", "") or "",
    )


def save_candidates(conn: sqlite3.Connection, candidates: Sequence[Candidate]) -> int:
    """Persist every candidate at `metadata` depth. Nothing is read yet; nothing is lost."""
    for candidate in candidates:
        save_paper(conn, to_paper(candidate), depth="metadata")
    return len(candidates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gather.py -v && ruff check jarvis/gather.py tests/test_gather.py`
Expected: PASS (19 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/gather.py tests/test_gather.py
git commit -m "feat: multi-source candidate fan-out with dedup and persistence"
```

---

### Task 6: Citation-graph expansion with recorded depth

**Files:**
- Modify: `jarvis/gather.py` (append)
- Test: `tests/test_gather.py` (append)

**Interfaces:**
- Consumes: `CitationWalker` and `paper_id` from `jarvis.citation_graph`; `Candidate` (Task 5).
- Produces: `expand_citations(seeds, neighbors, score_fn, *, threshold=0.5, max_depth=2, budget=200, already_seen=None) -> list[Candidate]`, `gather(question, planner, search_fn, *, neighbors=None, score_fn=None, seed_limit=20, threshold=0.5, max_depth=2, budget=200) -> list[Candidate]`.

`CitationWalker` already exists, is tested, and BFS-expands with a budget — but its `walk()` returns a flat list with no depth. `expand_citations` runs it **one hop at a time**, feeding each level's output back as the next level's seeds, so every candidate carries the exact hop count the gate needs as its graph-proximity signal.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gather.py`:

```python
from jarvis.gather import expand_citations, gather

GRAPH = {
    "seed": [{"arxiv_id": "hop1", "title": "One Hop", "abstract": "a"}],
    "hop1": [{"arxiv_id": "hop2", "title": "Two Hops", "abstract": "b"}],
    "hop2": [{"arxiv_id": "hop3", "title": "Three Hops", "abstract": "c"}],
}


def _neighbors():
    def refs(pid):
        return [dict(p) for p in GRAPH.get(pid, [])]

    def cites(pid):
        return []

    return refs, cites


SEEDS = [{"arxiv_id": "seed", "title": "Seed", "abstract": "s"}]


def test_expansion_records_the_hop_count_as_graph_depth():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=2)
    by_pid = {c.pid: c.graph_depth for c in found}
    assert by_pid == {"hop1": 1, "hop2": 2}
    assert all(c.origin == "citation" for c in found)


def test_expansion_stops_at_max_depth():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=1)
    assert {c.pid for c in found} == {"hop1"}


def test_the_seeds_themselves_are_never_returned():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=3)
    assert "seed" not in {c.pid for c in found}


def test_low_scoring_neighbours_are_not_walked_through():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 0.1, threshold=0.5, max_depth=3)
    assert found == []


def test_expansion_respects_its_budget():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=3, budget=1)
    assert len(found) == 1


def test_already_seen_papers_are_not_re_surfaced():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=2,
                             already_seen={"hop1"})
    assert "hop1" not in {c.pid for c in found}


def test_gather_merges_search_hits_and_citation_expansion():
    def search(query):
        return [dict(SEEDS[0])] if query == "seed topic" else []

    cands = gather("seed topic", SearchPlan(question="seed topic", queries=("seed topic",)),
                   search, neighbors=_neighbors(), score_fn=lambda p: 1.0, max_depth=1)
    assert {c.pid for c in cands} == {"seed", "hop1"}
    assert next(c for c in cands if c.pid == "seed").graph_depth == 0
    assert next(c for c in cands if c.pid == "hop1").graph_depth == 1


def test_gather_accepts_a_planner_and_builds_its_own_plan():
    calls = []

    def search(query):
        calls.append(query)
        return []

    gather("gust rejection", TemplatePlanner(), search)
    assert calls == list(TemplatePlanner().plan("gust rejection").queries)


def test_gather_without_a_graph_is_just_the_searches():
    def search(query):
        return [dict(SEEDS[0])] if query == "q" else []

    cands = gather("q", SearchPlan(question="q", queries=("q",)), search)
    assert {c.pid for c in cands} == {"seed"}


def test_gather_seeds_expansion_from_the_top_scoring_hits_only():
    walked = []

    def refs(pid):
        walked.append(pid)
        return []

    def search(query):
        return [{"arxiv_id": f"p{i}", "title": f"T{i}", "abstract": "x", "citation_count": i}
                for i in range(5)]

    gather("q", SearchPlan(question="q", queries=("q",)), search,
           neighbors=(refs, lambda pid: []), score_fn=lambda p: 1.0, seed_limit=2)
    assert len(walked) == 2, "only the top `seed_limit` hits are expanded from"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gather.py -v`
Expected: FAIL with `ImportError: cannot import name 'expand_citations' from 'jarvis.gather'`

- [ ] **Step 3: Write the implementation**

Add `from jarvis.citation_graph import CitationWalker, paper_id` to the imports (extending the existing line), then append:

```python
def expand_citations(seeds: Sequence[dict],
                     neighbors: tuple[Callable[[str], list[dict]],
                                      Callable[[str], list[dict]]],
                     score_fn: Callable[[dict], float],
                     *, threshold: float = 0.5, max_depth: int = 2, budget: int = 200,
                     already_seen: set[str] | None = None) -> list[Candidate]:
    """Walk references and citations outward, recording the exact hop count per paper.

    `CitationWalker` BFS-expands correctly but returns a flat list, so this drives it one
    hop at a time and feeds each level back as the next level's seeds. Hop count is the
    gate's graph-proximity signal, and a flattened result would throw it away.

    PaperQA2 found citation traversal materially improved retrieval recall, and recall
    correlated with final answer accuracy — this is a recall tool, not a retrieval
    substrate (spec §7A, §11).
    """
    fetch_refs, fetch_citations = neighbors
    seen: set[str] = set(already_seen or ())
    seen |= {pid for pid in (paper_id(s) for s in seeds) if pid}

    out: list[Candidate] = []
    frontier = list(seeds)
    for depth in range(1, max_depth + 1):
        if not frontier or len(out) >= budget:
            break
        walker = CitationWalker(
            fetch_refs_fn=fetch_refs, fetch_citations_fn=fetch_citations,
            score_fn=score_fn, threshold=threshold, max_depth=1,
            budget=budget - len(out), already_seen=set(seen),
        )
        next_frontier: list[dict] = []
        for paper in walker.walk(frontier):
            pid = paper_id(paper)
            if not pid or pid in seen:
                continue
            seen.add(pid)
            out.append(Candidate(paper=paper, origin="citation", graph_depth=depth))
            next_frontier.append(paper)
            if len(out) >= budget:
                break
        frontier = next_frontier
    return out


def gather(question: str, planner: Planner | SearchPlan,
           search_fn: Callable[[str], list[dict]], *,
           neighbors: tuple[Callable[[str], list[dict]],
                            Callable[[str], list[dict]]] | None = None,
           score_fn: Callable[[dict], float] | None = None,
           seed_limit: int = 20, threshold: float = 0.5, max_depth: int = 2,
           budget: int = 200) -> list[Candidate]:
    """Stage A end to end: plan, search every query, then walk out from the best hits.

    `planner` may be a `Planner` or an already-built `SearchPlan`, so a caller can inspect
    or hand-edit the plan before spending API calls on it.
    """
    plan = planner if isinstance(planner, SearchPlan) else planner.plan(question)
    found = run_searches(plan, search_fn)
    if neighbors is None or score_fn is None or not found:
        return found

    seeds = [c.paper for c in found[:seed_limit]]
    expanded = expand_citations(seeds, neighbors, score_fn, threshold=threshold,
                                max_depth=max_depth, budget=budget,
                                already_seen={c.pid for c in found})
    return found + expanded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gather.py -v && ruff check jarvis/gather.py tests/test_gather.py`
Expected: PASS (29 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/gather.py tests/test_gather.py
git commit -m "feat: citation-graph expansion with recorded hop depth"
```

---

### Task 7: Gate signals

**Files:**
- Create: `jarvis/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Consumes: `Candidate` from `jarvis.gather`; `Embedder` from `jarvis.embed`; `cosine` and `paper_text` from `jarvis.scoring`; `jarvis.llm.chat` through injection.
- Produces: `Signals` (frozen: `embedding: float`, `graph: float`, `keyword: float`, `llm_vote: float`, method `as_dict() -> dict[str, float]`, property `best`), `Voter` protocol with `vote(question, paper) -> float`, `FakeVoter(mapping, default=0.0)`, `LLMVoter(router, chat_fn=None)`, `keyword_overlap(question, paper) -> float`, `graph_proximity(candidate, max_depth=2) -> float`, `score_signals(candidate, question, question_vector, embedder, voter=None) -> Signals`.

Four signals, each computed independently and each cheap. Every one of them is fallible; the union in Task 8 is what makes the set survivable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate.py
"""Stage B — the gate. Spec §7B: the recall ceiling of the whole system."""
import pytest

from jarvis.embed import FakeEmbedder
from jarvis.gate import (
    FakeVoter,
    LLMVoter,
    Signals,
    Voter,
    graph_proximity,
    keyword_overlap,
    score_signals,
)
from jarvis.gather import Candidate

RELEVANT = {"arxiv_id": "p1", "title": "Gust rejection for quadrotors",
            "abstract": "We reject wind gusts on a quadrotor."}
IRRELEVANT = {"arxiv_id": "p2", "title": "Protein folding with transformers",
              "abstract": "We fold proteins."}
QUESTION = "how do quadrotors reject wind gusts?"


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_keyword_overlap_is_higher_for_the_relevant_paper():
    assert keyword_overlap(QUESTION, RELEVANT) > keyword_overlap(QUESTION, IRRELEVANT)


def test_keyword_overlap_is_bounded_and_ignores_stopwords():
    assert 0.0 <= keyword_overlap(QUESTION, RELEVANT) <= 1.0
    assert keyword_overlap("the and of", RELEVANT) == 0.0


def test_keyword_overlap_of_an_empty_paper_is_zero():
    assert keyword_overlap(QUESTION, {"title": "", "abstract": ""}) == 0.0


def test_a_direct_search_hit_carries_no_citation_graph_evidence():
    assert graph_proximity(Candidate(paper=RELEVANT, origin="search")) == 0.0


def test_graph_proximity_is_strongest_one_hop_from_a_relevant_seed():
    one = graph_proximity(Candidate(paper=RELEVANT, origin="citation", graph_depth=1))
    two = graph_proximity(Candidate(paper=RELEVANT, origin="citation", graph_depth=2))
    assert one == 1.0
    assert 0.0 < two < one


def test_graph_proximity_never_goes_negative():
    assert graph_proximity(Candidate(paper=RELEVANT, origin="citation",
                                     graph_depth=99)) == 0.0


def test_fake_voter_satisfies_the_protocol():
    assert isinstance(FakeVoter({}), Voter)


def test_signals_expose_themselves_as_a_dict_for_the_log():
    s = Signals(embedding=0.8, graph=1.0, keyword=0.5, llm_vote=1.0)
    assert s.as_dict() == {"embedding": 0.8, "graph": 1.0, "keyword": 0.5, "llm_vote": 1.0}
    assert s.best == 1.0


def test_score_signals_scores_the_relevant_paper_above_the_irrelevant_one():
    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    voter = FakeVoter({"p1": 1.0, "p2": 0.0})

    good = score_signals(Candidate(paper=RELEVANT), QUESTION, qvec, embedder, voter)
    bad = score_signals(Candidate(paper=IRRELEVANT), QUESTION, qvec, embedder, voter)

    assert good.embedding > bad.embedding
    assert good.keyword > bad.keyword
    assert good.llm_vote == 1.0
    assert bad.llm_vote == 0.0


def test_score_signals_without_a_voter_records_zero_not_a_crash():
    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    assert score_signals(Candidate(paper=RELEVANT), QUESTION, qvec, embedder).llm_vote == 0.0


def test_a_voter_that_raises_scores_zero_and_does_not_abort_screening():
    class Boom:
        def vote(self, question, paper):
            raise RuntimeError("rate limited")

    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    s = score_signals(Candidate(paper=RELEVANT), QUESTION, qvec, embedder, Boom())
    assert s.llm_vote == 0.0
    assert s.embedding > 0.0, "the other three signals still stand"


def test_llm_voter_parses_a_score_and_clamps_it():
    for reply, expected in (({"relevant": True, "score": 0.9}, 0.9),
                            ({"relevant": True}, 1.0),
                            ({"relevant": False}, 0.0),
                            ({"relevant": True, "score": 5}, 1.0),
                            ({"relevant": True, "score": -3}, 0.0)):
        voter = LLMVoter(_Router(), chat_fn=lambda *a, **k: reply)
        assert voter.vote(QUESTION, RELEVANT) == pytest.approx(expected)


def test_llm_voter_scores_zero_on_failure_never_raising():
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    assert LLMVoter(_Router(), chat_fn=boom).vote(QUESTION, RELEVANT) == 0.0
    assert LLMVoter(_Router(), chat_fn=lambda *a, **k: "junk").vote(QUESTION, RELEVANT) == 0.0


def test_llm_voter_routes_to_the_screen_vote_task():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {"relevant": True}

    LLMVoter(_Router(), chat_fn=spy).vote(QUESTION, RELEVANT)
    assert seen["task"] == "screen_vote"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.gate'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/gate.py
"""Stage B — the gate (spec §7B). The recall ceiling of the whole system.

SESR-Eval benchmarked 9 LLMs on title-abstract screening in software engineering, the
closest published domain to ours: GPT-4o reached 0.66 recall, Claude 3.7 Sonnet 0.46, and
the verdict was that no model managed high recall with reasonable precision. Medical and
environmental domains report >95% for the identical task — this is domain-dependent and
ours is the bad one.

So the gate is never a single LLM judgment. It is a union of four cheap, independent
signals, calibrated per project against a hand-labeled seed, with three outcomes and no
`exclude`. `defer` demotes a paper to metadata depth; it never removes it.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.gather import Candidate
from jarvis.scoring import cosine, paper_text

_WORD = re.compile(r"[A-Za-z0-9]+")

# Words that carry no topical signal. Small on purpose: an over-eager stoplist silently
# strips domain terms and this stage cannot afford lost recall.
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from how in is it its of on or that the their "
    "there these this to what when where which who why with".split()
)

_VOTE_PROMPT = (
    "You are screening literature for a research question. Answer only about topical "
    "relevance, not quality.\n"
    "Return JSON: {{\"relevant\": true|false, \"score\": 0.0-1.0}}.\n"
    "When uncertain, answer relevant:true — a missed relevant paper costs far more here "
    "than an extra one.\n\n"
    "Question: {question}\n\nTitle: {title}\nAbstract: {abstract}"
)


@dataclass(frozen=True)
class Signals:
    """One paper's four gate scores, all on [0, 1]. Written verbatim to `screen_log`."""
    embedding: float = 0.0
    graph: float = 0.0
    keyword: float = 0.0
    llm_vote: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"embedding": self.embedding, "graph": self.graph,
                "keyword": self.keyword, "llm_vote": self.llm_vote}

    @property
    def best(self) -> float:
        return max(self.as_dict().values())


@runtime_checkable
class Voter(Protocol):
    def vote(self, question: str, paper: dict) -> float: ...


class FakeVoter:
    """Deterministic voter for tests, keyed by whatever `paper_id` resolves to."""

    def __init__(self, mapping: Mapping[str, float] | None = None,
                 default: float = 0.0) -> None:
        self._mapping = dict(mapping or {})
        self._default = default

    def vote(self, question: str, paper: dict) -> float:
        from jarvis.citation_graph import paper_id
        return self._mapping.get(paper_id(paper), self._default)


class LLMVoter:
    """One LLM vote — one signal of four, never the decision. Routed to the cheap tier."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None) -> None:
        self._router = router
        self._chat = chat_fn

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def vote(self, question: str, paper: dict) -> float:
        prompt = _VOTE_PROMPT.format(question=question, title=paper.get("title", ""),
                                     abstract=(paper.get("abstract", "") or "")[:4000])
        try:
            raw = self._chat_fn()(self._router, "screen_vote", prompt, json_mode=True)
        except Exception:  # noqa: BLE001 - a dead model is one signal down, not a decision
            return 0.0
        if not isinstance(raw, dict):
            return 0.0
        if not raw.get("relevant", False):
            return 0.0
        try:
            score = float(raw.get("score", 1.0))
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, score))


def _terms(text: str) -> set[str]:
    return {w for w in (t.lower() for t in _WORD.findall(text or "")) if w not in _STOPWORDS}


def keyword_overlap(question: str, paper: dict) -> float:
    """Fraction of the question's content words present in the title+abstract."""
    q_terms = _terms(question)
    if not q_terms:
        return 0.0
    return len(q_terms & _terms(paper_text(paper))) / len(q_terms)


def graph_proximity(candidate: Candidate, max_depth: int = 2) -> float:
    """How much citation-graph evidence supports this paper (spec §7B).

    A direct search hit scores **0.0**: it was found by keyword match, which is already
    the `keyword` signal's job, and it carries no graph evidence at all. Scoring it 1.0
    would make the union keep every search result unconditionally and switch adaptive
    depth off entirely — the gate would stop gating.

    A paper reached by walking outward from a high-scoring seed does carry evidence. One
    hop from a confirmed-relevant paper is the strongest form of it, decaying with
    distance and reaching 0.0 past `max_depth`.
    """
    if candidate.origin != "citation" or candidate.graph_depth < 1:
        return 0.0
    return max(0.0, 1.0 - (candidate.graph_depth - 1) / max(1, max_depth))


def score_signals(candidate: Candidate, question: str, question_vector,
                  embedder, voter: Voter | None = None,
                  max_depth: int = 2) -> Signals:
    """Compute all four signals for one candidate. No signal may abort the others."""
    try:
        vec = embedder.encode([paper_text(candidate.paper)])[0]
        embedding = max(0.0, cosine(vec, question_vector))
    except Exception:  # noqa: BLE001
        embedding = 0.0

    llm_vote = 0.0
    if voter is not None:
        try:
            llm_vote = max(0.0, min(1.0, float(voter.vote(question, candidate.paper))))
        except Exception:  # noqa: BLE001 - one signal failing must not lose the paper
            llm_vote = 0.0

    return Signals(
        embedding=embedding,
        graph=graph_proximity(candidate, max_depth),
        keyword=keyword_overlap(question, candidate.paper),
        llm_vote=llm_vote,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate.py -v && ruff check jarvis/gate.py tests/test_gate.py`
Expected: PASS (13 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/gate.py tests/test_gate.py
git commit -m "feat: four independent gate signals with per-signal failure isolation"
```

---

### Task 8: The union decision and the screening log

**Files:**
- Modify: `jarvis/gate.py` (append)
- Test: `tests/test_gate.py` (append)

**Interfaces:**
- Consumes: `Signals`, `score_signals`, `Voter` (Task 7); `save_screen_decision` and `set_depth` from `jarvis.store`.
- Produces: `DECISIONS = ("read_deep", "unsure", "defer")`, `KEPT = ("read_deep", "unsure")`, `Thresholds` (frozen: `embedding=0.35`, `graph=0.5`, `keyword=0.30`, `llm_vote=0.50`, `unsure_ratio=0.60`, method `as_dict()`), `decide(signals, thresholds=None) -> str`, `screen(conn, candidates, question, embedder, voter=None, thresholds=None, run_id="", max_depth=2) -> dict[str, str]`.

The union rule, stated once so no implementer has to infer it:

1. If **any** signal is at or above its threshold → `read_deep`.
2. Else if **any** signal is at or above `unsure_ratio ×` its threshold → `unsure`.
3. Else → `defer`.

`unsure` escalates to deep read alongside `read_deep` (that is what `KEPT` means, and it matches `jarvis.evaluate.KEPT_DECISIONS`, which already contains exactly `{"read_deep", "unsure"}`). `defer` writes the decision and leaves the paper at metadata depth — recoverable when the question shifts.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:

```python
from jarvis.evaluate import KEPT_DECISIONS, gate_recall
from jarvis.gate import DECISIONS, KEPT, Thresholds, decide, screen
from jarvis.gather import save_candidates
from jarvis.store import (
    close_store,
    get_papers_by_depth,
    get_screen_decisions,
    get_screen_signals,
    open_store,
)


def test_the_three_outcomes_are_exactly_the_spec_ones():
    assert DECISIONS == ("read_deep", "unsure", "defer")
    assert "exclude" not in DECISIONS


def test_kept_matches_what_the_eval_harness_already_counts_as_kept():
    assert set(KEPT) == KEPT_DECISIONS


def test_any_single_signal_over_threshold_keeps_the_paper():
    t = Thresholds()
    assert decide(Signals(embedding=0.9), t) == "read_deep"
    assert decide(Signals(graph=1.0), t) == "read_deep"
    assert decide(Signals(keyword=0.9), t) == "read_deep"
    assert decide(Signals(llm_vote=1.0), t) == "read_deep"


def test_the_gate_is_a_union_not_an_intersection():
    t = Thresholds()
    only_one = Signals(embedding=0.0, graph=0.0, keyword=0.0, llm_vote=1.0)
    assert decide(only_one, t) == "read_deep", "one signal is enough; intersection loses papers"


def test_a_near_miss_is_unsure_not_deferred():
    t = Thresholds(embedding=0.5, unsure_ratio=0.6)
    assert decide(Signals(embedding=0.35), t) == "unsure"


def test_nothing_anywhere_near_threshold_is_deferred():
    assert decide(Signals(), Thresholds()) == "defer"


def test_a_signal_exactly_at_threshold_is_kept():
    assert decide(Signals(embedding=0.35), Thresholds(embedding=0.35)) == "read_deep"


def test_screen_writes_every_decision_with_its_signals(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=RELEVANT), Candidate(paper=IRRELEVANT, graph_depth=9)]
        save_candidates(conn, cands)
        decisions = screen(conn, cands, QUESTION, FakeEmbedder(),
                           voter=FakeVoter({"p1": 1.0, "p2": 0.0}), run_id="r1")

        assert decisions["p1"] == "read_deep"
        assert get_screen_decisions(conn, "r1") == decisions
        assert set(get_screen_signals(conn, "r1")["p1"]) == {
            "embedding", "graph", "keyword", "llm_vote"}
    finally:
        close_store(conn)


def test_a_deferred_paper_stays_in_the_corpus_at_metadata_depth(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=IRRELEVANT)]
        save_candidates(conn, cands)
        decisions = screen(conn, cands, QUESTION, FakeEmbedder(), voter=FakeVoter({}),
                           run_id="r1")

        assert decisions["p2"] == "defer"
        assert [p.paper_id for p in get_papers_by_depth(conn, "metadata")] == ["p2"]
        assert get_papers_by_depth(conn, "deep") == []
    finally:
        close_store(conn)


def test_screening_is_rerunnable_without_refetching(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=RELEVANT)]
        save_candidates(conn, cands)
        unreachable = Thresholds(embedding=1.1, graph=1.1, keyword=1.1, llm_vote=1.1)
        screen(conn, cands, QUESTION, FakeEmbedder(), thresholds=unreachable,
               run_id="strict")
        screen(conn, cands, QUESTION, FakeEmbedder(), thresholds=Thresholds(), run_id="loose")

        assert get_screen_decisions(conn, "strict")["p1"] != "read_deep"
        assert get_screen_decisions(conn, "loose")["p1"] == "read_deep"
    finally:
        close_store(conn)


def test_gate_recall_reads_the_decisions_this_gate_produces(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=RELEVANT), Candidate(paper=IRRELEVANT, graph_depth=9)]
        save_candidates(conn, cands)
        decisions = screen(conn, cands, QUESTION, FakeEmbedder(),
                           voter=FakeVoter({"p1": 1.0}), run_id="r1")
        assert gate_recall(decisions, {"p1": True, "p2": False}) == 1.0
    finally:
        close_store(conn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'DECISIONS' from 'jarvis.gate'`

- [ ] **Step 3: Write the implementation**

Add `import sqlite3` and `from collections.abc import Sequence` to `jarvis/gate.py`'s imports, plus `from jarvis.store import save_screen_decision, set_depth`.

This task introduces a fourth ingest depth, `pending_deep`. Update the stale comment on the `depth` column in `jarvis/store.py`'s `_SCHEMA` to match — it currently reads `-- metadata | abstract | deep` and must become `-- metadata | abstract | pending_deep | deep`. The column is free text so nothing breaks either way, which is exactly why the comment will otherwise drift unnoticed.

Then append:

```python
DECISIONS = ("read_deep", "unsure", "defer")
KEPT = ("read_deep", "unsure")   # matches jarvis.evaluate.KEPT_DECISIONS


@dataclass(frozen=True)
class Thresholds:
    """Per-signal keep thresholds. Defaults are a starting point; calibrate per project.

    `unsure_ratio` is the fraction of a threshold below which a signal still counts as a
    near miss. Spec §7B: `unsure` escalates to deep read, so the band is deliberately wide.
    """
    embedding: float = 0.35
    graph: float = 0.50
    keyword: float = 0.30
    llm_vote: float = 0.50
    unsure_ratio: float = 0.60

    def as_dict(self) -> dict[str, float]:
        return {"embedding": self.embedding, "graph": self.graph,
                "keyword": self.keyword, "llm_vote": self.llm_vote}


def decide(signals: Signals, thresholds: Thresholds | None = None) -> str:
    """Union rule. Any one signal clearing its bar keeps the paper.

    Intersection would lose whatever any single signal misses, and §7B's whole point is
    that every individual signal in this domain misses a lot.
    """
    t = thresholds or Thresholds()
    scores = signals.as_dict()
    bars = t.as_dict()

    if any(scores[name] >= bars[name] for name in bars):
        return "read_deep"
    if any(scores[name] >= bars[name] * t.unsure_ratio for name in bars):
        return "unsure"
    return "defer"


def screen(conn: sqlite3.Connection, candidates: Sequence[Candidate], question: str,
           embedder, voter: Voter | None = None, thresholds: Thresholds | None = None,
           run_id: str = "", max_depth: int = 2) -> dict[str, str]:
    """Score and decide every candidate, logging per-signal scores for every one.

    Papers are never removed: `read_deep` and `unsure` are promoted to `pending_deep`
    depth for Stage C to pick up, `defer` is left at `metadata` depth and stays
    recoverable when the question shifts.
    """
    t = thresholds or Thresholds()
    question_vector = embedder.encode([question])[0]

    out: dict[str, str] = {}
    for candidate in candidates:
        signals = score_signals(candidate, question, question_vector, embedder, voter,
                                max_depth=max_depth)
        decision = decide(signals, t)
        save_screen_decision(conn, candidate.pid, decision, signals.as_dict(), run_id=run_id)
        set_depth(conn, candidate.pid, "pending_deep" if decision in KEPT else "metadata")
        out[candidate.pid] = decision
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate.py tests/test_evaluate.py -v && ruff check jarvis/gate.py tests/test_gate.py`
Expected: PASS (24 gate tests + existing eval tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/gate.py tests/test_gate.py
git commit -m "feat: union gate decision with three outcomes and audit log"
```

---

### Task 9: Threshold calibration against a labeled seed

**Files:**
- Modify: `jarvis/gate.py` (append)
- Test: `tests/test_gate.py` (append)

**Interfaces:**
- Consumes: `Thresholds`, `Signals`, `decide` (Task 8); `gate_recall` and `GATE_RECALL_TARGET` from `jarvis.evaluate`.
- Produces: `calibrate(signal_rows, labels, target_recall=0.95, floor=0.0) -> Thresholds`, `calibration_report(signal_rows, labels, thresholds) -> dict`.

The algorithm, stated once:

> For each signal independently, take the scores of the **labeled-relevant** papers, sort them ascending, and set that signal's threshold to the score at index `floor((1 - target_recall) × n)`. By construction at least `target_recall` of relevant papers clear that bar **on that signal alone.**
>
> A union's recall is at least the recall of its best member, so the calibrated union clears the target too — and usually beats it, because different signals miss different papers.

This is deterministic, has no hyperparameters beyond the target, and is directly checkable: `calibration_report` re-runs `decide` over the same rows and reports the achieved recall, which the tests assert against.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gate.py`:

```python
from jarvis.evaluate import GATE_RECALL_TARGET
from jarvis.gate import calibrate, calibration_report


def _rows(n_relevant=20, n_irrelevant=80):
    """Relevant papers score high on embedding; one outlier scores near zero on everything."""
    rows = {}
    for i in range(n_relevant):
        score = 0.9 if i > 0 else 0.05      # paper r0 is the hard one every signal nearly misses
        rows[f"r{i}"] = Signals(embedding=score, graph=0.0, keyword=score, llm_vote=0.0)
    for i in range(n_irrelevant):
        rows[f"n{i}"] = Signals(embedding=0.02, graph=0.0, keyword=0.02, llm_vote=0.0)
    return rows


def _labels(rows):
    return {pid: pid.startswith("r") for pid in rows}


def test_calibration_hits_the_recall_target():
    rows = _rows()
    thresholds = calibrate(rows, _labels(rows), target_recall=0.95)
    achieved = calibration_report(rows, _labels(rows), thresholds)["recall"]
    assert achieved >= 0.95


def test_calibration_defaults_to_the_specs_target():
    rows = _rows()
    default = calibrate(rows, _labels(rows))
    explicit = calibrate(rows, _labels(rows), target_recall=GATE_RECALL_TARGET)
    assert default == explicit
    assert GATE_RECALL_TARGET == 0.95


def test_a_perfect_recall_target_lowers_thresholds_to_admit_the_outlier():
    rows = _rows()
    strict = calibrate(rows, _labels(rows), target_recall=1.0)
    assert calibration_report(rows, _labels(rows), strict)["recall"] == 1.0
    assert strict.embedding <= 0.05


def test_a_looser_target_produces_higher_thresholds():
    rows = _rows()
    loose = calibrate(rows, _labels(rows), target_recall=0.90)
    strict = calibrate(rows, _labels(rows), target_recall=1.0)
    assert loose.embedding >= strict.embedding


def test_calibration_reports_precision_and_the_kept_count():
    rows = _rows()
    rep = calibration_report(rows, _labels(rows), calibrate(rows, _labels(rows)))
    assert 0.0 <= rep["precision"] <= 1.0
    assert rep["kept"] >= rep["relevant_kept"]
    assert rep["relevant"] == 20


def test_calibration_with_no_labeled_relevant_papers_returns_the_defaults():
    rows = _rows()
    assert calibrate(rows, {pid: False for pid in rows}) == Thresholds()


def test_calibration_with_no_rows_returns_the_defaults():
    assert calibrate({}, {}) == Thresholds()


def test_calibration_respects_a_floor_so_a_signal_never_admits_everything():
    rows = _rows()
    thresholds = calibrate(rows, _labels(rows), target_recall=1.0, floor=0.2)
    assert thresholds.embedding >= 0.2


def test_calibration_ignores_unlabelled_papers():
    rows = _rows()
    partial = {"r1": True, "n1": False}
    assert calibrate(rows, partial).embedding == pytest.approx(0.9)


def test_calibrated_thresholds_are_a_frozen_thresholds_instance():
    rows = _rows()
    assert isinstance(calibrate(rows, _labels(rows)), Thresholds)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate.py -v`
Expected: FAIL with `ImportError: cannot import name 'calibrate' from 'jarvis.gate'`

- [ ] **Step 3: Write the implementation**

Add `from jarvis.evaluate import GATE_RECALL_TARGET` to `jarvis/gate.py`'s imports, then append:

```python
def calibrate(signal_rows: Mapping[str, Signals], labels: Mapping[str, bool],
              target_recall: float = GATE_RECALL_TARGET, floor: float = 0.0) -> Thresholds:
    """Fit per-signal thresholds to a hand-labeled seed set (spec §7B, §10).

    For each signal, sort the labeled-relevant papers' scores ascending and take the one
    at index floor((1 - target) * n). At least `target_recall` of relevant papers clear
    that bar on that signal alone; a union's recall is at least its best member's, so the
    union clears the target too.

    `floor` keeps a degenerate signal (one where even irrelevant papers score high) from
    being tuned down to admitting the entire gather set.
    """
    relevant = [pid for pid, is_relevant in labels.items()
                if is_relevant and pid in signal_rows]
    if not relevant:
        return Thresholds()

    default = Thresholds()
    fitted: dict[str, float] = {}
    for name in default.as_dict():
        scores = sorted(signal_rows[pid].as_dict()[name] for pid in relevant)
        index = int((1.0 - target_recall) * len(scores))
        index = max(0, min(index, len(scores) - 1))
        fitted[name] = max(floor, scores[index])
    return Thresholds(unsure_ratio=default.unsure_ratio, **fitted)


def calibration_report(signal_rows: Mapping[str, Signals], labels: Mapping[str, bool],
                       thresholds: Thresholds) -> dict:
    """Re-run the decision over the seed set and report what the thresholds actually achieve.

    Never trust a fitted threshold without this: the fit is per-signal, the gate is a
    union, and the number that matters is the union's recall on real labels.
    """
    labeled = {pid: labels[pid] for pid in labels if pid in signal_rows}
    decisions = {pid: decide(signal_rows[pid], thresholds) for pid in labeled}
    kept = [pid for pid, d in decisions.items() if d in KEPT]
    relevant = [pid for pid, is_relevant in labeled.items() if is_relevant]
    relevant_kept = [pid for pid in kept if labeled[pid]]

    return {
        "recall": len(relevant_kept) / len(relevant) if relevant else 1.0,
        "precision": len(relevant_kept) / len(kept) if kept else 0.0,
        "kept": len(kept),
        "relevant": len(relevant),
        "relevant_kept": len(relevant_kept),
        "labeled": len(labeled),
        "thresholds": thresholds.as_dict(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gate.py -v && ruff check jarvis/gate.py tests/test_gate.py`
Expected: PASS (34 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/gate.py tests/test_gate.py
git commit -m "feat: per-project gate calibration against a labeled seed set"
```

---

### Task 10: The seed labeling tool

**Files:**
- Create: `jarvis/label.py`
- Test: `tests/test_label.py`

**Interfaces:**
- Consumes: `Candidate` from `jarvis.gather`.
- Produces: `sample_seed(candidates, size=100, seed=0) -> list[Candidate]`, `write_label_sheet(path, candidates) -> int`, `read_labels(path) -> dict[str, bool]`, `label_progress(path) -> dict`.

This is the piece spec build step 5 names ("Seed labeling tool") that the single-paper core could not build — it has nothing to label until gathering exists. `gate_recall` in `jarvis/evaluate.py` and `calibrate` in Task 9 both take a `labels` mapping; this is where that mapping comes from.

The format is JSONL, one paper per line, with `label: null` until a human edits it to `true`/`false`. Plain text so it can be labeled in any editor, diffed in git, and re-read incrementally.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_label.py
"""Seed labeling — the ground truth calibration and gate_recall are both measured against."""
import json

import pytest

from jarvis.gather import Candidate
from jarvis.label import label_progress, read_labels, sample_seed, write_label_sheet


def _candidates(n):
    return [Candidate(paper={"arxiv_id": f"p{i}", "title": f"Paper {i}",
                             "abstract": f"Abstract {i}"}) for i in range(n)]


def test_sampling_is_deterministic_for_a_given_seed():
    cands = _candidates(50)
    assert [c.pid for c in sample_seed(cands, size=10, seed=7)] == \
           [c.pid for c in sample_seed(cands, size=10, seed=7)]


def test_a_different_seed_gives_a_different_sample():
    cands = _candidates(50)
    assert [c.pid for c in sample_seed(cands, size=10, seed=1)] != \
           [c.pid for c in sample_seed(cands, size=10, seed=2)]


def test_sampling_more_than_exists_returns_everything():
    cands = _candidates(5)
    assert len(sample_seed(cands, size=100)) == 5


def test_the_default_seed_size_matches_the_spec():
    assert len(sample_seed(_candidates(500))) == 100


def test_the_sheet_is_one_json_object_per_line_with_a_null_label(tmp_path):
    path = tmp_path / "seed.jsonl"
    assert write_label_sheet(path, _candidates(3)) == 3

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["paper_id"] == "p0"
    assert rows[0]["title"] == "Paper 0"
    assert rows[0]["abstract"] == "Abstract 0"
    assert rows[0]["label"] is None


def test_reading_an_unlabelled_sheet_yields_nothing(tmp_path):
    path = tmp_path / "seed.jsonl"
    write_label_sheet(path, _candidates(3))
    assert read_labels(path) == {}


def test_reading_a_labelled_sheet_yields_booleans(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"paper_id": "p0", "label": true}\n'
        '{"paper_id": "p1", "label": false}\n'
        '{"paper_id": "p2", "label": null}\n',
        encoding="utf-8")
    assert read_labels(path) == {"p0": True, "p1": False}


def test_common_hand_typed_label_spellings_are_accepted(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"paper_id": "a", "label": "yes"}\n'
        '{"paper_id": "b", "label": "no"}\n'
        '{"paper_id": "c", "label": 1}\n'
        '{"paper_id": "d", "label": 0}\n'
        '{"paper_id": "e", "label": "Y"}\n',
        encoding="utf-8")
    assert read_labels(path) == {"a": True, "b": False, "c": True, "d": False, "e": True}


def test_a_malformed_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text('{"paper_id": "a", "label": true}\nnot json\n\n'
                    '{"label": true}\n', encoding="utf-8")
    assert read_labels(path) == {"a": True}


def test_progress_reports_how_much_is_left(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"paper_id": "a", "label": true}\n'
        '{"paper_id": "b", "label": null}\n'
        '{"paper_id": "c", "label": false}\n',
        encoding="utf-8")
    progress = label_progress(path)
    assert progress == {"total": 3, "labeled": 2, "relevant": 1, "remaining": 1}


def test_progress_on_a_missing_file_is_all_zeros(tmp_path):
    assert label_progress(tmp_path / "nope.jsonl")["total"] == 0


def test_writing_a_sheet_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "seed.jsonl"
    write_label_sheet(path, _candidates(1))
    assert path.is_file()


def test_labels_feed_straight_into_gate_recall(tmp_path):
    from jarvis.evaluate import gate_recall
    path = tmp_path / "seed.jsonl"
    path.write_text('{"paper_id": "a", "label": true}\n'
                    '{"paper_id": "b", "label": true}\n', encoding="utf-8")
    assert gate_recall({"a": "read_deep", "b": "defer"}, read_labels(path)) == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_label.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.label'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/label.py
"""Seed-set labeling — the ground truth the gate is calibrated and measured against.

Spec §7B calibrates the gate against a hand-labeled seed of ~100 papers; spec §10 measures
gate recall against the same labels. Both need a human to actually read titles and
abstracts, so the format is deliberately dumb: JSONL, one paper per line, `label` starts
null and a human edits it to true/false in any editor. Diffable, resumable, no UI.

Note (spec §10): human citation lists are not ground truth. Do not substitute a paper's own
bibliography for these labels.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from jarvis.gather import Candidate

DEFAULT_SEED_SIZE = 100

_TRUE = {"true", "yes", "y", "1", "relevant"}
_FALSE = {"false", "no", "n", "0", "irrelevant"}


def sample_seed(candidates: Sequence[Candidate], size: int = DEFAULT_SEED_SIZE,
                seed: int = 0) -> list[Candidate]:
    """A deterministic, order-independent sample of the gathered set.

    Hashing the id rather than shuffling means the same corpus yields the same seed set on
    any machine and in any gather order — a labeling session survives a re-gather.
    """
    def rank(candidate: Candidate) -> str:
        return hashlib.sha256(f"{seed}:{candidate.pid}".encode()).hexdigest()

    return sorted(candidates, key=rank)[:size]


def write_label_sheet(path: str | Path, candidates: Sequence[Candidate]) -> int:
    """Write an unlabelled JSONL sheet for a human to fill in. Returns rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "paper_id": c.pid,
            "title": c.paper.get("title", ""),
            "year": c.paper.get("year"),
            "abstract": (c.paper.get("abstract", "") or "")[:2000],
            "label": None,
        }, ensure_ascii=False)
        for c in candidates
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


def _rows(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.is_file():
        return []
    out: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue          # a hand-edited sheet will have typos; skip, never crash
        if isinstance(row, dict) and row.get("paper_id"):
            out.append(row)
    return out


def read_labels(path: str | Path) -> dict[str, bool]:
    """Read the completed labels. Unlabelled and unparseable rows are simply absent."""
    out: dict[str, bool] = {}
    for row in _rows(path):
        label = _as_bool(row.get("label"))
        if label is not None:
            out[str(row["paper_id"])] = label
    return out


def label_progress(path: str | Path) -> dict:
    """How much of the sheet is done — the only thing a labeling session needs to know."""
    rows = _rows(path)
    labels = read_labels(path)
    return {
        "total": len(rows),
        "labeled": len(labels),
        "relevant": sum(1 for v in labels.values() if v),
        "remaining": len(rows) - len(labels),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_label.py -v && ruff check jarvis/label.py tests/test_label.py`
Expected: PASS (13 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/label.py tests/test_label.py
git commit -m "feat: seed-set sampling and hand-label round-trip"
```

---

### Task 11: Stage C — deep read into the corpus

**Files:**
- Create: `jarvis/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `Parser` from `jarvis.parse`; `build_units` and `DEFAULT_MAX_TOKENS` from `jarvis.units`; `apply_prefixes` and `TemplatePrefix` from `jarvis.context`; `index_units` and `Embedder` from `jarvis.embed`; `index_units_fts` from `jarvis.index`; `save_paper`, `save_units`, `set_depth`, `get_paper` from `jarvis.store`; `Candidate`, `to_paper` from `jarvis.gather`; `KEPT` from `jarvis.gate`.
- Produces: `IngestResult` (frozen: `paper_id: str`, `units: int`, `ok: bool`, `error: str = ""`), `ingest_paper(conn, paper, source_path, parser, embedder, prefix_generator=None, max_tokens=DEFAULT_MAX_TOKENS) -> IngestResult`, `ingest_decided(conn, decisions, candidates, parser, embedder, prefix_generator=None, path_for=None) -> list[IngestResult]`.

This is the pipeline the first plan built, wired end to end and driven from a decision set. Spec §14: **never silently ingest an empty parse.** A parse that yields no blocks is an error, recorded as one, with the paper left at its previous depth rather than marked deep and quietly empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
"""Stage C — deep read (spec §7C). Parse, type, prefix, embed, index."""
import pytest

from jarvis.embed import FakeEmbedder
from jarvis.gather import Candidate
from jarvis.ingest import IngestResult, ingest_decided, ingest_paper
from jarvis.models import Block, Paper
from jarvis.parse import FakeParser
from jarvis.retrieve import search
from jarvis.store import close_store, get_paper, get_papers_by_depth, get_units, open_store

BLOCKS = [
    Block(kind="heading", text="Results", page=2, section_path=("Results",)),
    Block(kind="paragraph", text="As shown in Table 1, we reach 94.2% accuracy under gust.",
          page=2, section_path=("Results",)),
    Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
          page=2, section_path=("Results",), label="Table 1"),
    Block(kind="caption", text="Table 1: Accuracy under wind.", page=2,
          section_path=("Results",), label="Table 1"),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / "c.db")
    yield c
    close_store(c)


def test_ingest_stores_layer_zero_and_marks_the_paper_deep(conn):
    result = ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    assert result.ok is True
    assert result.units > 0
    assert [p.paper_id for p in get_papers_by_depth(conn, "deep")] == ["p1"]


def test_ingest_produces_typed_units_with_prefixes(conn):
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    units = get_units(conn, "p1")
    assert {u.type.value for u in units} >= {"prose", "table"}
    assert all(u.context_prefix for u in units)


def test_the_prefix_never_leaks_into_verbatim_text(conn):
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    for unit in get_units(conn, "p1"):
        assert unit.context_prefix not in unit.verbatim_text


def test_an_ingested_paper_is_immediately_retrievable(conn):
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    hits = search(conn, "accuracy under wind", FakeEmbedder(), limit=3)
    assert any("94.2" in u.verbatim_text for u in hits)


def test_an_empty_parse_is_an_error_and_never_marked_deep(conn):
    result = ingest_paper(conn, PAPER, "p.pdf", FakeParser([]), FakeEmbedder())
    assert result.ok is False
    assert "empty parse" in result.error
    assert get_papers_by_depth(conn, "deep") == []


def test_a_parser_that_raises_is_recorded_not_propagated(conn):
    class Broken:
        def parse(self, path, paper_id):
            raise RuntimeError("corrupt pdf")

    result = ingest_paper(conn, PAPER, "p.pdf", Broken(), FakeEmbedder())
    assert result.ok is False
    assert "corrupt pdf" in result.error
    assert get_papers_by_depth(conn, "deep") == []


def test_reingesting_the_same_paper_is_idempotent(conn):
    first = ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    second = ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    assert first.units == second.units
    assert len(get_units(conn, "p1")) == first.units


def test_ingest_decided_reads_only_the_kept_papers(conn):
    cands = [
        Candidate(paper={"arxiv_id": "p1", "title": "Keep", "pdf_url": "a.pdf"}),
        Candidate(paper={"arxiv_id": "p2", "title": "Also keep", "pdf_url": "b.pdf"}),
        Candidate(paper={"arxiv_id": "p3", "title": "Defer", "pdf_url": "c.pdf"}),
    ]
    decisions = {"p1": "read_deep", "p2": "unsure", "p3": "defer"}
    results = ingest_decided(conn, decisions, cands, FakeParser(BLOCKS), FakeEmbedder())

    assert {r.paper_id for r in results} == {"p1", "p2"}
    assert {p.paper_id for p in get_papers_by_depth(conn, "deep")} == {"p1", "p2"}


def test_unsure_papers_are_read_exactly_like_read_deep_ones(conn):
    cands = [Candidate(paper={"arxiv_id": "p9", "title": "Unsure", "pdf_url": "u.pdf"})]
    results = ingest_decided(conn, {"p9": "unsure"}, cands, FakeParser(BLOCKS),
                             FakeEmbedder())
    assert results[0].ok is True
    assert results[0].units > 0


def test_a_deferred_paper_keeps_its_metadata_row(conn):
    cands = [Candidate(paper={"arxiv_id": "p3", "title": "Deferred", "abstract": "abs"})]
    ingest_decided(conn, {"p3": "defer"}, cands, FakeParser(BLOCKS), FakeEmbedder())
    assert get_paper(conn, "p3") is None or get_units(conn, "p3") == []


def test_one_broken_paper_does_not_stop_the_batch(conn):
    class Flaky:
        def parse(self, path, paper_id):
            if paper_id == "p1":
                raise RuntimeError("corrupt")
            return FakeParser(BLOCKS).parse(path, paper_id)

    cands = [Candidate(paper={"arxiv_id": "p1", "title": "A", "pdf_url": "a.pdf"}),
             Candidate(paper={"arxiv_id": "p2", "title": "B", "pdf_url": "b.pdf"})]
    results = ingest_decided(conn, {"p1": "read_deep", "p2": "read_deep"}, cands,
                             Flaky(), FakeEmbedder())
    assert {r.paper_id: r.ok for r in results} == {"p1": False, "p2": True}


def test_ingest_result_is_frozen():
    with pytest.raises(Exception):
        IngestResult(paper_id="p", units=1, ok=True).units = 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.ingest'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/ingest.py
"""Stage C — deep read (spec §7C).

For every paper the gate kept: parse (Layer 0) -> build typed units (Layer 1) ->
contextual prefixes -> embed -> index. This is the pipeline the single-paper core built,
wired end to end and driven from a decision set.

Spec §14: never silently ingest an empty parse. A paper that produces no blocks is
recorded as a parse failure and left at its previous depth, because a paper marked `deep`
with nothing in it is indistinguishable from a paper that genuinely says nothing — and
the second one is a claim about the literature.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import Embedder, index_units
from jarvis.gate import KEPT
from jarvis.gather import Candidate, to_paper
from jarvis.index import index_units_fts
from jarvis.models import Paper, Unit
from jarvis.parse import Parser
from jarvis.store import save_paper, save_units, set_depth
from jarvis.units import DEFAULT_MAX_TOKENS, build_units


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    units: int = 0
    ok: bool = False
    error: str = ""


def ingest_paper(conn: sqlite3.Connection, paper: Paper, source_path: str,
                 parser: Parser, embedder: Embedder, prefix_generator=None,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> IngestResult:
    """One paper, all the way into the corpus. Never raises; failures are returned."""
    try:
        parsed = parser.parse(source_path, paper.paper_id)
    except Exception as exc:  # noqa: BLE001 - a bad PDF is data, not a crash
        return IngestResult(paper_id=paper.paper_id, ok=False, error=f"parse failed: {exc}")

    if not parsed.blocks or not parsed.raw_text.strip():
        return IngestResult(paper_id=paper.paper_id, ok=False,
                            error="empty parse — escalate to a stronger parser (spec §5)")

    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")

    units: Sequence[Unit] = build_units(parsed, max_tokens=max_tokens)
    units = apply_prefixes(units, paper, prefix_generator or TemplatePrefix())
    save_units(conn, list(units))
    index_units_fts(conn, units)
    index_units(conn, units, embedder)

    return IngestResult(paper_id=paper.paper_id, units=len(units), ok=True)


def ingest_decided(conn: sqlite3.Connection, decisions: Mapping[str, str],
                   candidates: Sequence[Candidate], parser: Parser, embedder: Embedder,
                   prefix_generator=None,
                   path_for: Callable[[Candidate], str] | None = None,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> list[IngestResult]:
    """Deep-read every kept paper. `unsure` is read exactly like `read_deep` (spec §7B).

    One paper failing never stops the batch — a corrupt PDF in a 300-paper gather must
    cost one paper, not the run.
    """
    resolve = path_for or (lambda c: c.paper.get("pdf_url", "") or c.paper.get("url", ""))
    out: list[IngestResult] = []
    for candidate in candidates:
        if decisions.get(candidate.pid) not in KEPT:
            continue
        out.append(ingest_paper(conn, to_paper(candidate), resolve(candidate), parser,
                                embedder, prefix_generator, max_tokens=max_tokens))
    return out


def failed(results: Sequence[IngestResult]) -> list[IngestResult]:
    """The parse-failure log spec §14 requires. Never let these disappear silently."""
    return [r for r in results if not r.ok]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ingest.py -v && ruff check jarvis/ingest.py tests/test_ingest.py`
Expected: PASS (12 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/ingest.py tests/test_ingest.py
git commit -m "feat: stage c deep read with parse-failure isolation"
```

---

### Task 12: Layer 2 cards with verified bindings

**Files:**
- Create: `jarvis/card.py`
- Test: `tests/test_card.py`

**Interfaces:**
- Consumes: `Card`, `CardField`, `Paper`, `Unit`, `Claim` from `jarvis.models`; `quote_is_grounded` from `jarvis.verify`; `get_units` from `jarvis.store`; `jarvis.llm.chat` through injection.
- Produces: `CardExtractor` protocol with `extract(paper, units) -> Card`, `FakeCardExtractor(cards)`, `LLMCardExtractor(router, chat_fn=None, max_units=40)`, `verify_card(conn, card) -> Card`, `unverified_fields(card) -> list[tuple[str, CardField]]`, `extract_and_verify(conn, paper, extractor) -> Card`.

Spec §5, Layer 2, is unusually specific about why this component is *demoted*: *Diagnosing Structural Failures in LLM-Based Evidence Extraction* (arXiv 2602.10881) shows LLMs handle isolated entity extraction well but fail at preserving **roles, methods, and effect-size attribution** — the relational binding, which is exactly the part that carries the value. The counter-evidence (otto-SR at 93.1% extraction accuracy versus 79.7% for dual human reviewers) came from a tight schema with explicit verification.

So: the card is **never the ground for a claim**, every field carries a `unit_id` and a verbatim `quote`, and `verify_card` sets `binding_verified` by running each quote through the same deterministic Layer 0 matcher the verification stage uses. Unverified bindings are surfaced, not dropped.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_card.py
"""Layer 2 — the paper card. An index over evidence, never a replacement for it."""
import pytest

from jarvis.card import (
    CardExtractor,
    FakeCardExtractor,
    LLMCardExtractor,
    extract_and_verify,
    unverified_fields,
    verify_card,
)
from jarvis.embed import FakeEmbedder
from jarvis.ingest import ingest_paper
from jarvis.models import Block, Card, CardField, Paper
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store

BLOCKS = [
    Block(kind="heading", text="Results", page=2, section_path=("Results",)),
    Block(kind="paragraph", text="Our controller reaches 94.2% accuracy on the KITTI set.",
          page=2, section_path=("Results",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    yield conn
    close_store(conn)


def test_fake_extractor_satisfies_the_protocol():
    assert isinstance(FakeCardExtractor({}), CardExtractor)


def test_a_real_quote_verifies_the_binding(corpus):
    unit = get_units(corpus, "p1")[0]
    card = Card(paper_id="p1",
                metrics=(CardField(value="94.2", unit_id=unit.unit_id,
                                   quote="reaches 94.2% accuracy"),))
    verified = verify_card(corpus, card)
    assert verified.metrics[0].binding_verified is True


def test_a_fabricated_quote_leaves_the_binding_unverified(corpus):
    unit = get_units(corpus, "p1")[0]
    card = Card(paper_id="p1",
                metrics=(CardField(value="99.9", unit_id=unit.unit_id,
                                   quote="reaches 99.9% accuracy"),))
    verified = verify_card(corpus, card)
    assert verified.metrics[0].binding_verified is False


def test_a_quote_pointing_at_a_nonexistent_unit_is_unverified(corpus):
    card = Card(paper_id="p1",
                metrics=(CardField(value="94.2", unit_id="nope", quote="94.2"),))
    assert verify_card(corpus, card).metrics[0].binding_verified is False


def test_every_field_kind_gets_verified_not_just_metrics(corpus):
    unit = get_units(corpus, "p1")[0]
    good = CardField(value="v", unit_id=unit.unit_id, quote="Our controller")
    bad = CardField(value="v", unit_id=unit.unit_id, quote="never written")
    card = Card(paper_id="p1", problem=good, method=bad, datasets=(good,), claims=(bad,))

    verified = verify_card(corpus, card)
    assert verified.problem.binding_verified is True
    assert verified.method.binding_verified is False
    assert verified.datasets[0].binding_verified is True
    assert verified.claims[0].binding_verified is False


def test_unverified_fields_are_surfaced_with_their_names(corpus):
    unit = get_units(corpus, "p1")[0]
    card = verify_card(corpus, Card(
        paper_id="p1",
        problem=CardField("a", unit.unit_id, "Our controller"),
        metrics=(CardField("b", unit.unit_id, "fabricated"),),
    ))
    names = [name for name, _ in unverified_fields(card)]
    assert names == ["metrics"]


def test_a_fully_verified_card_surfaces_nothing(corpus):
    unit = get_units(corpus, "p1")[0]
    card = verify_card(corpus, Card(paper_id="p1",
                                    problem=CardField("a", unit.unit_id, "Our controller")))
    assert unverified_fields(card) == []


def test_verification_never_deletes_an_unverified_field(corpus):
    unit = get_units(corpus, "p1")[0]
    card = verify_card(corpus, Card(
        paper_id="p1", metrics=(CardField("x", unit.unit_id, "fabricated"),)))
    assert len(card.metrics) == 1, "unverified is surfaced as such, never silently dropped"


def test_llm_extractor_builds_a_card_from_model_json(corpus):
    unit = get_units(corpus, "p1")[0]
    reply = {
        "problem": {"value": "gust rejection", "unit_id": unit.unit_id,
                    "quote": "Our controller"},
        "metrics": [{"value": "94.2", "unit_id": unit.unit_id,
                     "quote": "reaches 94.2% accuracy"}],
        "datasets": [{"value": "KITTI", "unit_id": unit.unit_id, "quote": "KITTI set"}],
    }
    card = LLMCardExtractor(_Router(), chat_fn=lambda *a, **k: reply).extract(
        PAPER, get_units(corpus, "p1"))
    assert card.problem.value == "gust rejection"
    assert card.metrics[0].value == "94.2"
    assert card.method is None


def test_llm_extractor_drops_fields_citing_a_unit_that_does_not_exist(corpus):
    reply = {"metrics": [{"value": "94.2", "unit_id": "hallucinated", "quote": "q"}]}
    card = LLMCardExtractor(_Router(), chat_fn=lambda *a, **k: reply).extract(
        PAPER, get_units(corpus, "p1"))
    assert card.metrics == ()


def test_llm_extractor_returns_an_empty_card_on_failure(corpus):
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    card = LLMCardExtractor(_Router(), chat_fn=boom).extract(PAPER, get_units(corpus, "p1"))
    assert card == Card(paper_id="p1")


def test_llm_extractor_routes_to_card_extraction(corpus):
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {}

    LLMCardExtractor(_Router(), chat_fn=spy).extract(PAPER, get_units(corpus, "p1"))
    assert seen["task"] == "card_extraction"


def test_extract_and_verify_persists_a_verified_card(corpus):
    from jarvis.store import get_card
    unit = get_units(corpus, "p1")[0]
    card = Card(paper_id="p1",
                metrics=(CardField("94.2", unit.unit_id, "reaches 94.2% accuracy"),))
    extract_and_verify(corpus, PAPER, FakeCardExtractor({"p1": card}))

    stored = get_card(corpus, "p1")
    assert stored.metrics[0].binding_verified is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_card.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.card'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/card.py
"""Layer 2 — the paper card (spec §5).

Job: coverage bookkeeping and cross-paper comparison. **Never the ground for a claim.**

The card is deliberately demoted. LLMs extract isolated entities well but fail at
preserving roles, methods, and effect-size attribution — the relational binding, which is
exactly what makes a card worth having (arXiv 2602.10881). The counter-evidence for
keeping cards at all (otto-SR: 93.1% extraction accuracy versus 79.7% for dual human
reviewers) came from a tight schema with explicit verification, so that is what this is:
every field carries a unit_id and a verbatim quote, every quote is checked against Layer 0
by the same deterministic matcher verification uses, and unverified bindings are surfaced
rather than dropped.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from jarvis.models import Card, CardField, Claim, Paper, Unit
from jarvis.store import get_units, save_card
from jarvis.verify import quote_is_grounded

SINGLE_FIELDS = ("problem", "method")
LIST_FIELDS = ("datasets", "metrics", "claims", "limitations")

_EXTRACT_PROMPT = (
    "Extract a structured card from these evidence units of one paper.\n"
    "Return JSON with keys: problem, method (objects) and datasets, metrics, claims, "
    "limitations (arrays).\n"
    "Every object is {{\"value\": ..., \"unit_id\": ..., \"quote\": ...}} where `quote` is "
    "copied EXACTLY from the unit text — character for character, no paraphrase, no "
    "ellipsis. A quote that is not verbatim will be rejected automatically.\n"
    "Omit any field you cannot ground in a quote. An absent field is correct; an invented "
    "one is not.\n\n"
    "Paper: {title} ({year})\n\n{units}"
)


@runtime_checkable
class CardExtractor(Protocol):
    def extract(self, paper: Paper, units: Sequence[Unit]) -> Card: ...


class FakeCardExtractor:
    """Deterministic extractor for tests, keyed by paper_id."""

    def __init__(self, cards: Mapping[str, Card] | None = None) -> None:
        self._cards = dict(cards or {})

    def extract(self, paper: Paper, units: Sequence[Unit]) -> Card:
        return self._cards.get(paper.paper_id, Card(paper_id=paper.paper_id))


def _to_field(data, known_unit_ids: set[str]) -> CardField | None:
    """One JSON object -> a CardField, or None when it cannot be grounded.

    A field citing a unit_id that does not exist is a hallucinated citation. Dropping it
    here costs a row of bookkeeping; keeping it would put a fabricated pointer into the
    only structure the system uses for cross-paper comparison.
    """
    if not isinstance(data, dict):
        return None
    unit_id = str(data.get("unit_id", "") or "")
    quote = str(data.get("quote", "") or "")
    value = str(data.get("value", "") or "")
    if not unit_id or unit_id not in known_unit_ids or not quote or not value:
        return None
    return CardField(value=value, unit_id=unit_id, quote=quote)


class LLMCardExtractor:
    """Model-driven extraction, routed to the long-context reader tier."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None,
                 max_units: int = 40) -> None:
        self._router = router
        self._chat = chat_fn
        self._max_units = max_units

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def extract(self, paper: Paper, units: Sequence[Unit]) -> Card:
        empty = Card(paper_id=paper.paper_id)
        selected = list(units)[:self._max_units]
        if not selected:
            return empty

        rendered = "\n\n".join(f"[{u.unit_id}]\n{u.verbatim_text}" for u in selected)
        prompt = _EXTRACT_PROMPT.format(title=paper.title, year=paper.year or "n.d.",
                                        units=rendered)
        try:
            raw = self._chat_fn()(self._router, "card_extraction", prompt, json_mode=True)
        except Exception:  # noqa: BLE001 - an unextractable card is not a failed ingest
            return empty
        if not isinstance(raw, dict):
            return empty

        known = {u.unit_id for u in units}
        kwargs = {name: _to_field(raw.get(name), known) for name in SINGLE_FIELDS}
        for name in LIST_FIELDS:
            items = raw.get(name) or []
            fields = (_to_field(item, known) for item in items) if isinstance(items, list) \
                else ()
            kwargs[name] = tuple(f for f in fields if f is not None)
        return Card(paper_id=paper.paper_id, **kwargs)


def _verify_field(conn: sqlite3.Connection, field: CardField | None) -> CardField | None:
    if field is None:
        return None
    claim = Claim(claim_id=f"card:{field.unit_id}", text=field.value,
                  unit_id=field.unit_id, quote=field.quote)
    return replace(field, binding_verified=quote_is_grounded(conn, claim))


def verify_card(conn: sqlite3.Connection, card: Card) -> Card:
    """Set `binding_verified` on every field by matching its quote against Layer 0.

    Deterministic, free, no model — the same stage-1 matcher `verify_claim` uses.
    """
    kwargs = {name: _verify_field(conn, getattr(card, name)) for name in SINGLE_FIELDS}
    kwargs.update({
        name: tuple(f for f in (_verify_field(conn, x) for x in getattr(card, name))
                    if f is not None)
        for name in LIST_FIELDS
    })
    return Card(paper_id=card.paper_id, **kwargs)


def unverified_fields(card: Card) -> list[tuple[str, CardField]]:
    """Every field whose quote did not match Layer 0, for surfacing as unverified."""
    out: list[tuple[str, CardField]] = []
    for name in SINGLE_FIELDS:
        field = getattr(card, name)
        if field is not None and not field.binding_verified:
            out.append((name, field))
    for name in LIST_FIELDS:
        out += [(name, f) for f in getattr(card, name) if not f.binding_verified]
    return out


def extract_and_verify(conn: sqlite3.Connection, paper: Paper,
                       extractor: CardExtractor) -> Card:
    """Extract, verify every binding, persist. The only way a card should ever be written."""
    card = verify_card(conn, extractor.extract(paper, get_units(conn, paper.paper_id)))
    save_card(conn, card)
    return card
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_card.py -v && ruff check jarvis/card.py tests/test_card.py`
Expected: PASS (13 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/card.py tests/test_card.py
git commit -m "feat: layer 2 card extraction with verified quote bindings"
```

---

### Task 13: End to end — question in, corpus out

**Files:**
- Create: `tests/test_gather_end_to_end.py`
- Modify: `jarvis/__init__.py`

**Interfaces:**
- Consumes: everything this plan built.
- Produces: the extended public surface of the `jarvis` package.

This is the proof the plan exists to produce: one question, gathered into candidates, screened by a calibrated union gate, deep-read into the corpus, carded — and then the *measurement*, because spec §7B's whole argument is that a gate you cannot measure is a gate you cannot trust.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gather_end_to_end.py
"""The proof this plan exists to produce: a question becomes a measured corpus."""
import pytest

from jarvis.card import FakeCardExtractor, extract_and_verify
from jarvis.embed import FakeEmbedder
from jarvis.evaluate import gate_recall
from jarvis.gate import FakeVoter, Signals, calibrate, calibration_report, screen, score_signals
from jarvis.gather import SearchPlan, gather, save_candidates
from jarvis.ingest import failed, ingest_decided
from jarvis.label import read_labels, sample_seed, write_label_sheet
from jarvis.models import Block, Card, CardField, Paper
from jarvis.parse import FakeParser
from jarvis.retrieve import search
from jarvis.store import close_store, get_papers_by_depth, get_screen_signals, open_store

QUESTION = "how do quadrotors reject wind gusts?"

RELEVANT = [
    {"arxiv_id": "r1", "title": "Gust rejection for quadrotors",
     "abstract": "Wind gusts disturb quadrotors; we reject them.", "year": 2025},
    {"arxiv_id": "r2", "title": "Wind disturbance attenuation in UAVs",
     "abstract": "Quadrotors reject wind using adaptive control.", "year": 2024},
]
IRRELEVANT = [
    {"arxiv_id": "n1", "title": "Protein folding", "abstract": "We fold proteins.",
     "year": 2025},
    {"arxiv_id": "n2", "title": "Compiler optimization", "abstract": "We optimize loops.",
     "year": 2023},
]
CITED = {"arxiv_id": "r3", "title": "Gust tolerance benchmarks",
         "abstract": "Benchmarks for quadrotor gust tolerance.", "year": 2023}

BLOCKS = [
    Block(kind="heading", text="Results", page=2, section_path=("Results",)),
    Block(kind="paragraph",
          text="As shown in Table 1, the controller holds 94.2% tracking accuracy in gusts.",
          page=2, section_path=("Results",)),
    Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
          page=2, section_path=("Results",), label="Table 1"),
    Block(kind="caption", text="Table 1: Tracking accuracy under wind.", page=2,
          section_path=("Results",), label="Table 1"),
]

LABELS = {"r1": True, "r2": True, "r3": True, "n1": False, "n2": False}


def search_fn(query: str) -> list[dict]:
    return [dict(p) for p in RELEVANT + IRRELEVANT]


def neighbors():
    return (lambda pid: [dict(CITED)] if pid == "r1" else [], lambda pid: [])


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    yield conn
    close_store(conn)


@pytest.fixture
def gathered():
    return gather(QUESTION, SearchPlan(question=QUESTION, queries=(QUESTION,)), search_fn,
                  neighbors=neighbors(), score_fn=lambda p: 1.0, max_depth=1)


def test_gathering_finds_the_searched_and_the_cited_papers(gathered):
    assert {c.pid for c in gathered} == {"r1", "r2", "n1", "n2", "r3"}
    assert next(c for c in gathered if c.pid == "r3").graph_depth == 1


def test_the_gate_keeps_every_hand_labelled_relevant_paper(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0, "r2": 1.0, "r3": 1.0}), run_id="run1")
    assert gate_recall(decisions, LABELS) >= 0.95


def test_every_decision_carries_its_four_signals(corpus, gathered):
    save_candidates(corpus, gathered)
    screen(corpus, gathered, QUESTION, FakeEmbedder(), run_id="run1")
    logged = get_screen_signals(corpus, "run1")
    assert set(logged) == {c.pid for c in gathered}
    assert all(set(v) == {"embedding", "graph", "keyword", "llm_vote"}
               for v in logged.values())


def test_calibration_from_labels_meets_the_target(corpus, gathered):
    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    rows = {c.pid: score_signals(c, QUESTION, qvec, embedder,
                                 FakeVoter({"r1": 1.0, "r2": 1.0, "r3": 1.0}))
            for c in gathered}
    thresholds = calibrate(rows, LABELS)
    assert calibration_report(rows, LABELS, thresholds)["recall"] >= 0.95


def test_the_label_sheet_round_trips_through_a_file(tmp_path, gathered):
    path = tmp_path / "seed.jsonl"
    written = write_label_sheet(path, sample_seed(gathered, size=3))
    assert written == 3
    assert read_labels(path) == {}, "a fresh sheet is unlabelled by construction"


def test_kept_papers_are_deep_read_and_deferred_ones_are_not(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0, "r2": 1.0, "r3": 1.0}), run_id="run1")
    results = ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    assert failed(results) == []
    deep = {p.paper_id for p in get_papers_by_depth(corpus, "deep")}
    assert {"r1", "r2", "r3"} <= deep


def test_no_paper_is_ever_removed_from_the_corpus(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(), run_id="run1")
    ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    everywhere = {p.paper_id for depth in ("metadata", "pending_deep", "deep")
                  for p in get_papers_by_depth(corpus, depth)}
    assert everywhere == {c.pid for c in gathered}, "defer is demotion, never deletion"


def test_the_ingested_corpus_is_retrievable_and_the_card_is_verified(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0}), run_id="run1")
    ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    hits = search(corpus, "tracking accuracy under wind", FakeEmbedder(), limit=5)
    assert any("94.2" in u.verbatim_text for u in hits)

    unit = next(u for u in hits if "94.2" in u.verbatim_text)
    card = Card(paper_id=unit.paper_id,
                metrics=(CardField("94.2", unit.unit_id, "| ours | 94.2 |"),))
    verified = extract_and_verify(corpus, Paper(paper_id=unit.paper_id, title="T"),
                                  FakeCardExtractor({unit.paper_id: card}))
    assert verified.metrics[0].binding_verified is True


def test_a_fabricated_card_binding_is_caught_without_consulting_a_model(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0}), run_id="run1")
    ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    hits = search(corpus, "tracking accuracy", FakeEmbedder(), limit=5)
    unit = next(u for u in hits if "94.2" in u.verbatim_text)
    card = Card(paper_id=unit.paper_id,
                metrics=(CardField("99.9", unit.unit_id, "| ours | 99.9 |"),))
    verified = extract_and_verify(corpus, Paper(paper_id=unit.paper_id, title="T"),
                                  FakeCardExtractor({unit.paper_id: card}))
    assert verified.metrics[0].binding_verified is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gather_end_to_end.py -v`
Expected: FAIL — `screen` promotes to `pending_deep`, which `get_papers_by_depth` must return; if any assertion fails here, fix the *module*, not the test.

- [ ] **Step 3: Extend the package exports**

Add to `jarvis/__init__.py`, keeping both the import block and `__all__` alphabetically sorted (ruff's `I001` and `RUF022` both check this — run `ruff check --fix jarvis/__init__.py` after editing and confirm the diff is only sorting):

```python
from jarvis.card import (
    CardExtractor,
    FakeCardExtractor,
    LLMCardExtractor,
    extract_and_verify,
    unverified_fields,
    verify_card,
)
from jarvis.gate import (
    FakeVoter,
    LLMVoter,
    Signals,
    Thresholds,
    calibrate,
    calibration_report,
    decide,
    screen,
)
from jarvis.gather import (
    Candidate,
    LLMPlanner,
    SearchPlan,
    TemplatePlanner,
    expand_citations,
    gather,
    run_searches,
    save_candidates,
    to_paper,
)
from jarvis.ingest import IngestResult, failed, ingest_decided, ingest_paper
from jarvis.label import read_labels, sample_seed, write_label_sheet
from jarvis.sources import (
    enrich_provenance,
    make_arxiv_search,
    make_openalex_search,
    make_retraction_check,
    make_s2_search,
    normalize_openalex,
    normalize_s2,
)
from jarvis.store import (
    all_units,
    get_card,
    get_papers_by_depth,
    get_screen_decisions,
    get_screen_signals,
    save_card,
    save_screen_decision,
    set_depth,
)
```

Add every one of those names to `__all__`. Update the module docstring's status paragraph to say gather + gate is built and steps 7–10 are not.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass (185 existing + ~140 new). `ruff check .` reports exactly the **11 pre-existing** violations and nothing in any file this plan touched.

- [ ] **Step 5: Commit**

```bash
git add jarvis/__init__.py tests/test_gather_end_to_end.py
git commit -m "test: end-to-end gather, gate, deep read, and card verification"
```

---

## Definition of done

- `python -m pytest` passes with zero network access, no API keys, no model downloads.
- `test_the_gate_keeps_every_hand_labelled_relevant_paper` passes at ≥95% recall, and `test_calibration_from_labels_meets_the_target` proves the threshold fit reaches it on real labels rather than by assertion.
- `test_no_paper_is_ever_removed_from_the_corpus` passes — `defer` is demotion, never deletion.
- `test_a_fabricated_card_binding_is_caught_without_consulting_a_model` passes — Layer 2 inherits the same mechanical grounding as Layer 1.
- Every gate decision in the log carries all four per-signal scores, so any threshold can be re-fitted without re-fetching a single abstract.
- `ruff check .` reports exactly the 11 pre-existing violations.

## Where this stops

This plan produces a corpus. It does not answer anything from it. Q&A is `docs/plans/2026-08-14-compile-cited-qa.md` (spec step 7), and everything after that depends on this plan and that one.

Open questions this plan should answer empirically, and record when it does (spec §15):

- **Gate calibration transfer** — does a threshold set fitted on one project's seed transfer to the next, or is per-project labeling required every time? `calibration_report` on a second project's seed against the first project's thresholds answers this directly.
- **VLM descriptions for figures** — spec §15 asks whether they beat caption + referring text. Figure units currently carry caption + referring text only. Measure before adding a VLM, not after.
