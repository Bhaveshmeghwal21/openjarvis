# Compile — Cited Q&A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer a question from the corpus so that every sentence in the answer resolves to a verbatim span in a specific paper at a specific location — and any sentence that does not is removed before a human ever reads it.

**Architecture:** Spec §7 Stage D. Retrieval is now local and solved, so this stage optimizes purely for precision. A **retriever** calls `jarvis.retrieve.search` repeatedly with refined queries rather than once. The hits are passed through a **hard evidence cap** and re-ordered so the strongest evidence sits at the beginning and end of the context window. A **writer** drafts from that bounded set and emits explicit `(claim, unit_id, quote)` triples. A **verifier** — a separate pass with separate state, never the writer checking itself — runs the two-stage mechanical check already built in `jarvis/verify.py`. Claims whose quote is not in Layer 0 are **blocked**, not flagged; claims that ground but do not entail are flagged for a human.

**Tech Stack:** Python 3.10+, stdlib only in the new modules, `pytest`. The writer and the query refiner are `typing.Protocol`s with deterministic offline fakes; the NLI model and embedder come from the existing core.

**Prerequisite:** the verifiable single-paper core (`docs/plans/2026-08-11-verifiable-single-paper-core.md`), merged at `d7f8672`. This plan can be built and tested against a single-paper corpus and does **not** require `docs/plans/2026-08-14-gather-and-gate.md` — though it is far more useful once a real corpus exists.

## Global Constraints

- Python **>= 3.10**. Use `X | None`, not `Optional[X]`.
- **Never read `.env`.** Configuration is environment variables or `$JARVIS_CONFIG` JSON only.
- **Every test is offline.** No network, no API keys, no model downloads. Heavy dependencies are imported inside the function that needs them.
- All external models are consumed through a `typing.Protocol` with a deterministic fake used in tests.
- Line length **100**. Target `py310`. Run `ruff check .` against **both** the module and its test file before every commit.
- **`jarvis/store.py` is the only module that writes SQL.** This plan adds no SQL at all.
- **The writer never verifies its own output.** Separate function, separate inputs, no shared state. Spec §9 calls this the one multi-agent principle the literature is unambiguous about.
- **No LLM may judge citation support.** Verification is `jarvis.verify` and nothing else. GPT-3.5-as-judge correlates 0.101 with human judgment on this task; NLI correlates 0.638. A test already asserts no `verif*` task is routable to an LLM — do not add one.
- **A claim whose quote is not found is removed from the answer, never merely annotated.** Spec §8: any quote-match failure blocks the claim.
- Frozen dataclasses for all new types; tuples not lists in frozen types.
- Commit after every task with a `feat:`/`test:`/`fix:` prefix.
- Repo-wide `ruff check .` baseline is **11 pre-existing violations** in `citation_graph.py` (2), `config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1). Do not fix them; do not add to them.

## The measured effect this plan is designed around

Spec §7 Stage D names a counter-intuitive result and builds against it: *Parsing and Evaluating Source Attribution in LLM Deep Research* (arXiv 2605.06635) found that **increased search depth consistently degrades factual accuracy while surface-level citation metrics stay stable.** More retrieval makes the answer look better and be worse. Corroborating it: order-preserving retrieval with 48K well-chosen tokens beat full-context 117K by **13 F1 points** at one-seventh the budget, and lost-in-the-middle degradation is 20+ percentage points.

Three consequences, all load-bearing, all in Task 1:

1. A **hard cap** on units per synthesis call. Not a soft preference — a cap.
2. Many small, well-scoped calls rather than one large one.
3. The strongest evidence placed at the **beginning and end** of the context.

An implementer who "improves" this by raising the cap or by appending evidence in rank order has undone the plan.

## File Structure

| File | Responsibility |
|---|---|
| `jarvis/evidence.py` | Create. Hard cap, token budget, primacy/recency ordering, context rendering. |
| `jarvis/retriever.py` | Create. Iterative multi-round retrieval with query refinement. |
| `jarvis/writer.py` | Create. `Writer` protocol, `Draft`, claim triples validated against the evidence set. |
| `jarvis/answer.py` | Create. Wire retrieve → write → verify → assemble. Blocking and flagging live here. |
| `jarvis/evaluate.py` | **Modify.** Add ALCE-style citation precision and recall to the §10 metric set. |
| `jarvis/__init__.py` | **Modify.** Export the new public surface. |

Tests mirror module names: `tests/test_evidence.py`, `tests/test_retriever.py`, `tests/test_writer.py`, `tests/test_answer.py`, `tests/test_evaluate_citations.py`, `tests/test_qa_end_to_end.py`.

---

### Task 1: The evidence budget and its ordering

**Files:**
- Create: `jarvis/evidence.py`
- Test: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `Unit` from `jarvis.models`; `approx_tokens` from `jarvis.text`.
- Produces: `MAX_UNITS = 12`, `MAX_TOKENS = 6000`, `EvidenceSet` (frozen: `units: tuple[Unit, ...]`, `dropped: int = 0`, `tokens: int = 0`), `cap(units, max_units=MAX_UNITS, max_tokens=MAX_TOKENS) -> EvidenceSet`, `order_for_context(units) -> list[Unit]`, `render(units) -> str`.

`order_for_context` takes units **already ranked best-first** and returns them front-loaded and back-loaded: rank 0 first, rank 1 last, rank 2 second, rank 3 second-to-last, and so on, so the weakest evidence ends up in the middle where a model attends least.

`render` is the one place a unit becomes model-visible text, and it is the reason a writer can cite at all: every block is labeled with its `unit_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evidence.py
"""The evidence budget. More retrieval makes answers look better and be worse (spec §7D)."""
import pytest

from jarvis.evidence import (
    MAX_TOKENS,
    MAX_UNITS,
    EvidenceSet,
    cap,
    order_for_context,
    render,
)
from jarvis.models import Unit, UnitType


def _unit(i: int, text: str = "evidence") -> Unit:
    return Unit(unit_id=f"u{i}", paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=("Results",), verbatim_text=text, ordinal=i)


def test_the_cap_is_a_hard_number_not_a_suggestion():
    result = cap([_unit(i) for i in range(50)])
    assert len(result.units) == MAX_UNITS
    assert result.dropped == 50 - MAX_UNITS


def test_a_small_set_passes_through_untouched():
    units = [_unit(i) for i in range(3)]
    result = cap(units)
    assert list(result.units) == units
    assert result.dropped == 0


def test_the_token_budget_cuts_before_the_unit_count_when_units_are_long():
    long_units = [_unit(i, "word " * 3000) for i in range(MAX_UNITS)]
    result = cap(long_units)
    assert len(result.units) < MAX_UNITS
    assert result.tokens <= MAX_TOKENS


def test_a_single_oversized_unit_is_still_included():
    result = cap([_unit(0, "word " * 100000)])
    assert len(result.units) == 1, "never return an empty evidence set for a real hit"


def test_capping_preserves_rank_order():
    units = [_unit(i) for i in range(20)]
    assert [u.unit_id for u in cap(units).units] == [f"u{i}" for i in range(MAX_UNITS)]


def test_capping_an_empty_list_is_empty():
    result = cap([])
    assert result.units == ()
    assert result.dropped == 0


def test_the_strongest_evidence_lands_at_both_ends():
    units = [_unit(i) for i in range(5)]        # ranked best-first
    ordered = order_for_context(units)
    assert ordered[0].unit_id == "u0", "best evidence first (primacy)"
    assert ordered[-1].unit_id == "u1", "second-best evidence last (recency)"
    assert ordered[len(ordered) // 2].unit_id == "u4", "weakest evidence in the middle"


def test_ordering_keeps_every_unit_exactly_once():
    units = [_unit(i) for i in range(9)]
    ordered = order_for_context(units)
    assert len(ordered) == 9
    assert {u.unit_id for u in ordered} == {u.unit_id for u in units}


def test_ordering_handles_one_and_zero_units():
    assert order_for_context([]) == []
    assert [u.unit_id for u in order_for_context([_unit(0)])] == ["u0"]


def test_rendering_labels_every_block_with_its_unit_id():
    text = render([_unit(0, "the controller reaches 94.2%"), _unit(1, "under gusts")])
    assert "[u0]" in text
    assert "[u1]" in text
    assert "94.2" in text


def test_rendering_includes_the_contextual_prefix_when_there_is_one():
    unit = Unit(unit_id="u9", paper_id="p1", type=UnitType.TABLE, page=3,
                section_path=("Results",), verbatim_text="| ours | 94.2 |",
                ordinal=9, context_prefix="From \"Gust-Robust Control\", Table 3.")
    text = render([unit])
    assert "Gust-Robust Control" in text
    assert "| ours | 94.2 |" in text


def test_rendering_nothing_is_the_empty_string():
    assert render([]) == ""


def test_evidence_set_is_frozen():
    with pytest.raises(Exception):
        EvidenceSet(units=()).dropped = 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.evidence'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/evidence.py
"""The evidence budget — capped, ordered, and rendered (spec §7 Stage D).

Increased search depth consistently degrades factual accuracy while surface-level citation
metrics stay stable (arXiv 2605.06635): more evidence makes an answer look better and be
worse, and the metrics that would catch it do not move. Order-preserving retrieval with
48K well-chosen tokens beat full-context 117K by 13 F1 points at one-seventh the budget.

Hence: a hard cap, many small calls rather than one large one, and the strongest evidence
placed at the beginning and the end of the context to exploit primacy and recency and
avoid lost-in-the-middle degradation (20+ percentage points).

Raising `MAX_UNITS` is not a tuning knob. It is the failure this module exists to prevent.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.models import Unit
from jarvis.text import approx_tokens

MAX_UNITS = 12
MAX_TOKENS = 6000


@dataclass(frozen=True)
class EvidenceSet:
    """What actually reaches a synthesis call, plus how much was left out."""
    units: tuple[Unit, ...] = ()
    dropped: int = 0
    tokens: int = 0


def cap(units: Sequence[Unit], max_units: int = MAX_UNITS,
        max_tokens: int = MAX_TOKENS) -> EvidenceSet:
    """Truncate a ranked list to the budget, preserving rank order.

    The first unit is always kept even if it alone blows the token budget: returning an
    empty evidence set for a real retrieval hit would silently turn a groundable question
    into an ungroundable one.
    """
    kept: list[Unit] = []
    total = 0
    for unit in units:
        if len(kept) >= max_units:
            break
        size = approx_tokens(unit.verbatim_text)
        if kept and total + size > max_tokens:
            break
        kept.append(unit)
        total += size
    return EvidenceSet(units=tuple(kept), dropped=max(0, len(units) - len(kept)),
                       tokens=total)


def order_for_context(units: Sequence[Unit]) -> list[Unit]:
    """Re-order a best-first ranking so the strongest evidence sits at both ends.

    Rank 0 goes first, rank 1 last, rank 2 second, rank 3 second-to-last, and so on. The
    weakest evidence ends up in the middle, which is exactly where a model attends least.
    """
    front: list[Unit] = []
    back: list[Unit] = []
    for index, unit in enumerate(units):
        (front if index % 2 == 0 else back).append(unit)
    return front + list(reversed(back))


def render(units: Sequence[Unit]) -> str:
    """Evidence as model-visible text. The `[unit_id]` label is what makes citing possible."""
    blocks: list[str] = []
    for unit in units:
        header = f"[{unit.unit_id}]"
        if unit.context_prefix:
            header = f"{header} {unit.context_prefix}"
        blocks.append(f"{header}\n{unit.verbatim_text}")
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evidence.py -v && ruff check jarvis/evidence.py tests/test_evidence.py`
Expected: PASS (13 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/evidence.py tests/test_evidence.py
git commit -m "feat: capped and order-preserving evidence budget"
```

---

### Task 2: Iterative retrieval

**Files:**
- Create: `jarvis/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `search` and `Reranker` from `jarvis.retrieve`; `Embedder` from `jarvis.embed`; `Unit` from `jarvis.models`; `jarvis.llm.chat` through injection.
- Produces: `Refiner` protocol with `refine(question, queries, units) -> str | None`, `FakeRefiner(queries)`, `LLMRefiner(router, chat_fn=None)`, `Retrieval` (frozen: `question: str`, `units: tuple[Unit, ...]`, `queries: tuple[str, ...]`, `rounds: int`), `retrieve_iteratively(conn, question, embedder, *, refiner=None, rounds=3, limit=8, reranker=None) -> Retrieval`.

Spec §7 Stage D: *"Retrieval is agentic and iterative, not single-shot. Search exposed as tools a subagent calls repeatedly with refined queries."* This is the generalizable lesson from both Claude Code's agentic search and PaperQA2's tool decomposition, arrived at independently.

What does **not** generalize, and must not be imported here: grep-over-embeddings. That is a code-specific result driven by unique function names and exact import strings. Scientific prose has no such identifiers — "wind rejection", "disturbance attenuation", and "gust tolerance" are the same concept. Hybrid BM25+vector retrieval stays.

`refiner=None` means single-shot, which is the deterministic default and the behaviour the existing core already has.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_retriever.py
"""Agentic, iterative retrieval (spec §7 Stage D)."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Paper
from jarvis.parse import FakeParser
from jarvis.retriever import FakeRefiner, LLMRefiner, Refiner, Retrieval, retrieve_iteratively
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.units import build_units

BLOCKS = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy in gusts.",
          page=1, section_path=("Results",)),
    Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
    Block(kind="paragraph", text="Performance degrades sharply above 12 m/s wind speed.",
          page=2, section_path=("Limitations",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    parsed = FakeParser(BLOCKS).parse("p.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    yield conn
    close_store(conn)


def test_fake_refiner_satisfies_the_protocol():
    assert isinstance(FakeRefiner([]), Refiner)


def test_without_a_refiner_retrieval_is_single_shot(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder())
    assert result.rounds == 1
    assert result.queries == ("tracking accuracy",)
    assert len(result.units) > 0


def test_a_refiner_adds_rounds_and_records_every_query(corpus):
    refiner = FakeRefiner(["wind speed limitations"])
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=refiner, rounds=3)
    assert result.queries == ("tracking accuracy", "wind speed limitations")
    assert result.rounds == 2


def test_refinement_surfaces_evidence_the_first_query_missed(corpus):
    single = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(), limit=1)
    iterated = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                    refiner=FakeRefiner(["wind speed degrades"]),
                                    rounds=2, limit=1)
    assert len(iterated.units) > len(single.units)
    assert any("12 m/s" in u.verbatim_text for u in iterated.units)


def test_units_are_deduped_across_rounds(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner(["tracking accuracy"] * 3), rounds=4)
    ids = [u.unit_id for u in result.units]
    assert len(ids) == len(set(ids))


def test_a_refiner_returning_none_stops_the_loop_early(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner([None]), rounds=5)
    assert result.rounds == 1


def test_a_refiner_repeating_a_query_stops_the_loop(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner(["tracking accuracy"]), rounds=5)
    assert result.rounds == 1, "a repeated query means the refiner has nothing new"


def test_the_round_budget_is_respected(corpus):
    refiner = FakeRefiner([f"query {i}" for i in range(20)])
    result = retrieve_iteratively(corpus, "start", FakeEmbedder(), refiner=refiner, rounds=3)
    assert result.rounds == 3
    assert len(result.queries) == 3


def test_a_refiner_that_raises_ends_the_loop_without_losing_earlier_hits(corpus):
    class Boom:
        def refine(self, question, queries, units):
            raise RuntimeError("model down")

    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=Boom(), rounds=3)
    assert result.rounds == 1
    assert len(result.units) > 0


def test_llm_refiner_returns_the_models_query():
    refiner = LLMRefiner(_Router(), chat_fn=lambda *a, **k: {"query": "wind speed limits"})
    assert refiner.refine("q", ("q",), []) == "wind speed limits"


def test_llm_refiner_returns_none_when_the_model_says_it_is_done():
    for reply in ({"query": ""}, {"done": True}, {}, "junk", None):
        assert LLMRefiner(_Router(), chat_fn=lambda *a, **k: reply).refine("q", ("q",), []) \
            is None


def test_llm_refiner_returns_none_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    assert LLMRefiner(_Router(), chat_fn=boom).refine("q", ("q",), []) is None


def test_llm_refiner_routes_to_retrieval_refine():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {"query": "x"}

    LLMRefiner(_Router(), chat_fn=spy).refine("q", ("q",), [])
    assert seen["task"] == "retrieval_refine"


def test_retrieval_is_frozen(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder())
    with pytest.raises(Exception):
        result.question = "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.retriever'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/retriever.py
"""Agentic, iterative retrieval (spec §7 Stage D).

Search is called repeatedly with refined queries rather than once. This is the
generalizable lesson from both Claude Code's agentic search and PaperQA2's tool
decomposition, arrived at independently.

What does NOT generalize from Claude Code, and is deliberately not imported here:
grep-over-embeddings. That result is code-specific, driven by unique function names and
exact import strings. Scientific prose has no such identifiers — "wind rejection",
"disturbance attenuation", and "gust tolerance" are the same concept, which is why the
hybrid BM25+vector stack stays.

`refiner=None` is single-shot and is the deterministic default.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.embed import Embedder
from jarvis.models import Unit
from jarvis.retrieve import Reranker, search

_REFINE_PROMPT = (
    "You are retrieving evidence for a research question from a paper corpus.\n"
    "Queries already tried:\n{queries}\n\n"
    "Evidence found so far:\n{found}\n\n"
    "Question: {question}\n\n"
    "Return JSON: {{\"query\": \"<one new search query>\"}} for an aspect not yet covered, "
    "or {{\"done\": true}} if the evidence is sufficient. Vary vocabulary — the same "
    "concept is named differently across communities."
)


@dataclass(frozen=True)
class Retrieval:
    """Everything one retrieval session produced, ranked best-first across all rounds."""
    question: str
    units: tuple[Unit, ...] = ()
    queries: tuple[str, ...] = ()
    rounds: int = 0


@runtime_checkable
class Refiner(Protocol):
    def refine(self, question: str, queries: Sequence[str],
               units: Sequence[Unit]) -> str | None: ...


class FakeRefiner:
    """Deterministic refiner for tests: yields its queries in order, then None."""

    def __init__(self, queries: Sequence[str | None]) -> None:
        self._queries = list(queries)
        self._index = 0

    def refine(self, question: str, queries: Sequence[str],
               units: Sequence[Unit]) -> str | None:
        if self._index >= len(self._queries):
            return None
        out = self._queries[self._index]
        self._index += 1
        return out


class LLMRefiner:
    """Model-written follow-up queries, routed to the cheap tier."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None,
                 preview_units: int = 5) -> None:
        self._router = router
        self._chat = chat_fn
        self._preview_units = preview_units

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def refine(self, question: str, queries: Sequence[str],
               units: Sequence[Unit]) -> str | None:
        found = "\n".join(f"- {u.verbatim_text[:200]}"
                          for u in list(units)[:self._preview_units]) or "(nothing yet)"
        prompt = _REFINE_PROMPT.format(queries="\n".join(f"- {q}" for q in queries),
                                       found=found, question=question)
        try:
            raw = self._chat_fn()(self._router, "retrieval_refine", prompt, json_mode=True)
        except Exception:  # noqa: BLE001 - a dead refiner ends the loop, never the answer
            return None
        if not isinstance(raw, dict):
            return None
        query = " ".join(str(raw.get("query", "") or "").split())
        return query or None


def retrieve_iteratively(conn: sqlite3.Connection, question: str, embedder: Embedder, *,
                         refiner: Refiner | None = None, rounds: int = 3, limit: int = 8,
                         reranker: Reranker | None = None) -> Retrieval:
    """Search, refine, search again. Stops on the budget, a repeat, or a `None` refinement."""
    queries: list[str] = [question]
    seen_queries = {question}
    units: list[Unit] = []
    seen_units: set[str] = set()
    completed = 0

    while completed < max(1, rounds):
        query = queries[completed]
        for unit in search(conn, query, embedder, limit=limit, reranker=reranker):
            if unit.unit_id in seen_units:
                continue
            seen_units.add(unit.unit_id)
            units.append(unit)
        completed += 1

        if refiner is None or completed >= rounds:
            break
        try:
            nxt = refiner.refine(question, tuple(queries), tuple(units))
        except Exception:  # noqa: BLE001 - keep everything retrieved so far
            break
        if not nxt or nxt in seen_queries:
            break
        seen_queries.add(nxt)
        queries.append(nxt)

    return Retrieval(question=question, units=tuple(units), queries=tuple(queries),
                     rounds=completed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_retriever.py -v && ruff check jarvis/retriever.py tests/test_retriever.py`
Expected: PASS (14 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/retriever.py tests/test_retriever.py
git commit -m "feat: iterative retrieval with query refinement"
```

---

**Amended post-implementation** (Task 2 review, plan-conflict, human ruling: fix via
cross-round RRF fusion):

The reference code above accumulates each round's hits by simple round-block
concatenation — `units.append(unit)` inside the per-round loop, deduped by `unit_id` but
never reordered. `Retrieval.units`'s own docstring claims the result is "ranked best-first
across all rounds," which this does not deliver: round 2's top hit — the evidence the
refiner exists specifically to surface — always sits behind every one of round 1's hits,
however weak, purely because round 1 ran first. Task 3's `cap()` trusts its input is
already a valid global ranking and truncates from the front, so with the defaults `ask()`
uses (`rounds=2`, `limit=8`, against `MAX_UNITS=12`), a later round's most relevant unit
can be silently dropped in favor of an earlier round's weakest one.

The fix reuses `jarvis.retrieve.rrf` — the same Reciprocal Rank Fusion `search()` already
uses to fuse BM25 and vector rankings within one round — to fuse across rounds instead of
concatenating. Each round's search results are kept as their own ranked list of
`unit_id`s; at the end, `rrf(rankings)` fuses all of them into one genuine best-first
order. A unit ranked highly in two different rounds (found relevant by two different
sub-queries) is rewarded with a higher combined score, which is the correct behavior, not
an edge case to special-case away. `rrf` on a single-round list reproduces that round's
original order exactly (its score `1/(k+rank)` is strictly decreasing in rank), so
`refiner=None`'s single-shot path is unaffected.

Corrected `retrieve_iteratively`:

```python
def retrieve_iteratively(conn: sqlite3.Connection, question: str, embedder: Embedder, *,
                         refiner: Refiner | None = None, rounds: int = 3, limit: int = 8,
                         reranker: Reranker | None = None) -> Retrieval:
    """Search, refine, search again. Stops on the budget, a repeat, or a `None` refinement.

    Units are fused across rounds with the same Reciprocal Rank Fusion `search()` already
    uses to fuse BM25 and vector rankings within one round: a unit ranked highly by two
    different rounds is genuinely more likely relevant, and a later round's top hit is
    never buried behind an earlier round's weaker one just because it arrived first.
    """
    queries: list[str] = [question]
    seen_queries = {question}
    rankings: list[list[str]] = []
    by_id: dict[str, Unit] = {}
    accumulated: list[Unit] = []
    seen_units: set[str] = set()
    completed = 0

    while completed < max(1, rounds):
        query = queries[completed]
        round_ids: list[str] = []
        for unit in search(conn, query, embedder, limit=limit, reranker=reranker):
            round_ids.append(unit.unit_id)
            by_id.setdefault(unit.unit_id, unit)
            if unit.unit_id not in seen_units:
                seen_units.add(unit.unit_id)
                accumulated.append(unit)
        rankings.append(round_ids)
        completed += 1

        if refiner is None or completed >= rounds:
            break
        try:
            nxt = refiner.refine(question, tuple(queries), tuple(accumulated))
        except Exception:  # noqa: BLE001 - keep everything retrieved so far
            break
        if not nxt or nxt in seen_queries:
            break
        seen_queries.add(nxt)
        queries.append(nxt)

    ordered = [by_id[uid] for uid, _ in rrf(rankings) if uid in by_id]
    return Retrieval(question=question, units=tuple(ordered), queries=tuple(queries),
                     rounds=completed)
```

Add `from jarvis.retrieve import Reranker, rrf, search` (the module already imports
`Reranker` and `search` from the same line; `rrf` joins them). `accumulated` — the deduped,
round-arrival-order list — is what the refiner sees each round (unaffected by the fusion
change; a refiner reasoning about "what's been found so far" cares about coverage, not
final rank). `ordered` — the RRF-fused list — is what `Retrieval.units` actually holds,
now honestly matching its own docstring.

**Also amended** (same review, test-coverage gap in the reference test, not a code
defect): `test_units_are_deduped_across_rounds`'s reference code above used
`refiner=FakeRefiner(["tracking accuracy"] * 3)` — a refined query identical to the seed
question, which trips the repeated-query stop condition before a second round ever runs.
The test's assertion was trivially true: a single `search()` call already returns unique
`unit_id`s on its own, so the test passed without ever exercising the cross-round fusion
path it claims to cover. Corrected to force two rounds with genuinely different queries,
which — given this fixture's two-unit corpus — reliably return overlapping unit sets so
the dedup/fusion path is actually proven:

```python
def test_units_are_deduped_across_rounds(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner(["wind speed limitations"]), rounds=2)
    ids = [u.unit_id for u in result.units]
    assert len(ids) == len(set(ids))
    assert result.rounds == 2, "both rounds must actually run for this test to mean anything"
```

---

### Task 3: The writer and its claim triples

**Files:**
- Create: `jarvis/writer.py`
- Test: `tests/test_writer.py`

**Interfaces:**
- Consumes: `Claim`, `Unit` from `jarvis.models`; `render` from `jarvis.evidence`; `jarvis.llm.chat` through injection.
- Produces: `Draft` (frozen: `text: str`, `claims: tuple[Claim, ...]`), `Writer` protocol with `write(question, units) -> Draft`, `FakeWriter(drafts, default=None)`, `LLMWriter(router, chat_fn=None)`, `claims_from_json(data, units, prefix="c") -> list[Claim]`.

The writer emits **explicit triples** — claim text, `unit_id`, verbatim quote — rather than prose with citation markers that someone has to parse back out. That is the shape the verification stage already consumes (`Claim` has exactly those fields), and it removes an entire class of parser bugs between drafting and verification.

Two rules the writer cannot bend, both enforced mechanically in `claims_from_json`:

- A claim citing a `unit_id` that was not in its evidence set is dropped. That is a hallucinated citation, and the fact that the id looks plausible is the whole problem.
- A claim with no quote is dropped. There is nothing for the verifier to match, and an unmatched claim would sail through stage 1 by having nothing to fail.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_writer.py
"""The writer. Emits claim triples, never prose the verifier has to parse."""
import pytest

from jarvis.models import Unit, UnitType
from jarvis.writer import Draft, FakeWriter, LLMWriter, Writer, claims_from_json

UNITS = [
    Unit(unit_id="u1", paper_id="p1", type=UnitType.PROSE, page=1, section_path=(),
         verbatim_text="The controller reaches 94.2% tracking accuracy.", ordinal=0),
    Unit(unit_id="u2", paper_id="p1", type=UnitType.PROSE, page=2, section_path=(),
         verbatim_text="Performance degrades above 12 m/s.", ordinal=1),
]


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_fake_writer_satisfies_the_protocol():
    assert isinstance(FakeWriter({}), Writer)


def test_claims_are_built_from_the_models_triples():
    data = {"claims": [
        {"text": "It reaches 94.2% accuracy.", "unit_id": "u1", "quote": "94.2% tracking"},
        {"text": "It degrades in strong wind.", "unit_id": "u2", "quote": "12 m/s"},
    ]}
    claims = claims_from_json(data, UNITS)
    assert [c.unit_id for c in claims] == ["u1", "u2"]
    assert claims[0].quote == "94.2% tracking"


def test_claim_ids_are_unique_and_prefixed():
    data = {"claims": [{"text": "a", "unit_id": "u1", "quote": "q"},
                       {"text": "b", "unit_id": "u1", "quote": "q"}]}
    claims = claims_from_json(data, UNITS, prefix="sub3")
    assert [c.claim_id for c in claims] == ["sub3-0", "sub3-1"]


def test_a_claim_citing_a_unit_outside_the_evidence_set_is_dropped():
    data = {"claims": [{"text": "invented", "unit_id": "u99", "quote": "q"}]}
    assert claims_from_json(data, UNITS) == []


def test_a_claim_with_no_quote_is_dropped():
    data = {"claims": [{"text": "unbacked", "unit_id": "u1", "quote": ""},
                       {"text": "unbacked", "unit_id": "u1"}]}
    assert claims_from_json(data, UNITS) == []


def test_a_claim_with_no_text_is_dropped():
    assert claims_from_json({"claims": [{"unit_id": "u1", "quote": "q"}]}, UNITS) == []


def test_malformed_payloads_yield_no_claims_and_never_raise():
    for junk in (None, {}, "text", {"claims": None}, {"claims": "x"}, {"claims": [1, 2]}):
        assert claims_from_json(junk, UNITS) == []


def test_llm_writer_returns_answer_text_and_claims():
    reply = {"answer": "The controller is accurate in gusts.",
             "claims": [{"text": "It reaches 94.2% accuracy.", "unit_id": "u1",
                         "quote": "94.2% tracking"}]}
    draft = LLMWriter(_Router(), chat_fn=lambda *a, **k: reply).write("how accurate?", UNITS)
    assert draft.text == "The controller is accurate in gusts."
    assert len(draft.claims) == 1


def test_llm_writer_is_shown_the_unit_ids_it_must_cite():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["prompt"] = prompt
        return {"answer": "", "claims": []}

    LLMWriter(_Router(), chat_fn=spy).write("q", UNITS)
    assert "[u1]" in seen["prompt"]
    assert "[u2]" in seen["prompt"]


def test_llm_writer_routes_to_synthesis():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {}

    LLMWriter(_Router(), chat_fn=spy).write("q", UNITS)
    assert seen["task"] == "synthesis"


def test_llm_writer_returns_an_empty_draft_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    draft = LLMWriter(_Router(), chat_fn=boom).write("q", UNITS)
    assert draft == Draft(text="", claims=())


def test_a_writer_given_no_evidence_writes_nothing():
    draft = LLMWriter(_Router(), chat_fn=lambda *a, **k: {"answer": "invented"}).write("q", [])
    assert draft == Draft(text="", claims=()), "no evidence means no answer, not a guess"


def test_fake_writer_returns_the_draft_for_its_question():
    draft = Draft(text="answer", claims=())
    assert FakeWriter({"q": draft}).write("q", UNITS) is draft
    assert FakeWriter({}).write("q", UNITS) == Draft(text="", claims=())


def test_draft_is_frozen():
    with pytest.raises(Exception):
        Draft(text="a", claims=()).text = "b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.writer'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/writer.py
"""The writer subagent (spec §9).

Takes a question and a bounded evidence set; returns prose plus explicit
(claim, unit_id, quote) triples. Triples rather than prose-with-markers because that is
exactly the shape `jarvis.verify` consumes, which removes a whole class of parser bugs
between drafting and verification.

The writer NEVER verifies its own output. That is `jarvis.answer`, running a separate pass
with separate inputs — the one multi-agent principle the literature is unambiguous about.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.evidence import render
from jarvis.models import Claim, Unit

_WRITE_PROMPT = (
    "Answer the question using ONLY the evidence below. Do not use prior knowledge.\n\n"
    "Return JSON:\n"
    "{{\"answer\": \"<prose>\", \"claims\": [{{\"text\": \"<one factual sentence>\", "
    "\"unit_id\": \"<id from the evidence>\", \"quote\": \"<exact span copied from that "
    "unit>\"}}]}}\n\n"
    "Rules:\n"
    "- Every factual sentence in `answer` must appear as a claim.\n"
    "- `quote` must be copied character-for-character from the cited unit. No paraphrase, "
    "no ellipsis, no reformatting. A quote that is not verbatim is discarded automatically "
    "and its claim with it.\n"
    "- `unit_id` must be one of the ids shown in brackets below. An id not listed there is "
    "a fabricated citation.\n"
    "- If the evidence does not answer the question, say so. An incomplete answer is "
    "correct; an unsupported one is not.\n\n"
    "Question: {question}\n\nEvidence:\n{evidence}"
)


@dataclass(frozen=True)
class Draft:
    """Prose plus the claims it rests on. Unverified — nothing here has been checked yet."""
    text: str = ""
    claims: tuple[Claim, ...] = ()


@runtime_checkable
class Writer(Protocol):
    def write(self, question: str, units: Sequence[Unit]) -> Draft: ...


def claims_from_json(data, units: Sequence[Unit], prefix: str = "c") -> list[Claim]:
    """Model JSON -> validated claims. Anything ungroundable is dropped here, not later.

    Two rules, both mechanical:
      * a `unit_id` outside the evidence set is a hallucinated citation;
      * a claim with no quote has nothing for stage 1 to match, so it would pass
        verification by having nothing to fail.
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("claims")
    if not isinstance(raw, list):
        return []

    known = {u.unit_id for u in units}
    out: list[Claim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        unit_id = str(item.get("unit_id", "") or "").strip()
        quote = str(item.get("quote", "") or "").strip()
        if not text or not quote or unit_id not in known:
            continue
        out.append(Claim(claim_id=f"{prefix}-{len(out)}", text=text, unit_id=unit_id,
                         quote=quote))
    return out


class FakeWriter:
    """Deterministic writer for tests, keyed by question."""

    def __init__(self, drafts: Mapping[str, Draft] | None = None) -> None:
        self._drafts = dict(drafts or {})

    def write(self, question: str, units: Sequence[Unit]) -> Draft:
        return self._drafts.get(question, Draft())


class LLMWriter:
    """Model-written draft, routed to the frontier tier. Returns an empty draft on failure."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None) -> None:
        self._router = router
        self._chat = chat_fn

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def write(self, question: str, units: Sequence[Unit]) -> Draft:
        if not units:
            # No evidence means no answer. A model asked to answer from an empty context
            # will answer from its training data, which is the failure this whole system
            # is built to remove.
            return Draft()

        prompt = _WRITE_PROMPT.format(question=question, evidence=render(units))
        try:
            raw = self._chat_fn()(self._router, "synthesis", prompt, json_mode=True)
        except Exception:  # noqa: BLE001
            return Draft()
        if not isinstance(raw, dict):
            return Draft()
        return Draft(text=str(raw.get("answer", "") or "").strip(),
                     claims=tuple(claims_from_json(raw, units)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v && ruff check jarvis/writer.py tests/test_writer.py`
Expected: PASS (14 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/writer.py tests/test_writer.py
git commit -m "feat: writer subagent emitting validated claim triples"
```

---

### Task 4: Answer assembly — blocking, flagging, rendering

**Files:**
- Create: `jarvis/answer.py`
- Test: `tests/test_answer.py`

**Interfaces:**
- Consumes: `retrieve_iteratively`, `Refiner` from `jarvis.retriever`; `cap`, `order_for_context` from `jarvis.evidence`; `Writer`, `Draft` from `jarvis.writer`; `verify_claim`, `NLIModel` from `jarvis.verify`; `Verdict`, `Verification`, `Claim`, `Unit` from `jarvis.models`.
- Produces: `Answer` (frozen: `question`, `text`, `claims`, `verifications`, `units`, `dropped_evidence`, plus properties `supported`, `flagged`, `blocked`, `is_grounded`), `ask(conn, question, embedder, writer, nli, *, refiner=None, rounds=2, limit=8, reranker=None, max_units=MAX_UNITS, max_tokens=MAX_TOKENS, threshold=0.5) -> Answer`, `render_answer(answer) -> str`.

The three outcomes, and why they differ:

| Verdict | Meaning | What happens |
|---|---|---|
| `SUPPORTED` | quote is in Layer 0 **and** entails the claim | kept, cited |
| `NEUTRAL` / `CONTRADICTED` | quote is real, entailment is not established | **flagged** — surfaced with a warning, not silently passed |
| `QUOTE_NOT_FOUND` | the quote is not in Layer 0 | **blocked** — removed from the answer entirely |

Spec §8 is explicit that stage 1 failure blocks the claim, and equally explicit that the NLI stage *"is a filter, not an oracle"* — AttributionBench found even fine-tuned GPT-3.5 reaches only ~80% macro-F1 on binary attribution, so low-confidence results surface as flagged rather than either passed or deleted.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_answer.py
"""Answer assembly: retrieve, write, verify, block, flag."""
import pytest

from jarvis.answer import Answer, ask, render_answer
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

BLOCKS = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy in gusts.",
          page=1, section_path=("Results",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)
QUESTION = "how accurate is the controller?"
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})
NEUTRAL = FakeNLI(default={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    parsed = FakeParser(BLOCKS).parse("p.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    yield conn
    close_store(conn)


def _prose_unit(conn):
    return next(u for u in get_units(conn, "p1")
                if "94.2" in u.verbatim_text and u.type.value == "prose")


def _writer(conn, quote, text="It reaches 94.2% tracking accuracy."):
    unit = _prose_unit(conn)
    return FakeWriter({QUESTION: Draft(
        text="The controller is accurate under gusts.",
        claims=(Claim(claim_id="c-0", text=text, unit_id=unit.unit_id, quote=quote),),
    )})


def test_a_grounded_entailed_claim_is_supported(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), ENTAILS)
    assert len(answer.supported) == 1
    assert answer.blocked == ()
    assert answer.is_grounded is True


def test_a_fabricated_quote_blocks_the_claim(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 99.9% tracking accuracy"), ENTAILS)
    assert len(answer.blocked) == 1
    assert answer.supported == ()
    assert answer.blocked[0].verdict is Verdict.QUOTE_NOT_FOUND
    assert answer.is_grounded is False


def test_a_blocked_claim_never_appears_in_the_rendered_answer(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 99.9% tracking accuracy",
                         text="It reaches 99.9% accuracy."), ENTAILS)
    rendered = render_answer(answer)
    assert "99.9" not in rendered


def test_a_real_quote_that_does_not_entail_is_flagged_not_blocked(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), NEUTRAL)
    assert len(answer.flagged) == 1
    assert answer.blocked == ()
    assert answer.flagged[0].verdict is Verdict.NEUTRAL


def test_a_flagged_claim_is_rendered_with_a_warning(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), NEUTRAL)
    rendered = render_answer(answer)
    assert "unverified" in rendered.lower()


def test_a_supported_claim_is_rendered_with_its_unit_id(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), ENTAILS)
    unit_id = _prose_unit(corpus).unit_id
    assert unit_id in render_answer(answer)


def test_every_claim_gets_exactly_one_verification(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), ENTAILS)
    assert len(answer.verifications) == len(answer.claims) == 1


def test_the_writer_only_ever_sees_capped_ordered_evidence(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["units"] = list(units)
            return Draft()

    ask(corpus, QUESTION, FakeEmbedder(), SpyWriter(), ENTAILS, max_units=1)
    assert len(seen["units"]) == 1


def test_a_question_with_no_retrievable_evidence_answers_nothing(corpus):
    answer = ask(corpus, "zzzz nonexistent topic qqq", FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert answer.claims == ()
    assert answer.is_grounded is False


def test_an_empty_answer_renders_an_explicit_no_evidence_message(corpus):
    answer = ask(corpus, "zzzz nonexistent topic qqq", FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert "no" in render_answer(answer).lower()


def test_the_evidence_cap_is_reported_not_hidden(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS,
                 max_units=1, limit=8)
    assert answer.dropped_evidence >= 0


def test_answer_is_frozen(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS)
    with pytest.raises(Exception):
        answer.text = "rewritten"


def test_verification_does_not_consult_the_writer(corpus):
    """The writer is called once, for drafting, and never again during verification."""
    calls = []
    unit = _prose_unit(corpus)

    class CountingWriter:
        def write(self, question, units):
            calls.append(question)
            return Draft(text="t", claims=(Claim("c-0", "claim", unit.unit_id,
                                                 "reaches 94.2% tracking accuracy"),))

    ask(corpus, QUESTION, FakeEmbedder(), CountingWriter(), ENTAILS)
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_answer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.answer'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/answer.py
"""Compile — cited Q&A (spec §7 Stage D, §8).

Retrieve iteratively, cap and order the evidence, draft, then verify in a separate pass.

Three outcomes, deliberately different:
  * SUPPORTED                — quote is in Layer 0 and entails the claim. Kept and cited.
  * NEUTRAL / CONTRADICTED   — quote is real, entailment is not established. Flagged.
  * QUOTE_NOT_FOUND          — the quote is not in Layer 0. BLOCKED, removed entirely.

Blocking versus flagging is not a stylistic choice. Stage 1 is deterministic and exact, so
its failure is proof of fabrication and the claim cannot stand. Stage 2 is an NLI model,
and spec §8 is explicit that it is a filter and not an oracle — AttributionBench found even
fine-tuned GPT-3.5 reaches only ~80% macro-F1 on binary attribution — so a stage-2 failure
surfaces for a human instead of being silently passed or silently deleted.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.embed import Embedder
from jarvis.evidence import MAX_TOKENS, MAX_UNITS, cap, order_for_context
from jarvis.models import Claim, Unit, Verdict, Verification
from jarvis.retrieve import Reranker
from jarvis.retriever import Refiner, retrieve_iteratively
from jarvis.verify import NLIModel, verify_claim
from jarvis.writer import Writer

FLAGGED_VERDICTS = (Verdict.NEUTRAL, Verdict.CONTRADICTED)


@dataclass(frozen=True)
class Answer:
    """One answer plus the full record of how every sentence in it was checked."""
    question: str
    text: str = ""
    claims: tuple[Claim, ...] = ()
    verifications: tuple[Verification, ...] = ()
    units: tuple[Unit, ...] = ()
    queries: tuple[str, ...] = ()
    dropped_evidence: int = 0

    def _by_verdict(self, *verdicts: Verdict) -> tuple[Verification, ...]:
        return tuple(v for v in self.verifications if v.verdict in verdicts)

    @property
    def supported(self) -> tuple[Verification, ...]:
        return self._by_verdict(Verdict.SUPPORTED)

    @property
    def flagged(self) -> tuple[Verification, ...]:
        return self._by_verdict(*FLAGGED_VERDICTS)

    @property
    def blocked(self) -> tuple[Verification, ...]:
        return self._by_verdict(Verdict.QUOTE_NOT_FOUND)

    @property
    def is_grounded(self) -> bool:
        """True only when there is at least one claim and every one of them is supported."""
        return bool(self.supported) and not self.blocked and not self.flagged

    def claim_for(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)


def ask(conn: sqlite3.Connection, question: str, embedder: Embedder, writer: Writer,
        nli: NLIModel, *, refiner: Refiner | None = None, rounds: int = 2, limit: int = 8,
        reranker: Reranker | None = None, max_units: int = MAX_UNITS,
        max_tokens: int = MAX_TOKENS, threshold: float = 0.5) -> Answer:
    """One question, end to end. The writer drafts; a separate pass verifies."""
    retrieval = retrieve_iteratively(conn, question, embedder, refiner=refiner,
                                     rounds=rounds, limit=limit, reranker=reranker)
    budget = cap(retrieval.units, max_units=max_units, max_tokens=max_tokens)
    evidence = order_for_context(budget.units)

    draft = writer.write(question, evidence)
    verifications = tuple(verify_claim(conn, claim, nli, threshold=threshold)
                          for claim in draft.claims)

    return Answer(question=question, text=draft.text, claims=draft.claims,
                  verifications=verifications, units=tuple(evidence),
                  queries=retrieval.queries, dropped_evidence=budget.dropped)


def render_answer(answer: Answer) -> str:
    """Human-readable output. Blocked claims are absent; flagged ones carry a warning."""
    if not answer.claims:
        return "No evidence in this corpus answers that question."

    lines: list[str] = []
    for verification in answer.supported:
        claim = answer.claim_for(verification.claim_id)
        if claim is not None:
            lines.append(f"{claim.text} [{claim.unit_id}]")

    if answer.flagged:
        lines.append("")
        lines.append("Unverified — the quote is real but does not clearly support the claim:")
        for verification in answer.flagged:
            claim = answer.claim_for(verification.claim_id)
            if claim is not None:
                lines.append(f"  - {claim.text} [{claim.unit_id}] "
                             f"({verification.verdict.value})")

    if answer.blocked:
        lines.append("")
        lines.append(f"{len(answer.blocked)} claim(s) were removed: their quotes do not "
                     f"appear in any source paper.")

    if not answer.supported and not answer.flagged:
        return ("No claim in the draft could be grounded in the corpus. "
                f"{len(answer.blocked)} claim(s) were removed.")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_answer.py -v && ruff check jarvis/answer.py tests/test_answer.py`
Expected: PASS (13 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/answer.py tests/test_answer.py
git commit -m "feat: answer assembly with claim blocking and flagging"
```

---

**Amended post-implementation** (Task 4 review, plan-conflict, human ruling: fix now,
test-only, no production-code change):

The reference `BLOCKS` fixture above has exactly one paragraph block, so `build_units`
produces exactly one retrievable prose unit for the whole corpus. Two of the reference
tests — `test_the_writer_only_ever_sees_capped_ordered_evidence` (`max_units=1`, asserts
`len(seen["units"]) == 1`) and `test_the_evidence_cap_is_reported_not_hidden` (asserts
`answer.dropped_evidence >= 0`) — were written to prove the evidence-cap wiring
(`cap()`/`order_for_context()` genuinely applied before the writer sees anything), but with
only one candidate unit ever available, both pass identically whether or not `ask()` calls
`cap()`/`order_for_context()` at all. `jarvis/answer.py`'s own implementation is correct —
this is purely a test-coverage gap in the reference fixture, the same category as the
`FakeEmbedder` noise-floor fixture fix in the gather-and-gate branch's Task 8.

Fix: expand `BLOCKS` to three distinct prose-producing paragraphs across three sections,
so more than one unit is genuinely retrievable, and strengthen the two assertions to
require an actual drop:

```python
BLOCKS = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy in gusts.",
          page=1, section_path=("Results",)),
    Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
    Block(kind="paragraph", text="Performance degrades sharply above 12 m/s wind speed.",
          page=2, section_path=("Limitations",)),
    Block(kind="heading", text="Discussion", page=3, section_path=("Discussion",)),
    Block(kind="paragraph", text="Future work should explore adaptive gain scheduling for "
                                 "extreme wind conditions.", page=3, section_path=("Discussion",)),
]
```

`_prose_unit`'s existing filter (`"94.2" in u.verbatim_text and u.type.value == "prose"`)
still uniquely identifies the tracking-accuracy unit among the three — no other change
needed there.

```python
def test_the_writer_only_ever_sees_capped_ordered_evidence(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["units"] = list(units)
            return Draft()

    ask(corpus, QUESTION, FakeEmbedder(), SpyWriter(), ENTAILS, limit=8, max_units=1)
    assert len(seen["units"]) == 1


def test_the_evidence_cap_is_reported_not_hidden(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS,
                 max_units=1, limit=8)
    assert answer.dropped_evidence > 0, "with 3 candidate units and max_units=1, capping " \
                                        "must genuinely drop at least one"
```

If, after running the suite, `search()` with `limit=8` does not surface all three units for
`QUESTION` (e.g. because keyword matching alone doesn't hit the Discussion/Limitations
paragraphs and vector-only recall via `FakeEmbedder` ranks one of them outside the
practical top results), adjust `QUESTION` or the paragraph text so the fixture's own
`corpus_search`/`retrieve_iteratively` call demonstrably returns at least 2 candidates
before `cap()` — verify this against a real test run rather than assuming it; do not
weaken the assertion back to `>= 0` to make it pass.

---

### Task 5: ALCE-style citation precision and recall

**Files:**
- Modify: `jarvis/evaluate.py`
- Test: `tests/test_evaluate_citations.py`

**Interfaces:**
- Consumes: `Verification`, `Verdict` from `jarvis.models`.
- Produces: `citation_precision(verifications) -> float`, `citation_recall(verifications) -> float`; `EvalReport` gains `citation_precision: float | None = None` and `citation_recall: float | None = None`; `report()` populates both.

Spec §10 lists these as **tracked, no target in v1** — the point is measurement, not a gate. Definitions, stated once so they cannot drift:

- **Citation precision:** of all (claim, citation) pairs, the fraction whose cited quote actually supports the claim. Answers *"when this system cites something, is the citation doing its job?"*
- **Citation recall:** of all distinct claims, the fraction with **at least one** supporting citation. Answers *"is every sentence backed by something?"*

They differ whenever a claim carries several citations, which is why both are needed: a claim cited five times where one citation supports it has recall 1.0 and precision 0.2.

`EvalReport` gains two fields **with defaults, appended at the end**, so existing construction sites keep working. Run the existing `tests/test_evaluate.py` as part of Step 4 to prove that.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate_citations.py
"""ALCE-style citation precision and recall (spec §10, tracked not targeted)."""
import pytest

from jarvis.evaluate import citation_precision, citation_recall, report
from jarvis.models import Verdict, Verification


def _v(claim_id: str, unit_id: str, verdict: Verdict) -> Verification:
    return Verification(claim_id=claim_id, unit_id=unit_id,
                        quote_found=verdict is not Verdict.QUOTE_NOT_FOUND, verdict=verdict)


def test_precision_is_the_fraction_of_citations_that_support():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c1", "u2", Verdict.NEUTRAL),
          _v("c2", "u3", Verdict.SUPPORTED), _v("c3", "u4", Verdict.QUOTE_NOT_FOUND)]
    assert citation_precision(vs) == pytest.approx(0.5)


def test_recall_is_the_fraction_of_claims_with_any_supporting_citation():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c1", "u2", Verdict.NEUTRAL),
          _v("c2", "u3", Verdict.SUPPORTED), _v("c3", "u4", Verdict.QUOTE_NOT_FOUND)]
    assert citation_recall(vs) == pytest.approx(2 / 3)


def test_the_two_metrics_diverge_on_an_over_cited_claim():
    vs = [_v("c1", "u1", Verdict.SUPPORTED)] + \
         [_v("c1", f"u{i}", Verdict.NEUTRAL) for i in range(2, 6)]
    assert citation_recall(vs) == 1.0
    assert citation_precision(vs) == pytest.approx(0.2)


def test_a_perfectly_cited_answer_scores_one_on_both():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c2", "u2", Verdict.SUPPORTED)]
    assert citation_precision(vs) == 1.0
    assert citation_recall(vs) == 1.0


def test_an_answer_with_no_claims_scores_zero_on_both():
    assert citation_precision([]) == 0.0
    assert citation_recall([]) == 0.0


def test_a_contradicted_citation_does_not_count_as_support():
    assert citation_precision([_v("c1", "u1", Verdict.CONTRADICTED)]) == 0.0
    assert citation_recall([_v("c1", "u1", Verdict.CONTRADICTED)]) == 0.0


def test_the_report_carries_both_new_metrics():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c2", "u2", Verdict.NEUTRAL)]
    r = report(vs)
    assert r.citation_precision == pytest.approx(0.5)
    assert r.citation_recall == pytest.approx(0.5)


def test_the_report_still_carries_the_original_metrics():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c2", "u2", Verdict.QUOTE_NOT_FOUND)]
    r = report(vs)
    assert r.quote_fidelity == pytest.approx(0.5)
    assert r.statement_support == pytest.approx(0.5)
    assert r.meets_quote_target is False


def test_citation_metrics_have_no_target_only_a_number():
    r = report([_v("c1", "u1", Verdict.SUPPORTED)])
    assert not hasattr(r, "meets_citation_target"), "spec §10 tracks these, does not gate them"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate_citations.py -v`
Expected: FAIL with `ImportError: cannot import name 'citation_precision' from 'jarvis.evaluate'`

- [ ] **Step 3: Write the implementation**

In `jarvis/evaluate.py`, add these two functions after `statement_support`:

```python
def citation_precision(verifications: Sequence[Verification]) -> float:
    """ALCE-style: fraction of (claim, citation) pairs whose citation supports the claim.

    Answers "when this system cites something, is the citation doing its job?" Tracked,
    no target in v1 (spec §10).
    """
    if not verifications:
        return 0.0
    supported = sum(1 for v in verifications if v.verdict is Verdict.SUPPORTED)
    return supported / len(verifications)


def citation_recall(verifications: Sequence[Verification]) -> float:
    """ALCE-style: fraction of distinct claims with at least one supporting citation.

    Diverges from precision whenever a claim carries several citations: a claim cited five
    times where one supports it has recall 1.0 and precision 0.2. Both numbers are needed.
    """
    by_claim: dict[str, bool] = {}
    for v in verifications:
        by_claim[v.claim_id] = by_claim.get(v.claim_id, False) or \
            (v.verdict is Verdict.SUPPORTED)
    if not by_claim:
        return 0.0
    return sum(1 for ok in by_claim.values() if ok) / len(by_claim)
```

Extend `EvalReport` — append the two fields **after** `coverage` so no positional construction breaks:

```python
@dataclass(frozen=True)
class EvalReport:
    quote_fidelity: float
    statement_support: float
    gate_recall: float | None = None
    coverage: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
```

And populate them in `report()`, adding these two keyword arguments to the existing `EvalReport(...)` call:

```python
        citation_precision=citation_precision(verifications),
        citation_recall=citation_recall(verifications),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_evaluate_citations.py tests/test_evaluate.py -v && ruff check jarvis/evaluate.py tests/test_evaluate_citations.py`
Expected: PASS (9 new + every existing evaluate test still green), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/evaluate.py tests/test_evaluate_citations.py
git commit -m "feat: alce-style citation precision and recall metrics"
```

---

### Task 6: End to end — a question answered, and a fabrication stopped

**Files:**
- Create: `tests/test_qa_end_to_end.py`
- Modify: `jarvis/__init__.py`

**Interfaces:**
- Consumes: everything this plan built.
- Produces: the extended public surface of the `jarvis` package.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qa_end_to_end.py
"""The proof this plan exists to produce: a cited answer, and a fabrication stopped cold."""
import pytest

from jarvis.answer import ask, render_answer
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.evaluate import report
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.parse import FakeParser
from jarvis.retriever import FakeRefiner
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

BLOCKS = [
    Block(kind="heading", text="Results", page=3, section_path=("Results",)),
    Block(kind="paragraph",
          text="As shown in Table 3, our controller reaches 94.2% tracking accuracy under "
               "gust distur-\nbance.", page=3, section_path=("Results",)),
    Block(kind="table", text="| method | accuracy |\n|---|---|\n| ours | 94.2 |",
          page=3, section_path=("Results",), label="Table 3"),
    Block(kind="caption", text="Table 3: Tracking accuracy under wind.", page=3,
          section_path=("Results",), label="Table 3"),
    Block(kind="heading", text="Limitations", page=4, section_path=("Limitations",)),
    Block(kind="paragraph", text="Above 12 m/s the controller loses tracking entirely.",
          page=4, section_path=("Limitations",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Quadrotor Control", year=2025)
QUESTION = "how accurate is the controller under wind?"
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})


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


def _table_unit(conn):
    return next(u for u in get_units(conn, "p1") if u.type.value == "table")


def _limits_unit(conn):
    return next(u for u in get_units(conn, "p1") if "12 m/s" in u.verbatim_text)


def test_a_two_claim_answer_is_fully_supported(corpus):
    table, limits = _table_unit(corpus), _limits_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="Accurate in gusts, but not above 12 m/s.",
        claims=(
            Claim("c-0", "It reaches 94.2% tracking accuracy.", table.unit_id,
                  "| ours | 94.2 |"),
            Claim("c-1", "It fails above 12 m/s.", limits.unit_id,
                  "Above 12 m/s the controller loses tracking entirely."),
        ))})

    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS,
                 refiner=FakeRefiner(["wind speed limits"]), rounds=2)
    assert len(answer.supported) == 2
    assert answer.blocked == ()
    assert answer.is_grounded is True


def test_the_rendered_answer_cites_every_supported_claim(corpus):
    table = _table_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="Accurate.",
        claims=(Claim("c-0", "It reaches 94.2%.", table.unit_id, "| ours | 94.2 |"),))})
    rendered = render_answer(ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS))
    assert table.unit_id in rendered
    assert "94.2" in rendered


def test_a_fabricated_number_never_reaches_the_reader(corpus):
    table = _table_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="It reaches 99.9% accuracy.",
        claims=(Claim("c-0", "It reaches 99.9% accuracy.", table.unit_id,
                      "| ours | 99.9 |"),))})

    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS)
    assert answer.blocked[0].verdict is Verdict.QUOTE_NOT_FOUND
    assert "99.9" not in render_answer(answer)


def test_fabrication_is_caught_even_when_the_model_is_certain(corpus):
    """The NLI model insists the claim is entailed. Stage 1 never asks it."""
    table = _table_unit(corpus)
    certain = FakeNLI(default={"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0})
    writer = FakeWriter({QUESTION: Draft(
        text="x", claims=(Claim("c-0", "99.9%", table.unit_id, "| ours | 99.9 |"),))})
    assert ask(corpus, QUESTION, FakeEmbedder(), writer, certain).supported == ()


def test_a_quote_hyphenated_across_a_line_break_still_grounds(corpus):
    prose = next(u for u in get_units(corpus, "p1")
                 if u.type.value == "prose" and "gust" in u.verbatim_text)
    writer = FakeWriter({QUESTION: Draft(
        text="x", claims=(Claim("c-0", "Gusts disturb it.", prose.unit_id,
                                "under gust disturbance"),))})
    assert ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS).supported != ()


def test_the_eval_report_scores_the_answer(corpus):
    table = _table_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="x",
        claims=(Claim("c-0", "94.2%", table.unit_id, "| ours | 94.2 |"),
                Claim("c-1", "99.9%", table.unit_id, "| ours | 99.9 |")))})

    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS)
    r = report(list(answer.verifications))
    assert r.quote_fidelity == pytest.approx(0.5)
    assert r.meets_quote_target is False
    assert r.citation_recall == pytest.approx(0.5)


def test_the_evidence_reaching_the_writer_is_never_the_whole_corpus(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["count"] = len(units)
            return Draft()

    ask(corpus, QUESTION, FakeEmbedder(), SpyWriter(), ENTAILS, limit=20, max_units=3)
    assert seen["count"] <= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_qa_end_to_end.py -v`
Expected: FAIL on import until the exports land; if any assertion fails, fix the *module*, not the test.

- [ ] **Step 3: Extend the package exports**

Add to `jarvis/__init__.py`, keeping both the import block and `__all__` alphabetically sorted (run `ruff check --fix jarvis/__init__.py` afterwards and confirm the diff is only sorting):

```python
from jarvis.answer import Answer, ask, render_answer
from jarvis.evidence import EvidenceSet, cap, order_for_context
from jarvis.evaluate import citation_precision, citation_recall
from jarvis.retriever import FakeRefiner, LLMRefiner, Refiner, Retrieval, retrieve_iteratively
from jarvis.writer import Draft, FakeWriter, LLMWriter, Writer, claims_from_json
```

Add every one of those names to `__all__`, and update the module docstring to say cited Q&A is built.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass (everything previously green plus ~70 new). `ruff check .` reports exactly the **11 pre-existing** violations.

- [ ] **Step 5: Commit**

```bash
git add jarvis/__init__.py tests/test_qa_end_to_end.py
git commit -m "test: end-to-end cited question answering"
```

---

**Amended post-implementation** (Task 6 review, plan-conflict, human ruling: fix now,
test-only, scoped to this one test):

`test_the_evidence_reaching_the_writer_is_never_the_whole_corpus`, as given above, reuses
the file's shared `corpus` fixture — six blocks producing exactly **3** total units (2
prose + 1 table; the caption binds to the table rather than becoming its own unit). With
`max_units=3` against a 3-unit corpus, `cap()` has nothing to truncate: the assertion
`seen["count"] <= 3` passes identically whether or not `ask()` calls `cap()` at all. Same
category as the Task 4 finding — production code already correct, test-coverage gap only.

Fix: give this one test its own larger, self-contained corpus (5 sections, verified to
produce 5 units — well over `max_units=3`) instead of the shared fixture, so the cap has
something real to enforce, and strengthen the assertion accordingly. Scoped to this test
alone — the shared `corpus`/`BLOCKS` fixture and every other test in the file are
untouched:

```python
def test_the_evidence_reaching_the_writer_is_never_the_whole_corpus(tmp_path):
    """Its own 5-unit corpus, not the shared 3-unit fixture — the cap needs something to
    actually cut, or this test cannot tell a working cap from a deleted one."""
    blocks = [
        Block(kind="heading", text="Results", page=1, section_path=("Results",)),
        Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy.",
              page=1, section_path=("Results",)),
        Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
        Block(kind="paragraph", text="Performance degrades above 12 m/s wind speed.",
              page=2, section_path=("Limitations",)),
        Block(kind="heading", text="Related Work", page=3, section_path=("Related Work",)),
        Block(kind="paragraph", text="Prior controllers used fixed-gain PID schemes.",
              page=3, section_path=("Related Work",)),
        Block(kind="heading", text="Discussion", page=4, section_path=("Discussion",)),
        Block(kind="paragraph", text="Future work should explore adaptive gain scheduling.",
              page=4, section_path=("Discussion",)),
        Block(kind="heading", text="Conclusion", page=5, section_path=("Conclusion",)),
        Block(kind="paragraph", text="The approach generalizes to other wind regimes.",
              page=5, section_path=("Conclusion",)),
    ]
    conn = open_store(tmp_path / "big_corpus.db")
    paper = Paper(paper_id="p2", title="A Larger Paper", year=2025)
    parsed = FakeParser(blocks).parse("big.pdf", "p2")
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["count"] = len(units)
            return Draft()

    answer = ask(conn, "how accurate is the controller under wind?", FakeEmbedder(),
                 SpyWriter(), ENTAILS, limit=20, max_units=3)
    close_store(conn)

    assert seen["count"] == 3
    assert answer.dropped_evidence > 0, "5 candidate units and max_units=3 must genuinely " \
                                        "drop at least one"
```

Verified directly before writing this amendment: `build_units` on the blocks above produces
exactly 5 prose units (one per section), so with the generous `limit=20` used here, `search()`
returns all 5 before `cap()` truncates to 3 — `dropped_evidence` will be `2`. If a real test
run shows otherwise, investigate why (is `search()` genuinely retrieving fewer than 5 for
this question?) rather than weakening the assertion back to `<= 3`.

---

## Definition of done

- `python -m pytest` passes with zero network access, no API keys, no model downloads.
- `test_a_fabricated_number_never_reaches_the_reader` passes — a blocked claim is absent from the rendered output, not annotated in it.
- `test_fabrication_is_caught_even_when_the_model_is_certain` passes — stage 1 is deterministic and never consults the model.
- `test_the_evidence_reaching_the_writer_is_never_the_whole_corpus` passes — the cap is real.
- `jarvis.evaluate.report` returns citation precision and recall alongside the existing metrics.
- `ruff check .` reports exactly the 11 pre-existing violations.

## Where this stops

Single-question, single-pass answers. Not built here:

| Not built | Plan |
|---|---|
| Corpus gathering — this answers from whatever is already ingested | `docs/plans/2026-08-14-gather-and-gate.md` |
| Exposing `ask` as an MCP tool | `docs/plans/2026-08-14-mcp-server.md` |
| Cross-paper contradiction detection | `docs/plans/2026-08-14-contradiction-detection.md` |
| Multi-section long-form reports | `docs/plans/2026-08-14-longform-reports.md` |

One deliberate simplification to revisit with real usage: `ask` treats the question as a single unit. Spec §9 gives `retriever` and `writer` a **sub-question** each, with fan-out. The long-form report plan builds that decomposition, and `ask` should adopt it once outline-driven sub-questions exist rather than growing a second, parallel decomposer here.

---

## Final whole-branch review — fix wave (post-Task-6, human ruling)

The final whole-branch adversarial review, dispatched on the most capable available model
after all 6 tasks passed their own scoped reviews, found 1 Critical + 5 Important findings.
All traced to this plan's own reference code (verbatim, not implementer deviation). Two of
the reviewer's claims were independently re-verified before presenting anything to the
human partner: the Critical finding was reproduced directly against the real store, and the
Important finding about Task 2's fix having no real regression test was confirmed by
reverting `jarvis/retriever.py` to its pre-fix (round-block-concatenation) version and
observing all 14 `test_retriever.py` tests still pass.

Five of the six findings were ruled into this fix wave; one — Task 2's missing regression
test — was ruled **out** and is parked below, not fixed.

### Fix 1 (Critical): a blocked claim's text can render under a citation

**Finding.** `Answer.claim_for` (`jarvis/answer.py`) resolves a claim by `claim_id` via
first-match: `next((c for c in self.claims if c.claim_id == claim_id), None)`.
`verifications` is index-aligned to `claims` by construction in `ask()`
(`tuple(verify_claim(...) for claim in draft.claims)`), but `render_answer` looks claims up
by id through `claim_for` rather than by position. If two claims in one `Draft` ever share
a `claim_id`, `claim_for` always returns the *first* one — so a `SUPPORTED` verdict for the
second claim gets attached to the first claim's `Claim` object when rendering, and vice
versa for a `QUOTE_NOT_FOUND` verdict. The reviewer demonstrated this directly: a draft with
two claims sharing one id, one fabricated (blocked) and one grounded (supported), rendered
the *fabricated* claim's text under a citation in the "supported" section, while the footer
simultaneously reported "1 claim(s) were removed."

No writer shipped in this branch can trigger it today — `claims_from_json` numbers ids
`f"{prefix}-{len(out)}"`, unique within one call, and every `FakeWriter`/`FakeParser` test
fixture in this plan hand-writes unique ids. The exposure is that `Writer` is a
`typing.Protocol` — an untrusted boundary by design — and nothing on the consuming side
(`ask()`/`Answer`) enforces the uniqueness the rendering logic silently depends on. This
directly contradicts the plan's own binding constraint: *"A claim whose quote is not found
is removed from the answer, never merely annotated."*

**Fix.** Minimal and API-preserving, per the reviewer's own recommendation: guarantee
unique `claim_id`s on every `Draft` before verification runs, inside `ask()`, rather than
restructuring `Answer`'s public shape (which would touch every existing test and the not-
yet-built MCP-server plan, which already queued the same `claim_for` pattern twice).

Add to `jarvis/answer.py`, after the existing imports (extend `from dataclasses import
dataclass` to `from dataclasses import dataclass, replace`, and add
`from collections.abc import Sequence`):

```python
def _dedupe_claim_ids(claims: tuple[Claim, ...]) -> tuple[Claim, ...]:
    """Guarantee unique claim_ids within one draft.

    `Answer.claim_for` resolves by first-match id lookup; `verifications` is index-aligned
    to `claims` by construction. `Writer` is an untrusted `typing.Protocol` boundary — a
    well-behaved implementation (`claims_from_json`) already produces unique ids, but
    nothing enforces it structurally. Without this, two claims sharing an id let a later
    claim's verdict get attached to an earlier, unrelated claim's text via first-match
    lookup — including rendering a BLOCKED claim's fabricated text as if it were the
    SUPPORTED claim that happens to share its id.
    """
    seen: set[str] = set()
    out: list[Claim] = []
    for i, claim in enumerate(claims):
        if claim.claim_id in seen:
            out.append(replace(claim, claim_id=f"dedup-{i}"))
        else:
            seen.add(claim.claim_id)
            out.append(claim)
    return tuple(out)
```

### Fix 2 (Important): `ask()` never re-checks a citation against its own evidence set

**Finding.** The "no citation outside the bounded evidence set" rule lives only inside
`claims_from_json` — one implementation of the `Writer` protocol. `ask()` itself hands
`writer.write()` a capped, ordered evidence list but never verifies afterward that every
claim it gets back actually cites a `unit_id` from that list. The reviewer demonstrated a
writer citing a unit outside its shown evidence rendering successfully (the quote was real,
so nothing fabricated shipped, but the citation pointed at content the writer was never
given). This makes "every section gets its own bounded evidence set" a convention one
`Writer` implementation happens to honor, not a mechanical guarantee `ask()` enforces.

**Fix.** Add alongside `_dedupe_claim_ids`:

```python
def _drop_citations_outside_evidence(claims: tuple[Claim, ...],
                                     evidence: Sequence[Unit]) -> tuple[Claim, ...]:
    """Drop any claim citing a unit_id outside the bounded evidence set it was shown.

    `claims_from_json` already enforces this inside one `Writer` implementation, but
    `Writer` is an untrusted Protocol boundary. A real quote from an out-of-budget unit
    would ground and render as if it came from evidence `ask()` actually bounded and
    ordered — the same class of gap `claims_from_json`'s own rejection rules exist to
    close, just unenforced on the consuming side.
    """
    known = {u.unit_id for u in evidence}
    return tuple(c for c in claims if c.unit_id in known)
```

Wire both fixes into `ask()` — replace the body from `draft = writer.write(...)` onward:

```python
    draft = writer.write(question, evidence)
    claims = _dedupe_claim_ids(draft.claims)
    claims = _drop_citations_outside_evidence(claims, evidence)
    verifications = tuple(verify_claim(conn, claim, nli, threshold=threshold)
                          for claim in claims)

    return Answer(question=question, text=draft.text, claims=claims,
                  verifications=verifications, units=tuple(evidence),
                  queries=retrieval.queries, dropped_evidence=budget.dropped)
```

Add tests to `tests/test_answer.py` proving both fixes:

```python
def test_two_claims_sharing_an_id_never_let_one_render_as_the_other(corpus):
    unit = _prose_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="x",
        claims=(
            Claim("dup", "It reaches 99.9% accuracy.", unit.unit_id,
                  "reaches 99.9% tracking accuracy"),   # fabricated, will be blocked
            Claim("dup", "It reaches 94.2% accuracy.", unit.unit_id,
                  "reaches 94.2% tracking accuracy"),   # grounded, will be supported
        ))})
    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS)
    rendered = render_answer(answer)
    assert "99.9" not in rendered, "the blocked claim's fabricated text must never render"
    assert "94.2" in rendered


def test_a_claim_citing_a_unit_outside_the_evidence_set_is_dropped_before_verification(corpus):
    writer = FakeWriter({QUESTION: Draft(
        text="x",
        claims=(Claim("c-0", "invented", "not-a-real-unit-id", "some quote"),))})
    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS)
    assert answer.claims == ()
    assert answer.verifications == ()
```

### Fix 3 (Important): primacy/recency ordering inside `ask()` is unverified

**Finding.** `order_for_context` is unit-tested in isolation (`tests/test_evidence.py`),
but nothing tests that `ask()` actually calls it rather than, say, `reversed()` or nothing
at all. The reviewer swapped `order_for_context(budget.units)` for
`list(reversed(order_for_context(budget.units)))` in `jarvis/answer.py` and the full suite
stayed green. The plan calls this ordering "load-bearing" (evidence.py's own module
docstring); nothing in this plan's tests would notice it silently breaking.

**Fix.** Replace `test_the_writer_only_ever_sees_capped_ordered_evidence` in
`tests/test_answer.py` (the shared 3-unit `corpus` fixture from the Task 4 fix already has
exactly 3 units, which is enough to make front/back interleaving distinguishable from
plain rank order — no fixture change needed):

```python
def test_the_writer_only_ever_sees_capped_ordered_evidence(corpus):
    from jarvis.evidence import cap, order_for_context
    from jarvis.retriever import retrieve_iteratively

    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["units"] = list(units)
            return Draft()

    ask(corpus, QUESTION, FakeEmbedder(), SpyWriter(), ENTAILS, limit=8, max_units=3)

    retrieval = retrieve_iteratively(corpus, QUESTION, FakeEmbedder(), limit=8)
    expected = order_for_context(cap(retrieval.units, max_units=3).units)
    assert [u.unit_id for u in seen["units"]] == [u.unit_id for u in expected], \
        "ask() must hand the writer order_for_context's interleave, not raw rank order"
```

This independently recomputes the expected order using the same primitives `ask()` calls
internally and compares the writer's actual input against it — genuinely mutation-
sensitive, unlike the original assertion (which only checked a count).

### Fix 4 (Important): the "no retrievable evidence" tests never exercise empty retrieval

**Finding.** `test_a_question_with_no_retrievable_evidence_answers_nothing` and
`test_an_empty_answer_renders_an_explicit_no_evidence_message` both query the shared
3-unit `corpus` fixture with a nonsense string. The reviewer confirmed
`retrieve_iteratively` still returns **all 3 units** for that query — `FakeEmbedder`'s
vector search has no relevance floor, so it always ranks every candidate. Both tests only
pass because `FakeWriter({})` returns an empty `Draft()` for an unmapped question key, not
because retrieval was actually empty. This is the fourth instance of the same root cause
already fixed twice in this plan (Tasks 4 and 6): a fixture too small or too uniform to
distinguish working code from broken code.

**Fix.** Point both tests at a genuinely empty store instead of the shared fixture:

```python
def test_a_question_with_no_retrievable_evidence_answers_nothing(tmp_path):
    empty = open_store(tmp_path / "empty.db")
    try:
        answer = ask(empty, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS)
        assert answer.claims == ()
        assert answer.is_grounded is False
    finally:
        close_store(empty)


def test_an_empty_answer_renders_an_explicit_no_evidence_message(tmp_path):
    empty = open_store(tmp_path / "empty.db")
    try:
        answer = ask(empty, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS)
        assert "no" in render_answer(answer).lower()
    finally:
        close_store(empty)
```

An empty store has no rows in `units`, `units_fts`, or `embeddings`, so `search()` returns
`[]` from both the keyword and vector paths — this now exercises the real empty-retrieval
path, not just the writer's own empty-draft fallback.

### Fix 5 (Important): a dead model reads to the user as an empty corpus

**Finding.** `LLMWriter.write` and `LLMRefiner.refine` (both `except Exception:` blocks)
swallow every model failure — network errors, an expired API key, malformed JSON — with no
logging anywhere in either module. An operator whose `JARVIS_API_KEY` has expired gets back
`"No evidence in this corpus answers that question."`: a confident, specific, and false
statement about the corpus. Not a plan-conflict (the plan never mandated logging — this is
a robustness gap it never covered), so no arbitration was needed, but it was bundled into
this fix wave since it's small and touches files already being changed.

**Fix.** Add a module-level logger to `jarvis/writer.py` and `jarvis/retriever.py`:

```python
import logging

_LOGGER = logging.getLogger(__name__)
```

In `jarvis/writer.py`'s `LLMWriter.write`, change:

```python
        except Exception:  # noqa: BLE001
            return Draft()
```

to:

```python
        except Exception:  # noqa: BLE001
            _LOGGER.warning("writer model call failed; the corpus will read as having no "
                            "evidence for this question", exc_info=True)
            return Draft()
```

In `jarvis/retriever.py`'s `LLMRefiner.refine`, change:

```python
        except Exception:  # noqa: BLE001 - a dead refiner ends the loop, never the answer
            return None
```

to:

```python
        except Exception:  # noqa: BLE001 - a dead refiner ends the loop, never the answer
            _LOGGER.warning("retrieval refiner model call failed; stopping refinement early",
                            exc_info=True)
            return None
```

Add regression tests using pytest's `caplog` fixture — to `tests/test_writer.py`:

```python
def test_llm_writer_logs_a_warning_on_failure(caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    with caplog.at_level("WARNING"):
        LLMWriter(_Router(), chat_fn=boom).write("q", UNITS)
    assert any("failed" in r.message.lower() for r in caplog.records)
```

and to `tests/test_retriever.py`:

```python
def test_llm_refiner_logs_a_warning_on_failure(caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    with caplog.at_level("WARNING"):
        LLMRefiner(_Router(), chat_fn=boom).refine("q", ("q",), [])
    assert any("failed" in r.message.lower() for r in caplog.records)
```

### Parked, not fixed: Task 2's RRF fix has no regression test

**Finding.** Confirmed by direct reproduction: reverting `jarvis/retriever.py` to its
pre-fix state (`git show 9ce6a9b:jarvis/retriever.py`, round-block concatenation instead of
RRF fusion) and running `tests/test_retriever.py` leaves all 14 tests passing. The Task 2
amendment's own corrected dedup test (`test_units_are_deduped_across_rounds`) only asserts
id-uniqueness and round count, both of which round-block concatenation already satisfies —
it never asserts anything about *order*, so it cannot distinguish the fix from the bug it
was written to catch. This is a real gap in that amendment's own reasoning, not just the
original implementer's.

**Ruling: not fixed in this branch.** Human-ruled to exclude from this fix wave (2 of 3
Important items presented were selected; this was the one declined). Documented here so it
is never silently dropped. A genuinely discriminating test would need a `FakeReranker`
(already exists in `jarvis/retrieve.py`) forcing round 1 to rank a weak unit last and round
2 to rank a strong unit first, then asserting the round-2 unit precedes the round-1 tail
unit in `result.units` — left as a follow-up, not blocking this plan's completion.

### Explicitly out of scope: `verify.py`'s paper-level quote fallback

`jarvis.verify.quote_is_grounded`'s fallback to `find_span(quote, get_raw_text(conn,
unit.paper_id))` — pre-existing code from the already-merged single-paper-core plan,
unchanged by this branch's diff — lets a quote that exists only in a *different* unit of
the same paper still ground a claim citing the *wrong* unit. The reviewer confirmed this
directly, and correctly scoped it as **not a merge blocker for this branch**: the root
cause predates this plan entirely, and the fallback exists for a real reason (quotes
spanning parent/child unit boundaries). This branch is the first code that renders
`[unit_id]` to a human as a specific location claim, which is what turns a tolerable
internal looseness into a user-visible misattribution risk — worth a dedicated follow-up
task against `jarvis/verify.py` itself, not a change bundled into this plan.
