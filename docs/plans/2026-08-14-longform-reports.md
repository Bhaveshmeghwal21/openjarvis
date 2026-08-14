# Long-Form Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a structured multi-section report over the whole corpus where every claim is verified the same way a one-sentence answer is — and report honestly how much of the corpus the report actually used.

**Architecture:** Spec §7 Stage D, AutoSurvey decomposition, in four passes. **Outline** from paper-level cards (Layer 2 exists for exactly this: coverage bookkeeping and cross-paper comparison). **Parallel subsection drafting**, each section getting its own bounded evidence set and its own retrieval — never one giant context. **Integration**, which deduplicates claims that several sections independently found and orders the result. **Verification**, run over the assembled report, blocking ungrounded claims exactly as `jarvis.answer` does for a single question. Coverage — what fraction of the deep-read corpus got cited — is reported alongside, because citation count and coverage measure different things and only one of them is easy to inflate.

**Tech Stack:** Python 3.10+, stdlib only in the new modules, `pytest`. Outliner and writer are `typing.Protocol`s with deterministic offline fakes.

**Prerequisites:** all three of
- `docs/plans/2026-08-11-verifiable-single-paper-core.md` (merged at `d7f8672`)
- `docs/plans/2026-08-14-gather-and-gate.md` — `Card`, `get_card`, `get_papers_by_depth`, `all_units`
- `docs/plans/2026-08-14-compile-cited-qa.md` — `Writer`, `Draft`, `evidence.cap`, `retriever`, `answer`

This is spec build step 10, the last one, and it is last because it is the furthest thing from the verifiable core: a wrong report is longer, more confident, and harder to check than a wrong sentence.

## Global Constraints

- Python **>= 3.10**. Use `X | None`, not `Optional[X]`.
- **Never read `.env`.** Configuration is environment variables or `$JARVIS_CONFIG` JSON only.
- **Every test is offline.** No network, no API keys, no model downloads.
- All external models are consumed through a `typing.Protocol` with a deterministic fake used in tests.
- Line length **100**. Target `py310`. Run `ruff check .` against **both** the module and its test file before every commit.
- **`jarvis/store.py` is the only module that writes SQL.** This plan adds no SQL.
- **Every section gets its own bounded evidence set.** No code path may assemble one context containing every section's evidence. `jarvis.evidence.cap` is applied per section, not once at the end.
- **The writer never verifies its own output**, and the report-level verification pass is the same `jarvis.verify` machinery, unchanged.
- **A claim whose quote is not in Layer 0 is removed from the report**, not footnoted.
- **Coverage is reported even when it is bad.** A report citing 4 of 300 papers must say so. Suppressing it because it looks weak is the exact failure spec §7D describes.
- Frozen dataclasses for all new types; tuples not lists in frozen types.
- Commit after every task with a `feat:`/`test:`/`fix:` prefix.
- Repo-wide `ruff check .` baseline is **11 pre-existing violations** in `citation_graph.py` (2), `config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1). Do not fix them; do not add to them.

## The failure mode this plan is shaped to avoid

A long report is the easiest artifact in this system to make look excellent and be worthless. Spec §7 Stage D names the mechanism: **increased search depth consistently degrades factual accuracy while surface-level citation metrics stay stable** (arXiv 2605.06635). A twenty-page report with 180 citations reads as more thorough than a two-page one with 20, and the citation-density metric agrees, and it can still be less accurate.

Three design choices follow, and none of them is negotiable:

1. **Per-section evidence budgets, never a global one.** Each section retrieves for its own sub-question and gets its own `cap`. This is the "many small well-scoped calls rather than one large one" instruction, applied literally.
2. **Coverage tracked separately from citation count.** Spec §10 lists coverage as its own metric precisely because citation count can be inflated by citing the same eight papers repeatedly. Task 4 computes it over the `read_deep` corpus and Task 5 prints it whether or not it flatters the report.
3. **The same verification pass as a single answer.** A report gets no leniency for being long. `quote_fidelity` on a 200-claim report must still be 1.0.

## File Structure

| File | Responsibility |
|---|---|
| `jarvis/outline.py` | Create. `Section`, `Outline`, `Outliner` protocol, template and LLM outliners, card digest. |
| `jarvis/report.py` | Create. Section drafting, integration, assembly, coverage, rendering. |
| `jarvis/__init__.py` | **Modify.** Export the new surface. |

Tests: `tests/test_outline.py`, `tests/test_report.py`, `tests/test_report_end_to_end.py`.

---

### Task 1: The outline, built from cards

**Files:**
- Create: `jarvis/outline.py`
- Test: `tests/test_outline.py`

**Interfaces:**
- Consumes: `Card`, `CardField` from `jarvis.models`; `jarvis.llm.chat` through injection.
- Produces: `Section` (frozen: `title: str`, `question: str`, `paper_ids: tuple[str, ...] = ()`), `Outline` (frozen: `topic: str`, `sections: tuple[Section, ...] = ()`), `Outliner` protocol with `outline(topic, cards) -> Outline`, `TemplateOutliner`, `LLMOutliner(router, chat_fn=None, max_sections=8)`, `cards_digest(cards, max_papers=100) -> str`.

Layer 2 finally earns its keep here. Spec §5 gives the card exactly one job — *"coverage bookkeeping and cross-paper comparison"* — and an outline is the cross-paper comparison. The card is still never the ground for a claim: it decides what the sections are, and then each section retrieves its own Layer 1 evidence from scratch.

`TemplateOutliner` is deterministic and emits only sections the corpus can actually support: a "Datasets and benchmarks" section appears only if some card has a `datasets` field. A section with no evidence behind it is an invitation for a model to invent one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_outline.py
"""Report outlines, built from Layer 2 cards (spec §5, §7D)."""
import pytest

from jarvis.models import Card, CardField
from jarvis.outline import (
    LLMOutliner,
    Outline,
    Outliner,
    Section,
    TemplateOutliner,
    cards_digest,
)

CARDS = [
    Card(paper_id="p1",
         problem=CardField("gust rejection", "u1", "gusts"),
         method=CardField("adaptive control", "u2", "adaptive"),
         datasets=(CardField("KITTI", "u3", "KITTI"),),
         metrics=(CardField("94.2", "u4", "94.2", binding_verified=True),),
         limitations=(CardField("fails above 12 m/s", "u5", "12 m/s"),)),
    Card(paper_id="p2",
         problem=CardField("wind disturbance", "u6", "wind"),
         metrics=(CardField("61.0", "u7", "61.0"),)),
]
BARE = [Card(paper_id="p9", problem=CardField("something", "u1", "q"))]


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_template_outliner_satisfies_the_protocol():
    assert isinstance(TemplateOutliner(), Outliner)


def test_an_outline_has_sections_with_their_own_sub_questions():
    outline = TemplateOutliner().outline("gust rejection", CARDS)
    assert outline.topic == "gust rejection"
    assert len(outline.sections) >= 3
    assert all(s.question for s in outline.sections)
    assert all(s.title for s in outline.sections)


def test_sections_only_appear_when_the_corpus_can_support_them():
    rich = {s.title for s in TemplateOutliner().outline("t", CARDS).sections}
    bare = {s.title for s in TemplateOutliner().outline("t", BARE).sections}
    assert any("ataset" in t for t in rich)
    assert not any("ataset" in t for t in bare)
    assert not any("imitation" in t for t in bare)


def test_an_empty_corpus_still_yields_a_minimal_outline():
    outline = TemplateOutliner().outline("gust rejection", [])
    assert len(outline.sections) >= 1
    assert "gust rejection" in outline.sections[0].question


def test_the_topic_appears_in_every_sub_question():
    for section in TemplateOutliner().outline("gust rejection", CARDS).sections:
        assert "gust rejection" in section.question


def test_sections_record_which_papers_motivated_them():
    outline = TemplateOutliner().outline("t", CARDS)
    datasets = next(s for s in outline.sections if "ataset" in s.title)
    assert datasets.paper_ids == ("p1",)


def test_the_template_outliner_is_deterministic():
    assert TemplateOutliner().outline("t", CARDS) == TemplateOutliner().outline("t", CARDS)


def test_the_card_digest_names_papers_and_their_fields():
    digest = cards_digest(CARDS)
    assert "p1" in digest
    assert "gust rejection" in digest
    assert "KITTI" in digest


def test_the_card_digest_is_capped():
    many = [Card(paper_id=f"p{i}", problem=CardField(f"topic {i}", "u", "q"))
            for i in range(500)]
    assert "p499" not in cards_digest(many, max_papers=10)


def test_llm_outliner_uses_the_models_sections():
    reply = {"sections": [
        {"title": "Control strategies", "question": "what control strategies exist?",
         "paper_ids": ["p1"]},
        {"title": "Reported accuracy", "question": "what accuracy is reported?"},
    ]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply).outline("t", CARDS)
    assert [s.title for s in outline.sections] == ["Control strategies", "Reported accuracy"]
    assert outline.sections[0].paper_ids == ("p1",)
    assert outline.sections[1].paper_ids == ()


def test_llm_outliner_drops_sections_missing_a_title_or_question():
    reply = {"sections": [{"title": "Only a title"}, {"question": "only a question"},
                          {"title": "Good", "question": "good?"}]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply).outline("t", CARDS)
    assert [s.title for s in outline.sections] == ["Good"]


def test_llm_outliner_drops_paper_ids_not_in_the_corpus():
    reply = {"sections": [{"title": "T", "question": "q?", "paper_ids": ["p1", "ghost"]}]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply).outline("t", CARDS)
    assert outline.sections[0].paper_ids == ("p1",)


def test_llm_outliner_caps_the_section_count():
    reply = {"sections": [{"title": f"T{i}", "question": f"q{i}?"} for i in range(50)]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply,
                          max_sections=5).outline("t", CARDS)
    assert len(outline.sections) == 5


def test_llm_outliner_falls_back_to_the_template_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    assert LLMOutliner(_Router(), chat_fn=boom).outline("t", CARDS) == \
        TemplateOutliner().outline("t", CARDS)


def test_llm_outliner_falls_back_on_junk():
    for junk in (None, {}, "text", {"sections": []}, {"sections": "x"}):
        assert LLMOutliner(_Router(), chat_fn=lambda *a, **k: junk).outline("t", CARDS) == \
            TemplateOutliner().outline("t", CARDS)


def test_llm_outliner_routes_to_outline():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {}

    LLMOutliner(_Router(), chat_fn=spy).outline("t", CARDS)
    assert seen["task"] == "outline"


def test_outline_types_are_frozen():
    with pytest.raises(Exception):
        Section(title="a", question="b").title = "c"
    with pytest.raises(Exception):
        Outline(topic="t").topic = "u"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_outline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.outline'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/outline.py
"""Report outlines from Layer 2 cards (spec §5, §7 Stage D).

AutoSurvey's decomposition starts from paper-level structure rather than from retrieval,
and this is where the card finally earns its keep: spec §5 gives it exactly one job —
coverage bookkeeping and cross-paper comparison — and an outline is that comparison.

The card still never grounds a claim. It decides what the sections ARE; each section then
retrieves its own Layer 1 evidence from scratch.

`TemplateOutliner` emits only sections the corpus can support. A section with no evidence
behind it is an invitation for a model to invent one.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.models import Card

MAX_SECTIONS = 8

_OUTLINE_PROMPT = (
    "Plan a structured literature report on this topic from the paper summaries below.\n"
    "Return JSON: {{\"sections\": [{{\"title\": ..., \"question\": ..., "
    "\"paper_ids\": [...]}}]}}\n"
    "At most {max_sections} sections. `question` is the single question that section "
    "answers, phrased so it can be used as a search query against the corpus. "
    "`paper_ids` lists the papers that motivated the section — use only ids shown below.\n"
    "Propose a section only if the summaries show there is material for it.\n\n"
    "Topic: {topic}\n\n{digest}"
)

# (attribute, singular label, section title, sub-question template)
_TEMPLATE_SECTIONS = (
    ("problem", "problem", "Problem framing", "what problem does {topic} address?"),
    ("method", "method", "Methods", "what methods are used for {topic}?"),
    ("datasets", "dataset", "Datasets and benchmarks",
     "what datasets and benchmarks are used for {topic}?"),
    ("metrics", "metric", "Reported results", "what results are reported for {topic}?"),
    ("limitations", "limitation", "Limitations and open problems",
     "what are the limitations and open problems in {topic}?"),
)


@dataclass(frozen=True)
class Section:
    """One report section and the sub-question it answers."""
    title: str
    question: str
    paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outline:
    topic: str
    sections: tuple[Section, ...] = ()


@runtime_checkable
class Outliner(Protocol):
    def outline(self, topic: str, cards: Sequence[Card]) -> Outline: ...


def _values(card: Card, attribute: str) -> list[str]:
    value = getattr(card, attribute, None)
    if value is None:
        return []
    if isinstance(value, tuple):
        return [f.value for f in value if f.value]
    return [value.value] if value.value else []


def cards_digest(cards: Sequence[Card], max_papers: int = 100) -> str:
    """Compact cross-paper summary — the outliner's whole input."""
    lines: list[str] = []
    for card in list(cards)[:max_papers]:
        parts = []
        for attribute, _label, _title, _q in _TEMPLATE_SECTIONS:
            values = _values(card, attribute)
            if values:
                parts.append(f"{attribute}: {'; '.join(values[:4])}")
        lines.append(f"[{card.paper_id}] " + (" | ".join(parts) or "(no card fields)"))
    return "\n".join(lines)


class TemplateOutliner:
    """Deterministic, free, no model. The fallback and the test default."""

    def outline(self, topic: str, cards: Sequence[Card]) -> Outline:
        clean = " ".join((topic or "").split())
        sections: list[Section] = [Section(
            title="Overview",
            question=f"what is the state of the art in {clean}?",
        )]
        for attribute, _label, title, question in _TEMPLATE_SECTIONS:
            papers = tuple(c.paper_id for c in cards if _values(c, attribute))
            if papers:
                sections.append(Section(title=title, question=question.format(topic=clean),
                                        paper_ids=papers))
        return Outline(topic=clean, sections=tuple(sections))


class LLMOutliner:
    """Model-planned outline, routed to the frontier tier. Falls back to the template."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None,
                 max_sections: int = MAX_SECTIONS) -> None:
        self._router = router
        self._chat = chat_fn
        self._max_sections = max_sections
        self._fallback = TemplateOutliner()

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def outline(self, topic: str, cards: Sequence[Card]) -> Outline:
        fallback = self._fallback.outline(topic, cards)
        prompt = _OUTLINE_PROMPT.format(max_sections=self._max_sections,
                                        topic=fallback.topic, digest=cards_digest(cards))
        try:
            raw = self._chat_fn()(self._router, "outline", prompt, json_mode=True)
        except Exception:  # noqa: BLE001
            return fallback
        if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
            return fallback

        known = {c.paper_id for c in cards}
        sections: list[Section] = []
        for item in raw["sections"]:
            if not isinstance(item, dict) or len(sections) >= self._max_sections:
                continue
            title = " ".join(str(item.get("title", "") or "").split())
            question = " ".join(str(item.get("question", "") or "").split())
            if not title or not question:
                continue
            ids = item.get("paper_ids") or []
            papers = tuple(str(p) for p in ids if str(p) in known) \
                if isinstance(ids, list) else ()
            sections.append(Section(title=title, question=question, paper_ids=papers))

        return Outline(topic=fallback.topic, sections=tuple(sections)) if sections \
            else fallback
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_outline.py -v && ruff check jarvis/outline.py tests/test_outline.py`
Expected: PASS (17 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/outline.py tests/test_outline.py
git commit -m "feat: report outlines built from layer 2 cards"
```

---

### Task 2: Section drafting with its own evidence budget

**Files:**
- Create: `jarvis/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `Section`, `Outline` from `jarvis.outline`; `retrieve_iteratively`, `Refiner` from `jarvis.retriever`; `cap`, `order_for_context`, `MAX_UNITS`, `MAX_TOKENS` from `jarvis.evidence`; `Writer` from `jarvis.writer`; `verify_claim`, `NLIModel` from `jarvis.verify`; `Verdict` from `jarvis.models`.
- Produces: `SectionDraft` (frozen: `section`, `text`, `claims`, `verifications`, `units`, `dropped_evidence`, plus `supported` / `flagged` / `blocked` properties mirroring `Answer`), `draft_section(conn, section, embedder, writer, nli, *, refiner=None, rounds=2, limit=8, reranker=None, max_units=MAX_UNITS, max_tokens=MAX_TOKENS, threshold=0.5) -> SectionDraft`.

This is `jarvis.answer.ask` at section scope, and it is deliberately a separate function rather than a call into `ask`: a section carries its `Section` (title, motivating papers) through to rendering, and `Answer` has no place to put that. The retrieval, budgeting, and verification logic is identical and must stay identical — if the two ever diverge, the report has quietly become less checked than a one-line answer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
"""Section drafting: one bounded evidence set per section, verified like any answer."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.outline import Section
from jarvis.parse import FakeParser
from jarvis.report import SectionDraft, draft_section
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

BLOCKS_A = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
          page=1, section_path=("Results",)),
]
BLOCKS_B = [
    Block(kind="heading", text="Limitations", page=1, section_path=("Limitations",)),
    Block(kind="paragraph", text="Tracking degrades sharply above 12 m/s wind speed.",
          page=1, section_path=("Limitations",)),
]
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})
NEUTRAL = FakeNLI(default={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})
RESULTS = Section(title="Reported results", question="what accuracy is reported?")


def _ingest(conn, paper_id, blocks):
    paper = Paper(paper_id=paper_id, title=f"Paper {paper_id}", year=2025)
    parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    _ingest(conn, "p1", BLOCKS_A)
    _ingest(conn, "p2", BLOCKS_B)
    yield conn
    close_store(conn)


def _unit(conn, paper_id, needle):
    return next(u for u in get_units(conn, paper_id) if needle in u.verbatim_text)


def _writer(conn, quote, text="It reaches 94.2% accuracy."):
    unit = _unit(conn, "p1", "94.2")
    return FakeWriter({RESULTS.question: Draft(
        text="Accuracy is high.",
        claims=(Claim("c-0", text, unit.unit_id, quote),))})


def test_a_section_draft_carries_its_section(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), ENTAILS)
    assert draft.section is RESULTS


def test_a_grounded_section_claim_is_supported(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), ENTAILS)
    assert len(draft.supported) == 1
    assert draft.blocked == ()


def test_a_fabricated_section_claim_is_blocked(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "99.9% tracking accuracy"), ENTAILS)
    assert len(draft.blocked) == 1
    assert draft.blocked[0].verdict is Verdict.QUOTE_NOT_FOUND
    assert draft.supported == ()


def test_a_real_quote_that_does_not_entail_is_flagged(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), NEUTRAL)
    assert len(draft.flagged) == 1
    assert draft.blocked == ()


def test_the_section_is_searched_on_its_own_sub_question(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["question"] = question
            return Draft()

    draft_section(corpus, RESULTS, FakeEmbedder(), SpyWriter(), ENTAILS)
    assert seen["question"] == RESULTS.question


def test_each_section_gets_its_own_capped_evidence(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["count"] = len(units)
            return Draft()

    draft_section(corpus, RESULTS, FakeEmbedder(), SpyWriter(), ENTAILS,
                  limit=20, max_units=2)
    assert seen["count"] <= 2


def test_the_dropped_evidence_count_is_reported(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(), FakeWriter({}), ENTAILS,
                          limit=8, max_units=1)
    assert draft.dropped_evidence >= 0


def test_a_section_with_no_retrievable_evidence_drafts_nothing(corpus):
    empty = Section(title="Nothing", question="zzz nonexistent qqq topic")
    draft = draft_section(corpus, empty, FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert draft.claims == ()
    assert draft.text == ""


def test_section_draft_is_frozen(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(), FakeWriter({}), ENTAILS)
    with pytest.raises(Exception):
        draft.text = "rewritten"


def test_a_section_draft_records_the_units_it_saw(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), ENTAILS)
    assert len(draft.units) > 0
    assert all(hasattr(u, "unit_id") for u in draft.units)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.report'`

- [ ] **Step 3: Write the implementation**

```python
# jarvis/report.py
"""Long-form reports — AutoSurvey decomposition (spec §7 Stage D).

Outline from cards, draft each subsection against its OWN bounded evidence set, integrate,
verify. A report gets no leniency for being long: the verification pass is exactly the one
`jarvis.answer` runs for a single sentence.

Per-section budgets rather than one global context is the load-bearing choice. Increased
search depth consistently degrades factual accuracy while surface-level citation metrics
stay stable (arXiv 2605.06635) — a long report is the easiest artifact in this system to
make look excellent and be worthless.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.embed import Embedder
from jarvis.evidence import MAX_TOKENS, MAX_UNITS, cap, order_for_context
from jarvis.models import Claim, Unit, Verdict, Verification
from jarvis.outline import Section
from jarvis.retrieve import Reranker
from jarvis.retriever import Refiner, retrieve_iteratively
from jarvis.verify import NLIModel, verify_claim
from jarvis.writer import Writer

FLAGGED_VERDICTS = (Verdict.NEUTRAL, Verdict.CONTRADICTED)


@dataclass(frozen=True)
class SectionDraft:
    """One drafted, verified section. Mirrors `jarvis.answer.Answer` at section scope."""
    section: Section
    text: str = ""
    claims: tuple[Claim, ...] = ()
    verifications: tuple[Verification, ...] = ()
    units: tuple[Unit, ...] = ()
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

    def claim_for(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)


def draft_section(conn: sqlite3.Connection, section: Section, embedder: Embedder,
                  writer: Writer, nli: NLIModel, *, refiner: Refiner | None = None,
                  rounds: int = 2, limit: int = 8, reranker: Reranker | None = None,
                  max_units: int = MAX_UNITS, max_tokens: int = MAX_TOKENS,
                  threshold: float = 0.5) -> SectionDraft:
    """Retrieve for this section's sub-question only, cap, draft, verify.

    The cap is applied here, per section — never once over an assembled whole-report
    context. That is the difference between many small well-scoped calls and one large
    one, and the measured difference is 13 F1 points.
    """
    retrieval = retrieve_iteratively(conn, section.question, embedder, refiner=refiner,
                                     rounds=rounds, limit=limit, reranker=reranker)
    budget = cap(retrieval.units, max_units=max_units, max_tokens=max_tokens)
    evidence = order_for_context(budget.units)

    draft = writer.write(section.question, evidence)
    verifications = tuple(verify_claim(conn, claim, nli, threshold=threshold)
                          for claim in draft.claims)

    return SectionDraft(section=section, text=draft.text, claims=draft.claims,
                        verifications=verifications, units=tuple(evidence),
                        dropped_evidence=budget.dropped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v && ruff check jarvis/report.py tests/test_report.py`
Expected: PASS (10 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/report.py tests/test_report.py
git commit -m "feat: per-section drafting with its own evidence budget"
```

---

### Task 3: The integration pass

**Files:**
- Modify: `jarvis/report.py` (append)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Consumes: `SectionDraft` (Task 2).
- Produces: `integrate(drafts) -> list[SectionDraft]`, `duplicate_claims(drafts) -> list[tuple[str, str]]`.

AutoSurvey's integration pass, made concrete and deterministic: sections are drafted independently, so several of them retrieve the same unit and make the same claim. The first section to make a claim keeps it; later ones drop it, along with its verification.

"Same claim" is defined narrowly — **identical `unit_id` and identical normalized claim text**. Two sections legitimately citing the same table for different points must both keep their claim, and a fuzzy match would silently delete the second one. Under-merging here costs a little repetition; over-merging costs content that no reader ever sees.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report.py`:

```python
from jarvis.report import duplicate_claims, integrate

S1 = Section(title="One", question="q1")
S2 = Section(title="Two", question="q2")


def _draft(section, claims, verdicts=None):
    from jarvis.models import Verification
    verdicts = verdicts or [Verdict.SUPPORTED] * len(claims)
    return SectionDraft(
        section=section, text="t", claims=tuple(claims),
        verifications=tuple(Verification(claim_id=c.claim_id, unit_id=c.unit_id,
                                         quote_found=True, verdict=v)
                            for c, v in zip(claims, verdicts)))


def test_a_claim_repeated_across_sections_survives_only_in_the_first():
    shared = Claim("a-0", "The controller reaches 94.2%.", "u1", "94.2")
    later = Claim("b-0", "The controller reaches 94.2%.", "u1", "94.2")
    merged = integrate([_draft(S1, [shared]), _draft(S2, [later])])
    assert len(merged[0].claims) == 1
    assert merged[1].claims == ()


def test_dropping_a_duplicate_claim_drops_its_verification_too():
    shared = Claim("a-0", "same", "u1", "q")
    later = Claim("b-0", "same", "u1", "q")
    merged = integrate([_draft(S1, [shared]), _draft(S2, [later])])
    assert merged[1].verifications == ()


def test_the_same_unit_cited_for_two_different_points_keeps_both():
    merged = integrate([_draft(S1, [Claim("a-0", "It is accurate.", "u1", "q")]),
                        _draft(S2, [Claim("b-0", "It is fast.", "u1", "q")])])
    assert len(merged[0].claims) == 1
    assert len(merged[1].claims) == 1


def test_the_same_point_from_two_different_units_keeps_both():
    merged = integrate([_draft(S1, [Claim("a-0", "It is accurate.", "u1", "q")]),
                        _draft(S2, [Claim("b-0", "It is accurate.", "u2", "q")])])
    assert len(merged[1].claims) == 1


def test_duplicate_matching_ignores_whitespace_and_case():
    merged = integrate([_draft(S1, [Claim("a-0", "It  is Accurate.", "u1", "q")]),
                        _draft(S2, [Claim("b-0", "it is accurate.", "u1", "q")])])
    assert merged[1].claims == ()


def test_a_claim_repeated_inside_one_section_is_also_deduped():
    merged = integrate([_draft(S1, [Claim("a-0", "same", "u1", "q"),
                                    Claim("a-1", "same", "u1", "q")])])
    assert len(merged[0].claims) == 1


def test_integration_preserves_section_order_and_identity():
    merged = integrate([_draft(S1, []), _draft(S2, [])])
    assert [d.section.title for d in merged] == ["One", "Two"]


def test_duplicates_are_reportable_not_only_removed():
    shared = Claim("a-0", "same", "u1", "q")
    later = Claim("b-0", "same", "u1", "q")
    dupes = duplicate_claims([_draft(S1, [shared]), _draft(S2, [later])])
    assert dupes == [("Two", "b-0")]


def test_integrating_nothing_is_nothing():
    assert integrate([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'integrate' from 'jarvis.report'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/report.py`, adding `from dataclasses import dataclass, replace` and `from jarvis.text import normalize` to the imports:

```python
def _claim_key(claim: Claim) -> tuple[str, str]:
    """What makes two claims the same claim: same unit, same normalized text.

    Deliberately narrow. Two sections citing the same table for different points must both
    keep their claim — under-merging costs a little repetition, over-merging deletes
    content no reader will ever see.
    """
    return (claim.unit_id, normalize(claim.text).lower())


def integrate(drafts: Sequence[SectionDraft]) -> list[SectionDraft]:
    """AutoSurvey's integration pass: first occurrence of a claim wins, later ones drop.

    Sections are drafted independently, so several will retrieve the same unit and make
    the same point. A dropped claim takes its verification with it, so the report-level
    metrics count each claim once.
    """
    seen: set[tuple[str, str]] = set()
    out: list[SectionDraft] = []
    for draft in drafts:
        kept: list[Claim] = []
        for claim in draft.claims:
            key = _claim_key(claim)
            if key in seen:
                continue
            seen.add(key)
            kept.append(claim)
        kept_ids = {c.claim_id for c in kept}
        out.append(replace(
            draft, claims=tuple(kept),
            verifications=tuple(v for v in draft.verifications if v.claim_id in kept_ids),
        ))
    return out


def duplicate_claims(drafts: Sequence[SectionDraft]) -> list[tuple[str, str]]:
    """(section title, claim_id) for every claim integration would drop. For auditing."""
    seen: set[tuple[str, str]] = set()
    dropped: list[tuple[str, str]] = []
    for draft in drafts:
        for claim in draft.claims:
            key = _claim_key(claim)
            if key in seen:
                dropped.append((draft.section.title, claim.claim_id))
            else:
                seen.add(key)
    return dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v && ruff check jarvis/report.py tests/test_report.py`
Expected: PASS (19 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/report.py tests/test_report.py
git commit -m "feat: integration pass deduplicating claims across sections"
```

---

### Task 4: Report assembly and coverage

**Files:**
- Modify: `jarvis/report.py` (append)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Consumes: `Outline`, `Outliner` from `jarvis.outline`; `get_card`, `get_papers_by_depth`, `all_units` from `jarvis.store`; `coverage` and `report` from `jarvis.evaluate`; `integrate`, `draft_section` (Tasks 2–3).
- Produces: `Report` (frozen: `topic`, `outline`, `sections`, `coverage`, `corpus_unit_ids`, plus `corpus_units` / `all_verifications` / `all_claims` / `cited_unit_ids` / `cited_paper_ids` properties), `corpus_cards(conn) -> list[Card]`, `write_report(conn, topic, outliner, embedder, writer, nli, *, ...) -> Report`, `evaluate_report(report) -> EvalReport`.

Coverage is **the fraction of the deep-read corpus's units that got cited anywhere in the report** — spec §10's definition, computed against `all_units` restricted to `read_deep` papers. It is tracked, not targeted, and it is reported whether or not it flatters the result. A report citing 4 units out of 8,000 is a real fact about that report.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report.py`:

```python
from jarvis.models import Card, CardField
from jarvis.outline import Outline, TemplateOutliner
from jarvis.report import Report, corpus_cards, evaluate_report, write_report
from jarvis.store import save_card


def test_a_report_covers_every_section_of_its_outline(corpus):
    outline = Outline(topic="gusts", sections=(
        Section(title="Results", question="what accuracy is reported?"),
        Section(title="Limits", question="what are the wind speed limits?"),
    ))
    result = write_report(corpus, "gusts", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert [s.section.title for s in result.sections] == ["Results", "Limits"]


def test_a_report_can_build_its_own_outline_from_cards(corpus):
    save_card(corpus, Card(paper_id="p1",
                           problem=CardField("gust rejection", "u1", "q"),
                           metrics=(CardField("94.2", "u2", "q"),)))
    result = write_report(corpus, "gusts", TemplateOutliner(), FakeEmbedder(),
                          FakeWriter({}), ENTAILS)
    assert len(result.sections) >= 2
    assert result.outline.topic == "gusts"


def test_corpus_cards_reads_every_deep_paper_that_has_one(corpus):
    save_card(corpus, Card(paper_id="p1", problem=CardField("a", "u1", "q")))
    cards = corpus_cards(corpus)
    assert [c.paper_id for c in cards] == ["p1"]


def test_coverage_is_the_cited_fraction_of_the_deep_corpus(corpus):
    unit = _unit(corpus, "p1", "94.2")
    question = "what accuracy is reported?"
    writer = FakeWriter({question: Draft(
        text="t", claims=(Claim("c-0", "94.2%", unit.unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="R", question=question),))

    result = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert 0.0 < result.coverage < 1.0
    assert result.corpus_units > 1


def test_coverage_is_zero_when_nothing_is_cited(corpus):
    outline = Outline(topic="gusts", sections=(Section(title="R", question="q"),))
    result = write_report(corpus, "gusts", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert result.coverage == 0.0


def test_a_blocked_claim_does_not_count_toward_coverage(corpus):
    unit = _unit(corpus, "p1", "94.2")
    question = "what accuracy is reported?"
    writer = FakeWriter({question: Draft(
        text="t", claims=(Claim("c-0", "99.9%", unit.unit_id, "99.9% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="R", question=question),))

    result = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert result.coverage == 0.0, "an ungrounded citation is not coverage"


def test_the_report_deduplicates_claims_across_sections(corpus):
    unit = _unit(corpus, "p1", "94.2")
    claim_text = "The controller reaches 94.2%."
    writer = FakeWriter({
        "q1": Draft(text="t", claims=(Claim("a-0", claim_text, unit.unit_id,
                                            "94.2% tracking accuracy"),)),
        "q2": Draft(text="t", claims=(Claim("b-0", claim_text, unit.unit_id,
                                            "94.2% tracking accuracy"),)),
    })
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),
                                           Section(title="B", question="q2")))
    result = write_report(corpus, "t", outline, FakeEmbedder(), writer, ENTAILS)
    assert len(result.all_claims) == 1


def test_the_report_aggregates_every_verification(corpus):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="t", claims=(
        Claim("a-0", "good", unit.unit_id, "94.2% tracking accuracy"),
        Claim("a-1", "bad", unit.unit_id, "99.9% tracking accuracy")))})
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),))

    result = write_report(corpus, "t", outline, FakeEmbedder(), writer, ENTAILS)
    assert len(result.all_verifications) == 2


def test_the_report_evaluates_like_any_other_answer(corpus):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="t", claims=(
        Claim("a-0", "good", unit.unit_id, "94.2% tracking accuracy"),
        Claim("a-1", "bad", unit.unit_id, "99.9% tracking accuracy")))})
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),))

    evaluation = evaluate_report(write_report(corpus, "t", outline, FakeEmbedder(),
                                              writer, ENTAILS))
    assert evaluation.quote_fidelity == pytest.approx(0.5)
    assert evaluation.meets_quote_target is False
    assert evaluation.coverage is not None


def test_cited_paper_ids_lists_only_papers_with_a_supported_claim(corpus):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="t", claims=(
        Claim("a-0", "good", unit.unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),))
    result = write_report(corpus, "t", outline, FakeEmbedder(), writer, ENTAILS)
    assert result.cited_paper_ids == {"p1"}


def test_report_is_frozen(corpus):
    outline = Outline(topic="t", sections=())
    result = write_report(corpus, "t", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    with pytest.raises(Exception):
        result.coverage = 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'Report' from 'jarvis.report'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/report.py`, adding these imports:

```python
from jarvis.evaluate import EvalReport, coverage
from jarvis.evaluate import report as eval_report
from jarvis.models import Card
from jarvis.outline import Outline, Outliner
from jarvis.store import all_units, get_card, get_papers_by_depth
```

```python
@dataclass(frozen=True)
class Report:
    topic: str
    outline: Outline
    sections: tuple[SectionDraft, ...] = ()
    coverage: float = 0.0
    corpus_unit_ids: tuple[str, ...] = ()

    @property
    def corpus_units(self) -> int:
        return len(self.corpus_unit_ids)

    @property
    def all_claims(self) -> tuple[Claim, ...]:
        return tuple(c for s in self.sections for c in s.claims)

    @property
    def all_verifications(self) -> tuple[Verification, ...]:
        return tuple(v for s in self.sections for v in s.verifications)

    @property
    def cited_unit_ids(self) -> set[str]:
        """Units backing a SUPPORTED claim. A blocked citation is not coverage."""
        return {v.unit_id for v in self.all_verifications if v.verdict is Verdict.SUPPORTED}

    @property
    def cited_paper_ids(self) -> set[str]:
        """Papers behind a supported claim, resolved through the units the sections saw.

        Deliberately not parsed out of `unit_id`. That id is
        f"{paper_id}:{type}:{page}:{ordinal}", and `citation_graph.paper_id` falls back to
        a title prefix when a paper has no arXiv or S2 id — titles routinely contain
        colons ("Attention: All You Need"), so splitting on the first one would silently
        truncate the paper.
        """
        cited = self.cited_unit_ids
        return {u.paper_id for s in self.sections for u in s.units if u.unit_id in cited}


def corpus_cards(conn: sqlite3.Connection) -> list[Card]:
    """Every Layer 2 card in the deep-read corpus — the outliner's input."""
    cards: list[Card] = []
    for paper in get_papers_by_depth(conn, "deep"):
        card = get_card(conn, paper.paper_id)
        if card is not None:
            cards.append(card)
    return cards


def write_report(conn: sqlite3.Connection, topic: str, outliner: Outliner | Outline,
                 embedder: Embedder, writer: Writer, nli: NLIModel, *,
                 refiner: Refiner | None = None, rounds: int = 2, limit: int = 8,
                 reranker: Reranker | None = None, max_units: int = MAX_UNITS,
                 max_tokens: int = MAX_TOKENS, threshold: float = 0.5) -> Report:
    """Outline, draft each section independently, integrate, measure coverage.

    `outliner` may be an `Outliner` or an already-built `Outline`, so a caller can inspect
    or hand-edit the plan before spending a model call per section on it.
    """
    outline = outliner if isinstance(outliner, Outline) \
        else outliner.outline(topic, corpus_cards(conn))

    drafts = [draft_section(conn, section, embedder, writer, nli, refiner=refiner,
                            rounds=rounds, limit=limit, reranker=reranker,
                            max_units=max_units, max_tokens=max_tokens,
                            threshold=threshold)
              for section in outline.sections]
    sections = tuple(integrate(drafts))

    deep_ids = {p.paper_id for p in get_papers_by_depth(conn, "deep")}
    corpus_unit_ids = [u.unit_id for u in all_units(conn) if u.paper_id in deep_ids]
    cited = {v.unit_id for s in sections for v in s.supported}

    return Report(topic=outline.topic, outline=outline, sections=sections,
                  coverage=coverage(cited, corpus_unit_ids),
                  corpus_unit_ids=tuple(corpus_unit_ids))


def evaluate_report(report: Report) -> EvalReport:
    """Spec §10 metrics over the whole report. A long report gets no leniency."""
    return eval_report(list(report.all_verifications),
                       cited=report.cited_unit_ids,
                       corpus=report.corpus_unit_ids)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v && ruff check jarvis/report.py tests/test_report.py`
Expected: PASS (30 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/report.py tests/test_report.py
git commit -m "feat: report assembly with corpus coverage measurement"
```

---

### Task 5: Rendering with references

**Files:**
- Modify: `jarvis/report.py` (append)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Consumes: `Report`, `SectionDraft` (Tasks 2–4); `get_paper` from `jarvis.store`.
- Produces: `render_report(conn, report, *, include_flagged=True) -> str`.

Markdown out. Supported claims appear with a `[unit_id]` marker; flagged claims appear in a clearly-labeled block; blocked claims appear only as a count. A **References** section lists every cited paper with its metadata, and marks any retracted paper explicitly — spec §14 requires the retraction check to run again at compile time, and this is compile time.

The report ends with its own coverage and verification numbers. A reader who cannot see how much of the corpus a report used cannot judge it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report.py`:

```python
from jarvis.report import render_report


def _one_section_report(corpus, quote, text="The controller reaches 94.2%.", nli=ENTAILS):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="Summary prose.",
                                     claims=(Claim("a-0", text, unit.unit_id, quote),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question="q1"),))
    return write_report(corpus, "gusts", outline, FakeEmbedder(), writer, nli)


def test_the_report_renders_its_topic_and_section_titles(corpus):
    text = render_report(corpus, _one_section_report(corpus, "94.2% tracking accuracy"))
    assert "gusts" in text
    assert "## Results" in text


def test_supported_claims_are_rendered_with_their_unit_id(corpus):
    report = _one_section_report(corpus, "94.2% tracking accuracy")
    text = render_report(corpus, report)
    assert report.all_claims[0].unit_id in text


def test_blocked_claims_appear_only_as_a_count(corpus):
    report = _one_section_report(corpus, "99.9% tracking accuracy",
                                 text="The controller reaches 99.9%.")
    text = render_report(corpus, report)
    assert "99.9" not in text
    assert "removed" in text.lower()


def test_flagged_claims_are_labelled_as_unverified(corpus):
    report = _one_section_report(corpus, "94.2% tracking accuracy", nli=NEUTRAL)
    assert "unverified" in render_report(corpus, report).lower()


def test_flagged_claims_can_be_suppressed(corpus):
    report = _one_section_report(corpus, "94.2% tracking accuracy", nli=NEUTRAL)
    assert "unverified" not in render_report(corpus, report, include_flagged=False).lower()


def test_references_list_every_cited_paper(corpus):
    text = render_report(corpus, _one_section_report(corpus, "94.2% tracking accuracy"))
    assert "## References" in text
    assert "Paper p1" in text


def test_uncited_papers_are_absent_from_the_references(corpus):
    text = render_report(corpus, _one_section_report(corpus, "94.2% tracking accuracy"))
    assert "Paper p2" not in text


def test_a_retracted_cited_paper_is_marked(corpus):
    save_paper(corpus, Paper(paper_id="p1", title="Paper p1", year=2025, retracted=True),
               depth="deep")
    text = render_report(corpus, _one_section_report(corpus, "94.2% tracking accuracy"))
    assert "RETRACTED" in text


def test_the_report_states_its_own_coverage(corpus):
    text = render_report(corpus, _one_section_report(corpus, "94.2% tracking accuracy"))
    assert "overage" in text


def test_an_empty_report_renders_without_pretending_otherwise(corpus):
    outline = Outline(topic="gusts", sections=(Section(title="R", question="zzz qqq"),))
    report = write_report(corpus, "gusts", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    text = render_report(corpus, report)
    assert "## References" in text
    assert "no " in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_report' from 'jarvis.report'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/report.py`, adding `from jarvis.store import get_paper` to the imports:

```python
def _render_section(draft: SectionDraft, include_flagged: bool) -> list[str]:
    lines = [f"## {draft.section.title}", ""]

    body = [f"{draft.claim_for(v.claim_id).text} [{draft.claim_for(v.claim_id).unit_id}]"
            for v in draft.supported if draft.claim_for(v.claim_id) is not None]
    lines += [" ".join(body) if body else "_No verified evidence for this section._", ""]

    if include_flagged and draft.flagged:
        lines.append("**Unverified** — the quote is real but does not clearly support "
                     "the claim:")
        lines.append("")
        for verification in draft.flagged:
            claim = draft.claim_for(verification.claim_id)
            if claim is not None:
                lines.append(f"- {claim.text} [{claim.unit_id}] "
                             f"({verification.verdict.value})")
        lines.append("")

    if draft.blocked:
        lines += [f"_{len(draft.blocked)} claim(s) removed: quote not found in any source "
                  f"paper._", ""]
    return lines


def render_report(conn: sqlite3.Connection, report: Report, *,
                  include_flagged: bool = True) -> str:
    """Markdown. Blocked claims are absent, flagged ones labelled, coverage always stated."""
    lines = [f"# {report.topic}", ""]
    for draft in report.sections:
        lines += _render_section(draft, include_flagged)

    lines += ["## References", ""]
    cited = sorted(report.cited_paper_ids)
    if not cited:
        lines += ["_No papers were cited: no claim in this report could be grounded._", ""]
    for paper_id in cited:
        paper = get_paper(conn, paper_id)
        if paper is None:
            lines.append(f"- [{paper_id}] (not in corpus)")
            continue
        year = f" ({paper.year})" if paper.year else ""
        venue = f". {paper.venue}" if paper.venue else ""
        doi = f". doi:{paper.doi}" if paper.doi else ""
        flag = "  **RETRACTED**" if paper.retracted else ""
        lines.append(f"- [{paper.paper_id}] {paper.title}{year}{venue}{doi}{flag}")

    verifications = report.all_verifications
    supported = sum(1 for v in verifications if v.verdict is Verdict.SUPPORTED)
    lines += [
        "",
        "---",
        "",
        f"Coverage: {report.coverage:.1%} of {report.corpus_units} deep-read corpus units "
        f"were cited. {supported}/{len(verifications)} claims verified.",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v && ruff check jarvis/report.py tests/test_report.py`
Expected: PASS (40 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/report.py tests/test_report.py
git commit -m "feat: markdown report rendering with references and coverage"
```

---

### Task 6: End to end — a report over a real multi-paper corpus

**Files:**
- Create: `tests/test_report_end_to_end.py`
- Modify: `jarvis/__init__.py`

**Interfaces:**
- Consumes: everything this plan built.
- Produces: the extended public surface.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_end_to_end.py
"""Three papers, one outline, one verified report, one honest coverage number."""
import pytest

from jarvis.card import extract_and_verify, FakeCardExtractor
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Card, CardField, Claim, Paper
from jarvis.outline import Outline, Section, TemplateOutliner
from jarvis.parse import FakeParser
from jarvis.report import evaluate_report, render_report, write_report
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

PAPERS = {
    "p1": ("Gust-Robust Control", [
        Block(kind="heading", text="Results", page=1, section_path=("Results",)),
        Block(kind="paragraph",
              text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
              page=1, section_path=("Results",)),
    ]),
    "p2": ("Wind Disturbance Attenuation", [
        Block(kind="heading", text="Limitations", page=1, section_path=("Limitations",)),
        Block(kind="paragraph", text="Tracking degrades sharply above 12 m/s wind speed.",
              page=1, section_path=("Limitations",)),
    ]),
    "p3": ("Gust Tolerance Benchmarks", [
        Block(kind="heading", text="Datasets", page=1, section_path=("Datasets",)),
        Block(kind="paragraph", text="We evaluate on the WindBench gust tolerance suite.",
              page=1, section_path=("Datasets",)),
    ]),
}
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    for paper_id, (title, blocks) in PAPERS.items():
        paper = Paper(paper_id=paper_id, title=title, year=2025, venue="ICRA")
        parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
        save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
        units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
        save_units(conn, units)
        index_units_fts(conn, units)
        index_units(conn, units, FakeEmbedder())

        unit = units[0]
        extract_and_verify(conn, paper, FakeCardExtractor({paper_id: Card(
            paper_id=paper_id,
            problem=CardField("gust rejection", unit.unit_id, unit.verbatim_text[:20]),
            metrics=(CardField("accuracy", unit.unit_id, unit.verbatim_text[:20]),),
        )}))
    yield conn
    close_store(conn)


def _u(conn, paper_id, needle):
    return next(u for u in get_units(conn, paper_id) if needle in u.verbatim_text)


def test_an_outline_is_built_from_the_corpus_cards(corpus):
    report = write_report(corpus, "gust rejection", TemplateOutliner(), FakeEmbedder(),
                          FakeWriter({}), ENTAILS)
    titles = [s.section.title for s in report.sections]
    assert "Overview" in titles
    assert any("esult" in t for t in titles)


def test_a_multi_section_report_verifies_every_section(corpus):
    q1, q2 = "what accuracy is reported?", "what are the wind speed limits?"
    writer = FakeWriter({
        q1: Draft(text="Accurate.", claims=(Claim(
            "a-0", "It reaches 94.2% accuracy.", _u(corpus, "p1", "94.2").unit_id,
            "94.2% tracking accuracy"),)),
        q2: Draft(text="Limited.", claims=(Claim(
            "b-0", "It degrades above 12 m/s.", _u(corpus, "p2", "12 m/s").unit_id,
            "above 12 m/s wind speed"),)),
    })
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),
                                               Section(title="Limits", question=q2)))

    report = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert len(report.all_claims) == 2
    assert all(s.blocked == () for s in report.sections)
    assert report.cited_paper_ids == {"p1", "p2"}


def test_a_fabricated_claim_never_reaches_the_rendered_report(corpus):
    q1 = "what accuracy is reported?"
    writer = FakeWriter({q1: Draft(text="It reaches 99.9%.", claims=(Claim(
        "a-0", "It reaches 99.9% accuracy.", _u(corpus, "p1", "94.2").unit_id,
        "99.9% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    report = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert "99.9" not in render_report(corpus, report)
    assert evaluate_report(report).meets_quote_target is False


def test_quote_fidelity_is_not_relaxed_for_a_long_report(corpus):
    q1 = "what accuracy is reported?"
    unit = _u(corpus, "p1", "94.2")
    writer = FakeWriter({q1: Draft(text="t", claims=tuple(
        Claim(f"a-{i}", f"claim {i}", unit.unit_id, "94.2% tracking accuracy")
        for i in range(10)))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    evaluation = evaluate_report(write_report(corpus, "gusts", outline, FakeEmbedder(),
                                              writer, ENTAILS))
    assert evaluation.quote_fidelity == 1.0
    assert evaluation.meets_quote_target is True


def test_the_report_reports_low_coverage_honestly(corpus):
    q1 = "what accuracy is reported?"
    writer = FakeWriter({q1: Draft(text="t", claims=(Claim(
        "a-0", "94.2%", _u(corpus, "p1", "94.2").unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    report = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert report.coverage < 0.5, "one unit out of three papers is low coverage"
    assert "Coverage" in render_report(corpus, report)


def test_every_section_is_retrieved_and_budgeted_separately(corpus):
    seen = []

    class SpyWriter:
        def write(self, question, units):
            seen.append((question, len(units)))
            return Draft()

    outline = Outline(topic="t", sections=(
        Section(title="A", question="what accuracy is reported?"),
        Section(title="B", question="what are the wind speed limits?"),
        Section(title="C", question="what datasets are used?")))
    write_report(corpus, "t", outline, FakeEmbedder(), SpyWriter(), ENTAILS, max_units=2)

    assert len(seen) == 3, "one call per section, never one call for the whole report"
    assert all(count <= 2 for _q, count in seen)
    assert len({q for q, _c in seen}) == 3


def test_the_references_only_list_papers_actually_cited(corpus):
    q1 = "what accuracy is reported?"
    writer = FakeWriter({q1: Draft(text="t", claims=(Claim(
        "a-0", "94.2%", _u(corpus, "p1", "94.2").unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    rendered = render_report(corpus, write_report(corpus, "gusts", outline, FakeEmbedder(),
                                                  writer, ENTAILS))
    assert "Gust-Robust Control" in rendered
    assert "Gust Tolerance Benchmarks" not in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report_end_to_end.py -v`
Expected: FAIL on import until the exports land. If an assertion fails, fix the *module*.

- [ ] **Step 3: Extend the package exports**

Add to `jarvis/__init__.py` (keeping imports and `__all__` sorted):

```python
from jarvis.outline import (
    LLMOutliner,
    Outline,
    Outliner,
    Section,
    TemplateOutliner,
    cards_digest,
)
from jarvis.report import (
    Report,
    SectionDraft,
    corpus_cards,
    draft_section,
    duplicate_claims,
    evaluate_report,
    integrate,
    render_report,
    write_report,
)
```

Update the module docstring: the whole spec build order (steps 1–10) is now implemented.

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass. `ruff check .` reports exactly the **11 pre-existing** violations.

- [ ] **Step 5: Commit**

```bash
git add jarvis/__init__.py tests/test_report_end_to_end.py
git commit -m "test: end-to-end multi-section report with coverage"
```

---

## Definition of done

- `python -m pytest` passes with zero network access, no API keys, no model downloads.
- `test_every_section_is_retrieved_and_budgeted_separately` passes — one bounded call per section, never one call for the whole report.
- `test_quote_fidelity_is_not_relaxed_for_a_long_report` passes — a 10-claim report faces the same bar as a 1-claim answer.
- `test_a_fabricated_claim_never_reaches_the_rendered_report` passes.
- `test_the_report_reports_low_coverage_honestly` passes — the number is printed whether or not it flatters the report.
- `ruff check .` reports exactly the 11 pre-existing violations.

## Where this stops

This is the last plan in spec §13's build order. With it, all ten build steps exist.

Deliberately not built, and each one wants measurement before it is worth building:

- **Parallel section drafting.** `draft_section` is independent per section by construction, so wrapping the loop in a thread pool or async gather is mechanical. Not done here because it adds a concurrency surface to a codebase that currently has none, and a report over eight sections is not slow enough to justify that yet. Measure first.
- **A revision pass.** AutoSurvey iterates; this plan drafts once. Add it only if measured quote fidelity or coverage says the first draft is not good enough — a second model pass over an already-verified report is as likely to introduce ungrounded prose as to improve it.
- **Contradiction surfacing inside the report.** Once `docs/plans/2026-08-14-contradiction-detection.md` lands and its precision is measured against the 70% target, a "Disagreements in the literature" section built from `scan_corpus` over the report's own claims is the single highest-value addition here — and it is the thing no single-shot research assistant can produce at all.
