"""Stage A — recall-optimized gathering (spec §7A).

No answer is being written here, so nothing is filtered for precision. One question fans
out into many queries, several APIs answer them, and the citation graph is walked outward
from the best hits. Cost is paid once and amortized across every future query against the
corpus (spec §4).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.citation_graph import paper_id
from jarvis.models import Paper
from jarvis.sources import dedup_papers
from jarvis.store import save_paper

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
        except Exception:  # noqa: BLE001, S112 - a dead source costs less than no gather
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
