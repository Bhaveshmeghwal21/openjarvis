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

import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.embed import Embedder
from jarvis.models import Unit
from jarvis.retrieve import Reranker, rrf, search

_LOGGER = logging.getLogger(__name__)

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
        except Exception:  # a dead refiner ends the loop, never the answer
            _LOGGER.warning("retrieval refiner model call failed; stopping refinement early",
                            exc_info=True)
            return None
        if not isinstance(raw, dict):
            return None
        query = " ".join(str(raw.get("query", "") or "").split())
        return query or None


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
