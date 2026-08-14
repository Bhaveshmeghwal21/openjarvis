"""Hybrid retrieval: BM25 + vector, fused by RRF, reranked, expanded to parents.

RRF rather than score normalization because cosine and BM25 live on incompatible scales
and RRF needs no tuning (spec §7 Stage D). Children do the matching; parents do the
generating, so hits are swapped for their parents before the text reaches a model.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

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
