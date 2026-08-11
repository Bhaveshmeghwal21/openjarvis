"""Similarity primitives shared by the gate and the citation walker.

Deliberately dependency-free so it imports offline and stays trivially testable.
Extracted from NanoResearch/jarvis (entity_resolver.cosine, relevancy.make_cosine_scorer).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for empty, mismatched, or zero-magnitude vectors."""
    if a is None or b is None or len(a) != len(b) or not len(a):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def paper_text(paper: dict) -> str:
    """The text a paper is judged on before it is read: title + abstract."""
    return f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()


def make_cosine_scorer(embed_fn: Callable[[str], list[float]],
                       query: str) -> Callable[[dict], float]:
    """Cheap relevance scorer for citation-graph traversal: cosine vs the query embedding."""
    query_vec = embed_fn(query)

    def score(paper: dict) -> float:
        return cosine(embed_fn(paper_text(paper)), query_vec)

    return score


def recency(year, current_year: int) -> float:
    """1.0 for this year, decaying linearly to 0.0 at 10 years old."""
    if not year:
        return 0.0
    return max(0.0, 1 - max(0, current_year - int(year)) / 10)


def citation_weight(citation_count) -> float:
    """Log-compressed citation count in [0, 1]."""
    return min(math.log1p(max(0, citation_count or 0)) / 10, 1.0)
