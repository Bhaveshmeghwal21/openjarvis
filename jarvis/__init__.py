"""Jarvis — a verifiable research corpus agent.

Gathers hundreds of papers on a question, reads the ones worth reading, and answers from
them where every claim resolves to a verbatim span in a specific paper at a specific
location. See docs/specs/2026-08-11-research-corpus-agent-design.md.

Only the ported gather-stage and routing primitives exist so far; the corpus store,
parsing, retrieval, and verification layers are build steps 1-5 of the spec.
"""
from __future__ import annotations

__version__ = "0.0.1"

from jarvis.citation_graph import CitationWalker, make_s2_neighbors, paper_id
from jarvis.config import Config
from jarvis.router import ModelRouter, CostTracker
from jarvis.scoring import citation_weight, cosine, make_cosine_scorer, paper_text, recency
from jarvis.sources import (
    combine_sources,
    dedup_papers,
    make_core_search,
    make_crossref_search,
    make_unpaywall_pdf,
    normalize_crossref,
)

__all__ = [
    "CitationWalker",
    "Config",
    "CostTracker",
    "ModelRouter",
    "citation_weight",
    "combine_sources",
    "cosine",
    "dedup_papers",
    "make_core_search",
    "make_cosine_scorer",
    "make_crossref_search",
    "make_s2_neighbors",
    "make_unpaywall_pdf",
    "normalize_crossref",
    "paper_id",
    "paper_text",
    "recency",
]
