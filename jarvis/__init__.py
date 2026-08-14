"""Jarvis — a verifiable research corpus agent.

Gathers hundreds of papers on a question, reads the ones worth reading, and answers from
them where every claim resolves to a verbatim span in a specific paper at a specific
location. See docs/specs/2026-08-11-research-corpus-agent-design.md.

The verifiable single-paper core (spec build steps 1-5) is complete: storage, parsing,
typed units, hybrid retrieval, and two-stage verification. Gather-stage primitives are
ported from NanoResearch. Gather + gate, compile, and long-form reports (spec steps 6-10)
are not yet built.
"""
from __future__ import annotations

__version__ = "0.0.1"

from jarvis.citation_graph import CitationWalker, make_s2_neighbors, paper_id
from jarvis.config import Config
from jarvis.context import TemplatePrefix, apply_prefixes, embedding_text
from jarvis.embed import BGEEmbedder, FakeEmbedder, index_units, vector_search
from jarvis.evaluate import EvalReport, report
from jarvis.index import index_units_fts, keyword_search
from jarvis.models import (
    Block,
    Card,
    CardField,
    Claim,
    Paper,
    ParsedPaper,
    Unit,
    UnitType,
    Verdict,
    Verification,
)
from jarvis.parse import DoclingParser, FakeParser
from jarvis.retrieve import CrossEncoderReranker, rrf, search
from jarvis.router import CostTracker, ModelRouter
from jarvis.scoring import citation_weight, cosine, make_cosine_scorer, paper_text, recency
from jarvis.sources import (
    combine_sources,
    dedup_papers,
    make_core_search,
    make_crossref_search,
    make_unpaywall_pdf,
    normalize_crossref,
)
from jarvis.store import close_store, get_paper, get_units, open_store, save_paper, save_units
from jarvis.text import approx_tokens, find_span, normalize
from jarvis.units import build_units
from jarvis.verify import HFNLI, FakeNLI, verify_claim

__all__ = [
    "HFNLI",
    "BGEEmbedder",
    "Block",
    "Card",
    "CardField",
    "CitationWalker",
    "Claim",
    "Config",
    "CostTracker",
    "CrossEncoderReranker",
    "DoclingParser",
    "EvalReport",
    "FakeEmbedder",
    "FakeNLI",
    "FakeParser",
    "ModelRouter",
    "Paper",
    "ParsedPaper",
    "TemplatePrefix",
    "Unit",
    "UnitType",
    "Verdict",
    "Verification",
    "apply_prefixes",
    "approx_tokens",
    "build_units",
    "citation_weight",
    "close_store",
    "combine_sources",
    "cosine",
    "dedup_papers",
    "embedding_text",
    "find_span",
    "get_paper",
    "get_units",
    "index_units",
    "index_units_fts",
    "keyword_search",
    "make_core_search",
    "make_cosine_scorer",
    "make_crossref_search",
    "make_s2_neighbors",
    "make_unpaywall_pdf",
    "normalize",
    "normalize_crossref",
    "open_store",
    "paper_id",
    "paper_text",
    "recency",
    "report",
    "rrf",
    "save_paper",
    "save_units",
    "search",
    "vector_search",
    "verify_claim",
]
