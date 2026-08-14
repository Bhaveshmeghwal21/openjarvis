"""Jarvis — a verifiable research corpus agent.

Gathers hundreds of papers on a question, reads the ones worth reading, and answers from
them where every claim resolves to a verbatim span in a specific paper at a specific
location. See docs/specs/2026-08-11-research-corpus-agent-design.md.

The verifiable single-paper core (spec build steps 1-5) is complete: storage, parsing,
typed units, hybrid retrieval, and two-stage verification. Gather + gate (spec step 6) is
now also complete: multi-source search, citation-graph expansion, a calibrated union gate
with no exclude outcome, deep read into the corpus, and verified Layer 2 cards. Compile,
MCP, contradiction detection, and long-form reports (spec steps 7-10) are not yet built.
"""
from __future__ import annotations

__version__ = "0.0.1"

from jarvis.card import (
    CardExtractor,
    FakeCardExtractor,
    LLMCardExtractor,
    extract_and_verify,
    unverified_fields,
    verify_card,
)
from jarvis.citation_graph import CitationWalker, make_s2_neighbors, paper_id
from jarvis.config import Config
from jarvis.context import TemplatePrefix, apply_prefixes, embedding_text
from jarvis.embed import BGEEmbedder, FakeEmbedder, index_units, vector_search
from jarvis.evaluate import EvalReport, report
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
from jarvis.index import index_units_fts, keyword_search
from jarvis.ingest import IngestResult, failed, ingest_decided, ingest_paper
from jarvis.label import read_labels, sample_seed, write_label_sheet
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
    enrich_provenance,
    make_arxiv_search,
    make_core_search,
    make_crossref_search,
    make_openalex_search,
    make_retraction_check,
    make_s2_search,
    make_unpaywall_pdf,
    normalize_crossref,
    normalize_openalex,
    normalize_s2,
)
from jarvis.store import (
    all_units,
    close_store,
    get_card,
    get_paper,
    get_papers_by_depth,
    get_screen_decisions,
    get_screen_signals,
    get_units,
    open_store,
    save_card,
    save_paper,
    save_screen_decision,
    save_units,
    set_depth,
)
from jarvis.text import approx_tokens, find_span, normalize
from jarvis.units import build_units
from jarvis.verify import HFNLI, FakeNLI, verify_claim

__all__ = [
    "HFNLI",
    "BGEEmbedder",
    "Block",
    "Candidate",
    "Card",
    "CardExtractor",
    "CardField",
    "CitationWalker",
    "Claim",
    "Config",
    "CostTracker",
    "CrossEncoderReranker",
    "DoclingParser",
    "EvalReport",
    "FakeCardExtractor",
    "FakeEmbedder",
    "FakeNLI",
    "FakeParser",
    "FakeVoter",
    "IngestResult",
    "LLMCardExtractor",
    "LLMPlanner",
    "LLMVoter",
    "ModelRouter",
    "Paper",
    "ParsedPaper",
    "SearchPlan",
    "Signals",
    "TemplatePlanner",
    "TemplatePrefix",
    "Thresholds",
    "Unit",
    "UnitType",
    "Verdict",
    "Verification",
    "all_units",
    "apply_prefixes",
    "approx_tokens",
    "build_units",
    "calibrate",
    "calibration_report",
    "citation_weight",
    "close_store",
    "combine_sources",
    "cosine",
    "decide",
    "dedup_papers",
    "embedding_text",
    "enrich_provenance",
    "expand_citations",
    "extract_and_verify",
    "failed",
    "find_span",
    "gather",
    "get_card",
    "get_paper",
    "get_papers_by_depth",
    "get_screen_decisions",
    "get_screen_signals",
    "get_units",
    "index_units",
    "index_units_fts",
    "ingest_decided",
    "ingest_paper",
    "keyword_search",
    "make_arxiv_search",
    "make_core_search",
    "make_cosine_scorer",
    "make_crossref_search",
    "make_openalex_search",
    "make_retraction_check",
    "make_s2_neighbors",
    "make_s2_search",
    "make_unpaywall_pdf",
    "normalize",
    "normalize_crossref",
    "normalize_openalex",
    "normalize_s2",
    "open_store",
    "paper_id",
    "paper_text",
    "read_labels",
    "recency",
    "report",
    "rrf",
    "run_searches",
    "sample_seed",
    "save_candidates",
    "save_card",
    "save_paper",
    "save_screen_decision",
    "save_units",
    "screen",
    "search",
    "set_depth",
    "to_paper",
    "unverified_fields",
    "vector_search",
    "verify_card",
    "verify_claim",
    "write_label_sheet",
]
