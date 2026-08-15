"""Jarvis — a verifiable research corpus agent.

Gathers hundreds of papers on a question, reads the ones worth reading, and answers from
them where every claim resolves to a verbatim span in a specific paper at a specific
location. See docs/specs/2026-08-11-research-corpus-agent-design.md.

The verifiable single-paper core (spec build steps 1-5) is complete: storage, parsing,
typed units, hybrid retrieval, and two-stage verification. Gather + gate (spec step 6) is
now also complete: multi-source search, citation-graph expansion, a calibrated union gate
with no exclude outcome, deep read into the corpus, and verified Layer 2 cards. Compile
— cited Q&A (spec step 7) is now also complete: end-to-end question answering with
evidence capping, retrieval refinement, deterministic quote verification, and entailment
filtering. MCP server (spec step 8) is now also complete: the corpus exposed as tools
(`corpus_search`, `get_unit`, `get_paper`, `list_papers`, `verify_quote`, `ask`) over a
pure dispatcher in `jarvis.tools`, with a thin stdio adapter in `jarvis.mcp_server`
(imports `mcp` lazily, not exported from this package). Long-form reports (spec step 10)
is now also complete: `jarvis.outline` builds a report outline from Layer 2 cards, and
`jarvis.report` drafts each section against its own bounded evidence set, integrates
claims across sections, assembles the report, measures corpus coverage, and renders it
to markdown with references. Contradiction detection (spec step 9) is now also complete:
`jarvis.contradict` retrieves cross-paper evidence topically close to a claim, reuses the
verification pass's own NLI model to score disagreement, ranks candidates, and runs the
human review round-trip a precision metric is measured against — output is always ranked
candidates for review, never assertions.
"""
from __future__ import annotations

__version__ = "0.0.1"

from jarvis.answer import Answer, ask, render_answer
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
from jarvis.contradict import (
    Conflict,
    apply_reviews,
    opposing_units,
    rank,
    read_reviews,
    render_conflicts,
    scan_claim,
    scan_corpus,
    write_review_sheet,
)
from jarvis.embed import BGEEmbedder, FakeEmbedder, index_units, vector_search
from jarvis.evaluate import (
    EvalReport,
    citation_precision,
    citation_recall,
    contradiction_precision,
    report,
)
from jarvis.evidence import EvidenceSet, cap, order_for_context
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
from jarvis.outline import (
    LLMOutliner,
    Outline,
    Outliner,
    Section,
    TemplateOutliner,
    cards_digest,
)
from jarvis.parse import DoclingParser, FakeParser
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
from jarvis.retrieve import CrossEncoderReranker, rrf, search
from jarvis.retriever import FakeRefiner, LLMRefiner, Refiner, Retrieval, retrieve_iteratively
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
    get_contradiction_reviews,
    get_contradictions,
    get_paper,
    get_papers_by_depth,
    get_screen_decisions,
    get_screen_signals,
    get_units,
    open_store,
    save_card,
    save_contradictions,
    save_paper,
    save_screen_decision,
    save_units,
    set_contradiction_review,
    set_depth,
)
from jarvis.text import approx_tokens, find_span, normalize
from jarvis.tools import REGISTRY, ToolContext, ToolSpec, call_tool, tool_specs, unit_payload
from jarvis.units import build_units
from jarvis.verify import HFNLI, FakeNLI, verify_claim
from jarvis.writer import Draft, FakeWriter, LLMWriter, Writer, claims_from_json

__all__ = [
    "HFNLI",
    "REGISTRY",
    "Answer",
    "BGEEmbedder",
    "Block",
    "Candidate",
    "Card",
    "CardExtractor",
    "CardField",
    "CitationWalker",
    "Claim",
    "Config",
    "Conflict",
    "CostTracker",
    "CrossEncoderReranker",
    "DoclingParser",
    "Draft",
    "EvalReport",
    "EvidenceSet",
    "FakeCardExtractor",
    "FakeEmbedder",
    "FakeNLI",
    "FakeParser",
    "FakeRefiner",
    "FakeVoter",
    "FakeWriter",
    "IngestResult",
    "LLMCardExtractor",
    "LLMOutliner",
    "LLMPlanner",
    "LLMRefiner",
    "LLMVoter",
    "LLMWriter",
    "ModelRouter",
    "Outline",
    "Outliner",
    "Paper",
    "ParsedPaper",
    "Refiner",
    "Report",
    "Retrieval",
    "SearchPlan",
    "Section",
    "SectionDraft",
    "Signals",
    "TemplateOutliner",
    "TemplatePlanner",
    "TemplatePrefix",
    "Thresholds",
    "ToolContext",
    "ToolSpec",
    "Unit",
    "UnitType",
    "Verdict",
    "Verification",
    "Writer",
    "all_units",
    "apply_prefixes",
    "apply_reviews",
    "approx_tokens",
    "ask",
    "build_units",
    "calibrate",
    "calibration_report",
    "call_tool",
    "cap",
    "cards_digest",
    "citation_precision",
    "citation_recall",
    "citation_weight",
    "claims_from_json",
    "close_store",
    "combine_sources",
    "contradiction_precision",
    "corpus_cards",
    "cosine",
    "decide",
    "dedup_papers",
    "draft_section",
    "duplicate_claims",
    "embedding_text",
    "enrich_provenance",
    "evaluate_report",
    "expand_citations",
    "extract_and_verify",
    "failed",
    "find_span",
    "gather",
    "get_card",
    "get_contradiction_reviews",
    "get_contradictions",
    "get_paper",
    "get_papers_by_depth",
    "get_screen_decisions",
    "get_screen_signals",
    "get_units",
    "index_units",
    "index_units_fts",
    "ingest_decided",
    "ingest_paper",
    "integrate",
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
    "opposing_units",
    "order_for_context",
    "paper_id",
    "paper_text",
    "rank",
    "read_labels",
    "read_reviews",
    "recency",
    "render_answer",
    "render_conflicts",
    "render_report",
    "report",
    "retrieve_iteratively",
    "rrf",
    "run_searches",
    "sample_seed",
    "save_candidates",
    "save_card",
    "save_contradictions",
    "save_paper",
    "save_screen_decision",
    "save_units",
    "scan_claim",
    "scan_corpus",
    "screen",
    "search",
    "set_contradiction_review",
    "set_depth",
    "to_paper",
    "tool_specs",
    "unit_payload",
    "unverified_fields",
    "vector_search",
    "verify_card",
    "verify_claim",
    "write_label_sheet",
    "write_report",
    "write_review_sheet",
]
