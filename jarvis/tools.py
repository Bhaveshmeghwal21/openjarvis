# jarvis/tools.py
"""The corpus as callable tools (spec §3, build step 8).

A pure dispatcher: a registry of specs with JSON schemas, plus handlers that take a
context and return plain dicts. This module imports nothing from `mcp` — the protocol
adapter is `jarvis/mcp_server.py` and it contains no logic. Everything worth testing lives
here, where a test needs no protocol library and no transport.

Contract for every handler:
  * never raise — return `err(...)` instead;
  * return `ok(**payload)` on success;
  * hand back `unit_id` and verbatim text as separate fields, always.

That last rule is not cosmetic. The client on the other end is a general assistant, the
kind spec §1 measures hallucinating citations 78-90% of the time on scholarly queries.
Handing it loose text gets loose citations back.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MAX_LIMIT = 25


@dataclass
class ToolContext:
    """Everything the handlers need. Optional members gate the tools that require them."""
    conn: sqlite3.Connection
    embedder: Any
    writer: Any = None
    nli: Any = None
    reranker: Any = None
    max_limit: int = MAX_LIMIT


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict
    handler: Callable[[ToolContext, dict], dict]
    required: tuple[str, ...] = ()          # required argument keys
    requires: tuple[str, ...] = ()          # required ToolContext members


REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    REGISTRY[spec.name] = spec
    return spec


def ok(**payload) -> dict:
    return {"ok": True, **payload}


def err(message: str) -> dict:
    return {"ok": False, "error": message}


def tool_specs() -> list[dict]:
    """The listing an MCP client receives. JSON-serializable by construction."""
    return [
        {"name": spec.name, "description": spec.description, "inputSchema": spec.schema}
        for spec in REGISTRY.values()
    ]


def clamp_limit(value, default: int, maximum: int) -> int:
    """Coerce a client-supplied limit into a sane integer. Never trust the other side."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, maximum))


def call_tool(ctx: ToolContext, name: str, arguments: dict | None = None) -> dict:
    """Dispatch one tool call. Returns a dict in every case, including every failure."""
    spec = REGISTRY.get(name)
    if spec is None:
        return err(f"unknown tool: {name}")

    args = dict(arguments or {})
    for key in spec.required:
        if args.get(key) in (None, ""):
            return err(f"missing required argument: {key}")
    for member in spec.requires:
        if getattr(ctx, member, None) is None:
            return err(f"this tool needs a {member}, which is not configured on this server")

    try:
        return spec.handler(ctx, args)
    except Exception as exc:  # noqa: BLE001 - an exception here is a dead client session
        return err(f"{type(exc).__name__}: {exc}")



DEPTHS = ("deep", "pending_deep", "metadata", "abstract")

_CITE_NOTE = ("Quote text from `text` EXACTLY, character for character, when citing. A "
              "paraphrase will not verify.")


def unit_payload(unit) -> dict:
    """The one place a Unit becomes client-visible JSON.

    `unit_id` and `text` are separate fields on purpose: the id is the citation target and
    the text is the only thing that may be quoted.
    """
    return {
        "unit_id": unit.unit_id,
        "paper_id": unit.paper_id,
        "type": unit.type.value,
        "page": unit.page,
        "section_path": list(unit.section_path),
        "label": unit.label,
        "text": unit.verbatim_text,
        "context": unit.context_prefix,
    }


def _paper_payload(paper, unit_count: int) -> dict:
    return {
        "paper_id": paper.paper_id, "title": paper.title, "authors": list(paper.authors),
        "year": paper.year, "venue": paper.venue, "doi": paper.doi,
        "arxiv_id": paper.arxiv_id, "citation_count": paper.citation_count,
        "retracted": paper.retracted, "abstract": paper.abstract,
        "unit_count": unit_count,
    }


def _corpus_search(ctx: ToolContext, args: dict) -> dict:
    from jarvis.retrieve import search
    limit = clamp_limit(args.get("limit"), default=8, maximum=ctx.max_limit)
    hits = search(ctx.conn, str(args["query"]), ctx.embedder, limit=limit,
                  reranker=ctx.reranker)
    return ok(units=[unit_payload(u) for u in hits], count=len(hits))


register(ToolSpec(
    name="corpus_search",
    description=(
        "Search this research corpus for evidence units (prose passages, tables, figures, "
        "equations) relevant to a query. Hybrid BM25 + vector retrieval. Returns units with "
        f"their `unit_id` (the citation target) and verbatim `text`. {_CITE_NOTE}"
    ),
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "limit": {"type": "integer", "description": "Max units to return (default 8)."},
        },
        "required": ["query"],
    },
    handler=_corpus_search,
    required=("query",),
))


def _get_unit(ctx: ToolContext, args: dict) -> dict:
    from jarvis.store import get_unit as _load
    unit = _load(ctx.conn, str(args["unit_id"]))
    if unit is None:
        return err(f"no unit with id: {args['unit_id']}")
    return ok(unit=unit_payload(unit))


register(ToolSpec(
    name="get_unit",
    description=(
        "Fetch one evidence unit in full by its `unit_id`, as returned by corpus_search. "
        f"Use this when a search snippet is not enough to quote accurately. {_CITE_NOTE}"
    ),
    schema={
        "type": "object",
        "properties": {"unit_id": {"type": "string"}},
        "required": ["unit_id"],
    },
    handler=_get_unit,
    required=("unit_id",),
))


def _get_paper(ctx: ToolContext, args: dict) -> dict:
    from jarvis.store import get_paper as _load
    from jarvis.store import get_units
    paper = _load(ctx.conn, str(args["paper_id"]))
    if paper is None:
        return err(f"no paper with id: {args['paper_id']}")
    return ok(paper=_paper_payload(paper, len(get_units(ctx.conn, paper.paper_id))))


register(ToolSpec(
    name="get_paper",
    description=(
        "Fetch paper-level metadata by `paper_id`: title, authors, year, venue, DOI, "
        "citation count, and whether the paper has been RETRACTED. Check `retracted` "
        "before relying on a paper — citing retracted work is the worst failure this "
        "corpus can produce."
    ),
    schema={
        "type": "object",
        "properties": {"paper_id": {"type": "string"}},
        "required": ["paper_id"],
    },
    handler=_get_paper,
    required=("paper_id",),
))


def _list_papers(ctx: ToolContext, args: dict) -> dict:
    from jarvis.store import get_papers_by_depth, get_units
    depth = str(args.get("depth") or "deep")
    if depth not in DEPTHS:
        return err(f"unknown depth: {depth} (expected one of {', '.join(DEPTHS)})")
    limit = clamp_limit(args.get("limit"), default=50, maximum=500)
    papers = get_papers_by_depth(ctx.conn, depth)[:limit]
    return ok(papers=[_paper_payload(p, len(get_units(ctx.conn, p.paper_id)))
                      for p in papers],
              count=len(papers), depth=depth)


register(ToolSpec(
    name="list_papers",
    description=(
        "List papers in the corpus. `depth` selects how far each was ingested: 'deep' "
        "(fully read, searchable, the default), 'pending_deep' (kept by the screening gate "
        "but not yet read), or 'metadata' (deferred — still present and recoverable, but "
        "only title and abstract are known)."
    ),
    schema={
        "type": "object",
        "properties": {
            "depth": {"type": "string", "enum": list(DEPTHS)},
            "limit": {"type": "integer"},
        },
    },
    handler=_list_papers,
))



def _verify_quote(ctx: ToolContext, args: dict) -> dict:
    from jarvis.models import Claim
    from jarvis.verify import quote_is_grounded
    claim = Claim(claim_id="tool-check", text="", unit_id=str(args["unit_id"]),
                  quote=str(args["quote"]))
    return ok(grounded=quote_is_grounded(ctx.conn, claim), unit_id=claim.unit_id)


register(ToolSpec(
    name="verify_quote",
    description=(
        "Check whether a quote appears VERBATIM in a unit's source paper. Deterministic "
        "string match against the immutable parsed text — no model, no cost, no judgment "
        "call. Call this before asserting anything you quoted: if it returns "
        "grounded=false, the quote is not in the paper and the claim must not be made. "
        "Whitespace, ligatures, smart quotes, and hyphenation across line breaks are "
        "normalized, so a faithful copy will match even if the formatting differs."
    ),
    schema={
        "type": "object",
        "properties": {
            "unit_id": {"type": "string"},
            "quote": {"type": "string", "description": "The exact text you intend to quote."},
        },
        "required": ["unit_id", "quote"],
    },
    handler=_verify_quote,
    required=("unit_id", "quote"),
))


def _ask(ctx: ToolContext, args: dict) -> dict:
    from jarvis.answer import ask as _ask_impl
    from jarvis.answer import render_answer

    limit = clamp_limit(args.get("limit"), default=8, maximum=ctx.max_limit)
    answer = _ask_impl(ctx.conn, str(args["question"]), ctx.embedder, ctx.writer, ctx.nli,
                       limit=limit, reranker=ctx.reranker)

    supported = []
    for verification in answer.supported:
        claim = answer.claim_for(verification.claim_id)
        if claim is not None:
            supported.append({"text": claim.text, "unit_id": claim.unit_id,
                              "quote": claim.quote, "verdict": verification.verdict.value})
    flagged = []
    for verification in answer.flagged:
        claim = answer.claim_for(verification.claim_id)
        if claim is not None:
            flagged.append({"text": claim.text, "unit_id": claim.unit_id,
                            "quote": claim.quote, "verdict": verification.verdict.value})

    return ok(answer=render_answer(answer), claims=supported, flagged=flagged,
              blocked=len(answer.blocked), queries=list(answer.queries))


register(ToolSpec(
    name="ask",
    description=(
        "Answer a question from this corpus with verified citations. Retrieves evidence, "
        "drafts an answer, then mechanically verifies every claim: claims whose quote is "
        "not in the source are REMOVED from the answer and reported only as a count in "
        "`blocked`; claims that ground but do not clearly entail appear in `flagged`. "
        "Everything in `claims` has been verified. Requires a configured writer model."
    ),
    schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "limit": {"type": "integer", "description": "Max evidence units to consider."},
        },
        "required": ["question"],
    },
    handler=_ask,
    required=("question",),
    requires=("writer", "nli"),
))
