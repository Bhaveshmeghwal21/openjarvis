# MCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the corpus as MCP tools so Claude Code, Cursor, or any other MCP client can search it, read units, verify quotes, and ask cited questions — before any UI exists.

**Architecture:** Two layers, and the split is the whole point. `jarvis/tools.py` is a **pure dispatcher**: a registry of tool specs with JSON schemas, argument validation, and handlers that take a context object and return plain dicts. It imports nothing from the `mcp` package and is fully testable offline. `jarvis/mcp_server.py` is a thin stdio adapter that imports `mcp` lazily and does nothing but translate between the MCP protocol and the dispatcher. Every behaviour worth testing lives in the layer with no protocol dependency.

**Tech Stack:** Python 3.10+, stdlib only in `tools.py`, the `mcp` package (optional extra) in `mcp_server.py`, `pytest`.

**Prerequisites:** all three of
- `docs/plans/2026-08-11-verifiable-single-paper-core.md` (merged at `d7f8672`) — storage, retrieval, verification
- `docs/plans/2026-08-14-gather-and-gate.md` — `get_papers_by_depth`, used by `list_papers`
- `docs/plans/2026-08-14-compile-cited-qa.md` — `ask`, `Writer`, `Answer`, used by the `ask` tool

This is spec build step 8, and it comes after steps 6 and 7 for exactly these reasons: it is a surface over a core that must work first.

## Global Constraints

- Python **>= 3.10**. Use `X | None`, not `Optional[X]`.
- **Never read `.env`.** Configuration is environment variables or `$JARVIS_CONFIG` JSON only.
- **Every test is offline.** No network, no API keys, no model downloads. `mcp` is imported inside the function that needs it, never at module top level, and no test imports `jarvis.mcp_server` at all.
- **`jarvis/tools.py` must not import `mcp`.** Not at top level, not lazily, not in a type annotation. If a behaviour cannot be tested without the protocol library, it is in the wrong file.
- Line length **100**. Target `py310`. Run `ruff check .` against **both** the module and its test file before every commit.
- **`jarvis/store.py` is the only module that writes SQL.** This plan adds no SQL.
- **No tool handler may raise.** Every handler returns a dict; failures return `{"ok": False, "error": "..."}`. An exception crossing the MCP boundary is a broken session for the client.
- **`verify_quote` never consults a model.** It is the deterministic Layer 0 matcher and nothing else. No `verif*` task may be routed to an LLM — a test already asserts this.
- Frozen dataclasses for all new types; tuples not lists in frozen types.
- Commit after every task with a `feat:`/`test:`/`fix:` prefix.
- Repo-wide `ruff check .` baseline is **11 pre-existing violations** in `citation_graph.py` (2), `config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1). Do not fix them; do not add to them.

## What an MCP client is actually being handed

Spec §3 puts an MCP server in scope so that *"external agents (Claude Code, Cursor) query the corpus natively."* That framing has a consequence worth stating up front, because it shapes every tool description in Task 2 and Task 3:

The client on the other end is a general assistant — precisely the kind of system spec §1 measures hallucinating citations 78–90% of the time on scholarly queries. It will paraphrase, it will summarize, and if a tool hands it loose text it will cite that text loosely.

So every unit this server returns carries its `unit_id` and its verbatim text as separate fields, every tool description says in plain words that a quote must be copied exactly, and `verify_quote` exists specifically so a client can check itself mechanically before asserting anything. The server's job is not just to answer — it is to make the honest path the easy one.

## File Structure

| File | Responsibility |
|---|---|
| `jarvis/tools.py` | Create. Tool registry, JSON schemas, argument validation, handlers, dispatcher. No `mcp` import. |
| `jarvis/mcp_server.py` | Create. stdio MCP adapter and `main()`. Lazy `mcp` import, no logic. |
| `pyproject.toml` | **Modify.** Add the `mcp` optional extra and the `jarvis-mcp` console script. |
| `README.md` | **Modify.** Add a "Use it from Claude Code" section with the client config. |
| `jarvis/__init__.py` | **Modify.** Export the dispatcher surface. |

Tests: `tests/test_tools.py`, `tests/test_tools_dispatch.py`, `tests/test_mcp_end_to_end.py`.

---

### Task 1: The tool registry and dispatcher

**Files:**
- Create: `jarvis/tools.py`
- Test: `tests/test_tools_dispatch.py`

**Interfaces:**
- Consumes: `Embedder` from `jarvis.embed`; `Reranker` from `jarvis.retrieve`; `Writer` from `jarvis.writer`; `NLIModel` from `jarvis.verify`.
- Produces: `ToolContext` (mutable dataclass: `conn`, `embedder`, `writer=None`, `nli=None`, `reranker=None`, `max_limit=25`), `ToolSpec` (frozen: `name`, `description`, `schema: dict`, `handler`, `requires: tuple[str, ...] = ()`), `REGISTRY: dict[str, ToolSpec]`, `register(spec)`, `tool_specs() -> list[dict]`, `call_tool(ctx, name, arguments) -> dict`, `ok(**payload) -> dict`, `err(message) -> dict`.

The dispatcher's contract, fixed here so every handler in Tasks 2 and 3 can rely on it:

- Unknown tool name → `{"ok": False, "error": "unknown tool: <name>"}`.
- Missing required argument → `{"ok": False, "error": "missing required argument: <key>"}`.
- A tool whose `requires` names a `ToolContext` field that is `None` → a clear "not configured" error, never an `AttributeError`.
- A handler that raises anyway → caught, converted to `{"ok": False, "error": ...}`. Defense in depth: the no-raise rule is a rule for handlers *and* a guarantee from the dispatcher.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_dispatch.py
"""The dispatcher contract. Every failure is a dict; nothing crosses the boundary as an exception."""
import pytest

from jarvis.embed import FakeEmbedder
from jarvis.store import close_store, open_store
from jarvis.tools import REGISTRY, ToolContext, ToolSpec, call_tool, err, ok, tool_specs


@pytest.fixture
def ctx(tmp_path):
    conn = open_store(tmp_path / "c.db")
    yield ToolContext(conn=conn, embedder=FakeEmbedder())
    close_store(conn)


def test_ok_and_err_have_disjoint_shapes():
    assert ok(value=1) == {"ok": True, "value": 1}
    assert err("bad") == {"ok": False, "error": "bad"}


def test_every_registered_tool_has_a_name_description_and_schema():
    for spec in REGISTRY.values():
        assert spec.name
        assert len(spec.description) > 20, f"{spec.name} needs a description a client can act on"
        assert spec.schema["type"] == "object"
        assert "properties" in spec.schema


def test_tool_specs_are_serializable_for_a_client():
    import json
    listing = tool_specs()
    assert isinstance(listing, list)
    assert json.dumps(listing)
    assert {t["name"] for t in listing} == set(REGISTRY)
    assert all({"name", "description", "inputSchema"} <= set(t) for t in listing)


def test_an_unknown_tool_is_an_error_not_an_exception(ctx):
    result = call_tool(ctx, "no_such_tool", {})
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_a_missing_required_argument_is_reported_by_name(ctx):
    result = call_tool(ctx, "get_unit", {})
    assert result["ok"] is False
    assert "unit_id" in result["error"]


def test_a_handler_that_raises_is_converted_to_an_error(ctx):
    def boom(context, arguments):
        raise RuntimeError("kaboom")

    REGISTRY["_boom"] = ToolSpec(name="_boom", description="x" * 30,
                                 schema={"type": "object", "properties": {}},
                                 handler=boom)
    try:
        result = call_tool(ctx, "_boom", {})
        assert result["ok"] is False
        assert "kaboom" in result["error"]
    finally:
        del REGISTRY["_boom"]


def test_a_tool_needing_an_unconfigured_dependency_says_so(ctx):
    def handler(context, arguments):
        return ok()

    REGISTRY["_needs_writer"] = ToolSpec(name="_needs_writer", description="x" * 30,
                                         schema={"type": "object", "properties": {}},
                                         handler=handler, requires=("writer",))
    try:
        result = call_tool(ctx, "_needs_writer", {})
        assert result["ok"] is False
        assert "writer" in result["error"]
        assert "not configured" in result["error"]
    finally:
        del REGISTRY["_needs_writer"]


def test_arguments_defaulting_to_none_are_treated_as_absent(ctx):
    assert call_tool(ctx, "get_unit", {"unit_id": None})["ok"] is False


def test_the_dispatcher_never_mutates_the_callers_arguments(ctx):
    args = {"query": "x"}
    call_tool(ctx, "corpus_search", args)
    assert args == {"query": "x"}


def test_tools_module_does_not_import_mcp():
    import inspect

    import jarvis.tools
    source = inspect.getsource(jarvis.tools)
    assert "import mcp" not in source
    assert "from mcp" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools_dispatch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis.tools'`

- [ ] **Step 3: Write the implementation**

```python
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
from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools_dispatch.py -v && ruff check jarvis/tools.py tests/test_tools_dispatch.py`
Expected: FAIL on the tests that call `get_unit` and `corpus_search` — those tools land in Task 2. Every other test passes. Mark those two tests with `@pytest.mark.xfail(reason="tools land in Task 2", strict=False)` **only if** the reviewer's fix loop demands a green commit here; otherwise commit with Task 2 and note it. The straightforward path is to finish Task 2 before running the full file.

Run instead: `python -m pytest tests/test_tools_dispatch.py -v -k "not get_unit and not corpus_search" && ruff check jarvis/tools.py tests/test_tools_dispatch.py`
Expected: PASS (8 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/tools.py tests/test_tools_dispatch.py
git commit -m "feat: tool registry and no-raise dispatcher"
```

---

### Task 2: Corpus read tools

**Files:**
- Modify: `jarvis/tools.py` (append)
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `search` from `jarvis.retrieve`; `get_unit`, `get_paper`, `get_units`, `get_papers_by_depth` from `jarvis.store`.
- Produces: registered tools `corpus_search`, `get_unit`, `get_paper`, `list_papers`; helper `unit_payload(unit) -> dict`.

`unit_payload` is the single place a `Unit` becomes client-visible JSON, so the "unit_id and verbatim text as separate fields" rule is enforced once rather than in four handlers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py
"""The corpus read tools."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Paper
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.tools import ToolContext, call_tool, unit_payload
from jarvis.units import build_units

BLOCKS = [
    Block(kind="heading", text="Results", page=3, section_path=("Results",)),
    Block(kind="paragraph", text="As shown in Table 3, we reach 94.2% tracking accuracy.",
          page=3, section_path=("Results",)),
    Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
          page=3, section_path=("Results",), label="Table 3"),
    Block(kind="caption", text="Table 3: Tracking accuracy under wind.", page=3,
          section_path=("Results",), label="Table 3"),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025, venue="ICRA",
              doi="10.1/x", citation_count=42)


@pytest.fixture
def ctx(tmp_path):
    conn = open_store(tmp_path / "c.db")
    parsed = FakeParser(BLOCKS).parse("p.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    yield ToolContext(conn=conn, embedder=FakeEmbedder())
    close_store(conn)


def _table_unit(ctx):
    return next(u for u in get_units(ctx.conn, "p1") if u.type.value == "table")


def test_unit_payload_separates_the_id_from_the_text(ctx):
    payload = unit_payload(_table_unit(ctx))
    assert payload["unit_id"] == _table_unit(ctx).unit_id
    assert "94.2" in payload["text"]
    assert payload["paper_id"] == "p1"
    assert payload["page"] == 3
    assert payload["type"] == "table"


def test_unit_payload_carries_the_section_path_as_a_list(ctx):
    payload = unit_payload(_table_unit(ctx))
    assert payload["section_path"] == ["Results"]


def test_corpus_search_returns_ranked_units(ctx):
    result = call_tool(ctx, "corpus_search", {"query": "tracking accuracy under wind"})
    assert result["ok"] is True
    assert any("94.2" in u["text"] for u in result["units"])
    assert all("unit_id" in u for u in result["units"])


def test_corpus_search_honours_its_limit(ctx):
    assert len(call_tool(ctx, "corpus_search", {"query": "accuracy", "limit": 1})["units"]) == 1


def test_corpus_search_clamps_an_absurd_limit(ctx):
    result = call_tool(ctx, "corpus_search", {"query": "accuracy", "limit": 100000})
    assert len(result["units"]) <= ctx.max_limit


def test_corpus_search_survives_a_junk_limit(ctx):
    assert call_tool(ctx, "corpus_search", {"query": "accuracy", "limit": "lots"})["ok"] is True


def test_corpus_search_with_no_hits_is_an_empty_success(ctx):
    result = call_tool(ctx, "corpus_search", {"query": "zzz nonexistent qqq"})
    assert result["ok"] is True
    assert result["units"] == []


def test_corpus_search_requires_a_query(ctx):
    assert call_tool(ctx, "corpus_search", {})["ok"] is False


def test_fts_operators_in_a_query_do_not_break_the_tool(ctx):
    for query in ('accuracy AND NOT "', "table OR (", "*", "-94.2"):
        assert call_tool(ctx, "corpus_search", {"query": query})["ok"] is True


def test_get_unit_returns_the_full_verbatim_text(ctx):
    unit = _table_unit(ctx)
    result = call_tool(ctx, "get_unit", {"unit_id": unit.unit_id})
    assert result["ok"] is True
    assert result["unit"]["text"] == unit.verbatim_text


def test_get_unit_on_a_missing_id_is_a_clean_error(ctx):
    result = call_tool(ctx, "get_unit", {"unit_id": "nope"})
    assert result["ok"] is False
    assert "nope" in result["error"]


def test_get_paper_returns_metadata_and_a_unit_count(ctx):
    result = call_tool(ctx, "get_paper", {"paper_id": "p1"})
    assert result["ok"] is True
    assert result["paper"]["title"] == "Gust-Robust Control"
    assert result["paper"]["year"] == 2025
    assert result["paper"]["citation_count"] == 42
    assert result["paper"]["unit_count"] > 0


def test_get_paper_surfaces_the_retraction_flag(ctx):
    save_paper(ctx.conn, Paper(paper_id="p2", title="Retracted", retracted=True))
    assert call_tool(ctx, "get_paper", {"paper_id": "p2"})["paper"]["retracted"] is True


def test_get_paper_on_a_missing_id_is_a_clean_error(ctx):
    assert call_tool(ctx, "get_paper", {"paper_id": "nope"})["ok"] is False


def test_list_papers_defaults_to_the_deep_read_corpus(ctx):
    result = call_tool(ctx, "list_papers", {})
    assert result["ok"] is True
    assert [p["paper_id"] for p in result["papers"]] == ["p1"]


def test_list_papers_can_show_a_different_depth(ctx):
    save_paper(ctx.conn, Paper(paper_id="p3", title="Only metadata"), depth="metadata")
    result = call_tool(ctx, "list_papers", {"depth": "metadata"})
    assert [p["paper_id"] for p in result["papers"]] == ["p3"]


def test_list_papers_rejects_an_unknown_depth(ctx):
    result = call_tool(ctx, "list_papers", {"depth": "sideways"})
    assert result["ok"] is False
    assert "depth" in result["error"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL with `ImportError: cannot import name 'unit_payload' from 'jarvis.tools'`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/tools.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools.py tests/test_tools_dispatch.py -v && ruff check jarvis/tools.py tests/test_tools.py`
Expected: PASS (17 + 10 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/tools.py tests/test_tools.py
git commit -m "feat: corpus search, unit, paper, and listing tools"
```

---

### Task 3: Verification and question-answering tools

**Files:**
- Modify: `jarvis/tools.py` (append)
- Test: `tests/test_tools.py` (append)

**Interfaces:**
- Consumes: `quote_is_grounded` from `jarvis.verify`; `Claim` from `jarvis.models`; `ask` and `render_answer` from `jarvis.answer`.
- Produces: registered tools `verify_quote` and `ask`.

`verify_quote` is the tool that makes this server worth pointing a general assistant at. It is deterministic, free, requires no configured model, and answers exactly one question: *does this text actually appear in that unit's paper?* An assistant that calls it before asserting cannot fabricate a quotation.

`ask` requires a `writer` and an `nli` on the context. Where they are absent — a server started for search-only use — the dispatcher's `requires` check reports that cleanly rather than crashing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tools.py`:

```python
from jarvis.models import Claim
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})


def test_a_real_quote_verifies(ctx):
    unit = _table_unit(ctx)
    result = call_tool(ctx, "verify_quote", {"unit_id": unit.unit_id,
                                             "quote": "| ours | 94.2 |"})
    assert result["ok"] is True
    assert result["grounded"] is True


def test_a_fabricated_quote_does_not_verify(ctx):
    unit = _table_unit(ctx)
    result = call_tool(ctx, "verify_quote", {"unit_id": unit.unit_id,
                                             "quote": "| ours | 99.9 |"})
    assert result["ok"] is True
    assert result["grounded"] is False


def test_verification_survives_hyphenation_and_smart_punctuation(ctx):
    prose = next(u for u in get_units(ctx.conn, "p1") if u.type.value == "prose")
    result = call_tool(ctx, "verify_quote",
                       {"unit_id": prose.unit_id, "quote": "94.2% tracking accuracy"})
    assert result["grounded"] is True


def test_verification_needs_no_model_configured(ctx):
    assert ctx.nli is None
    assert ctx.writer is None
    unit = _table_unit(ctx)
    assert call_tool(ctx, "verify_quote",
                     {"unit_id": unit.unit_id, "quote": "| ours | 94.2 |"})["ok"] is True


def test_verify_quote_requires_both_arguments(ctx):
    unit = _table_unit(ctx)
    assert call_tool(ctx, "verify_quote", {"unit_id": unit.unit_id})["ok"] is False
    assert call_tool(ctx, "verify_quote", {"quote": "x"})["ok"] is False


def test_verifying_against_an_unknown_unit_is_not_grounded(ctx):
    result = call_tool(ctx, "verify_quote", {"unit_id": "nope", "quote": "anything"})
    assert result["ok"] is True
    assert result["grounded"] is False


def test_ask_is_unavailable_without_a_writer(ctx):
    result = call_tool(ctx, "ask", {"question": "how accurate?"})
    assert result["ok"] is False
    assert "writer" in result["error"]


def test_ask_returns_a_cited_answer_when_configured(ctx):
    unit = _table_unit(ctx)
    ctx.writer = FakeWriter({"how accurate?": Draft(
        text="It is accurate.",
        claims=(Claim("c-0", "It reaches 94.2%.", unit.unit_id, "| ours | 94.2 |"),))})
    ctx.nli = ENTAILS

    result = call_tool(ctx, "ask", {"question": "how accurate?"})
    assert result["ok"] is True
    assert result["claims"][0]["unit_id"] == unit.unit_id
    assert result["claims"][0]["verdict"] == "supported"
    assert unit.unit_id in result["answer"]


def test_ask_reports_blocked_claims_separately(ctx):
    unit = _table_unit(ctx)
    ctx.writer = FakeWriter({"how accurate?": Draft(
        text="It reaches 99.9%.",
        claims=(Claim("c-0", "It reaches 99.9%.", unit.unit_id, "| ours | 99.9 |"),))})
    ctx.nli = ENTAILS

    result = call_tool(ctx, "ask", {"question": "how accurate?"})
    assert result["ok"] is True
    assert result["blocked"] == 1
    assert result["claims"] == []
    assert "99.9" not in result["answer"]


def test_ask_requires_a_question(ctx):
    ctx.writer, ctx.nli = FakeWriter({}), ENTAILS
    assert call_tool(ctx, "ask", {})["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tools.py -v`
Expected: FAIL — `call_tool(ctx, "verify_quote", ...)` returns `unknown tool: verify_quote`

- [ ] **Step 3: Write the implementation**

Append to `jarvis/tools.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tools.py tests/test_tools_dispatch.py -v && ruff check jarvis/tools.py tests/test_tools.py`
Expected: PASS (27 + 10 tests), ruff clean

- [ ] **Step 5: Commit**

```bash
git add jarvis/tools.py tests/test_tools.py
git commit -m "feat: deterministic quote verification and cited ask tools"
```

---

### Task 4: The stdio MCP server and packaging

**Files:**
- Create: `jarvis/mcp_server.py`
- Modify: `pyproject.toml`
- Test: none — this file is protocol plumbing with no logic, and every behaviour it exposes is already covered by `tests/test_tools.py`. Its correctness contract is "it contains no logic," which Task 5 asserts.

**Interfaces:**
- Consumes: `REGISTRY`, `ToolContext`, `call_tool`, `tool_specs` from `jarvis.tools`; `Config` from `jarvis.config`; `open_store` from `jarvis.store`.
- Produces: `build_context(db_path, *, with_models=False) -> ToolContext`, `serve(ctx)`, `main(argv=None) -> int`.

**The rule for this file:** any line that is not translating between the MCP protocol and `call_tool` belongs in `jarvis/tools.py`. If a reviewer finds a conditional here that decides something about the corpus, that is a defect.

- [ ] **Step 1: Write the implementation**

```python
# jarvis/mcp_server.py
"""stdio MCP adapter (spec §3, build step 8).

Translation only. Every decision about the corpus lives in `jarvis/tools.py`, which has no
protocol dependency and is fully tested offline. `mcp` is imported inside `serve()` so the
package stays importable without the optional extra installed.

Run it:  jarvis-mcp --db ~/.jarvis/projects/gusts/corpus.db
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from jarvis.config import Config
from jarvis.store import open_store
from jarvis.tools import ToolContext, call_tool, tool_specs

SERVER_NAME = "jarvis-corpus"


def build_context(db_path: str | Path, *, with_models: bool = False) -> ToolContext:
    """Open the corpus and assemble a tool context.

    Without `--with-models` the server is search-and-verify only: no embedder download, no
    API key, no writer. `verify_quote` — the tool that actually stops fabrication — needs
    none of them.
    """
    conn = open_store(db_path)

    if not with_models:
        from jarvis.embed import FakeEmbedder
        return ToolContext(conn=conn, embedder=FakeEmbedder())

    from jarvis.embed import BGEEmbedder
    from jarvis.router import ModelRouter
    from jarvis.verify import HFNLI
    from jarvis.writer import LLMWriter

    config = Config.load()
    router = ModelRouter(overrides=config.model_overrides)
    return ToolContext(conn=conn, embedder=BGEEmbedder(), writer=LLMWriter(router),
                       nli=HFNLI())


def serve(ctx: ToolContext) -> None:
    """Run the stdio MCP loop until the client disconnects."""
    import anyio
    import mcp.types as types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server

    server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list() -> list[types.Tool]:
        return [types.Tool(**spec) for spec in tool_specs()]

    @server.call_tool()
    async def _call(name: str, arguments: dict | None) -> list[types.TextContent]:
        import json
        result = call_tool(ctx, name, arguments or {})
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-mcp",
                                     description="Serve a jarvis corpus over MCP.")
    parser.add_argument("--db", default=os.environ.get("JARVIS_DB", ""),
                        help="Path to the project's corpus.db (or set $JARVIS_DB).")
    parser.add_argument("--with-models", action="store_true",
                        help="Load the real embedder, NLI model, and writer. Enables `ask`.")
    args = parser.parse_args(argv)

    if not args.db:
        print("error: --db is required (or set $JARVIS_DB)", file=sys.stderr)
        return 2
    if not Path(args.db).is_file():
        print(f"error: no corpus at {args.db}", file=sys.stderr)
        return 2

    serve(build_context(args.db, with_models=args.with_models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Add the extra and the console script**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
mcp = ["mcp>=1.2", "anyio>=4.0"]
```

and add a new section:

```toml
[project.scripts]
jarvis-mcp = "jarvis.mcp_server:main"
```

- [ ] **Step 3: Verify the package still imports without `mcp` installed**

Run: `python -c "import jarvis; import jarvis.tools; print('ok')"`
Expected: `ok`. This must hold whether or not `mcp` is installed — `jarvis/tools.py` never imports it, and `jarvis/mcp_server.py` only imports it inside `serve()`.

Run: `python -m pytest -q && ruff check jarvis/mcp_server.py`
Expected: the whole suite still passes, ruff clean

- [ ] **Step 4: Commit**

```bash
git add jarvis/mcp_server.py pyproject.toml
git commit -m "feat: stdio mcp server and jarvis-mcp entry point"
```

---

### Task 5: End to end, exports, and the client config

**Files:**
- Create: `tests/test_mcp_end_to_end.py`
- Modify: `jarvis/__init__.py`, `README.md`

**Interfaces:**
- Consumes: everything this plan built.
- Produces: the extended public surface, and the documentation a user needs to actually point Claude Code at a corpus.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_end_to_end.py
"""A client session, played out against the dispatcher: search, read, verify, refuse."""
import json

import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Paper
from jarvis.parse import FakeParser
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.tools import REGISTRY, ToolContext, call_tool, tool_specs
from jarvis.units import build_units

BLOCKS = [
    Block(kind="heading", text="Results", page=3, section_path=("Results",)),
    Block(kind="paragraph", text="As shown in Table 3, we reach 94.2% tracking accuracy.",
          page=3, section_path=("Results",)),
    Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
          page=3, section_path=("Results",), label="Table 3"),
    Block(kind="caption", text="Table 3: Tracking accuracy under wind.", page=3,
          section_path=("Results",), label="Table 3"),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


@pytest.fixture
def ctx(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    parsed = FakeParser(BLOCKS).parse("p.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    yield ToolContext(conn=conn, embedder=FakeEmbedder())
    close_store(conn)


def test_the_full_honest_client_loop(ctx):
    """Search, open the best hit, quote it, verify the quote. This is the intended path."""
    hits = call_tool(ctx, "corpus_search", {"query": "tracking accuracy under wind"})
    assert hits["ok"] is True

    best = next(u for u in hits["units"] if "94.2" in u["text"])
    full = call_tool(ctx, "get_unit", {"unit_id": best["unit_id"]})
    assert full["ok"] is True

    check = call_tool(ctx, "verify_quote",
                      {"unit_id": best["unit_id"], "quote": "| ours | 94.2 |"})
    assert check["grounded"] is True


def test_the_dishonest_path_is_refused_at_the_last_step(ctx):
    hits = call_tool(ctx, "corpus_search", {"query": "tracking accuracy"})
    best = next(u for u in hits["units"] if "94.2" in u["text"])
    check = call_tool(ctx, "verify_quote",
                      {"unit_id": best["unit_id"], "quote": "| ours | 99.9 |"})
    assert check["grounded"] is False


def test_a_paraphrase_does_not_pass_as_a_quote(ctx):
    hits = call_tool(ctx, "corpus_search", {"query": "tracking accuracy"})
    best = hits["units"][0]
    check = call_tool(ctx, "verify_quote",
                      {"unit_id": best["unit_id"],
                       "quote": "the method achieved roughly ninety-four percent"})
    assert check["grounded"] is False


def test_every_tool_result_is_json_serializable(ctx):
    calls = [
        ("corpus_search", {"query": "accuracy"}),
        ("get_paper", {"paper_id": "p1"}),
        ("list_papers", {}),
        ("verify_quote", {"unit_id": "x", "quote": "y"}),
        ("nope", {}),
    ]
    for name, args in calls:
        assert json.dumps(call_tool(ctx, name, args))


def test_no_tool_call_can_raise(ctx):
    """Every tool, called with hostile arguments, returns a dict."""
    hostile = [{}, {"query": None}, {"unit_id": 12345}, {"limit": -1},
               {"query": "x" * 100000}, {"depth": "../../etc"}]
    for name in REGISTRY:
        for args in hostile:
            result = call_tool(ctx, name, args)
            assert isinstance(result, dict)
            assert "ok" in result


def test_the_tool_listing_tells_a_client_to_quote_exactly(ctx):
    listing = {t["name"]: t["description"] for t in tool_specs()}
    assert "exactly" in listing["corpus_search"].lower()
    assert "verbatim" in listing["verify_quote"].lower()


def test_the_expected_tools_are_all_registered():
    assert set(REGISTRY) == {"corpus_search", "get_unit", "get_paper", "list_papers",
                             "verify_quote", "ask"}


def test_the_server_module_contains_no_corpus_logic():
    """Everything decision-shaped belongs in tools.py, which is tested without a protocol."""
    import inspect

    import jarvis.mcp_server
    source = inspect.getsource(jarvis.mcp_server)
    for forbidden in ("find_span", "verify_claim", "rrf", "SELECT ", "quote_is_grounded"):
        assert forbidden not in source, f"{forbidden} belongs in jarvis/tools.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mcp_end_to_end.py -v`
Expected: FAIL — `test_no_tool_call_can_raise` is the one most likely to find a real defect. Fix the *handler*, not the test.

- [ ] **Step 3: Extend exports and document the client config**

Add to `jarvis/__init__.py` (keeping imports and `__all__` sorted):

```python
from jarvis.tools import REGISTRY, ToolContext, ToolSpec, call_tool, tool_specs, unit_payload
```

Do **not** export anything from `jarvis.mcp_server` — importing it from `jarvis/__init__.py` would drag the optional `mcp` dependency into the package's import path.

Add to `README.md`, after the existing usage material:

````markdown
## Use it from Claude Code

Install the extra and point a client at a project's corpus:

```bash
pip install -e ".[mcp]"
```

Then add to your MCP client config (`.mcp.json` for Claude Code):

```json
{
  "mcpServers": {
    "jarvis-corpus": {
      "command": "jarvis-mcp",
      "args": ["--db", "/absolute/path/to/corpus.db"]
    }
  }
}
```

Six tools become available: `corpus_search`, `get_unit`, `get_paper`, `list_papers`,
`verify_quote`, and `ask`.

`verify_quote` is the one that matters. It is a deterministic string match against the
immutable parsed paper — no model, no cost — so an assistant can check a quotation before
asserting it. Add `--with-models` to enable `ask`, which additionally needs a writer model
and a local NLI model.
````

- [ ] **Step 4: Run the whole suite**

Run: `python -m pytest -v && ruff check .`
Expected: all tests pass. `ruff check .` reports exactly the **11 pre-existing** violations.

- [ ] **Step 5: Commit**

```bash
git add jarvis/__init__.py README.md tests/test_mcp_end_to_end.py
git commit -m "test: mcp client loop, hostile-argument coverage, and client docs"
```

---

## Definition of done

- `python -m pytest` passes with zero network access, no API keys, no model downloads, and without the `mcp` package installed.
- `test_no_tool_call_can_raise` passes — every tool, called with hostile arguments, returns a dict.
- `test_the_server_module_contains_no_corpus_logic` passes — the protocol adapter is translation only.
- `test_the_dishonest_path_is_refused_at_the_last_step` passes — a plausible-looking fabricated quote is caught deterministically, with no model involved.
- `jarvis-mcp --db <path>` starts and serves; `python -c "import jarvis.tools"` works whether or not `mcp` is installed.
- `ruff check .` reports exactly the 11 pre-existing violations.

## Where this stops

Read-only tools over an existing corpus. Deliberately **not** exposed:

| Not exposed | Why |
|---|---|
| Gathering / screening | These spend money and rewrite the corpus. A read-only surface can be pointed at a shared project safely; a write surface cannot. |
| Card extraction | Same reason, plus it needs a model the search-only server does not load. |
| Contradiction scanning | `docs/plans/2026-08-14-contradiction-detection.md` builds it; expose it here afterwards if it proves useful interactively. |
| Long-form report generation | Minutes-long and multi-call. Wrong shape for a synchronous tool call. |

The natural follow-up, once the contradiction plan lands: a `find_contradictions` tool returning ranked candidates for human review — never assertions (spec §8).
