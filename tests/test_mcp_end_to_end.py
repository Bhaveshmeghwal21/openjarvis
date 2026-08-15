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
