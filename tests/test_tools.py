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


def test_corpus_search_with_no_hits_is_an_empty_success(tmp_path):
    # FakeEmbedder's vector search has no relevance floor (see
    # LEDGER-compile-cited-qa.md Finding 4 for the prior instance of this exact fixture
    # limitation) -- against a nonempty store, RRF fusion always ranks *some* candidate
    # regardless of query content. A genuinely empty store is the only way to exercise the
    # real empty-retrieval path rather than accidentally asserting on a lucky query string.
    conn = open_store(tmp_path / "empty.db")
    empty_ctx = ToolContext(conn=conn, embedder=FakeEmbedder())
    result = call_tool(empty_ctx, "corpus_search", {"query": "zzz nonexistent qqq"})
    close_store(conn)
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
