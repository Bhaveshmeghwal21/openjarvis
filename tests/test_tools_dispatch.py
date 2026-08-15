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


def test_clamp_limit_survives_an_infinite_float():
    # Final whole-branch adversarial review, Finding 3: int(float("inf")) raises
    # OverflowError, which the original except (TypeError, ValueError) missed. The
    # no-raise contract still held end-to-end via call_tool's own outer try/except, but
    # clamp_limit's own docstring claims to "never trust the other side" -- this closes
    # the gap at the source rather than relying only on the outer defense-in-depth layer.
    from jarvis.tools import clamp_limit
    assert clamp_limit(float("inf"), default=8, maximum=25) == 8
    assert clamp_limit(float("-inf"), default=8, maximum=25) == 8


def test_tools_module_does_not_import_mcp():
    import inspect

    import jarvis.tools
    source = inspect.getsource(jarvis.tools)
    assert "import mcp" not in source
    assert "from mcp" not in source
