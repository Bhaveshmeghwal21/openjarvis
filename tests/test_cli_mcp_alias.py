"""Task 8: `jarvis mcp` alias delegating to `mcp_server.main`. `jarvis-mcp` stays
unchanged and independently callable for existing client configs."""
from __future__ import annotations

from jarvis.cli import main


def test_jarvis_mcp_delegates_to_mcp_server_main_with_remaining_args(monkeypatch):
    captured = {}

    def fake_mcp_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("jarvis.mcp_server.main", fake_mcp_main)

    exit_code = main(["mcp", "--db", "/some/corpus.db", "--with-models"])
    assert exit_code == 0
    assert captured["argv"] == ["--db", "/some/corpus.db", "--with-models"]


def test_jarvis_mcp_propagates_the_delegate_exit_code(monkeypatch):
    monkeypatch.setattr("jarvis.mcp_server.main", lambda argv: 2)
    exit_code = main(["mcp", "--db", "/nonexistent.db"])
    assert exit_code == 2


def test_jarvis_dash_mcp_entry_point_is_untouched():
    # jarvis-mcp must still exist as its own independent console script, unaffected by
    # the new jarvis mcp alias -- existing MCP client configs reference it directly.
    from jarvis import mcp_server
    assert callable(mcp_server.main)
