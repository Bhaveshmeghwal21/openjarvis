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
    from mcp import types
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
