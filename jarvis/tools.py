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
