"""Stage C — deep read (spec §7C).

For every paper the gate kept: parse (Layer 0) -> build typed units (Layer 1) ->
contextual prefixes -> embed -> index. This is the pipeline the single-paper core built,
wired end to end and driven from a decision set.

Spec §14: never silently ingest an empty parse. A paper that produces no blocks is
recorded as a parse failure and left at its previous depth, because a paper marked `deep`
with nothing in it is indistinguishable from a paper that genuinely says nothing — and
the second one is a claim about the literature.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import Embedder, index_units
from jarvis.gate import KEPT
from jarvis.gather import Candidate, to_paper
from jarvis.index import index_units_fts
from jarvis.models import Paper, Unit
from jarvis.parse import Parser
from jarvis.store import save_paper, save_units
from jarvis.units import DEFAULT_MAX_TOKENS, build_units


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    units: int = 0
    ok: bool = False
    error: str = ""


def ingest_paper(conn: sqlite3.Connection, paper: Paper, source_path: str,
                 parser: Parser, embedder: Embedder, prefix_generator=None,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> IngestResult:
    """One paper, all the way into the corpus. Never raises; failures are returned."""
    try:
        parsed = parser.parse(source_path, paper.paper_id)
    except Exception as exc:  # noqa: BLE001 - a bad PDF is data, not a crash
        return IngestResult(paper_id=paper.paper_id, ok=False, error=f"parse failed: {exc}")

    if not parsed.blocks or not parsed.raw_text.strip():
        return IngestResult(paper_id=paper.paper_id, ok=False,
                            error="empty parse — escalate to a stronger parser (spec §5)")

    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")

    units: Sequence[Unit] = build_units(parsed, max_tokens=max_tokens)
    units = apply_prefixes(units, paper, prefix_generator or TemplatePrefix())
    save_units(conn, list(units))
    index_units_fts(conn, units)
    index_units(conn, units, embedder)

    return IngestResult(paper_id=paper.paper_id, units=len(units), ok=True)


def ingest_decided(conn: sqlite3.Connection, decisions: Mapping[str, str],
                   candidates: Sequence[Candidate], parser: Parser, embedder: Embedder,
                   prefix_generator=None,
                   path_for: Callable[[Candidate], str] | None = None,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> list[IngestResult]:
    """Deep-read every kept paper. `unsure` is read exactly like `read_deep` (spec §7B).

    One paper failing never stops the batch — a corrupt PDF in a 300-paper gather must
    cost one paper, not the run.
    """
    resolve = path_for or (lambda c: c.paper.get("pdf_url", "") or c.paper.get("url", ""))
    out: list[IngestResult] = []
    for candidate in candidates:
        if decisions.get(candidate.pid) not in KEPT:
            continue
        out.append(ingest_paper(conn, to_paper(candidate), resolve(candidate), parser,
                                embedder, prefix_generator, max_tokens=max_tokens))
    return out


def failed(results: Sequence[IngestResult]) -> list[IngestResult]:
    """The parse-failure log spec §14 requires. Never let these disappear silently."""
    return [r for r in results if not r.ok]
