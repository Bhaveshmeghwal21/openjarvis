"""Contextual prefixes for child units (spec §5).

Anthropic's Contextual Retrieval: a 50-100 token description of where a chunk sits in its
document, prepended before embedding. Measured 35% fewer retrieval failures alone, 49%
with BM25, 67% with reranking.

The prefix is embedded but never stored as part of verbatim_text — a claim must always
resolve to text that genuinely appears in Layer 0.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from jarvis.models import Paper, Unit

_PROMPT = (
    "Write one sentence of 50-100 tokens locating this excerpt within its paper, so it can "
    "be understood on its own. State the paper, the section, and what the excerpt reports. "
    "Do not add facts that are not present.\n\n"
    "Paper: {title} ({year})\nSection: {section}\nArtifact: {label}\n\nExcerpt:\n{text}"
)


@runtime_checkable
class PrefixGenerator(Protocol):
    def describe(self, paper: Paper, unit: Unit) -> str: ...


class TemplatePrefix:
    """Deterministic, free, no model. The fallback and the test default."""

    def describe(self, paper: Paper, unit: Unit) -> str:
        section = " > ".join(unit.section_path) if unit.section_path else "body"
        artifact = f" ({unit.label})" if unit.label else ""
        year = f", {paper.year}" if paper.year else ""
        return (f"From \"{paper.title}\"{year}, {unit.type.value} in section {section}"
                f"{artifact}, page {unit.page}.")


class LLMPrefix:
    """LLM-written prefix, routed to the cheap tier. Falls back to the template on failure."""

    def __init__(self, router, chat_fn: Callable[..., str] | None = None) -> None:
        self._router = router
        self._chat = chat_fn
        self._fallback = TemplatePrefix()

    def _chat_fn(self) -> Callable[..., str]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def describe(self, paper: Paper, unit: Unit) -> str:
        prompt = _PROMPT.format(
            title=paper.title, year=paper.year or "n.d.",
            section=" > ".join(unit.section_path) or "body",
            label=unit.label or "none", text=unit.verbatim_text[:2000],
        )
        try:
            out = self._chat_fn()(self._router, "contextual_prefix", prompt)
            return (out or "").strip() or self._fallback.describe(paper, unit)
        except Exception:  # noqa: BLE001
            return self._fallback.describe(paper, unit)


def apply_prefixes(units: Sequence[Unit], paper: Paper,
                   generator: PrefixGenerator) -> list[Unit]:
    """Return copies of `units` with `context_prefix` filled in. verbatim_text is untouched."""
    return [replace(u, context_prefix=generator.describe(paper, u)) for u in units]


def embedding_text(unit: Unit) -> str:
    """What actually gets embedded: prefix then verbatim text."""
    if not unit.context_prefix:
        return unit.verbatim_text
    return f"{unit.context_prefix}\n\n{unit.verbatim_text}"
