"""Layer 0 -> Layer 1: typed evidence units (spec §5).

Chunking is recursive ~512 tokens on section boundaries. Not semantic chunking: measured
69% vs 54% on academic papers, and 14x slower.
"""
from __future__ import annotations

from jarvis.models import ParsedPaper, Unit, UnitType
from jarvis.text import approx_tokens, normalize

PROSE_KINDS = {"paragraph", "text"}
DEFAULT_MAX_TOKENS = 512


def _split_to_budget(text: str, max_tokens: int) -> list[str]:
    """Split on word boundaries, flushing a piece as soon as it reaches the token budget."""
    if approx_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if approx_tokens(" ".join(current)) >= max_tokens:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))
    return pieces


def build_prose_units(parsed: ParsedPaper, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Unit]:
    """Group contiguous prose blocks by section, then split each group to the token budget."""
    groups: list[tuple[tuple[str, ...], int, list[str]]] = []
    for block in parsed.blocks:
        if block.kind not in PROSE_KINDS or not block.text.strip():
            continue
        if groups and groups[-1][0] == block.section_path and groups[-1][1] == block.page:
            groups[-1][2].append(block.text)
        else:
            groups.append((block.section_path, block.page, [block.text]))

    units: list[Unit] = []
    ordinal = 0
    for section_path, page, texts in groups:
        for piece in _split_to_budget(normalize(" ".join(texts)), max_tokens):
            unit = Unit(unit_id="", paper_id=parsed.paper_id, type=UnitType.PROSE,
                        page=page, section_path=section_path, verbatim_text=piece,
                        ordinal=ordinal)
            units.append(_with_id(unit))
            ordinal += 1
    return units


def _with_id(unit: Unit) -> Unit:
    from dataclasses import replace
    return replace(unit, unit_id=unit.key())
