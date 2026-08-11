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


import re
from collections.abc import Sequence

from jarvis.models import Block

ARTIFACT_KINDS = {"table": UnitType.TABLE, "figure": UnitType.FIGURE,
                  "equation": UnitType.EQUATION}


def _label_pattern(label: str) -> re.Pattern[str] | None:
    """Match 'Table 3' / 'Tab. 3' / 'Fig 3' but never 'Table 30'."""
    m = re.match(r"([A-Za-z]+)\.?\s*(\d+)", label.strip())
    if not m:
        return None
    word, number = m.group(1), m.group(2)
    stem = re.escape(word[:3])
    return re.compile(rf"\b{stem}[a-z]*\.?\s*{re.escape(number)}\b(?!\d)", re.IGNORECASE)


def find_references(blocks: Sequence[Block], label: str) -> list[str]:
    """Prose blocks that mention `label`. This is what keeps 'as shown in Figure 3' bound."""
    pattern = _label_pattern(label)
    if pattern is None:
        return []
    return [b.text for b in blocks
            if b.kind in PROSE_KINDS and pattern.search(b.text)]


def build_artifact_units(parsed: ParsedPaper, start_ordinal: int = 0) -> list[Unit]:
    """One unit per table/figure/equation: artifact + caption + referring prose, indivisible."""
    blocks = list(parsed.blocks)
    units: list[Unit] = []
    ordinal = start_ordinal

    for i, block in enumerate(blocks):
        unit_type = ARTIFACT_KINDS.get(block.kind)
        if unit_type is None:
            continue

        parts: list[str] = []
        if block.text.strip():
            parts.append(block.text)

        if block.label:
            parts += [b.text for b in blocks
                      if b.kind == "caption" and b.label == block.label]
            parts += find_references(blocks, block.label)
        else:
            # Unlabelled artifact (common for equations): take the preceding prose block.
            for previous in reversed(blocks[:i]):
                if previous.kind in PROSE_KINDS:
                    parts.insert(0, previous.text)
                    break

        if not parts:
            continue

        unit = Unit(unit_id="", paper_id=parsed.paper_id, type=unit_type, page=block.page,
                    section_path=block.section_path,
                    verbatim_text=normalize(" ".join(parts)), ordinal=ordinal,
                    label=block.label)
        units.append(_with_id(unit))
        ordinal += 1
    return units
