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


# The word[:3] stem heuristic works only when an abbreviation shares a 3+ letter prefix
# with the full word ("Table"/"Tab.", "Figure"/"Fig." both do). "Eq." does not share a
# 3-letter prefix with "Equation", so it needs an explicit override — without one,
# find_references(blocks, "Equation 5") would never match "Eq. 5 gives the result",
# silently dropping referencing prose for equations specifically.
_ABBREVIATIONS = {"equation": "eq"}


def _label_pattern(label: str) -> re.Pattern[str] | None:
    """Match 'Table 3' / 'Tab. 3' / 'Fig 3' / 'Eq. 5' but never 'Table 30' or 'Table 3.1'.

    The trailing lookahead rejects both a directly-following digit ("30") and a
    decimal continuation (".1", ".10") — without the latter, "Table 3" would wrongly
    match inside "Table 3.1 shows...", folding prose about a different, more specific
    table into Table 3's evidence unit.
    """
    m = re.match(r"([A-Za-z]+)\.?\s*(\d+)", label.strip())
    if not m:
        return None
    word, number = m.group(1), m.group(2)
    stem = re.escape(_ABBREVIATIONS.get(word.lower(), word[:3]))
    return re.compile(rf"\b{stem}[a-z]*\.?\s*{re.escape(number)}\b(?!\.?\d)", re.IGNORECASE)


def _nearest_of(position: int, candidates: Sequence[int]) -> int | None:
    """Which of `candidates` (block indices) is closest to `position`. Ties favor the earlier
    candidate, so a caption/reference sitting exactly between two same-labeled artifacts is
    deterministically attributed rather than binding to whichever the loop iterated to last.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda c: (abs(c - position), c))


def find_references(blocks: Sequence[Block], label: str, *, owner: int | None = None,
                    candidates: Sequence[int] | None = None) -> list[str]:
    """Prose blocks that mention `label`. This is what keeps 'as shown in Figure 3' bound.

    When `owner` and `candidates` are given, a match is only returned when `owner` (a block
    index) is the nearest of `candidates` (block indices sharing `label`) to the matching
    prose block — this disambiguates which of several same-labeled artifacts a reference
    belongs to. Omitted (the default), every match in the document is returned, which is
    correct whenever `label` is unique — the common case, and the only one the original
    single-artifact call sites need.
    """
    pattern = _label_pattern(label)
    if pattern is None:
        return []
    found: list[str] = []
    for i, b in enumerate(blocks):
        if b.kind not in PROSE_KINDS or not pattern.search(b.text):
            continue
        if owner is not None and candidates and _nearest_of(i, candidates) != owner:
            continue
        found.append(b.text)
    return found


def build_artifact_units(parsed: ParsedPaper, start_ordinal: int = 0) -> list[Unit]:
    """One unit per table/figure/equation: artifact + caption + referring prose, indivisible.

    Caption/reference binding is scoped to the *nearest* same-labeled artifact, not a
    document-wide exact-label scan. A document-wide scan would let two distinct artifacts
    that happen to share a label (e.g. an appendix restarting "Table 3" numbering) each
    absorb the other's caption — the text is genuinely present, but attributed to the wrong
    table. Binding by proximity keeps each artifact's evidence pointed at its own location
    even when labels collide; it is a no-op whenever a label is unique in the document,
    which is the common case and the one every existing test exercises.
    """
    blocks = list(parsed.blocks)
    artifact_positions = [i for i, b in enumerate(blocks) if b.kind in ARTIFACT_KINDS]
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
            same_label = [j for j in artifact_positions if blocks[j].label == block.label]
            parts += [b.text for j, b in enumerate(blocks)
                      if b.kind == "caption" and b.label == block.label
                      and _nearest_of(j, same_label) == i]
            parts += find_references(blocks, block.label, owner=i, candidates=same_label)
        else:
            # Unlabelled artifact (common for equations): take the preceding prose block.
            for previous in reversed(blocks[:i]):
                if previous.kind in PROSE_KINDS:
                    parts.insert(0, previous.text)
                    break

        if not parts:
            # A labeled-but-textless artifact (e.g. a figure with no matched caption and
            # no referencing prose) must still become a unit, never vanish silently —
            # fall back to its label, or a minimal placeholder if it has none.
            parts = [block.label] if block.label else [
                f"({unit_type.value} on page {block.page}, no extractable text or caption)"
            ]

        unit = Unit(unit_id="", paper_id=parsed.paper_id, type=unit_type, page=block.page,
                    section_path=block.section_path,
                    verbatim_text=normalize(" ".join(parts)), ordinal=ordinal,
                    label=block.label)
        units.append(_with_id(unit))
        ordinal += 1
    return units


from dataclasses import replace as _replace


def build_units(parsed: ParsedPaper, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Unit]:
    """Full Layer 1 build: prose children with section parents, plus artifact units.

    Children are what get embedded and matched; parents are what get sent to the model at
    answer time (spec §5, parent/child structure).
    """
    children = build_prose_units(parsed, max_tokens=max_tokens)
    artifacts = build_artifact_units(parsed, start_ordinal=len(children))

    by_section: dict[tuple[tuple[str, ...], int], list[Unit]] = {}
    for child in children:
        by_section.setdefault((child.section_path, child.page), []).append(child)

    out: list[Unit] = []
    parent_ordinal = len(children) + len(artifacts)
    for (section_path, page), group in by_section.items():
        if len(group) < 2:
            out.extend(group)          # a section that fits in one unit needs no parent
            continue
        parent = Unit(unit_id="", paper_id=parsed.paper_id, type=UnitType.PROSE, page=page,
                      section_path=section_path,
                      verbatim_text=" ".join(u.verbatim_text for u in group),
                      ordinal=parent_ordinal)
        parent = _with_id(parent)
        parent_ordinal += 1
        out.append(parent)
        out.extend(_replace(u, parent_id=parent.unit_id) for u in group)

    out.extend(artifacts)
    return out
