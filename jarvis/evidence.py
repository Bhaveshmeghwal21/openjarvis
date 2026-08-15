"""The evidence budget — capped, ordered, and rendered (spec §7 Stage D).

Increased search depth consistently degrades factual accuracy while surface-level citation
metrics stay stable (arXiv 2605.06635): more evidence makes an answer look better and be
worse, and the metrics that would catch it do not move. Order-preserving retrieval with
48K well-chosen tokens beat full-context 117K by 13 F1 points at one-seventh the budget.

Hence: a hard cap, many small calls rather than one large one, and the strongest evidence
placed at the beginning and the end of the context to exploit primacy and recency and
avoid lost-in-the-middle degradation (20+ percentage points).

Raising `MAX_UNITS` is not a tuning knob. It is the failure this module exists to prevent.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.models import Unit
from jarvis.text import approx_tokens

MAX_UNITS = 12
MAX_TOKENS = 6000


@dataclass(frozen=True)
class EvidenceSet:
    """What actually reaches a synthesis call, plus how much was left out."""
    units: tuple[Unit, ...] = ()
    dropped: int = 0
    tokens: int = 0


def cap(units: Sequence[Unit], max_units: int = MAX_UNITS,
        max_tokens: int = MAX_TOKENS) -> EvidenceSet:
    """Truncate a ranked list to the budget, preserving rank order.

    The first unit is always kept even if it alone blows the token budget: returning an
    empty evidence set for a real retrieval hit would silently turn a groundable question
    into an ungroundable one.
    """
    kept: list[Unit] = []
    total = 0
    for unit in units:
        if len(kept) >= max_units:
            break
        size = approx_tokens(unit.verbatim_text)
        if kept and total + size > max_tokens:
            break
        kept.append(unit)
        total += size
    return EvidenceSet(units=tuple(kept), dropped=max(0, len(units) - len(kept)),
                       tokens=total)


def order_for_context(units: Sequence[Unit]) -> list[Unit]:
    """Re-order a best-first ranking so the strongest evidence sits at both ends.

    Rank 0 goes first, rank 1 last, rank 2 second, rank 3 second-to-last, and so on. The
    weakest evidence ends up in the middle, which is exactly where a model attends least.
    """
    front: list[Unit] = []
    back: list[Unit] = []
    for index, unit in enumerate(units):
        (front if index % 2 == 0 else back).append(unit)
    return front + list(reversed(back))


def render(units: Sequence[Unit]) -> str:
    """Evidence as model-visible text. The `[unit_id]` label is what makes citing possible."""
    blocks: list[str] = []
    for unit in units:
        header = f"[{unit.unit_id}]"
        if unit.context_prefix:
            header = f"{header} {unit.context_prefix}"
        blocks.append(f"{header}\n{unit.verbatim_text}")
    return "\n\n".join(blocks)
