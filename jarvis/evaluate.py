"""Evaluation metrics (spec §10).

Built before capability expansion, deliberately: without these, "the corpus is good" is
unfalsifiable and every later build step rests on an unmeasured foundation.

Targets: quote fidelity 1.0 (any failure is fabrication), gate recall >= 0.95 (field
standard for screening tools), statement support >= 0.90.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from jarvis.models import Verdict, Verification

QUOTE_FIDELITY_TARGET = 1.0
GATE_RECALL_TARGET = 0.95
STATEMENT_SUPPORT_TARGET = 0.90

KEPT_DECISIONS = {"read_deep", "unsure"}


def quote_fidelity(verifications: Sequence[Verification]) -> float:
    """Fraction of claims whose quote was found verbatim in Layer 0. Target 1.0."""
    if not verifications:
        return 1.0
    return sum(1 for v in verifications if v.quote_found) / len(verifications)


def statement_support(verifications: Sequence[Verification]) -> float:
    """Fraction of claims entailed by their cited quote. Target >= 0.90."""
    if not verifications:
        return 0.0
    return sum(1 for v in verifications if v.verdict is Verdict.SUPPORTED) / len(verifications)


def gate_recall(decisions: Mapping[str, str], labels: Mapping[str, bool]) -> float:
    """Fraction of hand-labelled relevant papers the gate kept. Target >= 0.95.

    `unsure` counts as kept — spec §7B escalates it to deep read.
    """
    relevant = [pid for pid, is_relevant in labels.items() if is_relevant]
    if not relevant:
        return 1.0
    kept = sum(1 for pid in relevant if decisions.get(pid) in KEPT_DECISIONS)
    return kept / len(relevant)


def coverage(cited_unit_ids: Iterable[str], corpus_unit_ids: Iterable[str]) -> float:
    """Fraction of the deep-read corpus actually cited. Tracked, not targeted."""
    corpus = set(corpus_unit_ids)
    if not corpus:
        return 0.0
    return len(set(cited_unit_ids) & corpus) / len(corpus)


@dataclass(frozen=True)
class EvalReport:
    quote_fidelity: float
    statement_support: float
    gate_recall: float | None = None
    coverage: float | None = None

    @property
    def meets_quote_target(self) -> bool:
        return self.quote_fidelity >= QUOTE_FIDELITY_TARGET

    @property
    def meets_support_target(self) -> bool:
        return self.statement_support >= STATEMENT_SUPPORT_TARGET

    @property
    def meets_gate_target(self) -> bool | None:
        if self.gate_recall is None:
            return None
        return self.gate_recall >= GATE_RECALL_TARGET


def report(verifications: Sequence[Verification],
           decisions: Mapping[str, str] | None = None,
           labels: Mapping[str, bool] | None = None,
           cited: Iterable[str] | None = None,
           corpus: Iterable[str] | None = None) -> EvalReport:
    """Bundle the spec §10 metrics that the available data supports."""
    return EvalReport(
        quote_fidelity=quote_fidelity(verifications),
        statement_support=statement_support(verifications),
        gate_recall=(gate_recall(decisions, labels)
                     if decisions is not None and labels is not None else None),
        coverage=(coverage(cited, corpus)
                  if cited is not None and corpus is not None else None),
    )
