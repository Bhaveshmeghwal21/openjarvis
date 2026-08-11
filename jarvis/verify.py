"""Two-stage mechanical verification (spec §8).

Stage 1 — quote grounding. Deterministic string match against Layer 0. Free, exact, no
model. A claim whose quote is absent is blocked and never reaches stage 2.

Stage 2 — entailment. An NLI model over (quote -> claim). NOT LLM-as-judge: measured
Pearson correlation with human judgement is 0.101 for GPT-3.5-as-judge versus 0.638 for
AutoAIS/NLI. Contradiction detection is the same pass reading NLI's third label.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from jarvis.models import Claim, Unit, Verdict, Verification
from jarvis.store import get_raw_text, get_unit
from jarvis.text import find_span

LABELS = ("entailment", "neutral", "contradiction")


@runtime_checkable
class NLIModel(Protocol):
    def predict(self, premise: str, hypothesis: str) -> dict[str, float]: ...


class FakeNLI:
    """Deterministic NLI for tests. Looks up (premise, hypothesis), else returns `default`."""

    def __init__(self, mapping: Mapping[tuple[str, str], dict[str, float]] | None = None,
                 default: dict[str, float] | None = None) -> None:
        self._mapping = dict(mapping or {})
        self._default = default or {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0}

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self._mapping.get((premise, hypothesis), self._default)


class HFNLI:
    """Real adapter. `transformers` is imported lazily; runs locally, no API cost."""

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli") -> None:
        self._model_name = model_name
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self._model_name,
                                  top_k=None)
        return self._pipe

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        raw = self._load()(f"{premise}</s></s>{hypothesis}")[0]
        scores = {r["label"].lower(): float(r["score"]) for r in raw}
        return {label: scores.get(label, 0.0) for label in LABELS}


def quote_is_grounded(conn: sqlite3.Connection, claim: Claim) -> bool:
    """Stage 1. True only when the quote appears verbatim in the unit or its paper's Layer 0."""
    if not claim.quote.strip():
        return False
    unit = get_unit(conn, claim.unit_id)
    if unit is None:
        return False
    if find_span(claim.quote, unit.verbatim_text) is not None:
        return True
    return find_span(claim.quote, get_raw_text(conn, unit.paper_id)) is not None


def verify_claim(conn: sqlite3.Connection, claim: Claim, nli: NLIModel,
                 threshold: float = 0.5) -> Verification:
    """Run both stages. Stage 2 is never reached when stage 1 fails."""
    if not quote_is_grounded(conn, claim):
        return Verification(claim_id=claim.claim_id, unit_id=claim.unit_id,
                            quote_found=False, verdict=Verdict.QUOTE_NOT_FOUND)

    scores = nli.predict(claim.quote, claim.text)
    entail = float(scores.get("entailment", 0.0))
    contra = float(scores.get("contradiction", 0.0))

    if contra >= threshold and contra > entail:
        verdict = Verdict.CONTRADICTED
    elif entail >= threshold:
        verdict = Verdict.SUPPORTED
    else:
        verdict = Verdict.NEUTRAL

    return Verification(claim_id=claim.claim_id, unit_id=claim.unit_id, quote_found=True,
                        verdict=verdict, entailment_score=entail,
                        contradiction_score=contra)


def find_contradictions(conn: sqlite3.Connection, claim: Claim, units: Sequence[Unit],
                        nli: NLIModel, threshold: float = 0.5) -> list[tuple[str, float]]:
    """Cross-corpus conflicts, free from the same NLI pass (spec §8).

    Returns (unit_id, contradiction_score) above threshold, most confident first. These are
    ranked candidates for human review, never assertions.
    """
    found: list[tuple[str, float]] = []
    for unit in units:
        if unit.unit_id == claim.unit_id:
            continue
        score = float(nli.predict(unit.verbatim_text, claim.text).get("contradiction", 0.0))
        if score >= threshold:
            found.append((unit.unit_id, score))
    return sorted(found, key=lambda pair: pair[1], reverse=True)
