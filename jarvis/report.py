"""Long-form reports — AutoSurvey decomposition (spec §7 Stage D).

Outline from cards, draft each subsection against its OWN bounded evidence set, integrate,
verify. A report gets no leniency for being long: the verification pass is exactly the one
`jarvis.answer` runs for a single sentence.

Per-section budgets rather than one global context is the load-bearing choice. Increased
search depth consistently degrades factual accuracy while surface-level citation metrics
stay stable (arXiv 2605.06635) — a long report is the easiest artifact in this system to
make look excellent and be worthless.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from jarvis.embed import Embedder
from jarvis.evidence import MAX_TOKENS, MAX_UNITS, cap, order_for_context
from jarvis.models import Claim, Unit, Verdict, Verification
from jarvis.outline import Section
from jarvis.retrieve import Reranker
from jarvis.retriever import Refiner, retrieve_iteratively
from jarvis.verify import NLIModel, verify_claim
from jarvis.writer import Writer

FLAGGED_VERDICTS = (Verdict.NEUTRAL, Verdict.CONTRADICTED)


@dataclass(frozen=True)
class SectionDraft:
    """One drafted, verified section. Mirrors `jarvis.answer.Answer` at section scope."""
    section: Section
    text: str = ""
    claims: tuple[Claim, ...] = ()
    verifications: tuple[Verification, ...] = ()
    units: tuple[Unit, ...] = ()
    dropped_evidence: int = 0

    def _by_verdict(self, *verdicts: Verdict) -> tuple[Verification, ...]:
        return tuple(v for v in self.verifications if v.verdict in verdicts)

    @property
    def supported(self) -> tuple[Verification, ...]:
        return self._by_verdict(Verdict.SUPPORTED)

    @property
    def flagged(self) -> tuple[Verification, ...]:
        return self._by_verdict(*FLAGGED_VERDICTS)

    @property
    def blocked(self) -> tuple[Verification, ...]:
        return self._by_verdict(Verdict.QUOTE_NOT_FOUND)

    def claim_for(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)


def draft_section(conn: sqlite3.Connection, section: Section, embedder: Embedder,
                  writer: Writer, nli: NLIModel, *, refiner: Refiner | None = None,
                  rounds: int = 2, limit: int = 8, reranker: Reranker | None = None,
                  max_units: int = MAX_UNITS, max_tokens: int = MAX_TOKENS,
                  threshold: float = 0.5) -> SectionDraft:
    """Retrieve for this section's sub-question only, cap, draft, verify.

    The cap is applied here, per section — never once over an assembled whole-report
    context. That is the difference between many small well-scoped calls and one large
    one, and the measured difference is 13 F1 points.
    """
    retrieval = retrieve_iteratively(conn, section.question, embedder, refiner=refiner,
                                     rounds=rounds, limit=limit, reranker=reranker)
    budget = cap(retrieval.units, max_units=max_units, max_tokens=max_tokens)
    evidence = order_for_context(budget.units)

    draft = writer.write(section.question, evidence)
    verifications = tuple(verify_claim(conn, claim, nli, threshold=threshold)
                          for claim in draft.claims)

    return SectionDraft(section=section, text=draft.text, claims=draft.claims,
                        verifications=verifications, units=tuple(evidence),
                        dropped_evidence=budget.dropped)
