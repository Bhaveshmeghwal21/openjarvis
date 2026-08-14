"""Compile — cited Q&A (spec §7 Stage D, §8).

Retrieve iteratively, cap and order the evidence, draft, then verify in a separate pass.

Three outcomes, deliberately different:
  * SUPPORTED                — quote is in Layer 0 and entails the claim. Kept and cited.
  * NEUTRAL / CONTRADICTED   — quote is real, entailment is not established. Flagged.
  * QUOTE_NOT_FOUND          — the quote is not in Layer 0. BLOCKED, removed entirely.

Blocking versus flagging is not a stylistic choice. Stage 1 is deterministic and exact, so
its failure is proof of fabrication and the claim cannot stand. Stage 2 is an NLI model,
and spec §8 is explicit that it is a filter and not an oracle — AttributionBench found even
fine-tuned GPT-3.5 reaches only ~80% macro-F1 on binary attribution — so a stage-2 failure
surfaces for a human instead of being silently passed or silently deleted.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from jarvis.embed import Embedder
from jarvis.evidence import MAX_TOKENS, MAX_UNITS, cap, order_for_context
from jarvis.models import Claim, Unit, Verdict, Verification
from jarvis.retrieve import Reranker
from jarvis.retriever import Refiner, retrieve_iteratively
from jarvis.verify import NLIModel, verify_claim
from jarvis.writer import Writer

FLAGGED_VERDICTS = (Verdict.NEUTRAL, Verdict.CONTRADICTED)


@dataclass(frozen=True)
class Answer:
    """One answer plus the full record of how every sentence in it was checked."""
    question: str
    text: str = ""
    claims: tuple[Claim, ...] = ()
    verifications: tuple[Verification, ...] = ()
    units: tuple[Unit, ...] = ()
    queries: tuple[str, ...] = ()
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

    @property
    def is_grounded(self) -> bool:
        """True only when there is at least one claim and every one of them is supported."""
        return bool(self.supported) and not self.blocked and not self.flagged

    def claim_for(self, claim_id: str) -> Claim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)


def ask(conn: sqlite3.Connection, question: str, embedder: Embedder, writer: Writer,
        nli: NLIModel, *, refiner: Refiner | None = None, rounds: int = 2, limit: int = 8,
        reranker: Reranker | None = None, max_units: int = MAX_UNITS,
        max_tokens: int = MAX_TOKENS, threshold: float = 0.5) -> Answer:
    """One question, end to end. The writer drafts; a separate pass verifies."""
    retrieval = retrieve_iteratively(conn, question, embedder, refiner=refiner,
                                     rounds=rounds, limit=limit, reranker=reranker)
    budget = cap(retrieval.units, max_units=max_units, max_tokens=max_tokens)
    evidence = order_for_context(budget.units)

    draft = writer.write(question, evidence)
    verifications = tuple(verify_claim(conn, claim, nli, threshold=threshold)
                          for claim in draft.claims)

    return Answer(question=question, text=draft.text, claims=draft.claims,
                  verifications=verifications, units=tuple(evidence),
                  queries=retrieval.queries, dropped_evidence=budget.dropped)


def render_answer(answer: Answer) -> str:
    """Human-readable output. Blocked claims are absent; flagged ones carry a warning."""
    if not answer.claims:
        return "No evidence in this corpus answers that question."

    lines: list[str] = []
    for verification in answer.supported:
        claim = answer.claim_for(verification.claim_id)
        if claim is not None:
            lines.append(f"{claim.text} [{claim.unit_id}]")

    if answer.flagged:
        lines.append("")
        lines.append("Unverified — the quote is real but does not clearly support the claim:")
        for verification in answer.flagged:
            claim = answer.claim_for(verification.claim_id)
            if claim is not None:
                lines.append(f"  - {claim.text} [{claim.unit_id}] "
                             f"({verification.verdict.value})")

    if answer.blocked:
        lines.append("")
        lines.append(f"{len(answer.blocked)} claim(s) were removed: their quotes do not "
                     f"appear in any source paper.")

    if not answer.supported and not answer.flagged:
        return ("No claim in the draft could be grounded in the corpus. "
                f"{len(answer.blocked)} claim(s) were removed.")
    return "\n".join(lines)
