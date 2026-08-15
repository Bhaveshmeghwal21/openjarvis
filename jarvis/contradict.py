"""Cross-paper contradiction detection (spec §8).

Free from the verification pass: NLI emits entailment, neutral, and contradiction, and the
verifier has been computing all three and reading two. Running claims from one paper
against evidence from others surfaces cross-corpus conflicts at no additional model cost.

This is structurally impossible for any system that does not retain a corpus, and the
shape genuinely favours a machine: one-versus-many per claim, many-versus-many across a
corpus. But it is hard for LLMs and for humans alike (arXiv 2504.00180), and ContraCrow's
own precision against expert review was 70%. Output is therefore RANKED CANDIDATES FOR
HUMAN REVIEW, never assertions. Nothing in this module returns a contradiction as a fact.

Retrieval-first, not corpus-wide: 500 papers is ~100k units, and cross-checking every
claim against all of them is quadratic and unaffordable. Two papers that never discuss the
same thing cannot disagree, so retrieval loses nothing real.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from jarvis.embed import Embedder
from jarvis.models import Claim, Unit
from jarvis.retrieve import Reranker, search
from jarvis.store import get_unit, save_contradictions
from jarvis.verify import NLIModel, find_contradictions


def opposing_units(conn: sqlite3.Connection, claim: Claim, embedder: Embedder, *,
                   limit: int = 20, reranker: Reranker | None = None) -> list[Unit]:
    """Evidence from OTHER papers that is topically close to this claim.

    Excludes the claim's own paper entirely, not merely its own unit: a paper's Results
    section "contradicting" its own Limitations is a claim-extraction artifact, and a scan
    that reports those will bury the real signal in self-conflict noise.
    """
    own = get_unit(conn, claim.unit_id)
    if own is None:
        return []

    # Over-fetch: the claim's own paper is usually the best match for its own claim text,
    # so a tight limit would return nothing but self-hits before filtering.
    hits = search(conn, claim.text, embedder, limit=max(limit * 3, limit),
                  reranker=reranker, expand_parents=False)

    out: list[Unit] = []
    seen: set[str] = set()
    for unit in hits:
        if unit.paper_id == own.paper_id or unit.unit_id in seen:
            continue
        seen.add(unit.unit_id)
        out.append(unit)
        if len(out) >= limit:
            break
    return out



EVIDENCE_PREVIEW = 500


@dataclass(frozen=True)
class Conflict:
    """One candidate disagreement. A prompt for a human to look, never a finding."""
    claim_id: str
    claim_text: str
    claim_paper_id: str
    unit_id: str
    paper_id: str
    score: float
    evidence: str = ""


def scan_claim(conn: sqlite3.Connection, claim: Claim, nli: NLIModel, embedder: Embedder, *,
               limit: int = 20, threshold: float = 0.5,
               reranker: Reranker | None = None) -> list[Conflict]:
    """Candidates for one claim. Reuses `verify.find_contradictions` for the NLI pass."""
    own = get_unit(conn, claim.unit_id)
    if own is None:
        return []

    units = opposing_units(conn, claim, embedder, limit=limit, reranker=reranker)
    if not units:
        return []

    by_id = {u.unit_id: u for u in units}
    out: list[Conflict] = []
    for unit_id, score in find_contradictions(conn, claim, units, nli, threshold=threshold):
        unit = by_id.get(unit_id)
        if unit is None:
            continue
        out.append(Conflict(
            claim_id=claim.claim_id, claim_text=claim.text, claim_paper_id=own.paper_id,
            unit_id=unit.unit_id, paper_id=unit.paper_id, score=score,
            evidence=unit.verbatim_text[:EVIDENCE_PREVIEW],
        ))
    return out


def rank(conflicts: Sequence[Conflict]) -> list[Conflict]:
    """Most confident first, one row per (claim, unit) pair."""
    best: dict[tuple[str, str], Conflict] = {}
    for conflict in conflicts:
        key = (conflict.claim_id, conflict.unit_id)
        if key not in best or conflict.score > best[key].score:
            best[key] = conflict
    return sorted(best.values(), key=lambda c: (-c.score, c.claim_id, c.unit_id))


def scan_corpus(conn: sqlite3.Connection, claims: Sequence[Claim], nli: NLIModel,
                embedder: Embedder, *, limit: int = 20, threshold: float = 0.5,
                budget: int = 500, run_id: str = "",
                reranker: Reranker | None = None) -> list[Conflict]:
    """Scan every claim against the rest of the corpus.

    One unscannable claim never aborts the run — a scan over 300 papers that dies on claim
    40 is worth less than one that reports 299 papers' worth of candidates.

    Persists only when `run_id` is given, so an exploratory scan costs nothing permanent.
    """
    found: list[Conflict] = []
    for claim in claims:
        if len(found) >= budget:
            break
        try:
            found += scan_claim(conn, claim, nli, embedder, limit=limit,
                                threshold=threshold, reranker=reranker)
        except Exception:  # noqa: BLE001, S112 - one claim's failure is not the scan's
            continue

    ranked = rank(found)[:budget]
    if run_id:
        save_contradictions(
            conn,
            [{"claim_id": c.claim_id, "unit_id": c.unit_id, "score": c.score}
             for c in ranked],
            run_id=run_id,
        )
    return ranked
