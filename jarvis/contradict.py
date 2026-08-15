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

from jarvis.embed import Embedder
from jarvis.models import Claim, Unit
from jarvis.retrieve import Reranker, search
from jarvis.store import get_unit


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
