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

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from jarvis.embed import Embedder
from jarvis.models import Claim, Unit
from jarvis.retrieve import Reranker, search
from jarvis.store import get_unit, save_contradictions, set_contradiction_review
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


_TRUE = {"true", "valid", "yes", "y", "1"}
_FALSE = {"false", "invalid", "no", "n", "0"}


def write_review_sheet(path: str | Path, conflicts: Sequence[Conflict]) -> int:
    """Write candidates as JSONL for a human to adjudicate. Returns rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "claim_id": c.claim_id, "claim_text": c.claim_text,
            "claim_paper_id": c.claim_paper_id, "unit_id": c.unit_id,
            "paper_id": c.paper_id, "score": round(c.score, 4), "evidence": c.evidence,
            "verdict": None,
        }, ensure_ascii=False)
        for c in conflicts
    ]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _as_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def read_reviews(path: str | Path) -> dict[tuple[str, str], bool]:
    """Read adjudicated verdicts. Unreviewed and unparseable rows are absent, not False."""
    target = Path(path)
    if not target.is_file():
        return {}
    out: dict[tuple[str, str], bool] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not row.get("claim_id") or not row.get("unit_id"):
            continue
        verdict = _as_bool(row.get("verdict"))
        if verdict is not None:
            out[(str(row["claim_id"]), str(row["unit_id"]))] = verdict
    return out


def apply_reviews(conn: sqlite3.Connection, reviews: Mapping[tuple[str, str], bool],
                  run_id: str = "") -> int:
    """Write adjudicated verdicts back into the store. Returns rows updated."""
    for (claim_id, unit_id), valid in reviews.items():
        set_contradiction_review(conn, claim_id, unit_id,
                                 "valid" if valid else "invalid", run_id=run_id)
    return len(reviews)


def render_conflicts(conflicts: Sequence[Conflict], top_n: int = 20) -> str:
    """Human-readable queue. Every line is a question, never a finding (spec §8)."""
    ranked = rank(conflicts)[:top_n]
    if not ranked:
        return "No contradiction candidates found in this corpus."

    lines = [(f"{len(ranked)} contradiction candidate(s) for review — these are prompts "
             f"to look, not findings:"), ""]
    for index, conflict in enumerate(ranked, start=1):
        lines += [
            f"{index}. candidate (contradiction score {conflict.score:.2f})",
            f"   claim  [{conflict.claim_paper_id}]: {conflict.claim_text}",
            f"   versus [{conflict.paper_id}] {conflict.unit_id}: {conflict.evidence}",
            "",
        ]
    return "\n".join(lines)
