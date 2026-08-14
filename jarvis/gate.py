"""Stage B — the gate (spec §7B). The recall ceiling of the whole system.

SESR-Eval benchmarked 9 LLMs on title-abstract screening in software engineering, the
closest published domain to ours: GPT-4o reached 0.66 recall, Claude 3.7 Sonnet 0.46, and
the verdict was that no model managed high recall with reasonable precision. Medical and
environmental domains report >95% for the identical task — this is domain-dependent and
ours is the bad one.

So the gate is never a single LLM judgment. It is a union of four cheap, independent
signals, calibrated per project against a hand-labeled seed, with three outcomes and no
`exclude`. `defer` demotes a paper to metadata depth; it never removes it.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.evaluate import GATE_RECALL_TARGET
from jarvis.gather import Candidate
from jarvis.scoring import cosine, paper_text
from jarvis.store import save_screen_decision, set_depth

_WORD = re.compile(r"[A-Za-z0-9]+")

# Words that carry no topical signal. Small on purpose: an over-eager stoplist silently
# strips domain terms and this stage cannot afford lost recall.
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to", "what", "when", "where", "which", "who", "why", "with"]
)

_VOTE_PROMPT = (
    "You are screening literature for a research question. Answer only about topical "
    "relevance, not quality.\n"
    "Return JSON: {{\"relevant\": true|false, \"score\": 0.0-1.0}}.\n"
    "When uncertain, answer relevant:true — a missed relevant paper costs far more here "
    "than an extra one.\n\n"
    "Question: {question}\n\nTitle: {title}\nAbstract: {abstract}"
)


@dataclass(frozen=True)
class Signals:
    """One paper's four gate scores, all on [0, 1]. Written verbatim to `screen_log`."""
    embedding: float = 0.0
    graph: float = 0.0
    keyword: float = 0.0
    llm_vote: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"embedding": self.embedding, "graph": self.graph,
                "keyword": self.keyword, "llm_vote": self.llm_vote}

    @property
    def best(self) -> float:
        return max(self.as_dict().values())


@runtime_checkable
class Voter(Protocol):
    def vote(self, question: str, paper: dict) -> float: ...


class FakeVoter:
    """Deterministic voter for tests, keyed by whatever `paper_id` resolves to."""

    def __init__(self, mapping: Mapping[str, float] | None = None,
                 default: float = 0.0) -> None:
        self._mapping = dict(mapping or {})
        self._default = default

    def vote(self, question: str, paper: dict) -> float:
        from jarvis.citation_graph import paper_id
        return self._mapping.get(paper_id(paper), self._default)


class LLMVoter:
    """One LLM vote — one signal of four, never the decision. Routed to the cheap tier."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None) -> None:
        self._router = router
        self._chat = chat_fn

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def vote(self, question: str, paper: dict) -> float:
        prompt = _VOTE_PROMPT.format(question=question, title=paper.get("title", ""),
                                     abstract=(paper.get("abstract", "") or "")[:4000])
        try:
            raw = self._chat_fn()(self._router, "screen_vote", prompt, json_mode=True)
        except Exception:  # noqa: BLE001 - a dead model is one signal down, not a decision
            return 0.0
        if not isinstance(raw, dict):
            return 0.0
        if not raw.get("relevant", False):
            return 0.0
        try:
            score = float(raw.get("score", 1.0))
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, score))


def _terms(text: str) -> set[str]:
    return {w for w in (t.lower() for t in _WORD.findall(text or "")) if w not in _STOPWORDS}


def keyword_overlap(question: str, paper: dict) -> float:
    """Fraction of the question's content words present in the title+abstract."""
    q_terms = _terms(question)
    if not q_terms:
        return 0.0
    return len(q_terms & _terms(paper_text(paper))) / len(q_terms)


def graph_proximity(candidate: Candidate, max_depth: int = 2) -> float:
    """How much citation-graph evidence supports this paper (spec §7B).

    A direct search hit scores **0.0**: it was found by keyword match, which is already
    the `keyword` signal's job, and it carries no graph evidence at all. Scoring it 1.0
    would make the union keep every search result unconditionally and switch adaptive
    depth off entirely — the gate would stop gating.

    A paper reached by walking outward from a high-scoring seed does carry evidence. One
    hop from a confirmed-relevant paper is the strongest form of it, decaying with
    distance and reaching 0.0 past `max_depth`.
    """
    if candidate.origin != "citation" or candidate.graph_depth < 1:
        return 0.0
    return max(0.0, 1.0 - (candidate.graph_depth - 1) / max(1, max_depth))


def score_signals(candidate: Candidate, question: str, question_vector,
                  embedder, voter: Voter | None = None,
                  max_depth: int = 2) -> Signals:
    """Compute all four signals for one candidate. No signal may abort the others."""
    try:
        vec = embedder.encode([paper_text(candidate.paper)])[0]
        embedding = max(0.0, cosine(vec, question_vector))
    except Exception:  # noqa: BLE001
        embedding = 0.0

    llm_vote = 0.0
    if voter is not None:
        try:
            llm_vote = max(0.0, min(1.0, float(voter.vote(question, candidate.paper))))
        except Exception:  # noqa: BLE001 - one signal failing must not lose the paper
            llm_vote = 0.0

    return Signals(
        embedding=embedding,
        graph=graph_proximity(candidate, max_depth),
        keyword=keyword_overlap(question, candidate.paper),
        llm_vote=llm_vote,
    )



DECISIONS = ("read_deep", "unsure", "defer")
KEPT = ("read_deep", "unsure")   # matches jarvis.evaluate.KEPT_DECISIONS

# A calibrated threshold below this is indistinguishable from "no threshold at all" once
# decide()'s inclusive >= comparison is applied: a signal with zero variance among labeled
# relevant papers (e.g. graph proximity before any citation expansion has run) would
# otherwise calibrate to 0.0 and admit every candidate scoring 0.0 on that signal too,
# which is the entire gather set for most signals. See Finding 2d/7a in the final review.
MIN_CALIBRATED_THRESHOLD = 0.05


@dataclass(frozen=True)
class Thresholds:
    """Per-signal keep thresholds. Defaults are a starting point; calibrate per project.

    `unsure_ratio` is the fraction of a threshold below which a signal still counts as a
    near miss. Spec §7B: `unsure` escalates to deep read, so the band is deliberately wide.
    """
    embedding: float = 0.35
    graph: float = 0.50
    keyword: float = 0.30
    llm_vote: float = 0.50
    unsure_ratio: float = 0.60

    def as_dict(self) -> dict[str, float]:
        return {"embedding": self.embedding, "graph": self.graph,
                "keyword": self.keyword, "llm_vote": self.llm_vote}


def decide(signals: Signals, thresholds: Thresholds | None = None) -> str:
    """Union rule. Any one signal clearing its bar keeps the paper.

    Intersection would lose whatever any single signal misses, and §7B's whole point is
    that every individual signal in this domain misses a lot.
    """
    t = thresholds or Thresholds()
    scores = signals.as_dict()
    bars = t.as_dict()

    if any(scores[name] >= bars[name] for name in bars):
        return "read_deep"
    if any(scores[name] >= bars[name] * t.unsure_ratio for name in bars):
        return "unsure"
    return "defer"


def screen(conn: sqlite3.Connection, candidates: Sequence[Candidate], question: str,
           embedder, voter: Voter | None = None, thresholds: Thresholds | None = None,
           run_id: str = "", max_depth: int = 2) -> dict[str, str]:
    """Score and decide every candidate, logging per-signal scores for every one.

    Papers are never removed: `read_deep` and `unsure` are promoted to `pending_deep`
    depth for Stage C to pick up, `defer` is left at `metadata` depth and stays
    recoverable when the question shifts.
    """
    t = thresholds or Thresholds()
    question_vector = embedder.encode([question])[0]

    out: dict[str, str] = {}
    for candidate in candidates:
        signals = score_signals(candidate, question, question_vector, embedder, voter,
                                max_depth=max_depth)
        decision = decide(signals, t)
        save_screen_decision(conn, candidate.pid, decision, signals.as_dict(), run_id=run_id)
        set_depth(conn, candidate.pid, "pending_deep" if decision in KEPT else "metadata")
        out[candidate.pid] = decision
    return out


def calibrate(signal_rows: Mapping[str, Signals], labels: Mapping[str, bool],
              target_recall: float = GATE_RECALL_TARGET,
              floor: float | None = None) -> Thresholds:
    """Fit per-signal thresholds to a hand-labeled seed set (spec §7B, §10).

    For each signal, sort the labeled-relevant papers' scores ascending and take the one
    at index floor((1 - target) * n). At least `target_recall` of relevant papers clear
    that bar on that signal alone; a union's recall is at least its best member's, so the
    union clears the target too.

    `floor` protects against a *degenerate* signal (one where every labeled-relevant
    paper scores exactly 0.0, e.g. graph proximity before any citation expansion has
    produced a hit) being tuned down to a threshold of 0.0 — `decide()`'s comparison is
    `>=`, so a threshold of exactly 0.0 admits *everything* scoring 0.0 on that signal,
    defeating the whole point of screening.

    Left unset (the default), the floor is `MIN_CALIBRATED_THRESHOLD` and is applied
    *only* to a signal whose raw fit is exactly 0.0 — i.e. only to signals that are
    genuinely degenerate. A signal with real, fine-grained separation clustered below
    that value (small but non-zero scores that still distinguish relevant from
    irrelevant papers) is left alone, so the automatic floor never itself narrows recall.

    Passed explicitly, `floor` instead behaves as an unconditional minimum on every
    signal's threshold — the caller has stated an intent stronger than "rescue dead
    signals only," and every signal's fitted value is clamped up to at least `floor`
    regardless of whether it was already above it.
    """
    relevant = [pid for pid, is_relevant in labels.items()
                if is_relevant and pid in signal_rows]
    if not relevant:
        return Thresholds()

    explicit_floor = floor is not None
    effective_floor = floor if explicit_floor else MIN_CALIBRATED_THRESHOLD

    default = Thresholds()
    fitted: dict[str, float] = {}
    for name in default.as_dict():
        scores = sorted(signal_rows[pid].as_dict()[name] for pid in relevant)
        index = int((1.0 - target_recall) * len(scores))
        index = max(0, min(index, len(scores) - 1))
        raw = scores[index]

        if explicit_floor:
            fitted[name] = max(effective_floor, raw)
        else:
            fitted[name] = effective_floor if raw == 0.0 else raw
    return Thresholds(unsure_ratio=default.unsure_ratio, **fitted)


def calibration_report(signal_rows: Mapping[str, Signals], labels: Mapping[str, bool],
                       thresholds: Thresholds) -> dict:
    """Re-run the decision over the seed set and report what the thresholds actually achieve.

    Never trust a fitted threshold without this: the fit is per-signal, the gate is a
    union, and the number that matters is the union's recall on real labels.
    """
    labeled = {pid: labels[pid] for pid in labels if pid in signal_rows}
    decisions = {pid: decide(signal_rows[pid], thresholds) for pid in labeled}
    kept = [pid for pid, d in decisions.items() if d in KEPT]
    relevant = [pid for pid, is_relevant in labeled.items() if is_relevant]
    relevant_kept = [pid for pid in kept if labeled[pid]]

    return {
        "recall": len(relevant_kept) / len(relevant) if relevant else 1.0,
        "precision": len(relevant_kept) / len(kept) if kept else 0.0,
        "kept": len(kept),
        "relevant": len(relevant),
        "relevant_kept": len(relevant_kept),
        "labeled": len(labeled),
        "thresholds": thresholds.as_dict(),
    }
