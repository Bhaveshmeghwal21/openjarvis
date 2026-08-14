"""Seed-set labeling — the ground truth the gate is calibrated and measured against.

Spec §7B calibrates the gate against a hand-labeled seed of ~100 papers; spec §10 measures
gate recall against the same labels. Both need a human to actually read titles and
abstracts, so the format is deliberately dumb: JSONL, one paper per line, `label` starts
null and a human edits it to true/false in any editor. Diffable, resumable, no UI.

Note (spec §10): human citation lists are not ground truth. Do not substitute a paper's own
bibliography for these labels.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from jarvis.gather import Candidate

DEFAULT_SEED_SIZE = 100

_TRUE = {"true", "yes", "y", "1", "relevant"}
_FALSE = {"false", "no", "n", "0", "irrelevant"}


def sample_seed(candidates: Sequence[Candidate], size: int = DEFAULT_SEED_SIZE,
                seed: int = 0) -> list[Candidate]:
    """A deterministic, order-independent sample of the gathered set.

    Hashing the id rather than shuffling means the same corpus yields the same seed set on
    any machine and in any gather order — a labeling session survives a re-gather.
    """
    def rank(candidate: Candidate) -> str:
        return hashlib.sha256(f"{seed}:{candidate.pid}".encode()).hexdigest()

    return sorted(candidates, key=rank)[:size]


def write_label_sheet(path: str | Path, candidates: Sequence[Candidate]) -> int:
    """Write an unlabelled JSONL sheet for a human to fill in. Returns rows written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({
            "paper_id": c.pid,
            "title": c.paper.get("title", ""),
            "year": c.paper.get("year"),
            "abstract": (c.paper.get("abstract", "") or "")[:2000],
            "label": None,
        }, ensure_ascii=False)
        for c in candidates
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


def _rows(path: str | Path) -> list[dict]:
    target = Path(path)
    if not target.is_file():
        return []
    out: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue          # a hand-edited sheet will have typos; skip, never crash
        if isinstance(row, dict) and row.get("paper_id"):
            out.append(row)
    return out


def read_labels(path: str | Path) -> dict[str, bool]:
    """Read the completed labels. Unlabelled and unparseable rows are simply absent."""
    out: dict[str, bool] = {}
    for row in _rows(path):
        label = _as_bool(row.get("label"))
        if label is not None:
            out[str(row["paper_id"])] = label
    return out


def label_progress(path: str | Path) -> dict:
    """How much of the sheet is done — the only thing a labeling session needs to know."""
    rows = _rows(path)
    labels = read_labels(path)
    return {
        "total": len(rows),
        "labeled": len(labels),
        "relevant": sum(1 for v in labels.values() if v),
        "remaining": len(rows) - len(labels),
    }
