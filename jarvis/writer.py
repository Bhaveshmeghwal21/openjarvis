"""The writer subagent (spec §9).

Takes a question and a bounded evidence set; returns prose plus explicit
(claim, unit_id, quote) triples. Triples rather than prose-with-markers because that is
exactly the shape `jarvis.verify` consumes, which removes a whole class of parser bugs
between drafting and verification.

The writer NEVER verifies its own output. That is `jarvis.answer`, running a separate pass
with separate inputs — the one multi-agent principle the literature is unambiguous about.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.evidence import render
from jarvis.models import Claim, Unit

_WRITE_PROMPT = (
    "Answer the question using ONLY the evidence below. Do not use prior knowledge.\n\n"
    "Return JSON:\n"
    "{{\"answer\": \"<prose>\", \"claims\": [{{\"text\": \"<one factual sentence>\", "
    "\"unit_id\": \"<id from the evidence>\", \"quote\": \"<exact span copied from that "
    "unit>\"}}]}}\n\n"
    "Rules:\n"
    "- Every factual sentence in `answer` must appear as a claim.\n"
    "- `quote` must be copied character-for-character from the cited unit. No paraphrase, "
    "no ellipsis, no reformatting. A quote that is not verbatim is discarded automatically "
    "and its claim with it.\n"
    "- `unit_id` must be one of the ids shown in brackets below. An id not listed there is "
    "a fabricated citation.\n"
    "- If the evidence does not answer the question, say so. An incomplete answer is "
    "correct; an unsupported one is not.\n\n"
    "Question: {question}\n\nEvidence:\n{evidence}"
)


@dataclass(frozen=True)
class Draft:
    """Prose plus the claims it rests on. Unverified — nothing here has been checked yet."""
    text: str = ""
    claims: tuple[Claim, ...] = ()


@runtime_checkable
class Writer(Protocol):
    def write(self, question: str, units: Sequence[Unit]) -> Draft: ...


def claims_from_json(
    data,
    units: Sequence[Unit],
    prefix: str = "c",
) -> list[Claim]:
    """Model JSON -> validated claims. Anything ungroundable is dropped here, not later.

    Two rules, both mechanical:
      * a `unit_id` outside the evidence set is a hallucinated citation;
      * a claim with no quote has nothing for stage 1 to match, so it would pass
        verification by having nothing to fail.
    """
    if not isinstance(data, dict):
        return []
    raw = data.get("claims")
    if not isinstance(raw, list):
        return []

    known = {u.unit_id for u in units}
    out: list[Claim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        unit_id = str(item.get("unit_id", "") or "").strip()
        quote = str(item.get("quote", "") or "").strip()
        if not text or not quote or unit_id not in known:
            continue
        out.append(
            Claim(
                claim_id=f"{prefix}-{len(out)}",
                text=text,
                unit_id=unit_id,
                quote=quote,
            )
        )
    return out


class FakeWriter:
    """Deterministic writer for tests, keyed by question."""

    def __init__(self, drafts: Mapping[str, Draft] | None = None) -> None:
        self._drafts = dict(drafts or {})

    def write(self, question: str, units: Sequence[Unit]) -> Draft:
        return self._drafts.get(question, Draft())


class LLMWriter:
    """Model-written draft, routed to the frontier tier. Returns an empty draft on
    failure."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None) -> None:
        self._router = router
        self._chat = chat_fn

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def write(self, question: str, units: Sequence[Unit]) -> Draft:
        if not units:
            # No evidence means no answer. A model asked to answer from an empty context
            # will answer from its training data, which is the failure this whole system
            # is built to remove.
            return Draft()

        prompt = _WRITE_PROMPT.format(question=question, evidence=render(units))
        try:
            raw = self._chat_fn()(
                self._router,
                "synthesis",
                prompt,
                json_mode=True,
            )
        except Exception:  # noqa: BLE001
            return Draft()
        if not isinstance(raw, dict):
            return Draft()
        return Draft(
            text=str(raw.get("answer", "") or "").strip(),
            claims=tuple(claims_from_json(raw, units)),
        )
