"""Report outlines from Layer 2 cards (spec §5, §7 Stage D).

AutoSurvey's decomposition starts from paper-level structure rather than from retrieval,
and this is where the card finally earns its keep: spec §5 gives it exactly one job —
coverage bookkeeping and cross-paper comparison — and an outline is that comparison.

The card still never grounds a claim. It decides what the sections ARE; each section then
retrieves its own Layer 1 evidence from scratch.

`TemplateOutliner` emits only sections the corpus can support. A section with no evidence
behind it is an invitation for a model to invent one.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from jarvis.models import Card

MAX_SECTIONS = 8

_OUTLINE_PROMPT = (
    "Plan a structured literature report on this topic from the paper summaries below.\n"
    "Return JSON: {{\"sections\": [{{\"title\": ..., \"question\": ..., "
    "\"paper_ids\": [...]}}]}}\n"
    "At most {max_sections} sections. `question` is the single question that section "
    "answers, phrased so it can be used as a search query against the corpus. "
    "`paper_ids` lists the papers that motivated the section — use only ids shown below.\n"
    "Propose a section only if the summaries show there is material for it.\n\n"
    "Topic: {topic}\n\n{digest}"
)

# (attribute, singular label, section title, sub-question template)
_TEMPLATE_SECTIONS = (
    ("problem", "problem", "Problem framing", "what problem does {topic} address?"),
    ("method", "method", "Methods", "what methods are used for {topic}?"),
    ("datasets", "dataset", "Datasets and benchmarks",
     "what datasets and benchmarks are used for {topic}?"),
    ("metrics", "metric", "Reported results", "what results are reported for {topic}?"),
    ("limitations", "limitation", "Limitations and open problems",
     "what are the limitations and open problems in {topic}?"),
)


@dataclass(frozen=True)
class Section:
    """One report section and the sub-question it answers."""
    title: str
    question: str
    paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outline:
    topic: str
    sections: tuple[Section, ...] = ()


@runtime_checkable
class Outliner(Protocol):
    def outline(self, topic: str, cards: Sequence[Card]) -> Outline: ...


def _values(card: Card, attribute: str) -> list[str]:
    value = getattr(card, attribute, None)
    if value is None:
        return []
    if isinstance(value, tuple):
        return [f.value for f in value if f.value]
    return [value.value] if value.value else []


def cards_digest(cards: Sequence[Card], max_papers: int = 100) -> str:
    """Compact cross-paper summary — the outliner's whole input."""
    lines: list[str] = []
    for card in list(cards)[:max_papers]:
        parts = []
        for attribute, _label, _title, _q in _TEMPLATE_SECTIONS:
            values = _values(card, attribute)
            if values:
                parts.append(f"{attribute}: {'; '.join(values[:4])}")
        lines.append(f"[{card.paper_id}] " + (" | ".join(parts) or "(no card fields)"))
    return "\n".join(lines)


class TemplateOutliner:
    """Deterministic, free, no model. The fallback and the test default."""

    def outline(self, topic: str, cards: Sequence[Card]) -> Outline:
        clean = " ".join((topic or "").split())
        sections: list[Section] = [Section(
            title="Overview",
            question=f"what is the state of the art in {clean}?",
        )]
        for attribute, _label, title, question in _TEMPLATE_SECTIONS:
            papers = tuple(c.paper_id for c in cards if _values(c, attribute))
            if papers:
                sections.append(Section(title=title, question=question.format(topic=clean),
                                        paper_ids=papers))
        return Outline(topic=clean, sections=tuple(sections))


class LLMOutliner:
    """Model-planned outline, routed to the frontier tier. Falls back to the template."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None,
                 max_sections: int = MAX_SECTIONS) -> None:
        self._router = router
        self._chat = chat_fn
        self._max_sections = max_sections
        self._fallback = TemplateOutliner()

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def outline(self, topic: str, cards: Sequence[Card]) -> Outline:
        fallback = self._fallback.outline(topic, cards)
        prompt = _OUTLINE_PROMPT.format(max_sections=self._max_sections,
                                        topic=fallback.topic, digest=cards_digest(cards))
        try:
            return self._parsed_outline(fallback, prompt, cards)
        except Exception:  # noqa: BLE001 - chat_fn is an untrusted Protocol boundary
            return fallback

    def _parsed_outline(self, fallback: Outline, prompt: str,
                        cards: Sequence[Card]) -> Outline:
        """Everything that can fail on a hostile reply, wrapped by `outline`'s one catch.

        Not just the `chat_fn()` call itself: a double whose returned object's own `.get`
        raises, or whose title/question value has a raising `__str__`, must fall back to
        the template exactly like an outright `chat_fn` exception does — the docstring's
        "falls back to the template on failure" promise otherwise only held for the first
        of two ways a reply can misbehave.
        """
        raw = self._chat_fn()(self._router, "outline", prompt, json_mode=True)
        if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
            return fallback

        known = {c.paper_id for c in cards}
        sections: list[Section] = []
        for item in raw["sections"]:
            if not isinstance(item, dict) or len(sections) >= self._max_sections:
                continue
            title = " ".join(str(item.get("title", "") or "").split())
            question = " ".join(str(item.get("question", "") or "").split())
            if not title or not question:
                continue
            ids = item.get("paper_ids") or []
            papers = tuple(str(p) for p in ids if str(p) in known) \
                if isinstance(ids, list) else ()
            sections.append(Section(title=title, question=question, paper_ids=papers))

        return Outline(topic=fallback.topic, sections=tuple(sections)) if sections \
            else fallback
