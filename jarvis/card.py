"""Layer 2 — the paper card (spec §5).

Job: coverage bookkeeping and cross-paper comparison. **Never the ground for a claim.**

The card is deliberately demoted. LLMs extract isolated entities well but fail at
preserving roles, methods, and effect-size attribution — the relational binding, which is
exactly what makes a card worth having (arXiv 2602.10881). The counter-evidence for
keeping cards at all (otto-SR: 93.1% extraction accuracy versus 79.7% for dual human
reviewers) came from a tight schema with explicit verification, so that is what this is:
every field carries a unit_id and a verbatim quote, every quote is checked against Layer 0
by the same deterministic matcher verification uses, and unverified bindings are surfaced
rather than dropped.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from jarvis.models import Card, CardField, Claim, Paper, Unit
from jarvis.store import get_units, save_card
from jarvis.verify import quote_is_grounded

SINGLE_FIELDS = ("problem", "method")
LIST_FIELDS = ("datasets", "metrics", "claims", "limitations")

_EXTRACT_PROMPT = (
    "Extract a structured card from these evidence units of one paper.\n"
    "Return JSON with keys: problem, method (objects) and datasets, metrics, claims, "
    "limitations (arrays).\n"
    "Every object is {{\"value\": ..., \"unit_id\": ..., \"quote\": ...}} where `quote` is "
    "copied EXACTLY from the unit text — character for character, no paraphrase, no "
    "ellipsis. A quote that is not verbatim will be rejected automatically.\n"
    "Omit any field you cannot ground in a quote. An absent field is correct; an invented "
    "one is not.\n\n"
    "Paper: {title} ({year})\n\n{units}"
)


@runtime_checkable
class CardExtractor(Protocol):
    def extract(self, paper: Paper, units: Sequence[Unit]) -> Card: ...


class FakeCardExtractor:
    """Deterministic extractor for tests, keyed by paper_id."""

    def __init__(self, cards: Mapping[str, Card] | None = None) -> None:
        self._cards = dict(cards or {})

    def extract(self, paper: Paper, units: Sequence[Unit]) -> Card:
        return self._cards.get(paper.paper_id, Card(paper_id=paper.paper_id))


def _to_field(data, known_unit_ids: set[str]) -> CardField | None:
    """One JSON object -> a CardField, or None when it cannot be grounded.

    A field citing a unit_id that does not exist is a hallucinated citation. Dropping it
    here costs a row of bookkeeping; keeping it would put a fabricated pointer into the
    only structure the system uses for cross-paper comparison.
    """
    if not isinstance(data, dict):
        return None
    unit_id = str(data.get("unit_id", "") or "")
    quote = str(data.get("quote", "") or "")
    value = str(data.get("value", "") or "")
    if not unit_id or unit_id not in known_unit_ids or not quote or not value:
        return None
    return CardField(value=value, unit_id=unit_id, quote=quote)


class LLMCardExtractor:
    """Model-driven extraction, routed to the long-context reader tier."""

    def __init__(self, router, chat_fn: Callable[..., object] | None = None,
                 max_units: int = 40) -> None:
        self._router = router
        self._chat = chat_fn
        self._max_units = max_units

    def _chat_fn(self) -> Callable[..., object]:
        if self._chat is not None:
            return self._chat
        from jarvis.llm import chat
        return chat

    def extract(self, paper: Paper, units: Sequence[Unit]) -> Card:
        empty = Card(paper_id=paper.paper_id)
        selected = list(units)[:self._max_units]
        if not selected:
            return empty

        rendered = "\n\n".join(f"[{u.unit_id}]\n{u.verbatim_text}" for u in selected)
        prompt = _EXTRACT_PROMPT.format(title=paper.title, year=paper.year or "n.d.",
                                        units=rendered)
        try:
            raw = self._chat_fn()(self._router, "card_extraction", prompt, json_mode=True)
        except Exception:  # noqa: BLE001 - an unextractable card is not a failed ingest
            return empty
        if not isinstance(raw, dict):
            return empty

        known = {u.unit_id for u in units}
        kwargs = {name: _to_field(raw.get(name), known) for name in SINGLE_FIELDS}
        for name in LIST_FIELDS:
            items = raw.get(name) or []
            fields = (_to_field(item, known) for item in items) if isinstance(items, list) \
                else ()
            kwargs[name] = tuple(f for f in fields if f is not None)
        return Card(paper_id=paper.paper_id, **kwargs)


def _verify_field(conn: sqlite3.Connection, field: CardField | None) -> CardField | None:
    if field is None:
        return None
    claim = Claim(claim_id=f"card:{field.unit_id}", text=field.value,
                  unit_id=field.unit_id, quote=field.quote)
    return replace(field, binding_verified=quote_is_grounded(conn, claim))


def verify_card(conn: sqlite3.Connection, card: Card) -> Card:
    """Set `binding_verified` on every field by matching its quote against Layer 0.

    Deterministic, free, no model — the same stage-1 matcher `verify_claim` uses.
    """
    kwargs = {name: _verify_field(conn, getattr(card, name)) for name in SINGLE_FIELDS}
    kwargs.update({
        name: tuple(f for f in (_verify_field(conn, x) for x in getattr(card, name))
                    if f is not None)
        for name in LIST_FIELDS
    })
    return Card(paper_id=card.paper_id, **kwargs)


def unverified_fields(card: Card) -> list[tuple[str, CardField]]:
    """Every field whose quote did not match Layer 0, for surfacing as unverified."""
    out: list[tuple[str, CardField]] = []
    for name in SINGLE_FIELDS:
        field = getattr(card, name)
        if field is not None and not field.binding_verified:
            out.append((name, field))
    for name in LIST_FIELDS:
        out += [(name, f) for f in getattr(card, name) if not f.binding_verified]
    return out


def extract_and_verify(conn: sqlite3.Connection, paper: Paper,
                       extractor: CardExtractor) -> Card:
    """Extract, verify every binding, persist. The only way a card should ever be written."""
    card = verify_card(conn, extractor.extract(paper, get_units(conn, paper.paper_id)))
    save_card(conn, card)
    return card
