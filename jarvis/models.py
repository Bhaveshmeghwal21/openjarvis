"""Frozen domain types for Layers 0/1/2 and verification (spec §5).

No behaviour beyond identity helpers. Everything is frozen and uses tuples so instances
are hashable and safe to pass between subagents.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UnitType(str, Enum):
    """Kinds of evidence unit (spec §5, Layer 1)."""
    PROSE = "prose"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"


class Verdict(str, Enum):
    """Outcome of verifying one claim against one cited unit (spec §8)."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NEUTRAL = "neutral"
    QUOTE_NOT_FOUND = "quote_not_found"


@dataclass(frozen=True)
class Block:
    """One element emitted by a parser. Layer 0 is a sequence of these."""
    kind: str                       # heading | paragraph | table | figure | equation | caption
    text: str
    page: int = 1
    section_path: tuple[str, ...] = ()
    label: str = ""                 # "Table 3", "Figure 1" when the parser knows it


@dataclass(frozen=True)
class ParsedPaper:
    """Layer 0 — immutable. The only source of truth for any claim."""
    paper_id: str
    blocks: tuple[Block, ...] = ()
    raw_text: str = ""


@dataclass(frozen=True)
class Paper:
    """Paper-level record and provenance metadata (spec §5)."""
    paper_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    s2_id: str = ""
    abstract: str = ""
    citation_count: int = 0
    retracted: bool = False
    version: str = ""
    source_path: str = ""


@dataclass(frozen=True)
class Unit:
    """Layer 1 — a typed evidence unit. The retrieval surface and the citation target."""
    unit_id: str
    paper_id: str
    type: UnitType
    page: int
    section_path: tuple[str, ...]
    verbatim_text: str
    ordinal: int = 0
    context_prefix: str = ""
    parent_id: str | None = None
    label: str = ""

    def key(self) -> str:
        """Deterministic id: stable across re-ingest of the same parse."""
        return f"{self.paper_id}:{self.type.value}:{self.page}:{self.ordinal}"


@dataclass(frozen=True)
class CardField:
    """One card field, anchored to a unit and a verbatim quote (spec §5, Layer 2)."""
    value: str
    unit_id: str
    quote: str
    binding_verified: bool = False


@dataclass(frozen=True)
class Card:
    """Layer 2 — coverage ledger and comparison index. Never the ground for a claim."""
    paper_id: str
    problem: CardField | None = None
    method: CardField | None = None
    datasets: tuple[CardField, ...] = ()
    metrics: tuple[CardField, ...] = ()
    claims: tuple[CardField, ...] = ()
    limitations: tuple[CardField, ...] = ()


@dataclass(frozen=True)
class Claim:
    """A statement asserted by the system, with the unit and quote it rests on."""
    claim_id: str
    text: str
    unit_id: str
    quote: str


@dataclass(frozen=True)
class Verification:
    """Result of the two-stage verification pass (spec §8)."""
    claim_id: str
    unit_id: str
    quote_found: bool
    verdict: Verdict
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
