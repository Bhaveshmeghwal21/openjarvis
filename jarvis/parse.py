"""Layer 0 production — PDF to an immutable block sequence (spec §5).

Docling is the default (MIT, structured lossless output, CPU-capable). MinerU is the later
escalation path for formula-dense papers. Both sit behind the `Parser` protocol so tests
run with `FakeParser` and never touch a model.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jarvis.models import Block, ParsedPaper
from jarvis.text import normalize


@runtime_checkable
class Parser(Protocol):
    def parse(self, path: str, paper_id: str) -> ParsedPaper: ...


def blocks_to_raw_text(blocks: Sequence[Block]) -> str:
    """Concatenate blocks into the normalized Layer 0 text that quotes are matched against."""
    return normalize("\n".join(b.text for b in blocks))


class FakeParser:
    """Deterministic parser for tests. Returns the blocks it was constructed with."""

    def __init__(self, blocks: Sequence[Block]) -> None:
        self._blocks = tuple(blocks)

    def parse(self, path: str, paper_id: str) -> ParsedPaper:
        return ParsedPaper(paper_id=paper_id, blocks=self._blocks,
                           raw_text=blocks_to_raw_text(self._blocks))


_DOCLING_KINDS = {
    "section_header": "heading", "title": "heading", "paragraph": "paragraph",
    "text": "paragraph", "table": "table", "picture": "figure", "caption": "caption",
    "formula": "equation",
}


class DoclingParser:
    """Real adapter. `docling` is imported lazily so this module loads without it."""

    def parse(self, path: str, paper_id: str) -> ParsedPaper:
        from docling.document_converter import DocumentConverter

        doc = DocumentConverter().convert(path).document
        blocks: list[Block] = []
        section: list[str] = []
        for item, _level in doc.iterate_items():
            kind = _DOCLING_KINDS.get(getattr(item, "label", ""), "paragraph")
            text = getattr(item, "text", "") or ""
            if kind == "table" and hasattr(item, "export_to_markdown"):
                text = item.export_to_markdown()
            if not text.strip():
                continue
            if kind == "heading":
                section = [text.strip()]
            page = getattr(getattr(item, "prov", [None])[0], "page_no", 1) or 1
            blocks.append(Block(kind=kind, text=text, page=page,
                                section_path=tuple(section),
                                label=getattr(item, "label", "") or ""))
        return ParsedPaper(paper_id=paper_id, blocks=tuple(blocks),
                           raw_text=blocks_to_raw_text(blocks))
