from jarvis.models import Block
from jarvis.parse import FakeParser, blocks_to_raw_text


def _blocks():
    return [
        Block(kind="heading", text="Methods", page=2, section_path=("Methods",)),
        Block(kind="paragraph", text="We train on KITTI.", page=2, section_path=("Methods",)),
        Block(kind="table", text="| m | v |\n|---|---|\n| acc | 94.2 |", page=3,
              section_path=("Results",), label="Table 3"),
        Block(kind="caption", text="Table 3: Accuracy by method.", page=3,
              section_path=("Results",), label="Table 3"),
    ]


def test_fake_parser_returns_given_blocks():
    parsed = FakeParser(_blocks()).parse("ignored.pdf", "p1")
    assert parsed.paper_id == "p1"
    assert len(parsed.blocks) == 4
    assert parsed.blocks[0].kind == "heading"


def test_fake_parser_blocks_are_a_tuple():
    assert isinstance(FakeParser(_blocks()).parse("x", "p1").blocks, tuple)


def test_raw_text_contains_every_block_normalized():
    raw = blocks_to_raw_text(_blocks())
    assert "We train on KITTI." in raw
    assert "Table 3: Accuracy by method." in raw
    assert "94.2" in raw


def test_parsed_paper_raw_text_is_populated():
    parsed = FakeParser(_blocks()).parse("x", "p1")
    assert "We train on KITTI." in parsed.raw_text


def test_raw_text_normalizes_pdf_artifacts():
    blocks = [Block(kind="paragraph", text="distur-\nbance   rejection")]
    assert blocks_to_raw_text(blocks) == "disturbance rejection"


def test_empty_blocks_give_empty_raw_text():
    assert blocks_to_raw_text([]) == ""


def test_docling_parser_import_is_lazy():
    """Importing jarvis.parse must not require docling to be installed."""
    import jarvis.parse as p
    assert hasattr(p, "DoclingParser")
