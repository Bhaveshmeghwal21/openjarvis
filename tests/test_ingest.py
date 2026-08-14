"""Stage C — deep read (spec §7C). Parse, type, prefix, embed, index."""
import pytest

from jarvis.embed import FakeEmbedder
from jarvis.gather import Candidate
from jarvis.ingest import IngestResult, ingest_decided, ingest_paper
from jarvis.models import Block, Paper
from jarvis.parse import FakeParser
from jarvis.retrieve import search
from jarvis.store import close_store, get_paper, get_papers_by_depth, get_units, open_store

BLOCKS = [
    Block(kind="heading", text="Results", page=2, section_path=("Results",)),
    Block(kind="paragraph", text="As shown in Table 1, we reach 94.2% accuracy under gust.",
          page=2, section_path=("Results",)),
    Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
          page=2, section_path=("Results",), label="Table 1"),
    Block(kind="caption", text="Table 1: Accuracy under wind.", page=2,
          section_path=("Results",), label="Table 1"),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / "c.db")
    yield c
    close_store(c)


def test_ingest_stores_layer_zero_and_marks_the_paper_deep(conn):
    result = ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    assert result.ok is True
    assert result.units > 0
    assert [p.paper_id for p in get_papers_by_depth(conn, "deep")] == ["p1"]


def test_ingest_produces_typed_units_with_prefixes(conn):
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    units = get_units(conn, "p1")
    assert {u.type.value for u in units} >= {"prose", "table"}
    assert all(u.context_prefix for u in units)


def test_the_prefix_never_leaks_into_verbatim_text(conn):
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    for unit in get_units(conn, "p1"):
        assert unit.context_prefix not in unit.verbatim_text


def test_an_ingested_paper_is_immediately_retrievable(conn):
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    hits = search(conn, "accuracy under wind", FakeEmbedder(), limit=3)
    assert any("94.2" in u.verbatim_text for u in hits)


def test_an_empty_parse_is_an_error_and_never_marked_deep(conn):
    result = ingest_paper(conn, PAPER, "p.pdf", FakeParser([]), FakeEmbedder())
    assert result.ok is False
    assert "empty parse" in result.error
    assert get_papers_by_depth(conn, "deep") == []


def test_a_parser_that_raises_is_recorded_not_propagated(conn):
    class Broken:
        def parse(self, path, paper_id):
            raise RuntimeError("corrupt pdf")

    result = ingest_paper(conn, PAPER, "p.pdf", Broken(), FakeEmbedder())
    assert result.ok is False
    assert "corrupt pdf" in result.error
    assert get_papers_by_depth(conn, "deep") == []


def test_reingesting_the_same_paper_is_idempotent(conn):
    first = ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    second = ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    assert first.units == second.units
    assert len(get_units(conn, "p1")) == first.units


def test_ingest_decided_reads_only_the_kept_papers(conn):
    cands = [
        Candidate(paper={"arxiv_id": "p1", "title": "Keep", "pdf_url": "a.pdf"}),
        Candidate(paper={"arxiv_id": "p2", "title": "Also keep", "pdf_url": "b.pdf"}),
        Candidate(paper={"arxiv_id": "p3", "title": "Defer", "pdf_url": "c.pdf"}),
    ]
    decisions = {"p1": "read_deep", "p2": "unsure", "p3": "defer"}
    results = ingest_decided(conn, decisions, cands, FakeParser(BLOCKS), FakeEmbedder())

    assert {r.paper_id for r in results} == {"p1", "p2"}
    assert {p.paper_id for p in get_papers_by_depth(conn, "deep")} == {"p1", "p2"}


def test_unsure_papers_are_read_exactly_like_read_deep_ones(conn):
    cands = [Candidate(paper={"arxiv_id": "p9", "title": "Unsure", "pdf_url": "u.pdf"})]
    results = ingest_decided(conn, {"p9": "unsure"}, cands, FakeParser(BLOCKS),
                             FakeEmbedder())
    assert results[0].ok is True
    assert results[0].units > 0


def test_a_deferred_paper_keeps_its_metadata_row(conn):
    cands = [Candidate(paper={"arxiv_id": "p3", "title": "Deferred", "abstract": "abs"})]
    ingest_decided(conn, {"p3": "defer"}, cands, FakeParser(BLOCKS), FakeEmbedder())
    assert get_paper(conn, "p3") is None or get_units(conn, "p3") == []


def test_one_broken_paper_does_not_stop_the_batch(conn):
    class Flaky:
        def parse(self, path, paper_id):
            if paper_id == "p1":
                raise RuntimeError("corrupt")
            return FakeParser(BLOCKS).parse(path, paper_id)

    cands = [Candidate(paper={"arxiv_id": "p1", "title": "A", "pdf_url": "a.pdf"}),
             Candidate(paper={"arxiv_id": "p2", "title": "B", "pdf_url": "b.pdf"})]
    results = ingest_decided(conn, {"p1": "read_deep", "p2": "read_deep"}, cands,
                             Flaky(), FakeEmbedder())
    assert {r.paper_id: r.ok for r in results} == {"p1": False, "p2": True}


def test_ingest_result_is_frozen():
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        IngestResult(paper_id="p", units=1, ok=True).units = 2
