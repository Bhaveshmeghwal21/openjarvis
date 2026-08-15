"""Cross-paper contradiction candidates (spec §8)."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.contradict import opposing_units
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units

AGREES = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
          page=1, section_path=("Results",)),
    Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
    Block(kind="paragraph", text="Tracking accuracy degrades above 12 m/s gusts.",
          page=2, section_path=("Limitations",)),
]
DISAGREES = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Reproducing this controller, tracking accuracy never exceeded 61% under "
               "gust disturbance.", page=1, section_path=("Results",)),
]


def _ingest(conn, paper_id, title, blocks):
    paper = Paper(paper_id=paper_id, title=title, year=2025)
    parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    return units


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    _ingest(conn, "p1", "Gust-Robust Control", AGREES)
    _ingest(conn, "p2", "A Reproduction Study", DISAGREES)
    yield conn
    close_store(conn)


def _claim(conn):
    unit = next(u for u in get_units(conn, "p1") if "94.2" in u.verbatim_text)
    return Claim(claim_id="c1", text="The controller reaches 94.2% tracking accuracy.",
                 unit_id=unit.unit_id, quote="94.2% tracking accuracy")


def test_opposing_units_finds_the_other_papers_evidence(corpus):
    units = opposing_units(corpus, _claim(corpus), FakeEmbedder())
    assert any("61%" in u.verbatim_text for u in units)


def test_no_unit_from_the_claims_own_paper_is_returned(corpus):
    units = opposing_units(corpus, _claim(corpus), FakeEmbedder())
    assert all(u.paper_id != "p1" for u in units), \
        "a paper disagreeing with itself is a claim-extraction bug, not a finding"


def test_the_claims_own_unit_is_never_returned(corpus):
    claim = _claim(corpus)
    assert claim.unit_id not in {u.unit_id for u in opposing_units(corpus, claim,
                                                                  FakeEmbedder())}


def test_a_claim_whose_unit_is_unknown_yields_nothing(corpus):
    claim = Claim(claim_id="c9", text="anything", unit_id="ghost", quote="q")
    assert opposing_units(corpus, claim, FakeEmbedder()) == []


def test_a_single_paper_corpus_yields_no_candidates(tmp_path):
    conn = open_store(tmp_path / "solo.db")
    try:
        _ingest(conn, "p1", "Alone", AGREES)
        assert opposing_units(conn, _claim(conn), FakeEmbedder()) == []
    finally:
        close_store(conn)


def test_the_limit_is_respected(corpus):
    assert len(opposing_units(corpus, _claim(corpus), FakeEmbedder(), limit=1)) <= 1


def test_results_are_deduplicated(corpus):
    units = opposing_units(corpus, _claim(corpus), FakeEmbedder(), limit=20)
    assert len({u.unit_id for u in units}) == len(units)
