"""The proof the plan exists to produce: one paper, ingested, retrieved, verified."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.evaluate import report
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.parse import FakeParser
from jarvis.retrieve import search
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI, verify_claim

BLOCKS = [
    Block(kind="heading", text="Results", page=3, section_path=("Results",)),
    Block(kind="paragraph", text="As shown in Table 3, our controller reaches 94.2% "
                                 "tracking accuracy under gust distur-\nbance.",
          page=3, section_path=("Results",)),
    Block(kind="table", text="| method | accuracy |\n|---|---|\n| ours | 94.2 |",
          page=3, section_path=("Results",), label="Table 3"),
    Block(kind="caption", text="Table 3: Tracking accuracy under wind.", page=3,
          section_path=("Results",), label="Table 3"),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Quadrotor Control", year=2025)


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    parsed = FakeParser(BLOCKS).parse("paper.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")

    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    yield conn
    close_store(conn)


def test_ingest_produces_prose_and_table_units(corpus):
    kinds = {u.type.value for u in get_units(corpus, "p1")}
    assert "prose" in kinds
    assert "table" in kinds


def test_the_table_unit_carries_its_caption_and_referring_text(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    assert "94.2" in table.verbatim_text
    assert "Tracking accuracy under wind" in table.verbatim_text
    assert "As shown in Table 3" in table.verbatim_text


def test_retrieval_finds_the_evidence(corpus):
    hits = search(corpus, "tracking accuracy under wind", FakeEmbedder(), limit=3)
    assert any("94.2" in u.verbatim_text for u in hits)


def test_a_grounded_claim_verifies_as_supported(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    claim = Claim(claim_id="c1",
                  text="The controller reaches 94.2% tracking accuracy.",
                  unit_id=table.unit_id, quote="| ours | 94.2 |")
    nli = FakeNLI(default={"entailment": 0.93, "neutral": 0.05, "contradiction": 0.02})
    assert verify_claim(corpus, claim, nli).verdict == Verdict.SUPPORTED


def test_a_fabricated_number_is_caught_without_consulting_the_model(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    claim = Claim(claim_id="c2", text="The controller reaches 99.9% accuracy.",
                  unit_id=table.unit_id, quote="| ours | 99.9 |")
    nli = FakeNLI(default={"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0})
    result = verify_claim(corpus, claim, nli)
    assert result.verdict == Verdict.QUOTE_NOT_FOUND
    assert result.quote_found is False


def test_quote_survives_hyphenation_across_a_line_break(corpus):
    prose = next(u for u in get_units(corpus, "p1")
                 if u.type.value == "prose" and "gust" in u.verbatim_text)
    claim = Claim(claim_id="c3", text="Gusts disturb the controller.",
                  unit_id=prose.unit_id, quote="under gust disturbance")
    nli = FakeNLI(default={"entailment": 0.91, "neutral": 0.06, "contradiction": 0.03})
    assert verify_claim(corpus, claim, nli).quote_found is True


def test_eval_report_flags_the_fabricated_claim(corpus):
    table = next(u for u in get_units(corpus, "p1") if u.type.value == "table")
    nli = FakeNLI(default={"entailment": 0.93, "neutral": 0.05, "contradiction": 0.02})
    good = verify_claim(corpus, Claim("c1", "t", table.unit_id, "| ours | 94.2 |"), nli)
    bad = verify_claim(corpus, Claim("c2", "t", table.unit_id, "| ours | 99.9 |"), nli)

    r = report([good, bad])
    assert r.quote_fidelity == 0.5
    assert r.meets_quote_target is False
