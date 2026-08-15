"""Three papers, one disagreement, one review pass, one measured precision number."""
import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.contradict import apply_reviews, read_reviews, scan_corpus, write_review_sheet
from jarvis.embed import FakeEmbedder, index_units
from jarvis.evaluate import report
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper
from jarvis.parse import FakeParser
from jarvis.store import (
    close_store,
    get_contradiction_reviews,
    get_contradictions,
    get_units,
    open_store,
    save_paper,
    save_units,
)
from jarvis.units import build_units
from jarvis.verify import FakeNLI

ORIGINAL = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
          page=1, section_path=("Results",)),
]
REPRODUCTION = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Reproducing the controller, tracking accuracy never exceeded 61% under "
               "gust disturbance.", page=1, section_path=("Results",)),
]
UNRELATED = [
    Block(kind="heading", text="Method", page=1, section_path=("Method",)),
    Block(kind="paragraph", text="We fold proteins with a transformer.", page=1,
          section_path=("Method",)),
]


def _ingest(conn, paper_id, title, blocks):
    paper = Paper(paper_id=paper_id, title=title, year=2025)
    parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    _ingest(conn, "p1", "Gust-Robust Control", ORIGINAL)
    _ingest(conn, "p2", "A Reproduction Study", REPRODUCTION)
    _ingest(conn, "p3", "Protein Folding", UNRELATED)
    yield conn
    close_store(conn)


@pytest.fixture
def claim(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    return Claim("c1", "The controller reaches 94.2% tracking accuracy under gusts.",
                 unit.unit_id, "94.2% tracking accuracy")


def _nli_that_only_disputes_the_reproduction(corpus):
    reproduction = next(u for u in get_units(corpus, "p2") if "61%" in u.verbatim_text)
    return FakeNLI(
        mapping={(reproduction.verbatim_text,
                  "The controller reaches 94.2% tracking accuracy under gusts."):
                 {"entailment": 0.02, "neutral": 0.06, "contradiction": 0.92}},
        default={"entailment": 0.05, "neutral": 0.90, "contradiction": 0.05},
    )


def test_the_scan_finds_the_reproduction_and_ignores_the_unrelated_paper(corpus, claim):
    conflicts = scan_corpus(corpus, [claim], _nli_that_only_disputes_the_reproduction(corpus),
                            FakeEmbedder(), run_id="scan1")
    assert [c.paper_id for c in conflicts] == ["p2"]
    assert conflicts[0].score == pytest.approx(0.92)


def test_the_scan_never_reports_the_paper_disagreeing_with_itself(corpus, claim):
    always = FakeNLI(default={"entailment": 0.0, "neutral": 0.0, "contradiction": 1.0})
    conflicts = scan_corpus(corpus, [claim], always, FakeEmbedder())
    assert all(c.paper_id != "p1" for c in conflicts)


def test_the_review_loop_produces_a_precision_number(corpus, claim, tmp_path):
    conflicts = scan_corpus(corpus, [claim], _nli_that_only_disputes_the_reproduction(corpus),
                            FakeEmbedder(), run_id="scan1")

    sheet = tmp_path / "review.jsonl"
    write_review_sheet(sheet, conflicts)

    # A human reads the sheet and marks the one candidate as a genuine disagreement.
    lines = sheet.read_text(encoding="utf-8").replace('"verdict": null',
                                                      '"verdict": "valid"')
    sheet.write_text(lines, encoding="utf-8")

    reviews = read_reviews(sheet)
    apply_reviews(corpus, reviews, run_id="scan1")

    assert get_contradiction_reviews(corpus, "scan1") == reviews
    r = report([], contradiction_reviews=reviews)
    assert r.contradiction_precision == 1.0
    assert r.meets_contradiction_target is True


def test_a_scan_is_rerunnable_without_losing_earlier_judgments(corpus, claim):
    nli = _nli_that_only_disputes_the_reproduction(corpus)
    conflicts = scan_corpus(corpus, [claim], nli, FakeEmbedder(), run_id="scan1")
    apply_reviews(corpus, {(conflicts[0].claim_id, conflicts[0].unit_id): True},
                  run_id="scan1")

    scan_corpus(corpus, [claim], nli, FakeEmbedder(), run_id="scan1")
    assert get_contradiction_reviews(corpus, "scan1") != {}


def test_candidates_are_never_presented_as_facts(corpus, claim):
    from jarvis.contradict import render_conflicts
    conflicts = scan_corpus(corpus, [claim], _nli_that_only_disputes_the_reproduction(corpus),
                            FakeEmbedder())
    rendered = render_conflicts(conflicts).lower()
    assert "candidate" in rendered
    assert "contradicts" not in rendered, "spec §8: ranked candidates, never assertions"


def test_a_scan_over_a_corpus_with_no_disagreement_finds_nothing(corpus, claim):
    agreeable = FakeNLI(default={"entailment": 0.9, "neutral": 0.08, "contradiction": 0.02})
    assert scan_corpus(corpus, [claim], agreeable, FakeEmbedder(), run_id="quiet") == []
    assert get_contradictions(corpus, "quiet") == []
