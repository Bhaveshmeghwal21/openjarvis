"""Cross-paper contradiction candidates (spec §8)."""
from dataclasses import FrozenInstanceError

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



from jarvis.contradict import Conflict, rank, scan_claim, scan_corpus
from jarvis.store import get_contradictions
from jarvis.verify import FakeNLI

CONTRADICTS = FakeNLI(default={"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90})
AGREES_NLI = FakeNLI(default={"entailment": 0.90, "neutral": 0.08, "contradiction": 0.02})


def test_a_scan_finds_the_disagreeing_paper(corpus):
    conflicts = scan_claim(corpus, _claim(corpus), CONTRADICTS, FakeEmbedder())
    assert conflicts
    assert all(c.paper_id == "p2" for c in conflicts)
    assert conflicts[0].score == pytest.approx(0.90)


def test_a_conflict_carries_enough_context_to_review_it(corpus):
    conflict = scan_claim(corpus, _claim(corpus), CONTRADICTS, FakeEmbedder())[0]
    assert conflict.claim_text.startswith("The controller reaches")
    assert conflict.claim_paper_id == "p1"
    assert conflict.paper_id == "p2"
    assert conflict.evidence


def test_agreement_produces_no_candidates(corpus):
    assert scan_claim(corpus, _claim(corpus), AGREES_NLI, FakeEmbedder()) == []


def test_the_threshold_gates_what_is_reported(corpus):
    weak = FakeNLI(default={"entailment": 0.1, "neutral": 0.5, "contradiction": 0.40})
    assert scan_claim(corpus, _claim(corpus), weak, FakeEmbedder(), threshold=0.5) == []
    assert scan_claim(corpus, _claim(corpus), weak, FakeEmbedder(), threshold=0.3) != []


def test_conflicts_come_back_most_confident_first():
    conflicts = [
        Conflict("c1", "t", "p1", "u1", "p2", 0.6, "e"),
        Conflict("c1", "t", "p1", "u2", "p3", 0.9, "e"),
        Conflict("c1", "t", "p1", "u3", "p4", 0.7, "e"),
    ]
    assert [c.score for c in rank(conflicts)] == [0.9, 0.7, 0.6]


def test_ranking_deduplicates_a_repeated_pair_keeping_the_higher_score():
    conflicts = [Conflict("c1", "t", "p1", "u1", "p2", 0.6, "e"),
                 Conflict("c1", "t", "p1", "u1", "p2", 0.9, "e")]
    ranked = rank(conflicts)
    assert len(ranked) == 1
    assert ranked[0].score == pytest.approx(0.9)


def test_a_corpus_scan_covers_every_claim(corpus):
    unit1 = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    unit2 = next(u for u in get_units(corpus, "p1") if "12 m/s" in u.verbatim_text)
    claims = [
        Claim("c1", "It reaches 94.2% accuracy.", unit1.unit_id, "94.2% tracking accuracy"),
        Claim("c2", "It degrades above 12 m/s.", unit2.unit_id, "above 12 m/s"),
    ]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder())
    assert {c.claim_id for c in conflicts} == {"c1", "c2"}


def test_a_corpus_scan_persists_its_candidates(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder(), run_id="scan1")

    stored = get_contradictions(corpus, "scan1")
    assert stored
    assert stored[0]["claim_id"] == "c1"
    assert stored[0]["reviewed"] == ""


def test_a_scan_without_a_run_id_does_not_persist(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder())
    assert conflicts
    assert get_contradictions(corpus, "") == []


def test_the_budget_caps_the_scan(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim(f"c{i}", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")
              for i in range(20)]
    assert len(scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder(), budget=3)) <= 3


def test_one_bad_claim_does_not_abort_the_scan(corpus):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("bad", "x", "ghost-unit", "q"),
              Claim("good", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder())
    assert {c.claim_id for c in conflicts} == {"good"}


def test_conflict_is_frozen():
    with pytest.raises(FrozenInstanceError):
        Conflict("c1", "t", "p1", "u1", "p2", 0.6, "e").score = 1.0
