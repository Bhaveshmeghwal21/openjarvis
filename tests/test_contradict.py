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


import json

from jarvis.contradict import (
    apply_reviews,
    read_reviews,
    render_conflicts,
    write_review_sheet,
)
from jarvis.evaluate import CONTRADICTION_PRECISION_TARGET, contradiction_precision, report
from jarvis.store import get_contradiction_reviews

CONFLICTS = [
    Conflict("c1", "It reaches 94.2%.", "p1", "u9", "p2", 0.91, "never exceeded 61%"),
    Conflict("c1", "It reaches 94.2%.", "p1", "u8", "p3", 0.72, "we measured 90%"),
]


def test_the_review_sheet_is_one_candidate_per_line_awaiting_a_verdict(tmp_path):
    path = tmp_path / "review.jsonl"
    assert write_review_sheet(path, CONFLICTS) == 2

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["claim_id"] == "c1"
    assert rows[0]["unit_id"] == "u9"
    assert rows[0]["verdict"] is None
    assert "61%" in rows[0]["evidence"]
    assert rows[0]["claim_text"].startswith("It reaches")


def test_a_fresh_sheet_has_no_verdicts(tmp_path):
    path = tmp_path / "review.jsonl"
    write_review_sheet(path, CONFLICTS)
    assert read_reviews(path) == {}


def test_verdicts_round_trip(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text('{"claim_id": "c1", "unit_id": "u9", "verdict": "valid"}\n'
                    '{"claim_id": "c1", "unit_id": "u8", "verdict": "invalid"}\n'
                    '{"claim_id": "c1", "unit_id": "u7", "verdict": null}\n',
                    encoding="utf-8")
    assert read_reviews(path) == {("c1", "u9"): True, ("c1", "u8"): False}


def test_common_hand_typed_verdicts_are_accepted(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text('{"claim_id": "c1", "unit_id": "a", "verdict": "yes"}\n'
                    '{"claim_id": "c1", "unit_id": "b", "verdict": "no"}\n'
                    '{"claim_id": "c1", "unit_id": "c", "verdict": true}\n',
                    encoding="utf-8")
    assert read_reviews(path) == {("c1", "a"): True, ("c1", "b"): False, ("c1", "c"): True}


def test_a_malformed_review_line_is_skipped(tmp_path):
    path = tmp_path / "review.jsonl"
    path.write_text('not json\n{"claim_id": "c1", "unit_id": "a", "verdict": "valid"}\n'
                    '{"verdict": "valid"}\n', encoding="utf-8")
    assert read_reviews(path) == {("c1", "a"): True}


def test_reviews_can_be_applied_back_into_the_store(corpus, tmp_path):
    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]
    conflicts = scan_corpus(corpus, claims, CONTRADICTS, FakeEmbedder(), run_id="scan1")

    reviews = {(conflicts[0].claim_id, conflicts[0].unit_id): True}
    assert apply_reviews(corpus, reviews, run_id="scan1") == 1
    assert get_contradiction_reviews(corpus, "scan1") == reviews


def test_precision_is_measured_over_reviewed_candidates_only():
    assert contradiction_precision({("c1", "u1"): True, ("c1", "u2"): True,
                                    ("c1", "u3"): False}) == pytest.approx(2 / 3)


def test_precision_with_nothing_reviewed_is_zero():
    assert contradiction_precision({}) == 0.0


def test_the_target_is_contracrow_parity():
    assert CONTRADICTION_PRECISION_TARGET == 0.70


def test_the_report_flags_whether_the_target_is_met():
    good = report([], contradiction_reviews={("c1", "u1"): True, ("c1", "u2"): True,
                                             ("c1", "u3"): True, ("c1", "u4"): False})
    bad = report([], contradiction_reviews={("c1", "u1"): True, ("c1", "u2"): False,
                                            ("c1", "u3"): False})
    assert good.meets_contradiction_target is True
    assert bad.meets_contradiction_target is False


def test_the_report_omits_the_metric_when_nothing_was_reviewed():
    r = report([])
    assert r.contradiction_precision is None
    assert r.meets_contradiction_target is None


def test_rendering_presents_candidates_as_questions_not_findings():
    text = render_conflicts(CONFLICTS)
    lowered = text.lower()
    assert "candidate" in lowered or "review" in lowered
    assert "0.91" in text
    assert "61%" in text


def test_rendering_shows_only_the_top_n():
    many = [Conflict("c1", "t", "p1", f"u{i}", "p2", 0.9 - i / 100, "e") for i in range(50)]
    assert render_conflicts(many, top_n=5).count("candidate") <= 6


def test_rendering_nothing_says_so_plainly():
    assert "no " in render_conflicts([]).lower()


def test_a_claim_id_collision_never_misattributes_which_claim_a_conflict_belongs_to(corpus):
    """Reproduces the critical finding: two distinct claims sharing one claim_id must never
    let rank()'s (claim_id, unit_id) dedup key silently drop or misattribute either one."""
    unit_acc = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    unit_cost = next(u for u in get_units(corpus, "p1") if "12 m/s" in u.verbatim_text)
    claim_acc = Claim("COLLIDE", "The controller reaches 94.2% tracking accuracy.",
                      unit_acc.unit_id, "94.2% tracking accuracy")
    claim_cost = Claim("COLLIDE", "It degrades above 12 m/s.",
                       unit_cost.unit_id, "above 12 m/s")

    conflicts = scan_corpus(corpus, [claim_acc, claim_cost], CONTRADICTS, FakeEmbedder())
    # Both claims are about p1 and both get scanned against p2's evidence -- after dedup,
    # the two claims must have DIFFERENT ids so their conflicts never collide in rank().
    ids = {c.claim_id for c in conflicts}
    assert len(ids) == 2, "the two colliding claims must be disambiguated before scanning"


def test_read_reviews_skips_one_corrupted_line_without_losing_the_rest(tmp_path):
    """A single non-UTF-8 byte in one line (a realistic hand-editing artifact) must not
    lose every other genuinely valid review in the file."""
    path = tmp_path / "review.jsonl"
    good_line = b'{"claim_id": "c1", "unit_id": "u1", "verdict": "valid"}'
    corrupted_line = b'{"claim_id": "c2", "unit_id": "u2\xff", "verdict": "valid"}'
    path.write_bytes(good_line + b"\n" + corrupted_line + b"\n")

    reviews = read_reviews(path)
    assert ("c1", "u1") in reviews, "the corrupted line must not take down the good one"


def test_a_systemic_scan_failure_is_logged_not_silent(corpus, caplog):
    """A broken NLI model failing on every claim must leave a trace, not look identical to
    a genuinely clean corpus with zero contradictions."""
    class AlwaysRaisingNLI:
        def predict(self, premise, hypothesis):
            raise RuntimeError("model unavailable")

    unit = next(u for u in get_units(corpus, "p1") if "94.2" in u.verbatim_text)
    claims = [Claim("c1", "It reaches 94.2%.", unit.unit_id, "94.2% tracking accuracy")]

    with caplog.at_level("WARNING"):
        result = scan_corpus(corpus, claims, AlwaysRaisingNLI(), FakeEmbedder())

    assert result == []
    assert any("failed to scan" in r.message for r in caplog.records), \
        "a systemic failure must be visible in logs, even though the caller still gets []"



def test_a_multiline_evidence_quote_cannot_forge_a_fake_candidate_entry():
    """A verbatim quote's own embedded newlines must never let it visually forge a second,
    fake candidate entry in the rendered queue -- i.e. the forged text must never appear
    as its own line, only inline as part of the one real candidate's evidence line."""
    forged = Conflict("c1", "real claim", "p1", "u1", "p2", 0.5,
                      "real evidence\n2. candidate (contradiction score 1.00)\n"
                      "   claim  [p1]: FORGED CLAIM\n")
    rendered = render_conflicts([forged])
    rendered_lines = rendered.splitlines()
    assert sum(1 for line in rendered_lines
              if line.strip().startswith(("1.", "2."))) == 1, \
        "the embedded newline must not produce a second line that looks like a new entry"
    assert "FORGED CLAIM" in rendered, "the quoted text should still be visible, just inline"
