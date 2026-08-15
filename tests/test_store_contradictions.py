"""Persistence for contradiction candidates and their human review."""
import pytest

from jarvis.models import Paper
from jarvis.store import (
    close_store,
    get_contradiction_reviews,
    get_contradictions,
    open_store,
    save_contradictions,
    save_paper,
    set_contradiction_review,
)

ROWS = [
    {"claim_id": "c1", "unit_id": "p2:prose:1:0", "score": 0.91},
    {"claim_id": "c1", "unit_id": "p3:prose:2:1", "score": 0.72},
]


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / "c.db")
    yield c
    close_store(c)


def test_candidates_round_trip_with_their_scores(conn):
    assert save_contradictions(conn, ROWS, run_id="r1") == 2
    rows = get_contradictions(conn, "r1")
    assert {r["unit_id"] for r in rows} == {"p2:prose:1:0", "p3:prose:2:1"}
    assert rows[0]["score"] == pytest.approx(0.91)


def test_candidates_come_back_most_confident_first(conn):
    save_contradictions(conn, list(reversed(ROWS)), run_id="r1")
    scores = [r["score"] for r in get_contradictions(conn, "r1")]
    assert scores == sorted(scores, reverse=True)


def test_candidates_start_unreviewed(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    assert all(r["reviewed"] == "" for r in get_contradictions(conn, "r1"))
    assert get_contradiction_reviews(conn, "r1") == {}


def test_a_review_is_recorded_and_readable(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    set_contradiction_review(conn, "c1", "p2:prose:1:0", "valid", run_id="r1")
    set_contradiction_review(conn, "c1", "p3:prose:2:1", "invalid", run_id="r1")

    reviews = get_contradiction_reviews(conn, "r1")
    assert reviews == {("c1", "p2:prose:1:0"): True, ("c1", "p3:prose:2:1"): False}


def test_unreviewed_candidates_are_absent_from_reviews_not_false(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    set_contradiction_review(conn, "c1", "p2:prose:1:0", "valid", run_id="r1")
    reviews = get_contradiction_reviews(conn, "r1")
    assert len(reviews) == 1, "not-yet-looked-at is not the same fact as rejected"


def test_rescanning_the_same_pair_updates_its_score_and_keeps_the_review(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    set_contradiction_review(conn, "c1", "p2:prose:1:0", "valid", run_id="r1")
    save_contradictions(conn, [{"claim_id": "c1", "unit_id": "p2:prose:1:0", "score": 0.55}],
                        run_id="r1")

    row = next(r for r in get_contradictions(conn, "r1") if r["unit_id"] == "p2:prose:1:0")
    assert row["score"] == pytest.approx(0.55)
    assert row["reviewed"] == "valid", "human judgment survives a rescan"


def test_two_runs_are_independent(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    save_contradictions(conn, ROWS[:1], run_id="r2")
    assert len(get_contradictions(conn, "r1")) == 2
    assert len(get_contradictions(conn, "r2")) == 1


def test_an_invalid_review_verdict_is_rejected(conn):
    save_contradictions(conn, ROWS, run_id="r1")
    with pytest.raises(ValueError):
        set_contradiction_review(conn, "c1", "p2:prose:1:0", "maybe", run_id="r1")


def test_saving_nothing_is_zero_not_an_error(conn):
    assert save_contradictions(conn, [], run_id="r1") == 0
    assert get_contradictions(conn, "r1") == []


def test_candidates_do_not_require_a_papers_row(conn):
    """A candidate references units, not papers — no foreign key should block a scan."""
    save_paper(conn, Paper(paper_id="p1", title="T"))
    assert save_contradictions(conn, ROWS, run_id="r1") == 2
