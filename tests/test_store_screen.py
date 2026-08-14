"""Screening log, card, and run persistence — the bookkeeping the gate is audited from."""
import pytest

from jarvis.models import Card, CardField, Paper, Unit, UnitType
from jarvis.store import (
    all_units,
    close_store,
    get_card,
    get_papers_by_depth,
    get_run,
    get_screen_decisions,
    get_screen_signals,
    open_store,
    save_card,
    save_paper,
    save_run,
    save_screen_decision,
    save_units,
    set_depth,
)


@pytest.fixture
def conn(tmp_path):
    c = open_store(tmp_path / "corpus.db")
    yield c
    close_store(c)


def _unit(paper_id: str, ordinal: int) -> Unit:
    u = Unit(unit_id="", paper_id=paper_id, type=UnitType.PROSE, page=1,
             section_path=(), verbatim_text=f"text {ordinal}", ordinal=ordinal)
    return Unit(unit_id=u.key(), paper_id=u.paper_id, type=u.type, page=u.page,
                section_path=u.section_path, verbatim_text=u.verbatim_text, ordinal=u.ordinal)


def test_a_decision_round_trips_with_its_per_signal_scores(conn):
    save_screen_decision(conn, "p1", "read_deep",
                         {"embedding": 0.71, "graph": 0.0, "keyword": 0.4, "llm_vote": 1.0},
                         run_id="r1")
    assert get_screen_decisions(conn, "r1") == {"p1": "read_deep"}
    assert get_screen_signals(conn, "r1")["p1"]["embedding"] == pytest.approx(0.71)


def test_rescreening_the_same_paper_in_the_same_run_overwrites(conn):
    save_screen_decision(conn, "p1", "defer", {"embedding": 0.1}, run_id="r1")
    save_screen_decision(conn, "p1", "unsure", {"embedding": 0.4}, run_id="r1")
    assert get_screen_decisions(conn, "r1") == {"p1": "unsure"}


def test_the_same_paper_can_be_screened_differently_in_two_runs(conn):
    save_screen_decision(conn, "p1", "defer", {}, run_id="r1")
    save_screen_decision(conn, "p1", "read_deep", {}, run_id="r2")
    assert get_screen_decisions(conn, "r1") == {"p1": "defer"}
    assert get_screen_decisions(conn, "r2") == {"p1": "read_deep"}


def test_decisions_without_a_run_id_are_readable(conn):
    save_screen_decision(conn, "p1", "read_deep", {})
    assert get_screen_decisions(conn) == {"p1": "read_deep"}


def test_a_card_round_trips_with_every_field_and_flag(conn):
    save_paper(conn, Paper(paper_id="p1", title="T"))
    card = Card(
        paper_id="p1",
        problem=CardField(value="gust rejection", unit_id="u1", quote="gusts"),
        metrics=(CardField(value="94.2", unit_id="u2", quote="94.2", binding_verified=True),),
        datasets=(CardField(value="KITTI", unit_id="u3", quote="KITTI"),),
    )
    save_card(conn, card)

    got = get_card(conn, "p1")
    assert got.problem.value == "gust rejection"
    assert got.metrics[0].binding_verified is True
    assert got.datasets[0].unit_id == "u3"
    assert got.method is None
    assert got.claims == ()


def test_saving_a_card_twice_replaces_it(conn):
    save_paper(conn, Paper(paper_id="p1", title="T"))
    save_card(conn, Card(paper_id="p1", problem=CardField("a", "u1", "a")))
    save_card(conn, Card(paper_id="p1", problem=CardField("b", "u1", "b")))
    assert get_card(conn, "p1").problem.value == "b"


def test_a_missing_card_is_none(conn):
    assert get_card(conn, "nope") is None


def test_depth_can_be_promoted_without_touching_layer_zero(conn):
    save_paper(conn, Paper(paper_id="p1", title="T"), raw_text="ORIGINAL", depth="metadata")
    set_depth(conn, "p1", "deep")
    assert [p.paper_id for p in get_papers_by_depth(conn, "deep")] == ["p1"]
    assert get_papers_by_depth(conn, "metadata") == []


def test_all_units_can_exclude_one_paper(conn):
    save_paper(conn, Paper(paper_id="p1", title="A"))
    save_paper(conn, Paper(paper_id="p2", title="B"))
    save_units(conn, [_unit("p1", 0), _unit("p2", 0), _unit("p2", 1)])

    assert len(all_units(conn)) == 3
    assert {u.paper_id for u in all_units(conn, exclude_paper_id="p2")} == {"p1"}


def test_a_run_records_its_question_and_cost(conn):
    save_run(conn, "r1", question="how do quadrotors reject gusts?", cost_usd=1.25)
    assert get_run(conn, "r1")["question"] == "how do quadrotors reject gusts?"
    assert get_run(conn, "r1")["cost_usd"] == pytest.approx(1.25)




def test_set_depth_on_a_nonexistent_paper_raises_instead_of_silently_doing_nothing(conn):
    """Finding 1b: a bare UPDATE with no rowcount check would silently affect zero rows,
    letting screen() log a decision and signals for a paper the corpus has no record of
    at all -- the paper is then unreachable by get_paper/get_papers_by_depth despite the
    gate having 'kept' it. Loud failure here is strictly better than that silent gap."""
    with pytest.raises(ValueError, match="no paper row"):
        set_depth(conn, "never-saved", "deep")
