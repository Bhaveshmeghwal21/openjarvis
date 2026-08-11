import pytest

from jarvis.index import fts_escape, index_units_fts, keyword_search
from jarvis.models import Paper, Unit, UnitType
from jarvis.store import close_store, open_store, save_paper, save_units


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"))
    yield c
    close_store(c)


def _unit(uid, text, ordinal=0):
    return Unit(unit_id=uid, paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=("A",), verbatim_text=text, ordinal=ordinal)


def _seed(conn, pairs):
    units = [_unit(uid, text, i) for i, (uid, text) in enumerate(pairs)]
    save_units(conn, units)
    index_units_fts(conn, units)
    return units


def test_fts_escape_quotes_terms():
    assert fts_escape("wind rejection") == '"wind" "rejection"'


def test_fts_escape_neutralises_operators():
    escaped = fts_escape('NEAR(a b) OR "x" -y*')
    assert "NEAR" not in escaped or escaped.count('"') % 2 == 0
    assert escaped.startswith('"')


def test_fts_escape_of_empty_query_is_empty():
    assert fts_escape("   ") == ""


def test_index_returns_count(conn):
    assert index_units_fts(conn, _seed(conn, [("u1", "alpha")])) == 1


def test_keyword_search_finds_the_matching_unit(conn):
    _seed(conn, [("u1", "quadrotor wind rejection"), ("u2", "cake recipes")])
    hits = keyword_search(conn, "wind rejection")
    assert hits[0][0] == "u1"


def test_keyword_search_scores_are_positive_higher_is_better(conn):
    _seed(conn, [("u1", "wind wind wind"), ("u2", "wind once")])
    hits = keyword_search(conn, "wind")
    assert all(score > 0 for _, score in hits)
    assert hits[0][1] >= hits[-1][1]


def test_keyword_search_respects_limit(conn):
    _seed(conn, [(f"u{i}", "wind") for i in range(5)])
    assert len(keyword_search(conn, "wind", limit=2)) == 2


def test_keyword_search_with_no_match_returns_empty(conn):
    _seed(conn, [("u1", "alpha")])
    assert keyword_search(conn, "zebra") == []


def test_keyword_search_does_not_crash_on_operator_input(conn):
    _seed(conn, [("u1", "alpha")])
    assert keyword_search(conn, 'AND OR NOT "(' ) == []


def test_reindexing_does_not_duplicate_rows(conn):
    units = _seed(conn, [("u1", "wind")])
    index_units_fts(conn, units)
    assert len(keyword_search(conn, "wind")) == 1
