import pytest

from jarvis.models import Paper, Unit, UnitType
from jarvis.store import (
    close_store,
    get_paper,
    get_raw_text,
    get_unit,
    get_units,
    open_store,
    save_paper,
    save_units,
)


@pytest.fixture
def conn():
    c = open_store(":memory:")
    yield c
    close_store(c)


def _unit(uid="u1", paper="p1", ordinal=0, parent=None):
    return Unit(unit_id=uid, paper_id=paper, type=UnitType.PROSE, page=2,
                section_path=("Methods", "Setup"), verbatim_text="text body",
                ordinal=ordinal, parent_id=parent)


def test_save_and_get_paper_roundtrips_all_fields(conn):
    p = Paper(paper_id="p1", title="T", authors=("A", "B"), year=2025, venue="V",
              doi="10.1/x", arxiv_id="2501.1", abstract="abs", citation_count=9,
              retracted=True, version="v2")
    save_paper(conn, p)
    got = get_paper(conn, "p1")
    assert got == p


def test_get_paper_returns_none_when_absent(conn):
    assert get_paper(conn, "nope") is None


def test_raw_text_is_stored_and_readable(conn):
    save_paper(conn, Paper("p1", "T"), raw_text="LAYER ZERO")
    assert get_raw_text(conn, "p1") == "LAYER ZERO"


def test_saving_paper_again_does_not_erase_raw_text(conn):
    """Layer 0 is immutable: re-saving metadata must not blank the parsed text."""
    save_paper(conn, Paper("p1", "T"), raw_text="LAYER ZERO")
    save_paper(conn, Paper("p1", "T updated"))
    assert get_raw_text(conn, "p1") == "LAYER ZERO"
    assert get_paper(conn, "p1").title == "T updated"


def test_save_and_get_units_roundtrips(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    got = get_units(conn, "p1")
    assert got == [_unit()]
    assert got[0].section_path == ("Methods", "Setup")


def test_units_are_returned_in_ordinal_order(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit("u2", ordinal=2), _unit("u0", ordinal=0), _unit("u1", ordinal=1)])
    assert [u.unit_id for u in get_units(conn, "p1")] == ["u0", "u1", "u2"]


def test_save_units_is_idempotent(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    save_units(conn, [_unit()])
    assert len(get_units(conn, "p1")) == 1


def test_get_unit_by_id(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    assert get_unit(conn, "u1").verbatim_text == "text body"
    assert get_unit(conn, "missing") is None


def test_deleting_paper_cascades_to_units(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit()])
    conn.execute("DELETE FROM papers WHERE paper_id='p1'")
    assert get_units(conn, "p1") == []


def test_parent_id_survives_roundtrip(conn):
    save_paper(conn, Paper("p1", "T"))
    save_units(conn, [_unit("parent"), _unit("child", ordinal=1, parent="parent")])
    child = get_unit(conn, "child")
    assert child.parent_id == "parent"
