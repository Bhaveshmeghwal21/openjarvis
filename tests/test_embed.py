import pytest

from jarvis.embed import FakeEmbedder, index_units, vector_search
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


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=8)
    assert e.encode(["hello"]) == e.encode(["hello"])


def test_fake_embedder_respects_dim_and_name():
    e = FakeEmbedder(dim=16)
    assert e.dim == 16
    assert len(e.encode(["x"])[0]) == 16
    assert e.name


def test_fake_embedder_gives_different_vectors_for_different_text():
    e = FakeEmbedder()
    assert e.encode(["alpha"]) != e.encode(["beta"])


def test_index_units_stores_one_row_per_unit(conn):
    e = FakeEmbedder()
    units = [_unit("u1", "alpha"), _unit("u2", "beta", 1)]
    save_units(conn, units)
    assert index_units(conn, units, e) == 2
    count = conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"]
    assert count == 2


def test_index_units_is_idempotent(conn):
    e = FakeEmbedder()
    units = [_unit("u1", "alpha")]
    save_units(conn, units)
    index_units(conn, units, e)
    index_units(conn, units, e)
    assert conn.execute("SELECT COUNT(*) c FROM embeddings").fetchone()["c"] == 1


def test_vector_search_ranks_the_matching_unit_first(conn):
    e = FakeEmbedder()
    units = [_unit("u_alpha", "alpha alpha alpha"), _unit("u_beta", "beta beta beta", 1)]
    save_units(conn, units)
    index_units(conn, units, e)
    hits = vector_search(conn, e.encode(["alpha alpha alpha"])[0], e.name, limit=2)
    assert hits[0][0] == "u_alpha"


def test_vector_search_respects_limit(conn):
    e = FakeEmbedder()
    units = [_unit(f"u{i}", f"text {i}", i) for i in range(5)]
    save_units(conn, units)
    index_units(conn, units, e)
    assert len(vector_search(conn, e.encode(["text 1"])[0], e.name, limit=3)) == 3


def test_vector_search_ignores_other_models(conn):
    e = FakeEmbedder()
    units = [_unit("u1", "alpha")]
    save_units(conn, units)
    index_units(conn, units, e)
    assert vector_search(conn, e.encode(["alpha"])[0], "some-other-model") == []


def test_vector_search_on_empty_index_returns_empty(conn):
    assert vector_search(conn, [0.1] * 64, "fake-64") == []
