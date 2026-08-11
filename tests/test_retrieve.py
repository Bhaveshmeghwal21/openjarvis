import pytest

from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Paper, ParsedPaper, Unit, UnitType
from jarvis.retrieve import FakeReranker, rrf, search
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.units import build_units


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"))
    yield c
    close_store(c)


def _seed(conn, texts):
    units = [
        Unit(unit_id=f"u{i}", paper_id="p1", type=UnitType.PROSE, page=1,
             section_path=("A",), verbatim_text=t, ordinal=i)
        for i, t in enumerate(texts)
    ]
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    return units


# --- RRF ---------------------------------------------------------------------

def test_rrf_rewards_agreement_across_rankings():
    fused = dict(rrf([["a", "b", "c"], ["a", "c", "b"]]))
    assert fused["a"] > fused["b"]
    assert fused["a"] > fused["c"]


def test_rrf_includes_items_present_in_only_one_ranking():
    ids = [uid for uid, _ in rrf([["a"], ["b"]])]
    assert set(ids) == {"a", "b"}


def test_rrf_uses_k_60_by_default():
    fused = dict(rrf([["a"]]))
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_of_nothing_is_empty():
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_rrf_output_is_sorted_descending():
    scores = [s for _, s in rrf([["a", "b", "c"], ["a", "b", "c"]])]
    assert scores == sorted(scores, reverse=True)


# --- search ------------------------------------------------------------------

def test_search_returns_units_not_ids(conn):
    _seed(conn, ["quadrotor wind rejection", "cake"])
    results = search(conn, "wind rejection", FakeEmbedder(), limit=1)
    assert isinstance(results[0], Unit)


def test_search_finds_the_relevant_unit(conn):
    _seed(conn, ["quadrotor wind rejection under gusts", "sourdough baking"])
    results = search(conn, "wind rejection", FakeEmbedder(), limit=1)
    assert "wind rejection" in results[0].verbatim_text


def test_search_respects_limit(conn):
    _seed(conn, [f"wind {i}" for i in range(5)])
    assert len(search(conn, "wind", FakeEmbedder(), limit=2)) == 2


def test_search_on_empty_corpus_returns_empty(conn):
    assert search(conn, "anything", FakeEmbedder()) == []


def test_reranker_reorders_results(conn):
    _seed(conn, ["wind alpha", "wind beta"])
    reranked = search(conn, "wind", FakeEmbedder(), limit=2,
                      reranker=FakeReranker(order=["u1", "u0"]))
    assert [u.unit_id for u in reranked] == ["u1", "u0"]


def test_search_expands_children_to_parents(conn):
    parsed = ParsedPaper(paper_id="p1", blocks=(
        Block(kind="paragraph", text="wind " * 2000, page=1, section_path=("Methods",)),
    ))
    units = build_units(parsed)
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    results = search(conn, "wind", FakeEmbedder(), limit=3, expand_parents=True)
    assert all(u.parent_id is None for u in results), "children should be swapped for parents"


def test_parent_expansion_dedupes_siblings(conn):
    parsed = ParsedPaper(paper_id="p1", blocks=(
        Block(kind="paragraph", text="wind " * 2000, page=1, section_path=("Methods",)),
    ))
    units = build_units(parsed)
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    results = search(conn, "wind", FakeEmbedder(), limit=5, expand_parents=True)
    assert len({u.unit_id for u in results}) == len(results)


def test_expand_parents_off_returns_children(conn):
    parsed = ParsedPaper(paper_id="p1", blocks=(
        Block(kind="paragraph", text="wind " * 2000, page=1, section_path=("Methods",)),
    ))
    units = build_units(parsed)
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    results = search(conn, "wind", FakeEmbedder(), limit=3, expand_parents=False)
    assert any(u.parent_id is not None for u in results)
