"""Agentic, iterative retrieval (spec §7 Stage D)."""
import dataclasses

import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Paper
from jarvis.parse import FakeParser
from jarvis.retriever import FakeRefiner, LLMRefiner, Refiner, retrieve_iteratively
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.units import build_units

BLOCKS = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy in gusts.",
          page=1, section_path=("Results",)),
    Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
    Block(kind="paragraph", text="Performance degrades sharply above 12 m/s wind speed.",
          page=2, section_path=("Limitations",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    parsed = FakeParser(BLOCKS).parse("p.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    yield conn
    close_store(conn)


def test_fake_refiner_satisfies_the_protocol():
    assert isinstance(FakeRefiner([]), Refiner)


def test_without_a_refiner_retrieval_is_single_shot(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder())
    assert result.rounds == 1
    assert result.queries == ("tracking accuracy",)
    assert len(result.units) > 0


def test_a_refiner_adds_rounds_and_records_every_query(corpus):
    refiner = FakeRefiner(["wind speed limitations"])
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=refiner, rounds=3)
    assert result.queries == ("tracking accuracy", "wind speed limitations")
    assert result.rounds == 2


def test_refinement_surfaces_evidence_the_first_query_missed(corpus):
    single = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(), limit=1)
    iterated = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                    refiner=FakeRefiner(["wind speed degrades"]),
                                    rounds=2, limit=1)
    assert len(iterated.units) > len(single.units)
    assert any("12 m/s" in u.verbatim_text for u in iterated.units)


def test_units_are_deduped_across_rounds(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner(["wind speed limitations"]), rounds=2)
    ids = [u.unit_id for u in result.units]
    assert len(ids) == len(set(ids))
    assert result.rounds == 2, "both rounds must actually run for this test to mean anything"


def test_a_refiner_returning_none_stops_the_loop_early(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner([None]), rounds=5)
    assert result.rounds == 1


def test_a_refiner_repeating_a_query_stops_the_loop(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=FakeRefiner(["tracking accuracy"]), rounds=5)
    assert result.rounds == 1, "a repeated query means the refiner has nothing new"


def test_the_round_budget_is_respected(corpus):
    refiner = FakeRefiner([f"query {i}" for i in range(20)])
    result = retrieve_iteratively(corpus, "start", FakeEmbedder(), refiner=refiner, rounds=3)
    assert result.rounds == 3
    assert len(result.queries) == 3


def test_a_refiner_that_raises_ends_the_loop_without_losing_earlier_hits(corpus):
    class Boom:
        def refine(self, question, queries, units):
            raise RuntimeError("model down")

    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder(),
                                  refiner=Boom(), rounds=3)
    assert result.rounds == 1
    assert len(result.units) > 0


def test_llm_refiner_returns_the_models_query():
    refiner = LLMRefiner(_Router(), chat_fn=lambda *a, **k: {"query": "wind speed limits"})
    assert refiner.refine("q", ("q",), []) == "wind speed limits"


def test_llm_refiner_returns_none_when_the_model_says_it_is_done():
    for reply in ({"query": ""}, {"done": True}, {}, "junk", None):
        def _mock_chat(*a, r=reply, **k):
            return r
        assert LLMRefiner(_Router(), chat_fn=_mock_chat).refine("q", ("q",), []) \
            is None


def test_llm_refiner_returns_none_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    assert LLMRefiner(_Router(), chat_fn=boom).refine("q", ("q",), []) is None


def test_llm_refiner_routes_to_retrieval_refine():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {"query": "x"}

    LLMRefiner(_Router(), chat_fn=spy).refine("q", ("q",), [])
    assert seen["task"] == "retrieval_refine"


def test_retrieval_is_frozen(corpus):
    result = retrieve_iteratively(corpus, "tracking accuracy", FakeEmbedder())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.question = "other"


def test_llm_refiner_logs_a_warning_on_failure(caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    with caplog.at_level("WARNING"):
        LLMRefiner(_Router(), chat_fn=boom).refine("q", ("q",), [])
    assert any("failed" in r.message.lower() for r in caplog.records)
