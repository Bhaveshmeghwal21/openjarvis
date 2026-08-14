"""Stage B — the gate. Spec §7B: the recall ceiling of the whole system."""
import pytest

from jarvis.embed import FakeEmbedder
from jarvis.gate import (
    FakeVoter,
    LLMVoter,
    Signals,
    Voter,
    graph_proximity,
    keyword_overlap,
    score_signals,
)
from jarvis.gather import Candidate

RELEVANT = {"arxiv_id": "p1", "title": "Gust rejection for quadrotors",
            "abstract": "We reject wind gusts on a quadrotor."}
IRRELEVANT = {"arxiv_id": "p2", "title": "Protein folding with transformers",
              "abstract": "We fold proteins."}
QUESTION = "how do quadrotors reject wind gusts?"


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_keyword_overlap_is_higher_for_the_relevant_paper():
    assert keyword_overlap(QUESTION, RELEVANT) > keyword_overlap(QUESTION, IRRELEVANT)


def test_keyword_overlap_is_bounded_and_ignores_stopwords():
    assert 0.0 <= keyword_overlap(QUESTION, RELEVANT) <= 1.0
    assert keyword_overlap("the and of", RELEVANT) == 0.0


def test_keyword_overlap_of_an_empty_paper_is_zero():
    assert keyword_overlap(QUESTION, {"title": "", "abstract": ""}) == 0.0


def test_a_direct_search_hit_carries_no_citation_graph_evidence():
    assert graph_proximity(Candidate(paper=RELEVANT, origin="search")) == 0.0


def test_graph_proximity_is_strongest_one_hop_from_a_relevant_seed():
    one = graph_proximity(Candidate(paper=RELEVANT, origin="citation", graph_depth=1))
    two = graph_proximity(Candidate(paper=RELEVANT, origin="citation", graph_depth=2))
    assert one == 1.0
    assert 0.0 < two < one


def test_graph_proximity_never_goes_negative():
    assert graph_proximity(Candidate(paper=RELEVANT, origin="citation",
                                     graph_depth=99)) == 0.0


def test_fake_voter_satisfies_the_protocol():
    assert isinstance(FakeVoter({}), Voter)


def test_signals_expose_themselves_as_a_dict_for_the_log():
    s = Signals(embedding=0.8, graph=1.0, keyword=0.5, llm_vote=1.0)
    assert s.as_dict() == {"embedding": 0.8, "graph": 1.0, "keyword": 0.5, "llm_vote": 1.0}
    assert s.best == 1.0


def test_score_signals_scores_the_relevant_paper_above_the_irrelevant_one():
    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    voter = FakeVoter({"p1": 1.0, "p2": 0.0})

    good = score_signals(Candidate(paper=RELEVANT), QUESTION, qvec, embedder, voter)
    bad = score_signals(Candidate(paper=IRRELEVANT), QUESTION, qvec, embedder, voter)

    assert good.embedding > bad.embedding
    assert good.keyword > bad.keyword
    assert good.llm_vote == 1.0
    assert bad.llm_vote == 0.0


def test_score_signals_without_a_voter_records_zero_not_a_crash():
    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    assert score_signals(Candidate(paper=RELEVANT), QUESTION, qvec, embedder).llm_vote == 0.0


def test_a_voter_that_raises_scores_zero_and_does_not_abort_screening():
    class Boom:
        def vote(self, question, paper):
            raise RuntimeError("rate limited")

    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    s = score_signals(Candidate(paper=RELEVANT), QUESTION, qvec, embedder, Boom())
    assert s.llm_vote == 0.0
    assert s.embedding > 0.0, "the other three signals still stand"


def test_llm_voter_parses_a_score_and_clamps_it():
    for reply, expected in (({"relevant": True, "score": 0.9}, 0.9),
                            ({"relevant": True}, 1.0),
                            ({"relevant": False}, 0.0),
                            ({"relevant": True, "score": 5}, 1.0),
                            ({"relevant": True, "score": -3}, 0.0)):
        voter = LLMVoter(_Router(), chat_fn=lambda *a, reply=reply, **k: reply)
        assert voter.vote(QUESTION, RELEVANT) == pytest.approx(expected)


def test_llm_voter_scores_zero_on_failure_never_raising():
    def boom(*args, **kwargs):
        raise RuntimeError("down")

    assert LLMVoter(_Router(), chat_fn=boom).vote(QUESTION, RELEVANT) == 0.0
    assert LLMVoter(_Router(), chat_fn=lambda *a, **k: "junk").vote(QUESTION, RELEVANT) == 0.0


def test_llm_voter_routes_to_the_screen_vote_task():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {"relevant": True}

    LLMVoter(_Router(), chat_fn=spy).vote(QUESTION, RELEVANT)
    assert seen["task"] == "screen_vote"
