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



from jarvis.evaluate import KEPT_DECISIONS, gate_recall
from jarvis.gate import DECISIONS, KEPT, Thresholds, decide, screen
from jarvis.gather import save_candidates
from jarvis.store import (
    close_store,
    get_papers_by_depth,
    get_screen_decisions,
    get_screen_signals,
    open_store,
)


def test_the_three_outcomes_are_exactly_the_spec_ones():
    assert DECISIONS == ("read_deep", "unsure", "defer")
    assert "exclude" not in DECISIONS


def test_kept_matches_what_the_eval_harness_already_counts_as_kept():
    assert set(KEPT) == KEPT_DECISIONS


def test_any_single_signal_over_threshold_keeps_the_paper():
    t = Thresholds()
    assert decide(Signals(embedding=0.9), t) == "read_deep"
    assert decide(Signals(graph=1.0), t) == "read_deep"
    assert decide(Signals(keyword=0.9), t) == "read_deep"
    assert decide(Signals(llm_vote=1.0), t) == "read_deep"


def test_the_gate_is_a_union_not_an_intersection():
    t = Thresholds()
    only_one = Signals(embedding=0.0, graph=0.0, keyword=0.0, llm_vote=1.0)
    assert decide(only_one, t) == "read_deep", "one signal is enough; intersection loses papers"


def test_a_near_miss_is_unsure_not_deferred():
    t = Thresholds(embedding=0.5, unsure_ratio=0.6)
    assert decide(Signals(embedding=0.35), t) == "unsure"


def test_nothing_anywhere_near_threshold_is_deferred():
    assert decide(Signals(), Thresholds()) == "defer"


def test_a_signal_exactly_at_threshold_is_kept():
    assert decide(Signals(embedding=0.35), Thresholds(embedding=0.35)) == "read_deep"


def test_screen_writes_every_decision_with_its_signals(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=RELEVANT), Candidate(paper=IRRELEVANT, graph_depth=9)]
        save_candidates(conn, cands)
        decisions = screen(conn, cands, QUESTION, FakeEmbedder(),
                           voter=FakeVoter({"p1": 1.0, "p2": 0.0}), run_id="r1")

        assert decisions["p1"] == "read_deep"
        assert get_screen_decisions(conn, "r1") == decisions
        assert set(get_screen_signals(conn, "r1")["p1"]) == {
            "embedding", "graph", "keyword", "llm_vote"}
    finally:
        close_store(conn)


def test_a_deferred_paper_stays_in_the_corpus_at_metadata_depth(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=IRRELEVANT)]
        save_candidates(conn, cands)

        class ZeroEmbedder:
            """FakeEmbedder's hash-bucket noise floor (~0.2-0.3 cosine on any two short
            texts) always clears the unsure band, so no signal-isolation test can use it
            to assert `defer`. This returns orthogonal unit vectors keyed by text so the
            question and an unrelated paper never overlap."""

            def encode(self, texts):
                return [[1.0, 0.0] if text == QUESTION else [0.0, 1.0] for text in texts]

        decisions = screen(conn, cands, QUESTION, ZeroEmbedder(), voter=FakeVoter({}),
                           run_id="r1")

        assert decisions["p2"] == "defer"
        assert [p.paper_id for p in get_papers_by_depth(conn, "metadata")] == ["p2"]
        assert get_papers_by_depth(conn, "deep") == []
    finally:
        close_store(conn)


def test_screening_is_rerunnable_without_refetching(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=RELEVANT)]
        save_candidates(conn, cands)
        unreachable = Thresholds(embedding=1.1, graph=1.1, keyword=1.1, llm_vote=1.1)
        screen(conn, cands, QUESTION, FakeEmbedder(), thresholds=unreachable,
               run_id="strict")
        screen(conn, cands, QUESTION, FakeEmbedder(), thresholds=Thresholds(), run_id="loose")

        assert get_screen_decisions(conn, "strict")["p1"] != "read_deep"
        assert get_screen_decisions(conn, "loose")["p1"] == "read_deep"
    finally:
        close_store(conn)


def test_gate_recall_reads_the_decisions_this_gate_produces(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = [Candidate(paper=RELEVANT), Candidate(paper=IRRELEVANT, graph_depth=9)]
        save_candidates(conn, cands)
        decisions = screen(conn, cands, QUESTION, FakeEmbedder(),
                           voter=FakeVoter({"p1": 1.0}), run_id="r1")
        assert gate_recall(decisions, {"p1": True, "p2": False}) == 1.0
    finally:
        close_store(conn)
