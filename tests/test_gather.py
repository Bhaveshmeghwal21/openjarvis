"""Stage A — recall-optimized gathering (spec §7A)."""
import pytest

from jarvis.gather import LLMPlanner, Planner, SearchPlan, TemplatePlanner


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_template_planner_satisfies_the_protocol():
    assert isinstance(TemplatePlanner(), Planner)


def test_template_planner_keeps_the_question_verbatim_as_a_query():
    plan = TemplatePlanner().plan("how do quadrotors reject gusts?")
    assert plan.question == "how do quadrotors reject gusts?"
    assert "how do quadrotors reject gusts?" in plan.queries


def test_template_planner_fans_out_to_several_distinct_queries():
    plan = TemplatePlanner().plan("gust rejection")
    assert len(plan.queries) >= 4
    assert len(set(plan.queries)) == len(plan.queries)


def test_template_planner_collapses_whitespace():
    assert TemplatePlanner().plan("  a   b\n").question == "a b"


def test_template_planner_is_deterministic():
    assert TemplatePlanner().plan("x") == TemplatePlanner().plan("x")


def test_llm_planner_uses_the_models_sub_questions_and_queries():
    def fake_chat(router, task, prompt, **kwargs):
        assert task == "query_expansion"
        assert kwargs.get("json_mode") is True
        return {"sub_questions": ["what disturbs a quadrotor?"],
                "queries": ["quadrotor wind disturbance", "gust rejection control"]}

    plan = LLMPlanner(_Router(), chat_fn=fake_chat).plan("gust rejection")
    assert plan.sub_questions == ("what disturbs a quadrotor?",)
    assert "gust rejection control" in plan.queries
    assert "gust rejection" in plan.queries, "the raw question is always searched too"


def test_llm_planner_falls_back_to_the_template_when_the_model_raises():
    def boom(router, task, prompt, **kwargs):
        raise RuntimeError("no api key")

    plan = LLMPlanner(_Router(), chat_fn=boom).plan("gust rejection")
    assert plan == TemplatePlanner().plan("gust rejection")


def test_llm_planner_falls_back_when_the_model_returns_junk():
    for junk in (None, {}, {"queries": []}, "not a dict", {"queries": ["", "  "]}):
        plan = LLMPlanner(_Router(), chat_fn=lambda *a, junk=junk, **k: junk).plan("q")
        assert plan == TemplatePlanner().plan("q")


def test_llm_planner_caps_the_fan_out():
    many = {"sub_questions": [f"s{i}" for i in range(50)],
            "queries": [f"q{i}" for i in range(50)]}
    plan = LLMPlanner(_Router(), chat_fn=lambda *a, **k: many, max_sub=4, per_sub=3).plan("q")
    assert len(plan.sub_questions) == 4
    assert len(plan.queries) <= 4 * 3 + 1


def test_search_plan_is_frozen():
    import dataclasses

    plan = SearchPlan(question="q")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.question = "other"



from jarvis.gather import Candidate, run_searches, save_candidates, to_paper
from jarvis.store import close_store, get_paper, get_papers_by_depth, open_store

CORPUS = {
    "gust rejection": [
        {"arxiv_id": "2501.00001", "title": "Gust-Robust Control", "abstract": "a",
         "year": 2025, "citation_count": 42, "doi": "10.1/a"},
    ],
    "gust rejection survey": [
        {"arxiv_id": "2501.00001", "title": "Gust-Robust Control", "abstract": "a"},
        {"arxiv_id": "2501.00002", "title": "A Survey of Wind Rejection", "abstract": "b"},
    ],
}


def fake_search(query: str) -> list[dict]:
    return [dict(p) for p in CORPUS.get(query, [])]


def test_run_searches_visits_every_query_in_the_plan():
    seen = []

    def spy(query):
        seen.append(query)
        return []

    plan = TemplatePlanner().plan("gust rejection")
    run_searches(plan, spy)
    assert seen == list(plan.queries)


def test_a_paper_found_by_two_queries_appears_once_carrying_both():
    plan = SearchPlan(question="gust rejection",
                      queries=("gust rejection", "gust rejection survey"))
    cands = run_searches(plan, fake_search)
    assert len(cands) == 2
    first = next(c for c in cands if c.pid == "2501.00001")
    assert first.queries == ("gust rejection", "gust rejection survey")


def test_search_candidates_are_at_graph_depth_zero():
    cands = run_searches(SearchPlan(question="q", queries=("gust rejection",)), fake_search)
    assert cands[0].origin == "search"
    assert cands[0].graph_depth == 0


def test_a_failing_source_does_not_abort_the_fan_out():
    def flaky(query):
        if query == "gust rejection":
            raise RuntimeError("rate limited")
        return fake_search(query)

    plan = SearchPlan(question="q", queries=("gust rejection", "gust rejection survey"))
    assert len(run_searches(plan, flaky)) == 2


def test_records_without_any_identity_are_dropped():
    plan = SearchPlan(question="q", queries=("x",))
    assert run_searches(plan, lambda q: [{"title": "", "abstract": "orphan"}]) == []


def test_to_paper_maps_the_source_dict_onto_the_domain_type():
    p = to_paper(Candidate(paper=CORPUS["gust rejection"][0]))
    assert p.paper_id == "2501.00001"
    assert p.title == "Gust-Robust Control"
    assert p.year == 2025
    assert p.citation_count == 42
    assert p.doi == "10.1/a"
    assert p.retracted is False


def test_to_paper_carries_the_retracted_flag_through():
    c = Candidate(paper={"arxiv_id": "x1", "title": "T", "retracted": True})
    assert to_paper(c).retracted is True


def test_candidates_are_saved_at_metadata_depth(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = run_searches(SearchPlan(question="q", queries=("gust rejection survey",)),
                             fake_search)
        assert save_candidates(conn, cands) == 2
        assert len(get_papers_by_depth(conn, "metadata")) == 2
        assert get_paper(conn, "2501.00002").title == "A Survey of Wind Rejection"
    finally:
        close_store(conn)


def test_saving_candidates_twice_does_not_duplicate_them(tmp_path):
    conn = open_store(tmp_path / "c.db")
    try:
        cands = run_searches(SearchPlan(question="q", queries=("gust rejection survey",)),
                             fake_search)
        save_candidates(conn, cands)
        save_candidates(conn, cands)
        assert len(get_papers_by_depth(conn, "metadata")) == 2
    finally:
        close_store(conn)



from jarvis.gather import expand_citations, gather

GRAPH = {
    "seed": [{"arxiv_id": "hop1", "title": "One Hop", "abstract": "a"}],
    "hop1": [{"arxiv_id": "hop2", "title": "Two Hops", "abstract": "b"}],
    "hop2": [{"arxiv_id": "hop3", "title": "Three Hops", "abstract": "c"}],
}


def _neighbors():
    def refs(pid):
        return [dict(p) for p in GRAPH.get(pid, [])]

    def cites(pid):
        return []

    return refs, cites


SEEDS = [{"arxiv_id": "seed", "title": "Seed", "abstract": "s"}]


def test_expansion_records_the_hop_count_as_graph_depth():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=2)
    by_pid = {c.pid: c.graph_depth for c in found}
    assert by_pid == {"hop1": 1, "hop2": 2}
    assert all(c.origin == "citation" for c in found)


def test_expansion_stops_at_max_depth():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=1)
    assert {c.pid for c in found} == {"hop1"}


def test_the_seeds_themselves_are_never_returned():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=3)
    assert "seed" not in {c.pid for c in found}


def test_low_scoring_neighbours_are_not_walked_through():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 0.1, threshold=0.5, max_depth=3)
    assert found == []


def test_expansion_respects_its_budget():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=3, budget=1)
    assert len(found) == 1


def test_already_seen_papers_are_not_re_surfaced():
    found = expand_citations(SEEDS, _neighbors(), lambda p: 1.0, max_depth=2,
                             already_seen={"hop1"})
    assert "hop1" not in {c.pid for c in found}


def test_gather_merges_search_hits_and_citation_expansion():
    def search(query):
        return [dict(SEEDS[0])] if query == "seed topic" else []

    cands = gather("seed topic", SearchPlan(question="seed topic", queries=("seed topic",)),
                   search, neighbors=_neighbors(), score_fn=lambda p: 1.0, max_depth=1)
    assert {c.pid for c in cands} == {"seed", "hop1"}
    assert next(c for c in cands if c.pid == "seed").graph_depth == 0
    assert next(c for c in cands if c.pid == "hop1").graph_depth == 1


def test_gather_accepts_a_planner_and_builds_its_own_plan():
    calls = []

    def search(query):
        calls.append(query)
        return []

    gather("gust rejection", TemplatePlanner(), search)
    assert calls == list(TemplatePlanner().plan("gust rejection").queries)


def test_gather_without_a_graph_is_just_the_searches():
    def search(query):
        return [dict(SEEDS[0])] if query == "q" else []

    cands = gather("q", SearchPlan(question="q", queries=("q",)), search)
    assert {c.pid for c in cands} == {"seed"}


def test_gather_seeds_expansion_from_the_top_scoring_hits_only():
    walked = []

    def refs(pid):
        walked.append(pid)
        return []

    def search(query):
        return [{"arxiv_id": f"p{i}", "title": f"T{i}", "abstract": "x", "citation_count": i}
                for i in range(5)]

    gather("q", SearchPlan(question="q", queries=("q",)), search,
           neighbors=(refs, lambda pid: []), score_fn=lambda p: 1.0, seed_limit=2)
    assert len(walked) == 2, "only the top `seed_limit` hits are expanded from"
