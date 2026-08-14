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
