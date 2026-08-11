"""Tests for the ported gather-stage primitives. All offline, no network, no keys."""
from __future__ import annotations

import math

from jarvis.citation_graph import CitationWalker, paper_id
from jarvis.router import ModelRouter
from jarvis.scoring import citation_weight, cosine, make_cosine_scorer, recency
from jarvis.sources import combine_sources, dedup_papers, normalize_crossref


# --- sources -----------------------------------------------------------------

def test_dedup_by_arxiv_doi_and_title():
    papers = [
        {"arxiv_id": "2301.1", "title": "A"},
        {"arxiv_id": "2301.1", "title": "A duplicate"},
        {"doi": "10.1/x", "title": "B"},
        {"doi": "10.1/X", "title": "B again"},
        {"title": "C"},
        {"title": "C"},
    ]
    assert len(dedup_papers(papers)) == 3


def test_dedup_keeps_distinct_arxiv_versions_apart():
    # 2301.1 and 2301.2 are different papers, not versions of one.
    assert len(dedup_papers([{"arxiv_id": "2301.1"}, {"arxiv_id": "2301.2"}])) == 2


def test_combine_sources_skips_failing_source():
    def ok(_):
        return [{"title": "A"}]

    def boom(_):
        raise RuntimeError("source down")

    combined = combine_sources(ok, boom, ok)
    assert [p["title"] for p in combined("q")] == ["A"]


def test_normalize_crossref_strips_html_and_reads_year():
    item = {
        "DOI": "10.1/x",
        "title": ["T"],
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "issued": {"date-parts": [[2021, 5]]},
        "container-title": ["Venue"],
        "abstract": "<jats:p>Body</jats:p>",
        "is-referenced-by-count": 7,
        "URL": "http://x",
    }
    got = normalize_crossref(item)
    assert got["year"] == 2021
    assert got["abstract"] == "Body"
    assert got["authors"] == ["Ada Lovelace"]
    assert got["citation_count"] == 7


def test_normalize_crossref_tolerates_missing_fields():
    got = normalize_crossref({})
    assert got["title"] == ""
    assert got["year"] is None
    assert got["citation_count"] == 0


# --- scoring -----------------------------------------------------------------

def test_cosine_identical_and_orthogonal():
    assert math.isclose(cosine([1, 0], [1, 0]), 1.0)
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0)


def test_cosine_degenerate_inputs_are_zero():
    assert cosine([], []) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0
    assert cosine(None, [1]) == 0.0


def test_recency_decays_over_ten_years():
    assert recency(2026, 2026) == 1.0
    assert math.isclose(recency(2021, 2026), 0.5)
    assert recency(2000, 2026) == 0.0
    assert recency(None, 2026) == 0.0


def test_citation_weight_is_bounded():
    assert citation_weight(0) == 0.0
    assert 0 < citation_weight(100) <= 1.0
    assert citation_weight(10**9) == 1.0
    assert citation_weight(None) == 0.0


def test_cosine_scorer_ranks_matching_paper_higher():
    vectors = {"drone wind": [1.0, 0.0], "drone wind rejection": [0.9, 0.1], "cake": [0.0, 1.0]}
    score = make_cosine_scorer(lambda t: vectors[t.strip()], "drone wind")
    near = score({"title": "drone wind rejection", "abstract": ""})
    far = score({"title": "cake", "abstract": ""})
    assert near > far


# --- citation walker ---------------------------------------------------------

def _walker(graph, *, threshold=0.5, max_depth=2, budget=500, seen=None):
    def refs(pid):
        return graph.get(pid, {}).get("refs", [])

    def cites(pid):
        return graph.get(pid, {}).get("cites", [])

    return CitationWalker(
        fetch_refs_fn=refs,
        fetch_citations_fn=cites,
        score_fn=lambda p: p.get("score", 1.0),
        threshold=threshold,
        max_depth=max_depth,
        budget=budget,
        already_seen=seen or set(),
    )


def test_paper_id_prefers_arxiv_then_s2_then_title():
    assert paper_id({"arxiv_id": "a", "s2_id": "b", "title": "c"}) == "a"
    assert paper_id({"s2_id": "b", "title": "c"}) == "b"
    assert paper_id({"title": "c"}) == "c"


def test_walker_expands_and_excludes_seeds():
    graph = {"seed": {"refs": [{"arxiv_id": "r1"}], "cites": [{"arxiv_id": "c1"}]}}
    got = _walker(graph).walk([{"arxiv_id": "seed"}])
    assert sorted(paper_id(p) for p in got) == ["c1", "r1"]


def test_walker_applies_relevance_threshold():
    graph = {"seed": {"refs": [{"arxiv_id": "keep", "score": 0.9},
                               {"arxiv_id": "drop", "score": 0.1}]}}
    got = _walker(graph, threshold=0.5).walk([{"arxiv_id": "seed"}])
    assert [paper_id(p) for p in got] == ["keep"]


def test_walker_respects_budget():
    graph = {"seed": {"refs": [{"arxiv_id": f"r{i}"} for i in range(10)]}}
    assert len(_walker(graph, budget=3).walk([{"arxiv_id": "seed"}])) == 3


def test_walker_respects_max_depth():
    graph = {
        "seed": {"refs": [{"arxiv_id": "d1"}]},
        "d1": {"refs": [{"arxiv_id": "d2"}]},
        "d2": {"refs": [{"arxiv_id": "d3"}]},
    }
    got = [paper_id(p) for p in _walker(graph, max_depth=1).walk([{"arxiv_id": "seed"}])]
    assert got == ["d1"]


def test_walker_skips_already_seen():
    graph = {"seed": {"refs": [{"arxiv_id": "old"}, {"arxiv_id": "new"}]}}
    got = _walker(graph, seen={"old"}).walk([{"arxiv_id": "seed"}])
    assert [paper_id(p) for p in got] == ["new"]


def test_walker_terminates_on_cycles():
    graph = {"a": {"refs": [{"arxiv_id": "b"}]}, "b": {"refs": [{"arxiv_id": "a"}]}}
    got = _walker(graph, max_depth=5).walk([{"arxiv_id": "a"}])
    assert [paper_id(p) for p in got] == ["b"]


# --- router ------------------------------------------------------------------

def test_router_maps_task_to_tier_model():
    r = ModelRouter()
    assert r.tier_for("screen_vote") == "cheap_structured"
    assert r.route("screen_vote") == "gemini-flash-lite"
    assert r.route("synthesis") == "gpt-4.1"


def test_router_unknown_task_falls_back_to_cheap_tier():
    assert ModelRouter().route("some_new_task") == "gemini-flash-lite"


def test_router_override_wins_over_tier():
    r = ModelRouter(overrides={"synthesis": "my-local-model"})
    assert r.route("synthesis") == "my-local-model"


def test_router_logs_measured_usage_not_estimates():
    r = ModelRouter()
    r.log_usage("synthesis", 1_000_000, 0)
    summary = r.cost.summary()
    assert summary["calls"] == 1
    assert summary["by_task"]["synthesis"]["input_tokens"] == 1_000_000
    assert summary["total_cost"] > 0


def test_router_cost_summary_groups_by_task():
    r = ModelRouter()
    r.log_usage("screen_vote", 100, 10)
    r.log_usage("screen_vote", 100, 10)
    r.log_usage("synthesis", 100, 10)
    by_task = r.cost.summary()["by_task"]
    assert by_task["screen_vote"]["calls"] == 2
    assert by_task["synthesis"]["calls"] == 1


def test_verification_is_not_routed_to_an_llm():
    """Spec §8: verification runs on a local NLI model, never an LLM judge."""
    from jarvis.router import DEFAULT_ROUTING
    assert not any("verif" in task for task in DEFAULT_ROUTING)
