"""Report outlines, built from Layer 2 cards (spec §5, §7D)."""
from dataclasses import FrozenInstanceError

import pytest

from jarvis.models import Card, CardField
from jarvis.outline import (
    LLMOutliner,
    Outline,
    Outliner,
    Section,
    TemplateOutliner,
    cards_digest,
)

CARDS = [
    Card(paper_id="p1",
         problem=CardField("gust rejection", "u1", "gusts"),
         method=CardField("adaptive control", "u2", "adaptive"),
         datasets=(CardField("KITTI", "u3", "KITTI"),),
         metrics=(CardField("94.2", "u4", "94.2", binding_verified=True),),
         limitations=(CardField("fails above 12 m/s", "u5", "12 m/s"),)),
    Card(paper_id="p2",
         problem=CardField("wind disturbance", "u6", "wind"),
         metrics=(CardField("61.0", "u7", "61.0"),)),
]
BARE = [Card(paper_id="p9", problem=CardField("something", "u1", "q"))]


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_template_outliner_satisfies_the_protocol():
    assert isinstance(TemplateOutliner(), Outliner)


def test_an_outline_has_sections_with_their_own_sub_questions():
    outline = TemplateOutliner().outline("gust rejection", CARDS)
    assert outline.topic == "gust rejection"
    assert len(outline.sections) >= 3
    assert all(s.question for s in outline.sections)
    assert all(s.title for s in outline.sections)


def test_sections_only_appear_when_the_corpus_can_support_them():
    rich = {s.title for s in TemplateOutliner().outline("t", CARDS).sections}
    bare = {s.title for s in TemplateOutliner().outline("t", BARE).sections}
    assert any("ataset" in t for t in rich)
    assert not any("ataset" in t for t in bare)
    assert not any("imitation" in t for t in bare)


def test_an_empty_corpus_still_yields_a_minimal_outline():
    outline = TemplateOutliner().outline("gust rejection", [])
    assert len(outline.sections) >= 1
    assert "gust rejection" in outline.sections[0].question


def test_the_topic_appears_in_every_sub_question():
    for section in TemplateOutliner().outline("gust rejection", CARDS).sections:
        assert "gust rejection" in section.question


def test_sections_record_which_papers_motivated_them():
    outline = TemplateOutliner().outline("t", CARDS)
    datasets = next(s for s in outline.sections if "ataset" in s.title)
    assert datasets.paper_ids == ("p1",)


def test_the_template_outliner_is_deterministic():
    assert TemplateOutliner().outline("t", CARDS) == TemplateOutliner().outline("t", CARDS)


def test_the_card_digest_names_papers_and_their_fields():
    digest = cards_digest(CARDS)
    assert "p1" in digest
    assert "gust rejection" in digest
    assert "KITTI" in digest


def test_the_card_digest_is_capped():
    many = [Card(paper_id=f"p{i}", problem=CardField(f"topic {i}", "u", "q"))
            for i in range(500)]
    assert "p499" not in cards_digest(many, max_papers=10)


def test_llm_outliner_uses_the_models_sections():
    reply = {"sections": [
        {"title": "Control strategies", "question": "what control strategies exist?",
         "paper_ids": ["p1"]},
        {"title": "Reported accuracy", "question": "what accuracy is reported?"},
    ]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply).outline("t", CARDS)
    assert [s.title for s in outline.sections] == ["Control strategies", "Reported accuracy"]
    assert outline.sections[0].paper_ids == ("p1",)
    assert outline.sections[1].paper_ids == ()


def test_llm_outliner_drops_sections_missing_a_title_or_question():
    reply = {"sections": [{"title": "Only a title"}, {"question": "only a question"},
                          {"title": "Good", "question": "good?"}]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply).outline("t", CARDS)
    assert [s.title for s in outline.sections] == ["Good"]


def test_llm_outliner_drops_paper_ids_not_in_the_corpus():
    reply = {"sections": [{"title": "T", "question": "q?", "paper_ids": ["p1", "ghost"]}]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply).outline("t", CARDS)
    assert outline.sections[0].paper_ids == ("p1",)


def test_llm_outliner_caps_the_section_count():
    reply = {"sections": [{"title": f"T{i}", "question": f"q{i}?"} for i in range(50)]}
    outline = LLMOutliner(_Router(), chat_fn=lambda *a, **k: reply,
                          max_sections=5).outline("t", CARDS)
    assert len(outline.sections) == 5


def test_llm_outliner_falls_back_to_the_template_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    assert LLMOutliner(_Router(), chat_fn=boom).outline("t", CARDS) == \
        TemplateOutliner().outline("t", CARDS)


def test_llm_outliner_falls_back_on_junk():
    for junk in (None, {}, "text", {"sections": []}, {"sections": "x"}):
        outliner = LLMOutliner(_Router(), chat_fn=lambda *a, _junk=junk, **k: _junk)
        assert outliner.outline("t", CARDS) == TemplateOutliner().outline("t", CARDS)


def test_llm_outliner_routes_to_outline():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {}

    LLMOutliner(_Router(), chat_fn=spy).outline("t", CARDS)
    assert seen["task"] == "outline"


def test_outline_types_are_frozen():
    with pytest.raises(FrozenInstanceError):
        Section(title="a", question="b").title = "c"
    with pytest.raises(FrozenInstanceError):
        Outline(topic="t").topic = "u"
