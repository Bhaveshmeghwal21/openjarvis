from jarvis.context import LLMPrefix, TemplatePrefix, apply_prefixes, embedding_text
from jarvis.models import Paper, Unit, UnitType

PAPER = Paper(paper_id="p1", title="Wind Rejection for Quadrotors", year=2025)
UNIT = Unit(unit_id="u1", paper_id="p1", type=UnitType.TABLE, page=3,
            section_path=("Results",), verbatim_text="| acc | 94.2 |", label="Table 3")


def test_template_prefix_names_paper_section_and_artifact():
    prefix = TemplatePrefix().describe(PAPER, UNIT)
    assert "Wind Rejection for Quadrotors" in prefix
    assert "Results" in prefix
    assert "Table 3" in prefix


def test_template_prefix_handles_missing_section_and_label():
    bare = Unit(unit_id="u", paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=(), verbatim_text="x")
    assert TemplatePrefix().describe(PAPER, bare)


def test_apply_prefixes_sets_prefix_on_every_unit():
    out = apply_prefixes([UNIT], PAPER, TemplatePrefix())
    assert out[0].context_prefix
    assert out[0].unit_id == UNIT.unit_id


def test_apply_prefixes_does_not_mutate_verbatim_text():
    out = apply_prefixes([UNIT], PAPER, TemplatePrefix())
    assert out[0].verbatim_text == "| acc | 94.2 |"


def test_embedding_text_is_prefix_then_verbatim():
    unit = apply_prefixes([UNIT], PAPER, TemplatePrefix())[0]
    text = embedding_text(unit)
    assert text.startswith(unit.context_prefix)
    assert text.endswith(unit.verbatim_text)


def test_embedding_text_without_prefix_is_just_verbatim():
    assert embedding_text(UNIT) == "| acc | 94.2 |"


def test_llm_prefix_uses_injected_chat_and_routes_to_cheap_task():
    calls = []

    def fake_chat(router, task, prompt, **kwargs):
        calls.append((task, prompt))
        return "This table reports accuracy in the Results section."

    prefix = LLMPrefix(router=None, chat_fn=fake_chat).describe(PAPER, UNIT)
    assert prefix == "This table reports accuracy in the Results section."
    assert calls[0][0] == "contextual_prefix"
    assert "Wind Rejection for Quadrotors" in calls[0][1]


def test_llm_prefix_falls_back_to_template_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    prefix = LLMPrefix(router=None, chat_fn=boom).describe(PAPER, UNIT)
    assert "Wind Rejection for Quadrotors" in prefix
