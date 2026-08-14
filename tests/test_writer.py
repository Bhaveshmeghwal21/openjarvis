"""The writer. Emits claim triples, never prose the verifier has to parse."""
from dataclasses import FrozenInstanceError

import pytest

from jarvis.models import Unit, UnitType
from jarvis.writer import Draft, FakeWriter, LLMWriter, Writer, claims_from_json

UNITS = [
    Unit(unit_id="u1", paper_id="p1", type=UnitType.PROSE, page=1, section_path=(),
         verbatim_text="The controller reaches 94.2% tracking accuracy.", ordinal=0),
    Unit(unit_id="u2", paper_id="p1", type=UnitType.PROSE, page=2, section_path=(),
         verbatim_text="Performance degrades above 12 m/s.", ordinal=1),
]


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


def test_fake_writer_satisfies_the_protocol():
    assert isinstance(FakeWriter({}), Writer)


def test_claims_are_built_from_the_models_triples():
    data = {"claims": [
        {"text": "It reaches 94.2% accuracy.", "unit_id": "u1", "quote": "94.2% tracking"},
        {"text": "It degrades in strong wind.", "unit_id": "u2", "quote": "12 m/s"},
    ]}
    claims = claims_from_json(data, UNITS)
    assert [c.unit_id for c in claims] == ["u1", "u2"]
    assert claims[0].quote == "94.2% tracking"


def test_claim_ids_are_unique_and_prefixed():
    data = {"claims": [{"text": "a", "unit_id": "u1", "quote": "q"},
                       {"text": "b", "unit_id": "u1", "quote": "q"}]}
    claims = claims_from_json(data, UNITS, prefix="sub3")
    assert [c.claim_id for c in claims] == ["sub3-0", "sub3-1"]


def test_a_claim_citing_a_unit_outside_the_evidence_set_is_dropped():
    data = {"claims": [{"text": "invented", "unit_id": "u99", "quote": "q"}]}
    assert claims_from_json(data, UNITS) == []


def test_a_claim_with_no_quote_is_dropped():
    data = {"claims": [{"text": "unbacked", "unit_id": "u1", "quote": ""},
                       {"text": "unbacked", "unit_id": "u1"}]}
    assert claims_from_json(data, UNITS) == []


def test_a_claim_with_no_text_is_dropped():
    assert claims_from_json({"claims": [{"unit_id": "u1", "quote": "q"}]}, UNITS) == []


def test_malformed_payloads_yield_no_claims_and_never_raise():
    for junk in (None, {}, "text", {"claims": None}, {"claims": "x"}, {"claims": [1, 2]}):
        assert claims_from_json(junk, UNITS) == []


def test_llm_writer_returns_answer_text_and_claims():
    reply = {"answer": "The controller is accurate in gusts.",
             "claims": [{"text": "It reaches 94.2% accuracy.", "unit_id": "u1",
                         "quote": "94.2% tracking"}]}
    draft = LLMWriter(_Router(), chat_fn=lambda *a, **k: reply).write("how accurate?", UNITS)
    assert draft.text == "The controller is accurate in gusts."
    assert len(draft.claims) == 1


def test_llm_writer_is_shown_the_unit_ids_it_must_cite():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["prompt"] = prompt
        return {"answer": "", "claims": []}

    LLMWriter(_Router(), chat_fn=spy).write("q", UNITS)
    assert "[u1]" in seen["prompt"]
    assert "[u2]" in seen["prompt"]


def test_llm_writer_routes_to_synthesis():
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {}

    LLMWriter(_Router(), chat_fn=spy).write("q", UNITS)
    assert seen["task"] == "synthesis"


def test_llm_writer_returns_an_empty_draft_on_failure():
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    draft = LLMWriter(_Router(), chat_fn=boom).write("q", UNITS)
    assert draft == Draft(text="", claims=())


def test_a_writer_given_no_evidence_writes_nothing():
    draft = LLMWriter(_Router(), chat_fn=lambda *a, **k: {"answer": "invented"}).write("q", [])
    assert draft == Draft(text="", claims=()), "no evidence means no answer, not a guess"


def test_fake_writer_returns_the_draft_for_its_question():
    draft = Draft(text="answer", claims=())
    assert FakeWriter({"q": draft}).write("q", UNITS) is draft
    assert FakeWriter({}).write("q", UNITS) == Draft(text="", claims=())


def test_draft_is_frozen():
    with pytest.raises(FrozenInstanceError):
        Draft(text="a", claims=()).text = "b"
