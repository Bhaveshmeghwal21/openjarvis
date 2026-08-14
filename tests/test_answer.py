"""Answer assembly: retrieve, write, verify, block, flag."""
from dataclasses import FrozenInstanceError

import pytest

from jarvis.answer import ask, render_answer
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

BLOCKS = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy in gusts.",
          page=1, section_path=("Results",)),
    Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
    Block(kind="paragraph", text="Performance degrades sharply above 12 m/s wind speed.",
          page=2, section_path=("Limitations",)),
    Block(kind="heading", text="Discussion", page=3, section_path=("Discussion",)),
    Block(kind="paragraph", text="Future work should explore adaptive gain scheduling for "
                                 "extreme wind conditions.", page=3, section_path=("Discussion",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)
QUESTION = "how accurate is the controller?"
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})
NEUTRAL = FakeNLI(default={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})


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


def _prose_unit(conn):
    return next(u for u in get_units(conn, "p1")
                if "94.2" in u.verbatim_text and u.type.value == "prose")


def _writer(conn, quote, text="It reaches 94.2% tracking accuracy."):
    unit = _prose_unit(conn)
    return FakeWriter({QUESTION: Draft(
        text="The controller is accurate under gusts.",
        claims=(Claim(claim_id="c-0", text=text, unit_id=unit.unit_id, quote=quote),),
    )})


def test_a_grounded_entailed_claim_is_supported(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), ENTAILS)
    assert len(answer.supported) == 1
    assert answer.blocked == ()
    assert answer.is_grounded is True


def test_a_fabricated_quote_blocks_the_claim(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 99.9% tracking accuracy"), ENTAILS)
    assert len(answer.blocked) == 1
    assert answer.supported == ()
    assert answer.blocked[0].verdict is Verdict.QUOTE_NOT_FOUND
    assert answer.is_grounded is False


def test_a_blocked_claim_never_appears_in_the_rendered_answer(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 99.9% tracking accuracy",
                         text="It reaches 99.9% accuracy."), ENTAILS)
    rendered = render_answer(answer)
    assert "99.9" not in rendered


def test_a_real_quote_that_does_not_entail_is_flagged_not_blocked(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), NEUTRAL)
    assert len(answer.flagged) == 1
    assert answer.blocked == ()
    assert answer.flagged[0].verdict is Verdict.NEUTRAL


def test_a_flagged_claim_is_rendered_with_a_warning(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), NEUTRAL)
    rendered = render_answer(answer)
    assert "unverified" in rendered.lower()


def test_a_supported_claim_is_rendered_with_its_unit_id(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), ENTAILS)
    unit_id = _prose_unit(corpus).unit_id
    assert unit_id in render_answer(answer)


def test_every_claim_gets_exactly_one_verification(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(),
                 _writer(corpus, "reaches 94.2% tracking accuracy"), ENTAILS)
    assert len(answer.verifications) == len(answer.claims) == 1


def test_the_writer_only_ever_sees_capped_ordered_evidence(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["units"] = list(units)
            return Draft()

    ask(corpus, QUESTION, FakeEmbedder(), SpyWriter(), ENTAILS, limit=8, max_units=1)
    assert len(seen["units"]) == 1


def test_a_question_with_no_retrievable_evidence_answers_nothing(corpus):
    answer = ask(corpus, "zzzz nonexistent topic qqq", FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert answer.claims == ()
    assert answer.is_grounded is False


def test_an_empty_answer_renders_an_explicit_no_evidence_message(corpus):
    answer = ask(corpus, "zzzz nonexistent topic qqq", FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert "no" in render_answer(answer).lower()


def test_the_evidence_cap_is_reported_not_hidden(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS,
                 max_units=1, limit=8)
    assert answer.dropped_evidence > 0, "with 3 candidate units and max_units=1, capping " \
                                        "must genuinely drop at least one"


def test_answer_is_frozen(corpus):
    answer = ask(corpus, QUESTION, FakeEmbedder(), FakeWriter({}), ENTAILS)
    with pytest.raises(FrozenInstanceError):
        answer.text = "rewritten"


def test_verification_does_not_consult_the_writer(corpus):
    """The writer is called once, for drafting, and never again during verification."""
    calls = []
    unit = _prose_unit(corpus)

    class CountingWriter:
        def write(self, question, units):
            calls.append(question)
            return Draft(text="t", claims=(Claim("c-0", "claim", unit.unit_id,
                                                 "reaches 94.2% tracking accuracy"),))

    ask(corpus, QUESTION, FakeEmbedder(), CountingWriter(), ENTAILS)
    assert len(calls) == 1
