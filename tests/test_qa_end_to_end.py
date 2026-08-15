"""The proof this plan exists to produce: a cited answer, and a fabrication stopped."""
import pytest

from jarvis.answer import ask, render_answer
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.evaluate import report
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.parse import FakeParser
from jarvis.retriever import FakeRefiner
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

BLOCKS = [
    Block(kind="heading", text="Results", page=3, section_path=("Results",)),
    Block(kind="paragraph",
          text="As shown in Table 3, our controller reaches 94.2% tracking accuracy under "
               "gust distur-\nbance.", page=3, section_path=("Results",)),
    Block(kind="table", text="| method | accuracy |\n|---|---|\n| ours | 94.2 |",
          page=3, section_path=("Results",), label="Table 3"),
    Block(kind="caption", text="Table 3: Tracking accuracy under wind.", page=3,
          section_path=("Results",), label="Table 3"),
    Block(kind="heading", text="Limitations", page=4, section_path=("Limitations",)),
    Block(kind="paragraph", text="Above 12 m/s the controller loses tracking entirely.",
          page=4, section_path=("Limitations",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Quadrotor Control", year=2025)
QUESTION = "how accurate is the controller under wind?"
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    parsed = FakeParser(BLOCKS).parse("paper.pdf", "p1")
    save_paper(conn, PAPER, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), PAPER, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())
    yield conn
    close_store(conn)


def _table_unit(conn):
    return next(u for u in get_units(conn, "p1") if u.type.value == "table")


def _limits_unit(conn):
    return next(u for u in get_units(conn, "p1") if "12 m/s" in u.verbatim_text)


def test_a_two_claim_answer_is_fully_supported(corpus):
    table, limits = _table_unit(corpus), _limits_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="Accurate in gusts, but not above 12 m/s.",
        claims=(
            Claim("c-0", "It reaches 94.2% tracking accuracy.", table.unit_id,
                  "| ours | 94.2 |"),
            Claim("c-1", "It fails above 12 m/s.", limits.unit_id,
                  "Above 12 m/s the controller loses tracking entirely."),
        ))})

    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS,
                 refiner=FakeRefiner(["wind speed limits"]), rounds=2)
    assert len(answer.supported) == 2
    assert answer.blocked == ()
    assert answer.is_grounded is True


def test_the_rendered_answer_cites_every_supported_claim(corpus):
    table = _table_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="Accurate.",
        claims=(Claim("c-0", "It reaches 94.2%.", table.unit_id, "| ours | 94.2 |"),))})
    rendered = render_answer(ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS))
    assert table.unit_id in rendered
    assert "94.2" in rendered


def test_a_fabricated_number_never_reaches_the_reader(corpus):
    table = _table_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="It reaches 99.9% accuracy.",
        claims=(Claim("c-0", "It reaches 99.9% accuracy.", table.unit_id,
                      "| ours | 99.9 |"),))})

    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS)
    assert answer.blocked[0].verdict is Verdict.QUOTE_NOT_FOUND
    assert "99.9" not in render_answer(answer)


def test_fabrication_is_caught_even_when_the_model_is_certain(corpus):
    """The NLI model insists the claim is entailed. Stage 1 never asks it."""
    table = _table_unit(corpus)
    certain = FakeNLI(default={"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0})
    writer = FakeWriter({QUESTION: Draft(
        text="x", claims=(Claim("c-0", "99.9%", table.unit_id, "| ours | 99.9 |"),))})
    assert ask(corpus, QUESTION, FakeEmbedder(), writer, certain).supported == ()


def test_a_quote_hyphenated_across_a_line_break_still_grounds(corpus):
    prose = next(u for u in get_units(corpus, "p1")
                 if u.type.value == "prose" and "gust" in u.verbatim_text)
    writer = FakeWriter({QUESTION: Draft(
        text="x", claims=(Claim("c-0", "Gusts disturb it.", prose.unit_id,
                                "under gust disturbance"),))})
    assert ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS).supported != ()


def test_the_eval_report_scores_the_answer(corpus):
    table = _table_unit(corpus)
    writer = FakeWriter({QUESTION: Draft(
        text="x",
        claims=(Claim("c-0", "94.2%", table.unit_id, "| ours | 94.2 |"),
                Claim("c-1", "99.9%", table.unit_id, "| ours | 99.9 |")))})

    answer = ask(corpus, QUESTION, FakeEmbedder(), writer, ENTAILS)
    r = report(list(answer.verifications))
    assert r.quote_fidelity == pytest.approx(0.5)
    assert r.meets_quote_target is False
    assert r.citation_recall == pytest.approx(0.5)


def test_the_evidence_reaching_the_writer_is_never_the_whole_corpus(tmp_path):
    """Its own 5-unit corpus, not the shared 3-unit fixture — the cap needs something to
    actually cut, or this test cannot tell a working cap from a deleted one."""
    blocks = [
        Block(kind="heading", text="Results", page=1, section_path=("Results",)),
        Block(kind="paragraph", text="The controller reaches 94.2% tracking accuracy.",
              page=1, section_path=("Results",)),
        Block(kind="heading", text="Limitations", page=2, section_path=("Limitations",)),
        Block(kind="paragraph", text="Performance degrades above 12 m/s wind speed.",
              page=2, section_path=("Limitations",)),
        Block(kind="heading", text="Related Work", page=3, section_path=("Related Work",)),
        Block(kind="paragraph", text="Prior controllers used fixed-gain PID schemes.",
              page=3, section_path=("Related Work",)),
        Block(kind="heading", text="Discussion", page=4, section_path=("Discussion",)),
        Block(kind="paragraph", text="Future work should explore adaptive gain scheduling.",
              page=4, section_path=("Discussion",)),
        Block(kind="heading", text="Conclusion", page=5, section_path=("Conclusion",)),
        Block(kind="paragraph", text="The approach generalizes to other wind regimes.",
              page=5, section_path=("Conclusion",)),
    ]
    conn = open_store(tmp_path / "big_corpus.db")
    paper = Paper(paper_id="p2", title="A Larger Paper", year=2025)
    parsed = FakeParser(blocks).parse("big.pdf", "p2")
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())

    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["count"] = len(units)
            return Draft()

    answer = ask(conn, "how accurate is the controller under wind?", FakeEmbedder(),
                 SpyWriter(), ENTAILS, limit=20, max_units=3)
    close_store(conn)

    assert seen["count"] == 3
    assert answer.dropped_evidence > 0, "5 candidate units and max_units=3 must genuinely " \
                                        "drop at least one"
