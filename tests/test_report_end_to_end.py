"""Three papers, one outline, one verified report, one honest coverage number."""
import pytest

from jarvis.card import FakeCardExtractor, extract_and_verify
from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Card, CardField, Claim, Paper
from jarvis.outline import Outline, Section, TemplateOutliner
from jarvis.parse import FakeParser
from jarvis.report import evaluate_report, render_report, write_report
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

PAPERS = {
    "p1": ("Gust-Robust Control", [
        Block(kind="heading", text="Results", page=1, section_path=("Results",)),
        Block(kind="paragraph",
              text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
              page=1, section_path=("Results",)),
    ]),
    "p2": ("Wind Disturbance Attenuation", [
        Block(kind="heading", text="Limitations", page=1, section_path=("Limitations",)),
        Block(kind="paragraph", text="Tracking degrades sharply above 12 m/s wind speed.",
              page=1, section_path=("Limitations",)),
    ]),
    "p3": ("Gust Tolerance Benchmarks", [
        Block(kind="heading", text="Datasets", page=1, section_path=("Datasets",)),
        Block(kind="paragraph", text="We evaluate on the WindBench gust tolerance suite.",
              page=1, section_path=("Datasets",)),
    ]),
}
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    for paper_id, (title, blocks) in PAPERS.items():
        paper = Paper(paper_id=paper_id, title=title, year=2025, venue="ICRA")
        parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
        save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
        units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
        save_units(conn, units)
        index_units_fts(conn, units)
        index_units(conn, units, FakeEmbedder())

        unit = units[0]
        extract_and_verify(conn, paper, FakeCardExtractor({paper_id: Card(
            paper_id=paper_id,
            problem=CardField("gust rejection", unit.unit_id, unit.verbatim_text[:20]),
            metrics=(CardField("accuracy", unit.unit_id, unit.verbatim_text[:20]),),
        )}))
    yield conn
    close_store(conn)


def _u(conn, paper_id, needle):
    return next(u for u in get_units(conn, paper_id) if needle in u.verbatim_text)


def test_an_outline_is_built_from_the_corpus_cards(corpus):
    report = write_report(corpus, "gust rejection", TemplateOutliner(), FakeEmbedder(),
                          FakeWriter({}), ENTAILS)
    titles = [s.section.title for s in report.sections]
    assert "Overview" in titles
    assert any("esult" in t for t in titles)


def test_a_multi_section_report_verifies_every_section(corpus):
    q1, q2 = "what accuracy is reported?", "what are the wind speed limits?"
    writer = FakeWriter({
        q1: Draft(text="Accurate.", claims=(Claim(
            "a-0", "It reaches 94.2% accuracy.", _u(corpus, "p1", "94.2").unit_id,
            "94.2% tracking accuracy"),)),
        q2: Draft(text="Limited.", claims=(Claim(
            "b-0", "It degrades above 12 m/s.", _u(corpus, "p2", "12 m/s").unit_id,
            "above 12 m/s wind speed"),)),
    })
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),
                                               Section(title="Limits", question=q2)))

    report = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert len(report.all_claims) == 2
    assert all(s.blocked == () for s in report.sections)
    assert report.cited_paper_ids == {"p1", "p2"}


def test_a_fabricated_claim_never_reaches_the_rendered_report(corpus):
    q1 = "what accuracy is reported?"
    writer = FakeWriter({q1: Draft(text="It reaches 99.9%.", claims=(Claim(
        "a-0", "It reaches 99.9% accuracy.", _u(corpus, "p1", "94.2").unit_id,
        "99.9% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    report = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert "99.9" not in render_report(corpus, report)
    assert evaluate_report(report).meets_quote_target is False


def test_quote_fidelity_is_not_relaxed_for_a_long_report(corpus):
    q1 = "what accuracy is reported?"
    unit = _u(corpus, "p1", "94.2")
    writer = FakeWriter({q1: Draft(text="t", claims=tuple(
        Claim(f"a-{i}", f"claim {i}", unit.unit_id, "94.2% tracking accuracy")
        for i in range(10)))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    evaluation = evaluate_report(write_report(corpus, "gusts", outline, FakeEmbedder(),
                                              writer, ENTAILS))
    assert evaluation.quote_fidelity == 1.0
    assert evaluation.meets_quote_target is True


def test_the_report_reports_low_coverage_honestly(corpus):
    q1 = "what accuracy is reported?"
    writer = FakeWriter({q1: Draft(text="t", claims=(Claim(
        "a-0", "94.2%", _u(corpus, "p1", "94.2").unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    report = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert report.coverage < 0.5, "one unit out of three papers is low coverage"
    assert "Coverage" in render_report(corpus, report)


def test_every_section_is_retrieved_and_budgeted_separately(corpus):
    seen = []

    class SpyWriter:
        def write(self, question, units):
            seen.append((question, len(units)))
            return Draft()

    outline = Outline(topic="t", sections=(
        Section(title="A", question="what accuracy is reported?"),
        Section(title="B", question="what are the wind speed limits?"),
        Section(title="C", question="what datasets are used?")))
    write_report(corpus, "t", outline, FakeEmbedder(), SpyWriter(), ENTAILS, max_units=2)

    assert len(seen) == 3, "one call per section, never one call for the whole report"
    assert all(count <= 2 for _q, count in seen)
    assert len({q for q, _c in seen}) == 3


def test_the_references_only_list_papers_actually_cited(corpus):
    q1 = "what accuracy is reported?"
    writer = FakeWriter({q1: Draft(text="t", claims=(Claim(
        "a-0", "94.2%", _u(corpus, "p1", "94.2").unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="Results", question=q1),))

    rendered = render_report(corpus, write_report(corpus, "gusts", outline, FakeEmbedder(),
                                                  writer, ENTAILS))
    assert "Gust-Robust Control" in rendered
    assert "Gust Tolerance Benchmarks" not in rendered
