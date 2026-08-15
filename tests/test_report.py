"""Section drafting: one bounded evidence set per section, verified like any answer."""
from dataclasses import FrozenInstanceError

import pytest

from jarvis.context import TemplatePrefix, apply_prefixes
from jarvis.embed import FakeEmbedder, index_units
from jarvis.index import index_units_fts
from jarvis.models import Block, Claim, Paper, Verdict
from jarvis.outline import Section
from jarvis.parse import FakeParser
from jarvis.report import SectionDraft, draft_section
from jarvis.store import close_store, get_units, open_store, save_paper, save_units
from jarvis.units import build_units
from jarvis.verify import FakeNLI
from jarvis.writer import Draft, FakeWriter

BLOCKS_A = [
    Block(kind="heading", text="Results", page=1, section_path=("Results",)),
    Block(kind="paragraph",
          text="Our controller reaches 94.2% tracking accuracy under gust disturbance.",
          page=1, section_path=("Results",)),
]
BLOCKS_B = [
    Block(kind="heading", text="Limitations", page=1, section_path=("Limitations",)),
    Block(kind="paragraph", text="Tracking degrades sharply above 12 m/s wind speed.",
          page=1, section_path=("Limitations",)),
]
ENTAILS = FakeNLI(default={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02})
NEUTRAL = FakeNLI(default={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})
RESULTS = Section(title="Reported results", question="what accuracy is reported?")


def _ingest(conn, paper_id, blocks):
    paper = Paper(paper_id=paper_id, title=f"Paper {paper_id}", year=2025)
    parsed = FakeParser(blocks).parse(f"{paper_id}.pdf", paper_id)
    save_paper(conn, paper, raw_text=parsed.raw_text, depth="deep")
    units = apply_prefixes(build_units(parsed), paper, TemplatePrefix())
    save_units(conn, units)
    index_units_fts(conn, units)
    index_units(conn, units, FakeEmbedder())


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    _ingest(conn, "p1", BLOCKS_A)
    _ingest(conn, "p2", BLOCKS_B)
    yield conn
    close_store(conn)


def _unit(conn, paper_id, needle):
    return next(u for u in get_units(conn, paper_id) if needle in u.verbatim_text)


def _writer(conn, quote, text="It reaches 94.2% accuracy."):
    unit = _unit(conn, "p1", "94.2")
    return FakeWriter({RESULTS.question: Draft(
        text="Accuracy is high.",
        claims=(Claim("c-0", text, unit.unit_id, quote),))})


def test_a_section_draft_carries_its_section(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), ENTAILS)
    assert draft.section is RESULTS


def test_a_grounded_section_claim_is_supported(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), ENTAILS)
    assert len(draft.supported) == 1
    assert draft.blocked == ()


def test_a_fabricated_section_claim_is_blocked(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "99.9% tracking accuracy"), ENTAILS)
    assert len(draft.blocked) == 1
    assert draft.blocked[0].verdict is Verdict.QUOTE_NOT_FOUND
    assert draft.supported == ()


def test_a_real_quote_that_does_not_entail_is_flagged(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), NEUTRAL)
    assert len(draft.flagged) == 1
    assert draft.blocked == ()


def test_the_section_is_searched_on_its_own_sub_question(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["question"] = question
            return Draft()

    draft_section(corpus, RESULTS, FakeEmbedder(), SpyWriter(), ENTAILS)
    assert seen["question"] == RESULTS.question


def test_each_section_gets_its_own_capped_evidence(corpus):
    seen = {}

    class SpyWriter:
        def write(self, question, units):
            seen["count"] = len(units)
            return Draft()

    draft_section(corpus, RESULTS, FakeEmbedder(), SpyWriter(), ENTAILS,
                  limit=20, max_units=2)
    assert seen["count"] <= 2


def test_the_dropped_evidence_count_is_reported(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(), FakeWriter({}), ENTAILS,
                          limit=8, max_units=1)
    assert draft.dropped_evidence >= 0


def test_a_section_with_no_retrievable_evidence_drafts_nothing(corpus):
    empty = Section(title="Nothing", question="zzz nonexistent qqq topic")
    draft = draft_section(corpus, empty, FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert draft.claims == ()
    assert draft.text == ""


def test_section_draft_is_frozen(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(), FakeWriter({}), ENTAILS)
    with pytest.raises(FrozenInstanceError):
        draft.text = "rewritten"


def test_a_section_draft_records_the_units_it_saw(corpus):
    draft = draft_section(corpus, RESULTS, FakeEmbedder(),
                          _writer(corpus, "94.2% tracking accuracy"), ENTAILS)
    assert len(draft.units) > 0
    assert all(hasattr(u, "unit_id") for u in draft.units)



from jarvis.report import duplicate_claims, integrate

S1 = Section(title="One", question="q1")
S2 = Section(title="Two", question="q2")


def _draft(section, claims, verdicts=None):
    from jarvis.models import Verification
    verdicts = verdicts or [Verdict.SUPPORTED] * len(claims)
    return SectionDraft(
        section=section, text="t", claims=tuple(claims),
        verifications=tuple(Verification(claim_id=c.claim_id, unit_id=c.unit_id,
                                         quote_found=True, verdict=v)
                            for c, v in zip(claims, verdicts)))


def test_a_claim_repeated_across_sections_survives_only_in_the_first():
    shared = Claim("a-0", "The controller reaches 94.2%.", "u1", "94.2")
    later = Claim("b-0", "The controller reaches 94.2%.", "u1", "94.2")
    merged = integrate([_draft(S1, [shared]), _draft(S2, [later])])
    assert len(merged[0].claims) == 1
    assert merged[1].claims == ()


def test_dropping_a_duplicate_claim_drops_its_verification_too():
    shared = Claim("a-0", "same", "u1", "q")
    later = Claim("b-0", "same", "u1", "q")
    merged = integrate([_draft(S1, [shared]), _draft(S2, [later])])
    assert merged[1].verifications == ()


def test_the_same_unit_cited_for_two_different_points_keeps_both():
    merged = integrate([_draft(S1, [Claim("a-0", "It is accurate.", "u1", "q")]),
                        _draft(S2, [Claim("b-0", "It is fast.", "u1", "q")])])
    assert len(merged[0].claims) == 1
    assert len(merged[1].claims) == 1


def test_the_same_point_from_two_different_units_keeps_both():
    merged = integrate([_draft(S1, [Claim("a-0", "It is accurate.", "u1", "q")]),
                        _draft(S2, [Claim("b-0", "It is accurate.", "u2", "q")])])
    assert len(merged[1].claims) == 1


def test_duplicate_matching_ignores_whitespace_and_case():
    merged = integrate([_draft(S1, [Claim("a-0", "It  is Accurate.", "u1", "q")]),
                        _draft(S2, [Claim("b-0", "it is accurate.", "u1", "q")])])
    assert merged[1].claims == ()


def test_a_claim_repeated_inside_one_section_is_also_deduped():
    merged = integrate([_draft(S1, [Claim("a-0", "same", "u1", "q"),
                                    Claim("a-1", "same", "u1", "q")])])
    assert len(merged[0].claims) == 1


def test_integration_preserves_section_order_and_identity():
    merged = integrate([_draft(S1, []), _draft(S2, [])])
    assert [d.section.title for d in merged] == ["One", "Two"]


def test_duplicates_are_reportable_not_only_removed():
    shared = Claim("a-0", "same", "u1", "q")
    later = Claim("b-0", "same", "u1", "q")
    dupes = duplicate_claims([_draft(S1, [shared]), _draft(S2, [later])])
    assert dupes == [("Two", "b-0")]


def test_integrating_nothing_is_nothing():
    assert integrate([]) == []


from jarvis.models import Card, CardField
from jarvis.outline import Outline, TemplateOutliner
from jarvis.report import corpus_cards, evaluate_report, write_report
from jarvis.store import save_card


def test_a_report_covers_every_section_of_its_outline(corpus):
    outline = Outline(topic="gusts", sections=(
        Section(title="Results", question="what accuracy is reported?"),
        Section(title="Limits", question="what are the wind speed limits?"),
    ))
    result = write_report(corpus, "gusts", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert [s.section.title for s in result.sections] == ["Results", "Limits"]


def test_a_report_can_build_its_own_outline_from_cards(corpus):
    save_card(corpus, Card(paper_id="p1",
                           problem=CardField("gust rejection", "u1", "q"),
                           metrics=(CardField("94.2", "u2", "q"),)))
    result = write_report(corpus, "gusts", TemplateOutliner(), FakeEmbedder(),
                          FakeWriter({}), ENTAILS)
    assert len(result.sections) >= 2
    assert result.outline.topic == "gusts"


def test_corpus_cards_reads_every_deep_paper_that_has_one(corpus):
    save_card(corpus, Card(paper_id="p1", problem=CardField("a", "u1", "q")))
    cards = corpus_cards(corpus)
    assert [c.paper_id for c in cards] == ["p1"]


def test_coverage_is_the_cited_fraction_of_the_deep_corpus(corpus):
    unit = _unit(corpus, "p1", "94.2")
    question = "what accuracy is reported?"
    writer = FakeWriter({question: Draft(
        text="t", claims=(Claim("c-0", "94.2%", unit.unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="R", question=question),))

    result = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert 0.0 < result.coverage < 1.0
    assert result.corpus_units > 1


def test_coverage_is_zero_when_nothing_is_cited(corpus):
    outline = Outline(topic="gusts", sections=(Section(title="R", question="q"),))
    result = write_report(corpus, "gusts", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    assert result.coverage == 0.0


def test_a_blocked_claim_does_not_count_toward_coverage(corpus):
    unit = _unit(corpus, "p1", "94.2")
    question = "what accuracy is reported?"
    writer = FakeWriter({question: Draft(
        text="t", claims=(Claim("c-0", "99.9%", unit.unit_id, "99.9% tracking accuracy"),))})
    outline = Outline(topic="gusts", sections=(Section(title="R", question=question),))

    result = write_report(corpus, "gusts", outline, FakeEmbedder(), writer, ENTAILS)
    assert result.coverage == 0.0, "an ungrounded citation is not coverage"


def test_the_report_deduplicates_claims_across_sections(corpus):
    unit = _unit(corpus, "p1", "94.2")
    claim_text = "The controller reaches 94.2%."
    writer = FakeWriter({
        "q1": Draft(text="t", claims=(Claim("a-0", claim_text, unit.unit_id,
                                            "94.2% tracking accuracy"),)),
        "q2": Draft(text="t", claims=(Claim("b-0", claim_text, unit.unit_id,
                                            "94.2% tracking accuracy"),)),
    })
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),
                                           Section(title="B", question="q2")))
    result = write_report(corpus, "t", outline, FakeEmbedder(), writer, ENTAILS)
    assert len(result.all_claims) == 1


def test_the_report_aggregates_every_verification(corpus):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="t", claims=(
        Claim("a-0", "good", unit.unit_id, "94.2% tracking accuracy"),
        Claim("a-1", "bad", unit.unit_id, "99.9% tracking accuracy")))})
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),))

    result = write_report(corpus, "t", outline, FakeEmbedder(), writer, ENTAILS)
    assert len(result.all_verifications) == 2


def test_the_report_evaluates_like_any_other_answer(corpus):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="t", claims=(
        Claim("a-0", "good", unit.unit_id, "94.2% tracking accuracy"),
        Claim("a-1", "bad", unit.unit_id, "99.9% tracking accuracy")))})
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),))

    evaluation = evaluate_report(write_report(corpus, "t", outline, FakeEmbedder(),
                                              writer, ENTAILS))
    assert evaluation.quote_fidelity == pytest.approx(0.5)
    assert evaluation.meets_quote_target is False
    assert evaluation.coverage is not None


def test_cited_paper_ids_lists_only_papers_with_a_supported_claim(corpus):
    unit = _unit(corpus, "p1", "94.2")
    writer = FakeWriter({"q1": Draft(text="t", claims=(
        Claim("a-0", "good", unit.unit_id, "94.2% tracking accuracy"),))})
    outline = Outline(topic="t", sections=(Section(title="A", question="q1"),))
    result = write_report(corpus, "t", outline, FakeEmbedder(), writer, ENTAILS)
    assert result.cited_paper_ids == {"p1"}


def test_report_is_frozen(corpus):
    outline = Outline(topic="t", sections=())
    result = write_report(corpus, "t", outline, FakeEmbedder(), FakeWriter({}), ENTAILS)
    with pytest.raises(FrozenInstanceError):
        result.coverage = 1.0
