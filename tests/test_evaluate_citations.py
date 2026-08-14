"""ALCE-style citation precision and recall (spec §10, tracked not targeted)."""
import pytest

from jarvis.evaluate import citation_precision, citation_recall, report
from jarvis.models import Verdict, Verification


def _v(claim_id: str, unit_id: str, verdict: Verdict) -> Verification:
    return Verification(claim_id=claim_id, unit_id=unit_id,
                        quote_found=verdict is not Verdict.QUOTE_NOT_FOUND, verdict=verdict)


def test_precision_is_the_fraction_of_citations_that_support():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c1", "u2", Verdict.NEUTRAL),
          _v("c2", "u3", Verdict.SUPPORTED), _v("c3", "u4", Verdict.QUOTE_NOT_FOUND)]
    assert citation_precision(vs) == pytest.approx(0.5)


def test_recall_is_the_fraction_of_claims_with_any_supporting_citation():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c1", "u2", Verdict.NEUTRAL),
          _v("c2", "u3", Verdict.SUPPORTED), _v("c3", "u4", Verdict.QUOTE_NOT_FOUND)]
    assert citation_recall(vs) == pytest.approx(2 / 3)


def test_the_two_metrics_diverge_on_an_over_cited_claim():
    vs = [_v("c1", "u1", Verdict.SUPPORTED)] + \
         [_v("c1", f"u{i}", Verdict.NEUTRAL) for i in range(2, 6)]
    assert citation_recall(vs) == 1.0
    assert citation_precision(vs) == pytest.approx(0.2)


def test_a_perfectly_cited_answer_scores_one_on_both():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c2", "u2", Verdict.SUPPORTED)]
    assert citation_precision(vs) == 1.0
    assert citation_recall(vs) == 1.0


def test_an_answer_with_no_claims_scores_zero_on_both():
    assert citation_precision([]) == 0.0
    assert citation_recall([]) == 0.0


def test_a_contradicted_citation_does_not_count_as_support():
    assert citation_precision([_v("c1", "u1", Verdict.CONTRADICTED)]) == 0.0
    assert citation_recall([_v("c1", "u1", Verdict.CONTRADICTED)]) == 0.0


def test_the_report_carries_both_new_metrics():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c2", "u2", Verdict.NEUTRAL)]
    r = report(vs)
    assert r.citation_precision == pytest.approx(0.5)
    assert r.citation_recall == pytest.approx(0.5)


def test_the_report_still_carries_the_original_metrics():
    vs = [_v("c1", "u1", Verdict.SUPPORTED), _v("c2", "u2", Verdict.QUOTE_NOT_FOUND)]
    r = report(vs)
    assert r.quote_fidelity == pytest.approx(0.5)
    assert r.statement_support == pytest.approx(0.5)
    assert r.meets_quote_target is False


def test_citation_metrics_have_no_target_only_a_number():
    r = report([_v("c1", "u1", Verdict.SUPPORTED)])
    assert not hasattr(r, "meets_citation_target"), "spec §10 tracks these, does not gate them"
