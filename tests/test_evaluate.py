import pytest

from jarvis.evaluate import (
    EvalReport,
    coverage,
    gate_recall,
    quote_fidelity,
    report,
    statement_support,
)
from jarvis.models import Verdict, Verification


def _v(claim_id, verdict, quote_found=True):
    return Verification(claim_id=claim_id, unit_id="u1", quote_found=quote_found,
                        verdict=verdict)


def test_quote_fidelity_is_one_when_every_quote_grounded():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.NEUTRAL)]
    assert quote_fidelity(vs) == 1.0


def test_quote_fidelity_drops_with_a_fabricated_quote():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.QUOTE_NOT_FOUND, quote_found=False)]
    assert quote_fidelity(vs) == 0.5


def test_quote_fidelity_of_nothing_is_one():
    assert quote_fidelity([]) == 1.0


def test_statement_support_counts_only_supported():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.NEUTRAL),
          _v("c", Verdict.CONTRADICTED), _v("d", Verdict.SUPPORTED)]
    assert statement_support(vs) == 0.5


def test_statement_support_of_nothing_is_zero():
    assert statement_support([]) == 0.0


def test_gate_recall_counts_relevant_papers_kept():
    decisions = {"p1": "read_deep", "p2": "unsure", "p3": "defer", "p4": "defer"}
    labels = {"p1": True, "p2": True, "p3": True, "p4": False}
    # 3 relevant; read_deep and unsure both count as kept -> 2/3
    assert gate_recall(decisions, labels) == pytest.approx(2 / 3)


def test_gate_recall_treats_unsure_as_kept():
    assert gate_recall({"p1": "unsure"}, {"p1": True}) == 1.0


def test_gate_recall_with_no_relevant_papers_is_one():
    assert gate_recall({"p1": "defer"}, {"p1": False}) == 1.0


def test_gate_recall_ignores_papers_without_labels():
    assert gate_recall({"p1": "defer", "p2": "read_deep"}, {"p2": True}) == 1.0


def test_coverage_is_fraction_of_corpus_cited():
    assert coverage({"u1", "u2"}, {"u1", "u2", "u3", "u4"}) == 0.5


def test_coverage_of_empty_corpus_is_zero():
    assert coverage(set(), set()) == 0.0


def test_coverage_ignores_citations_outside_the_corpus():
    assert coverage({"u1", "ghost"}, {"u1", "u2"}) == 0.5


def test_report_bundles_metrics_and_flags_targets():
    vs = [_v("a", Verdict.SUPPORTED), _v("b", Verdict.QUOTE_NOT_FOUND, quote_found=False)]
    r = report(vs, decisions={"p1": "read_deep"}, labels={"p1": True})
    assert isinstance(r, EvalReport)
    assert r.quote_fidelity == 0.5
    assert r.gate_recall == 1.0
    assert r.meets_quote_target is False   # target is 1.0
    assert r.meets_gate_target is True     # target is 0.95


def test_report_without_gate_data_leaves_gate_recall_none():
    r = report([_v("a", Verdict.SUPPORTED)])
    assert r.gate_recall is None
    assert r.meets_gate_target is None
