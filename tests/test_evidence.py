"""The evidence budget. More retrieval makes answers look better and be worse (spec §7D)."""
from dataclasses import FrozenInstanceError

import pytest

from jarvis.evidence import (
    MAX_TOKENS,
    MAX_UNITS,
    EvidenceSet,
    cap,
    order_for_context,
    render,
)
from jarvis.models import Unit, UnitType


def _unit(i: int, text: str = "evidence") -> Unit:
    return Unit(unit_id=f"u{i}", paper_id="p1", type=UnitType.PROSE, page=1,
                section_path=("Results",), verbatim_text=text, ordinal=i)


def test_the_cap_is_a_hard_number_not_a_suggestion():
    result = cap([_unit(i) for i in range(50)])
    assert len(result.units) == MAX_UNITS
    assert result.dropped == 50 - MAX_UNITS


def test_a_small_set_passes_through_untouched():
    units = [_unit(i) for i in range(3)]
    result = cap(units)
    assert list(result.units) == units
    assert result.dropped == 0


def test_the_token_budget_cuts_before_the_unit_count_when_units_are_long():
    long_units = [_unit(i, "word " * 3000) for i in range(MAX_UNITS)]
    result = cap(long_units)
    assert len(result.units) < MAX_UNITS
    assert result.tokens <= MAX_TOKENS


def test_a_single_oversized_unit_is_still_included():
    result = cap([_unit(0, "word " * 100000)])
    assert len(result.units) == 1, "never return an empty evidence set for a real hit"


def test_capping_preserves_rank_order():
    units = [_unit(i) for i in range(20)]
    assert [u.unit_id for u in cap(units).units] == [f"u{i}" for i in range(MAX_UNITS)]


def test_capping_an_empty_list_is_empty():
    result = cap([])
    assert result.units == ()
    assert result.dropped == 0


def test_the_strongest_evidence_lands_at_both_ends():
    units = [_unit(i) for i in range(5)]        # ranked best-first
    ordered = order_for_context(units)
    assert ordered[0].unit_id == "u0", "best evidence first (primacy)"
    assert ordered[-1].unit_id == "u1", "second-best evidence last (recency)"
    assert ordered[len(ordered) // 2].unit_id == "u4", "weakest evidence in the middle"


def test_ordering_keeps_every_unit_exactly_once():
    units = [_unit(i) for i in range(9)]
    ordered = order_for_context(units)
    assert len(ordered) == 9
    assert {u.unit_id for u in ordered} == {u.unit_id for u in units}


def test_ordering_handles_one_and_zero_units():
    assert order_for_context([]) == []
    assert [u.unit_id for u in order_for_context([_unit(0)])] == ["u0"]


def test_rendering_labels_every_block_with_its_unit_id():
    text = render([_unit(0, "the controller reaches 94.2%"), _unit(1, "under gusts")])
    assert "[u0]" in text
    assert "[u1]" in text
    assert "94.2" in text


def test_rendering_includes_the_contextual_prefix_when_there_is_one():
    unit = Unit(unit_id="u9", paper_id="p1", type=UnitType.TABLE, page=3,
                section_path=("Results",), verbatim_text="| ours | 94.2 |",
                ordinal=9, context_prefix="From \"Gust-Robust Control\", Table 3.")
    text = render([unit])
    assert "Gust-Robust Control" in text
    assert "| ours | 94.2 |" in text


def test_rendering_nothing_is_the_empty_string():
    assert render([]) == ""


def test_evidence_set_is_frozen():
    with pytest.raises(FrozenInstanceError):
        EvidenceSet(units=()).dropped = 3
