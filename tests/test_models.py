# tests/test_models.py
import dataclasses

import pytest

from jarvis.models import (
    Block,
    Card,
    CardField,
    Claim,
    Paper,
    ParsedPaper,
    Unit,
    UnitType,
    Verdict,
    Verification,
)


def test_unit_types_cover_the_four_evidence_kinds():
    assert {t.value for t in UnitType} == {"prose", "table", "figure", "equation"}


def test_verdicts_include_quote_not_found():
    assert {v.value for v in Verdict} == {
        "supported", "contradicted", "neutral", "quote_not_found",
    }


def test_domain_types_are_frozen():
    for cls in (Block, ParsedPaper, Paper, Unit, CardField, Card, Claim, Verification):
        assert dataclasses.fields(cls) is not None
        assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


def test_paper_requires_only_id_and_title():
    p = Paper(paper_id="p1", title="T")
    assert p.year is None
    assert p.retracted is False
    assert p.authors == ()


def test_unit_key_is_deterministic_and_position_scoped():
    u = Unit(unit_id="", paper_id="p1", type=UnitType.PROSE, page=3,
             section_path=("Methods",), verbatim_text="x", ordinal=7)
    assert u.key() == "p1:prose:3:7"


def test_unit_is_immutable():
    u = Unit(unit_id="u", paper_id="p1", type=UnitType.PROSE, page=1,
             section_path=(), verbatim_text="x", ordinal=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.verbatim_text = "y"


def test_card_field_defaults_to_unverified_binding():
    f = CardField(value="94.2", unit_id="u1", quote="94.2% on KITTI")
    assert f.binding_verified is False


def test_card_holds_tuples_not_lists():
    c = Card(paper_id="p1", metrics=(CardField("94.2", "u1", "q"),))
    assert isinstance(c.metrics, tuple)
    assert c.datasets == ()
