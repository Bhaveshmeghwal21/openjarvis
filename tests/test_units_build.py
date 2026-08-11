# tests/test_units_build.py
from jarvis.models import Block, ParsedPaper, UnitType
from jarvis.units import build_units


def _parsed(blocks):
    return ParsedPaper(paper_id="p1", blocks=tuple(blocks))


def _long_section(name, word, page):
    return Block(kind="paragraph", text=f"{word} " * 2000, page=page, section_path=(name,))


def test_every_child_has_a_parent():
    units = build_units(_parsed([_long_section("Methods", "alpha", 1)]))
    children = [u for u in units if u.parent_id is not None]
    parents = {u.unit_id for u in units if u.parent_id is None}
    assert children
    assert all(c.parent_id in parents for c in children)


def test_parent_holds_the_whole_section_text():
    units = build_units(_parsed([_long_section("Methods", "alpha", 1)]))
    parent = next(u for u in units if u.parent_id is None)
    child = next(u for u in units if u.parent_id is not None)
    assert len(parent.verbatim_text) > len(child.verbatim_text)
    assert child.verbatim_text.split()[0] in parent.verbatim_text


def test_children_of_different_sections_have_different_parents():
    units = build_units(_parsed([
        _long_section("Intro", "alpha", 1),
        _long_section("Methods", "beta", 2),
    ]))
    parents = {u.parent_id for u in units if u.parent_id is not None}
    assert len(parents) == 2


def test_artifact_units_are_included_and_are_never_split():
    table = Block(kind="table", text="| m | v |\n| acc | 94.2 |", page=3, label="Table 1")
    caption = Block(kind="caption", text="Table 1: Results.", page=3, label="Table 1")
    units = build_units(_parsed([table, caption]))
    tables = [u for u in units if u.type == UnitType.TABLE]
    assert len(tables) == 1
    assert "94.2" in tables[0].verbatim_text


def test_all_unit_ids_are_unique():
    units = build_units(_parsed([
        _long_section("Intro", "alpha", 1),
        Block(kind="table", text="| a |", page=2, label="Table 1"),
        _long_section("Methods", "beta", 3),
    ]))
    assert len({u.unit_id for u in units}) == len(units)


def test_build_is_deterministic():
    parsed = _parsed([_long_section("Methods", "alpha", 1)])
    assert [u.unit_id for u in build_units(parsed)] == [u.unit_id for u in build_units(parsed)]


def test_short_section_needs_no_parent():
    units = build_units(_parsed([
        Block(kind="paragraph", text="Just one line.", page=1, section_path=("Intro",)),
    ]))
    assert len(units) == 1
    assert units[0].parent_id is None


def test_empty_paper_builds_nothing():
    assert build_units(_parsed([])) == []
