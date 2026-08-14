from jarvis.models import Block, ParsedPaper, UnitType
from jarvis.text import approx_tokens
from jarvis.units import build_prose_units


def _parsed(blocks):
    return ParsedPaper(paper_id="p1", blocks=tuple(blocks))


def test_short_section_becomes_one_unit():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="A short paragraph.", page=1, section_path=("Intro",)),
    ]))
    assert len(units) == 1
    assert units[0].type == UnitType.PROSE
    assert units[0].section_path == ("Intro",)


def test_units_never_span_two_sections():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="alpha " * 10, page=1, section_path=("Intro",)),
        Block(kind="paragraph", text="beta " * 10, page=2, section_path=("Methods",)),
    ]))
    sections = {u.section_path for u in units}
    assert sections == {("Intro",), ("Methods",)}
    for u in units:
        assert not ("alpha" in u.verbatim_text and "beta" in u.verbatim_text)


def test_long_section_is_split_under_the_token_budget():
    long_text = "word " * 2000
    units = build_prose_units(
        _parsed([Block(kind="paragraph", text=long_text, page=1, section_path=("Methods",))]),
        max_tokens=512,
    )
    assert len(units) > 1
    for u in units:
        assert approx_tokens(u.verbatim_text) <= 512


def test_split_preserves_all_content():
    text = " ".join(f"w{i}" for i in range(3000))
    units = build_prose_units(
        _parsed([Block(kind="paragraph", text=text, page=1, section_path=("M",))]),
        max_tokens=512,
    )
    joined = " ".join(u.verbatim_text for u in units)
    assert "w0" in joined and "w2999" in joined


def test_non_prose_blocks_are_ignored_here():
    units = build_prose_units(_parsed([
        Block(kind="table", text="| a |", page=1),
        Block(kind="figure", text="", page=1),
        Block(kind="heading", text="Methods", page=1),
    ]))
    assert units == []


def test_units_get_deterministic_ids_and_ordinals():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="one", page=1, section_path=("A",)),
        Block(kind="paragraph", text="two", page=2, section_path=("B",)),
    ]))
    assert [u.ordinal for u in units] == [0, 1]
    assert units[0].unit_id == "p1:prose:1:0"
    assert len({u.unit_id for u in units}) == 2


def test_rebuilding_the_same_parse_yields_identical_ids():
    parsed = _parsed([Block(kind="paragraph", text="x", page=1, section_path=("A",))])
    assert [u.unit_id for u in build_prose_units(parsed)] == \
           [u.unit_id for u in build_prose_units(parsed)]


def test_verbatim_text_is_normalized():
    units = build_prose_units(_parsed([
        Block(kind="paragraph", text="distur-\nbance   rejection", page=1),
    ]))
    assert units[0].verbatim_text == "disturbance rejection"


def test_empty_paper_yields_no_units():
    assert build_prose_units(_parsed([])) == []
