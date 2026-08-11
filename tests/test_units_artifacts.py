from jarvis.models import Block, ParsedPaper, UnitType
from jarvis.units import build_artifact_units, find_references


def _parsed(blocks):
    return ParsedPaper(paper_id="p1", blocks=tuple(blocks))


TABLE = Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
              page=3, section_path=("Results",), label="Table 3")
CAPTION = Block(kind="caption", text="Table 3: Accuracy on KITTI.", page=3,
                section_path=("Results",), label="Table 3")
REFERRING = Block(kind="paragraph", text="As shown in Table 3, ours reaches 94.2%.",
                  page=2, section_path=("Results",))
UNRELATED = Block(kind="paragraph", text="Weather was mild.", page=1)


def test_table_unit_includes_markdown_caption_and_referring_text():
    units = build_artifact_units(_parsed([REFERRING, TABLE, CAPTION, UNRELATED]))
    table = next(u for u in units if u.type == UnitType.TABLE)
    assert "94.2" in table.verbatim_text            # the artifact
    assert "Accuracy on KITTI" in table.verbatim_text  # the caption
    assert "As shown in Table 3" in table.verbatim_text  # the referring prose
    assert "Weather was mild" not in table.verbatim_text


def test_table_unit_carries_label_and_page():
    units = build_artifact_units(_parsed([TABLE, CAPTION]))
    table = next(u for u in units if u.type == UnitType.TABLE)
    assert table.label == "Table 3"
    assert table.page == 3


def test_figure_unit_includes_caption_and_referring_text():
    fig = Block(kind="figure", text="", page=4, label="Figure 1")
    cap = Block(kind="caption", text="Figure 1: Architecture.", page=4, label="Figure 1")
    ref = Block(kind="paragraph", text="Figure 1 shows the encoder.", page=4)
    units = build_artifact_units(_parsed([fig, cap, ref]))
    figure = next(u for u in units if u.type == UnitType.FIGURE)
    assert "Architecture" in figure.verbatim_text
    assert "shows the encoder" in figure.verbatim_text


def test_equation_unit_includes_surrounding_prose():
    eq = Block(kind="equation", text="E = mc^2", page=5, section_path=("Theory",))
    before = Block(kind="paragraph", text="Energy is given by", page=5,
                   section_path=("Theory",))
    units = build_artifact_units(_parsed([before, eq]))
    equation = next(u for u in units if u.type == UnitType.EQUATION)
    assert "E = mc^2" in equation.verbatim_text
    assert "Energy is given by" in equation.verbatim_text


def test_artifact_without_caption_still_becomes_a_unit():
    units = build_artifact_units(_parsed([TABLE]))
    assert len(units) == 1
    assert "94.2" in units[0].verbatim_text


def test_find_references_matches_label_variants():
    blocks = [
        Block(kind="paragraph", text="see Table 3 for details"),
        Block(kind="paragraph", text="Tab. 3 confirms this"),
        Block(kind="paragraph", text="Table 30 is different"),
        Block(kind="paragraph", text="nothing here"),
    ]
    found = find_references(blocks, "Table 3")
    assert len(found) == 2
    assert all("30" not in f for f in found)


def test_find_references_with_empty_label_returns_nothing():
    assert find_references([Block(kind="paragraph", text="x")], "") == []


def test_ordinals_continue_from_start_ordinal():
    units = build_artifact_units(_parsed([TABLE, CAPTION]), start_ordinal=10)
    assert units[0].ordinal == 10
    assert units[0].unit_id == "p1:table:3:10"


def test_captions_are_not_emitted_as_standalone_units():
    units = build_artifact_units(_parsed([TABLE, CAPTION]))
    assert len(units) == 1


def test_find_references_matches_the_eq_abbreviation():
    # "Eq." doesn't share a 3-letter prefix with "Equation" the way "Tab."/"Fig." do
    # with their full words, so it needs an explicit override.
    blocks = [Block(kind="paragraph", text="Eq. 5 gives the result.")]
    assert find_references(blocks, "Equation 5") == ["Eq. 5 gives the result."]


def test_find_references_rejects_a_decimal_numbered_label():
    # "Table 3" must not match "Table 3.1" / "Table 3.2" / "Table 3.10" — those are
    # different, more specific tables, and folding their prose into Table 3's unit
    # would fabricate an association the paper never made.
    blocks = [
        Block(kind="paragraph", text="Table 3.1 shows the breakdown by class."),
        Block(kind="paragraph", text="Table 3.2 provides more detail."),
        Block(kind="paragraph", text="See Table 3.10 for the ablation."),
        Block(kind="paragraph", text="Table 3 confirms this."),
    ]
    found = find_references(blocks, "Table 3")
    assert found == ["Table 3 confirms this."]


def test_labeled_artifact_with_no_text_caption_or_references_still_becomes_a_unit():
    figure = Block(kind="figure", text="", page=4, label="Figure 7")
    units = build_artifact_units(_parsed([figure]))
    assert len(units) == 1
    assert units[0].label == "Figure 7"
    assert units[0].verbatim_text  # never empty — falls back to the label
