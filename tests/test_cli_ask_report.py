"""Task 5: `jarvis ask` and `jarvis report`. Thin wrappers over ask()/write_report().

`report` must fail with a named error when `corpus_cards(conn)` is empty, rather than
emitting an empty report that looks like a report bug (design spec §5.2).
"""
from __future__ import annotations

from jarvis.card import extract_and_verify
from jarvis.cli import main
from jarvis.embed import FakeEmbedder
from jarvis.gather import Candidate, save_candidates, to_paper
from jarvis.ingest import ingest_paper
from jarvis.models import Card
from jarvis.parse import FakeParser
from jarvis.store import close_store, open_store


def _seed_deep_paper(db_path, *, with_card: bool):
    from jarvis.models import Block

    conn = open_store(db_path)
    candidate = Candidate(paper={
        "id": "p1", "arxiv_id": "p1", "title": "A Paper About Widgets",
        "abstract": "widgets are studied here", "pdf_url": "http://x/p1.pdf",
    })
    save_candidates(conn, [candidate])
    embedder = FakeEmbedder()
    blocks = [Block(kind="paragraph", text="Widgets are a well studied engineering topic.",
                    page=1, section_path=("Intro",))]
    result = ingest_paper(conn, to_paper(candidate), "fake-path", FakeParser(blocks),
                          embedder)
    assert result.ok
    if with_card:
        from jarvis.store import get_units
        units = get_units(conn, "p1")
        card = extract_and_verify(conn, to_paper(candidate), _CardStub(units))
        assert card is not None
    close_store(conn)


class _CardStub:
    """A CardExtractor double that anchors `problem` to a real unit + verbatim quote,
    so the card survives `verify_card`'s mechanical binding check."""

    def __init__(self, units):
        self._units = units

    def extract(self, paper, units):
        from jarvis.models import CardField
        first = self._units[0] if self._units else None
        if first is None:
            return Card(paper_id=paper.paper_id)
        quote = first.verbatim_text[:20]
        field = CardField(value=quote, unit_id=first.unit_id, quote=quote)
        return Card(paper_id=paper.paper_id, problem=field)


def test_ask_on_an_empty_corpus_renders_the_honest_no_evidence_answer(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriter())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: _FakeNLI())

    exit_code = main(["ask", "what about widgets?", "--project", "alpha"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "no evidence" in out.lower() or "no claim" in out.lower()


def test_ask_answers_from_a_seeded_corpus(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "beta" / "corpus.db"
    _seed_deep_paper(db_path, with_card=False)

    from jarvis import cli as cli_module
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriterWithClaim())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: _FakeNLI())

    exit_code = main(["ask", "what about widgets?", "--project", "beta"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.strip() != ""


def test_report_fails_loud_when_corpus_cards_is_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "gamma" / "corpus.db"
    _seed_deep_paper(db_path, with_card=False)  # deep paper, but no card extracted

    from jarvis import cli as cli_module
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriter())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: _FakeNLI())
    monkeypatch.setattr(cli_module, "build_outliner", lambda config, router: _TemplateOutliner())

    exit_code = main(["report", "widgets", "--project", "gamma"])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "card" in err.lower()


def test_report_succeeds_when_cards_exist(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "delta" / "corpus.db"
    _seed_deep_paper(db_path, with_card=True)

    from jarvis import cli as cli_module
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriter())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: _FakeNLI())
    monkeypatch.setattr(cli_module, "build_outliner", lambda config, router: _TemplateOutliner())

    exit_code = main(["report", "widgets", "--project", "delta"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "widgets" in out.lower()


def test_report_out_flag_writes_markdown_to_a_file(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "epsilon" / "corpus.db"
    _seed_deep_paper(db_path, with_card=True)

    from jarvis import cli as cli_module
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriter())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: _FakeNLI())
    monkeypatch.setattr(cli_module, "build_outliner", lambda config, router: _TemplateOutliner())

    out_file = tmp_path / "report.md"
    exit_code = main(["report", "widgets", "--project", "epsilon", "--out", str(out_file)])
    assert exit_code == 0
    assert out_file.is_file()
    assert "widgets" in out_file.read_text(encoding="utf-8").lower()


class _FakeWriter:
    def write(self, question, units):
        from jarvis.writer import Draft
        return Draft()


class _FakeWriterWithClaim:
    def write(self, question, units):
        from jarvis.models import Claim
        from jarvis.writer import Draft
        if not units:
            return Draft()
        unit = units[0]
        quote = unit.verbatim_text[:15]
        return Draft(text=f"Widgets are discussed. [{unit.unit_id}]",
                    claims=(Claim(claim_id="c-0", text="Widgets are discussed.",
                                  unit_id=unit.unit_id, quote=quote),))


class _FakeNLI:
    def predict(self, premise, hypothesis):
        return {"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05}


class _TemplateOutliner:
    def outline(self, topic, cards):
        from jarvis.outline import TemplateOutliner
        return TemplateOutliner().outline(topic, cards)
