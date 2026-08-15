"""Every subcommand that owns a ModelRouter must build it through `build_router(config)`,
never construct `ModelRouter(...)` directly -- a direct construction silently drops
`config.provider`/`config.provider_overrides`, so `JARVIS_PROVIDER` would have zero effect
on that subcommand even though it works everywhere else. Found live, in this codebase,
by exactly this gap: `cmd_gather`/`cmd_ask`/`cmd_report` each had their own direct
`ModelRouter(overrides=config.model_overrides)` construction that bypassed provider
routing entirely.
"""
from __future__ import annotations

from jarvis import cli as cli_module
from jarvis.cli import main
from jarvis.embed import FakeEmbedder
from jarvis.verify import FakeNLI


class _FakeWriter:
    def write(self, question, units):
        from jarvis.writer import Draft
        return Draft()


class _TemplateOutliner:
    def outline(self, topic, cards):
        from jarvis.outline import Outline
        return Outline(topic=topic, sections=())


def _spy_build_router(monkeypatch):
    """Replaces `build_router` with a spy that records every `Config` it was called with
    and returns a unique marker router, so a test can assert both that `build_router` (not
    a bare `ModelRouter(...)`) was used, and that the exact same router instance reached
    every downstream `build_*` call."""
    calls = []
    from jarvis.router import ModelRouter

    def spy(config):
        calls.append(config)
        return ModelRouter(overrides=config.model_overrides, provider=config.provider,
                           provider_overrides=config.provider_overrides)

    monkeypatch.setattr(cli_module, "build_router", spy)
    return calls


def test_ask_builds_its_router_through_build_router_not_directly(
    tmp_path, monkeypatch,
):
    calls = _spy_build_router(monkeypatch)
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriter())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: FakeNLI())
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("JARVIS_PROVIDER", "gcp")

    exit_code = main(["ask", "what about widgets?", "--project", "wiring-ask"])

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0].provider == "gcp"


def test_report_builds_its_router_through_build_router_not_directly(
    tmp_path, monkeypatch,
):
    from jarvis.card import extract_and_verify
    from jarvis.gather import Candidate, save_candidates, to_paper
    from jarvis.ingest import ingest_paper
    from jarvis.models import Block, CardField
    from jarvis.parse import FakeParser
    from jarvis.store import close_store, open_store

    db_path = tmp_path / "wiring-report" / "corpus.db"
    db_path.parent.mkdir(parents=True)
    conn = open_store(db_path)
    candidate = Candidate(paper={"id": "p1", "arxiv_id": "p1", "title": "Widgets",
                                 "abstract": "widgets", "pdf_url": "http://x/p1.pdf"})
    save_candidates(conn, [candidate])
    blocks = [Block(kind="paragraph", text="Widgets are studied here.", page=1,
                    section_path=("Intro",))]
    result = ingest_paper(conn, to_paper(candidate), "fake-path",
                          FakeParser(blocks), FakeEmbedder())
    assert result.ok

    class _CardStub:
        def extract(self, paper, units):
            first = units[0] if units else None
            field = (CardField(value="v", unit_id=first.unit_id, quote=first.verbatim_text)
                     if first else None)
            from jarvis.models import Card
            return Card(paper_id=paper.paper_id, problem=field)

    extract_and_verify(conn, to_paper(candidate), _CardStub())
    close_store(conn)

    calls = _spy_build_router(monkeypatch)
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_writer", lambda config, router: _FakeWriter())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: FakeNLI())
    monkeypatch.setattr(cli_module, "build_outliner",
                        lambda config, router: _TemplateOutliner())
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("JARVIS_PROVIDER", "azure")

    exit_code = main(["report", "widgets", "--project", "wiring-report"])

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0].provider == "azure"


def test_gather_builds_its_router_through_build_router_not_directly(
    tmp_path, monkeypatch,
):
    calls = _spy_build_router(monkeypatch)

    def fake_search(topic):
        return []

    monkeypatch.setattr(cli_module, "make_arxiv_search", lambda limit: fake_search)
    monkeypatch.setattr(cli_module, "make_s2_search", lambda limit: fake_search)
    monkeypatch.setattr(cli_module, "make_openalex_search",
                        lambda limit, mailto: fake_search)
    monkeypatch.setattr(cli_module, "make_crossref_search", lambda rows: fake_search)
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("JARVIS_PROVIDER", "gcp")

    exit_code = main(["gather", "widgets", "--project", "wiring-gather"])

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0].provider == "gcp"
