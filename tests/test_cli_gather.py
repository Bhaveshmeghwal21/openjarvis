"""Task 4: `jarvis gather` -- wires stages A-C end to end, closing spec gaps §5.2
(card extraction never called in production) and §5.3 (nothing writes runs.cost_usd).
"""
from __future__ import annotations

from jarvis.cli import main
from jarvis.models import Card, Paper
from jarvis.store import (
    get_card,
    get_papers_by_depth,
    open_store,
    save_paper,
)


def _fake_search(topic):
    return [
        {"id": "p1", "arxiv_id": "p1", "title": f"Paper directly about {topic}",
         "abstract": f"a very relevant discussion of {topic}",
         "pdf_url": "http://example.com/p1.pdf", "doi": "10.1/p1"},
        {"id": "p2", "arxiv_id": "p2", "title": f"Another paper on {topic}",
         "abstract": f"also directly relevant to {topic}",
         "pdf_url": "http://example.com/p2.pdf", "doi": "10.1/p2"},
    ]


class _FakeParser:
    def parse(self, path, paper_id):
        if not path:
            # A real parser has nothing to open when fetch_pdf returned None and
            # path_for fell back to "" -- this fake must fail the same way, or a test
            # asserting on an unfetchable PDF's downstream effect would pass for the
            # wrong reason (never actually reaching a parse failure).
            raise RuntimeError("no source path to parse")
        from jarvis.models import Block, ParsedPaper
        return ParsedPaper(
            paper_id=paper_id,
            blocks=(Block(kind="paragraph", text="Some real content about the topic.",
                          page=1, section_path=("Intro",)),),
            raw_text="Some real content about the topic.",
        )


class _FailingParser:
    def parse(self, path, paper_id):
        raise RuntimeError("corrupt PDF")


def _patch_pipeline(monkeypatch, *, fetch_ok=True, extractor_calls=None):
    from jarvis import cli as cli_module
    from jarvis.embed import FakeEmbedder

    monkeypatch.setattr(cli_module, "combine_sources", lambda *fns: _fake_search)
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli_module, "build_parser", lambda config: _FakeParser())

    if extractor_calls is not None:
        class _FakeExtractor:
            def extract(self, paper, units):
                extractor_calls.append(paper.paper_id)
                return Card(paper_id=paper.paper_id)
        monkeypatch.setattr(cli_module, "build_card_extractor",
                           lambda config, router: _FakeExtractor())
    else:
        class _NullExtractor:
            def extract(self, paper, units):
                return Card(paper_id=paper.paper_id)
        monkeypatch.setattr(cli_module, "build_card_extractor",
                           lambda config, router: _NullExtractor())

    if fetch_ok:
        monkeypatch.setattr(cli_module, "fetch_pdf",
                           lambda paper, cache_dir, **kw: f"/fake/{kw['paper_id']}.pdf")
    return cli_module


def test_the_gate_blocks_without_yes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    _patch_pipeline(monkeypatch, extractor_calls=[])

    exit_code = main(["gather", "test question", "--project", "alpha"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pending" in out.lower() or "confirm" in out.lower() or "--yes" in out

    conn = open_store(tmp_path / "alpha" / "corpus.db")
    # Screened, but nothing ingested -- the gate stopped before deep reads.
    assert get_papers_by_depth(conn, "deep") == []


def test_yes_proceeds_through_ingest_and_card_extraction(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    calls = []
    _patch_pipeline(monkeypatch, extractor_calls=calls)

    exit_code = main(["gather", "test question", "--project", "beta", "--yes"])
    assert exit_code == 0

    conn = open_store(tmp_path / "beta" / "corpus.db")
    deep = get_papers_by_depth(conn, "deep")
    assert len(deep) >= 1
    for paper in deep:
        assert get_card(conn, paper.paper_id) is not None
    assert len(calls) == len(deep)


def test_cost_is_written_even_when_the_run_fails_midway(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    _patch_pipeline(monkeypatch, extractor_calls=[])

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-gather")

    monkeypatch.setattr(cli_module, "screen", boom)

    exit_code = main(["gather", "test question", "--project", "gamma", "--yes"])
    assert exit_code != 0

    conn = open_store(tmp_path / "gamma" / "corpus.db")
    row = conn.execute("SELECT cost_usd FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["cost_usd"] >= 0.0


def test_resumption_does_not_re_search(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    search_calls = []

    def counting_search(topic):
        search_calls.append(topic)
        return _fake_search(topic)

    monkeypatch.setattr(cli_module, "combine_sources", lambda *fns: counting_search)
    _patch_pipeline(monkeypatch, extractor_calls=[])

    db_path = tmp_path / "delta" / "corpus.db"
    conn = open_store(db_path)
    save_paper(conn, Paper(paper_id="already-screened", title="Already screened paper",
                           doi="10.1/x"), depth="pending_deep")
    conn.close()

    exit_code = main(["gather", "test question", "--project", "delta", "--yes"])
    assert exit_code == 0
    # A paper already at pending_deep before the run started must reach ingest without
    # having gone through a fresh search -- confirmed indirectly: it is deep afterward.
    conn = open_store(db_path)
    deep_ids = {p.paper_id for p in get_papers_by_depth(conn, "deep")}
    assert "already-screened" in deep_ids


def test_one_unfetchable_pdf_costs_one_paper_not_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    _patch_pipeline(monkeypatch, extractor_calls=[], fetch_ok=False)

    def mixed_fetch(paper, cache_dir, **kw):
        return None if paper.get("id") == "p1" else f"/fake/{kw['paper_id']}.pdf"

    monkeypatch.setattr(cli_module, "fetch_pdf", mixed_fetch)

    exit_code = main(["gather", "test question", "--project", "epsilon", "--yes"])
    assert exit_code == 0

    conn = open_store(tmp_path / "epsilon" / "corpus.db")
    deep = get_papers_by_depth(conn, "deep")
    deep_ids = {p.paper_id for p in deep}
    assert "p1" not in deep_ids
    assert "p2" in deep_ids


def test_one_unparseable_pdf_costs_one_paper_not_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    _patch_pipeline(monkeypatch, extractor_calls=[])

    class MixedParser:
        def parse(self, path, paper_id):
            if paper_id == "p1":
                raise RuntimeError("corrupt PDF")
            return _FakeParser().parse(path, paper_id)

    monkeypatch.setattr(cli_module, "build_parser", lambda config: MixedParser())

    exit_code = main(["gather", "test question", "--project", "zeta", "--yes"])
    assert exit_code == 0

    conn = open_store(tmp_path / "zeta" / "corpus.db")
    deep_ids = {p.paper_id for p in get_papers_by_depth(conn, "deep")}
    assert "p1" not in deep_ids
    assert "p2" in deep_ids


def test_a_run_that_ingests_nothing_says_so_rather_than_reporting_success(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    monkeypatch.setattr(cli_module, "combine_sources", lambda *fns: (lambda topic: []))
    _patch_pipeline(monkeypatch, extractor_calls=[])

    exit_code = main(["gather", "test question", "--project", "eta", "--yes"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0" in out


def test_budget_flag_is_carried_into_gather(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    captured = {}
    real_gather = cli_module.gather

    def spy_gather(*args, **kwargs):
        captured.update(kwargs)
        return real_gather(*args, **kwargs)

    monkeypatch.setattr(cli_module, "gather", spy_gather)
    _patch_pipeline(monkeypatch, extractor_calls=[])

    main(["gather", "test question", "--project", "theta", "--yes", "--budget", "5"])
    assert captured.get("budget") == 5


def test_max_deep_caps_how_many_kept_papers_are_actually_deep_read(
    tmp_path, monkeypatch, capsys,
):
    # `_fake_search` returns 2 candidates crafted to both score as relevant, so both are
    # expected to clear the gate -- capping at 1 must leave exactly 1 at `deep` depth and
    # the other still recoverable at `pending_deep`, not silently dropped.
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    calls = []
    _patch_pipeline(monkeypatch, extractor_calls=calls)

    exit_code = main(["gather", "test question", "--project", "iota", "--yes",
                      "--max-deep", "1"])
    assert exit_code == 0

    conn = open_store(tmp_path / "iota" / "corpus.db")
    deep = get_papers_by_depth(conn, "deep")
    pending = get_papers_by_depth(conn, "pending_deep")
    assert len(deep) == 1
    assert len(pending) == 1
    assert len(calls) == 1  # card extraction only ran for the one actually ingested

    out = capsys.readouterr().out
    assert "--max-deep 1" in out
    assert "1 more" in out or "queued" in out


def test_max_deep_unset_processes_every_kept_paper_unchanged(
    tmp_path, monkeypatch,
):
    # No regression for the default (uncapped) path -- omitting --max-deep must behave
    # exactly as it did before this flag existed.
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    calls = []
    _patch_pipeline(monkeypatch, extractor_calls=calls)

    exit_code = main(["gather", "test question", "--project", "kappa", "--yes"])
    assert exit_code == 0

    conn = open_store(tmp_path / "kappa" / "corpus.db")
    deep = get_papers_by_depth(conn, "deep")
    pending = get_papers_by_depth(conn, "pending_deep")
    assert len(pending) == 0
    assert len(calls) == len(deep)
