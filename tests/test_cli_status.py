"""Task 1: project resolution, store lifecycle, `jarvis status`."""
from __future__ import annotations

import pytest

from jarvis.cli import main, resolve_db_path
from jarvis.models import Paper, Unit, UnitType
from jarvis.store import close_store, open_store, save_paper, save_run, save_units


def test_resolve_db_path_uses_project_name(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    path = resolve_db_path(project="alpha", db=None)
    assert path == tmp_path / "alpha" / "corpus.db"


def test_resolve_db_path_db_override_wins_over_project(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    explicit = tmp_path / "elsewhere" / "custom.db"
    path = resolve_db_path(project="alpha", db=str(explicit))
    assert path == explicit


def test_resolve_db_path_requires_project_or_db():
    with pytest.raises(SystemExit):
        resolve_db_path(project=None, db=None)


def test_resolve_db_path_rejects_a_project_name_that_escapes_the_project_root(
    tmp_path, monkeypatch
):
    # --project is a name, not a path -- Config.project_dir does a plain Path.__truediv__
    # with no sanitization, so an unsanitized value here would let a hostile or
    # accidental --project "../../../etc" (or an absolute path) point every subsequent
    # subcommand's file I/O at an arbitrary location entirely outside
    # $JARVIS_PROJECT_ROOT. Found by a final whole-branch adversarial review;
    # reproduced independently before fixing.
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    with pytest.raises(SystemExit):
        resolve_db_path(project="../../../etc", db=None)


def test_resolve_db_path_rejects_an_absolute_path_as_a_project_name(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    escape_target = tmp_path.parent / "elsewhere"
    with pytest.raises(SystemExit):
        resolve_db_path(project=str(escape_target), db=None)


def test_resolve_db_path_accepts_an_ordinary_project_name(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    path = resolve_db_path(project="my-question", db=None)
    assert path == (tmp_path / "my-question" / "corpus.db").resolve()


def test_status_on_nonexistent_project_is_a_clean_named_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    # A nonexistent project should not raise a traceback -- open_store would happily
    # create an empty db, so "nonexistent" here means the parent project root itself
    # doesn't exist and isn't writable, simulated via a file in its place.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    exit_code = main(["status", "--project", "blocked/inner"])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "blocked" in err.lower() or "error" in err.lower()


def test_status_reports_counts_against_a_seeded_store(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "beta" / "corpus.db"
    conn = open_store(db_path)
    save_paper(conn, Paper(paper_id="p1", title="Paper One"), depth="deep")
    save_paper(conn, Paper(paper_id="p2", title="Paper Two"), depth="pending_deep")
    save_paper(conn, Paper(paper_id="p3", title="Paper Three"), depth="metadata")
    save_units(conn, [
        Unit(unit_id="p1:prose:1:0", paper_id="p1", type=UnitType.PROSE, page=1,
             section_path=(), verbatim_text="hello"),
    ])
    save_run(conn, "r1", question="what is x?", started_at="2026-01-01T00:00:00",
             cost_usd=1.23)
    close_store(conn)

    exit_code = main(["status", "--project", "beta"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "deep" in out and "1" in out
    assert "pending_deep" in out
    assert "metadata" in out
    assert "1.23" in out or "r1" in out


def test_status_reports_zero_counts_on_a_freshly_created_empty_project(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    exit_code = main(["status", "--project", "gamma"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0" in out


def test_store_is_closed_even_when_the_subcommand_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    closed = []
    original_close = cli_module.close_store

    def spy_close(conn):
        closed.append(conn)
        original_close(conn)

    def boom(conn, args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "close_store", spy_close)
    monkeypatch.setitem(cli_module.COMMANDS, "status", boom)

    exit_code = main(["status", "--project", "delta"])
    assert exit_code != 0
    assert len(closed) == 1
