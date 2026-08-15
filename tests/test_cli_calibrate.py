"""Task 7: `jarvis calibrate` -- the gate's hand-label round trip (design spec §7B).

sample_seed -> write_label_sheet -> [human edits] -> read_labels -> calibrate ->
calibration_report, plus label_progress for a partially-completed sheet.
"""
from __future__ import annotations

import json

from jarvis.cli import main
from jarvis.gate import Signals
from jarvis.gather import Candidate, save_candidates
from jarvis.store import close_store, open_store, save_screen_decision, set_depth


def _seed_screened_papers(db_path, run_id="r1"):
    conn = open_store(db_path)
    candidates = [
        Candidate(paper={"id": f"p{i}", "arxiv_id": f"p{i}", "title": f"Paper {i}",
                         "abstract": f"about topic {i}"})
        for i in range(5)
    ]
    save_candidates(conn, candidates)
    for i, candidate in enumerate(candidates):
        signals = Signals(embedding=0.5 + i * 0.05, graph=0.0, keyword=0.4, llm_vote=0.0)
        save_screen_decision(conn, candidate.pid, "read_deep", signals.as_dict(),
                             run_id=run_id)
        set_depth(conn, candidate.pid, "pending_deep")
    close_store(conn)
    return candidates


def test_calibrate_writes_a_label_sheet_from_the_seed_set(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "alpha" / "corpus.db"
    _seed_screened_papers(db_path)

    exit_code = main(["calibrate", "--project", "alpha", "seed", "--run-id", "r1",
                      "--size", "3"])
    assert exit_code == 0

    sheet_path = tmp_path / "alpha" / "labels" / "seed.jsonl"
    assert sheet_path.is_file()
    lines = sheet_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert row["label"] is None
    assert "title" in row


def test_calibrate_seed_is_deterministic_across_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "beta" / "corpus.db"
    _seed_screened_papers(db_path)

    main(["calibrate", "--project", "beta", "seed", "--run-id", "r1", "--size", "3"])
    first = (tmp_path / "beta" / "labels" / "seed.jsonl").read_text(encoding="utf-8")

    (tmp_path / "beta" / "labels" / "seed.jsonl").unlink()
    main(["calibrate", "--project", "beta", "seed", "--run-id", "r1", "--size", "3"])
    second = (tmp_path / "beta" / "labels" / "seed.jsonl").read_text(encoding="utf-8")

    assert first == second


def test_calibrate_fit_reports_recall_and_precision_against_labels(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "gamma" / "corpus.db"
    candidates = _seed_screened_papers(db_path)

    labels_path = tmp_path / "gamma" / "labels" / "seed.jsonl"
    labels_path.parent.mkdir(parents=True)
    rows = [
        json.dumps({"paper_id": c.pid, "title": c.paper["title"], "label": i < 3})
        for i, c in enumerate(candidates)
    ]
    labels_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    exit_code = main(["calibrate", "--project", "gamma", "fit", "--run-id", "r1",
                      str(labels_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "recall" in out.lower()
    assert "precision" in out.lower()


def test_calibrate_fit_with_no_labeled_relevant_papers_is_a_named_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "delta" / "corpus.db"
    candidates = _seed_screened_papers(db_path)

    labels_path = tmp_path / "delta" / "labels" / "seed.jsonl"
    labels_path.parent.mkdir(parents=True)
    rows = [json.dumps({"paper_id": c.pid, "label": False}) for c in candidates]
    labels_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    exit_code = main(["calibrate", "--project", "delta", "fit", "--run-id", "r1",
                      str(labels_path)])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "relevant" in err.lower() or "label" in err.lower()


def test_calibrate_progress_reports_partial_completion(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    db_path = tmp_path / "epsilon" / "corpus.db"
    candidates = _seed_screened_papers(db_path)

    labels_path = tmp_path / "epsilon" / "labels" / "seed.jsonl"
    labels_path.parent.mkdir(parents=True)
    rows = [
        json.dumps({"paper_id": c.pid, "label": True if i == 0 else None})
        for i, c in enumerate(candidates)
    ]
    labels_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    exit_code = main(["calibrate", "--project", "epsilon", "progress", str(labels_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1" in out
    assert str(len(candidates)) in out
