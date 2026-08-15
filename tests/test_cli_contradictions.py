"""Task 6: `jarvis contradictions` and `jarvis review`.

Claims come from the most recent report (design spec §9), persisted as a small JSON
sidecar next to the report's own markdown -- there is no report-persistence layer
anywhere else in the codebase (`write_report` returns a `Report` held only in memory),
so `jarvis report` must be the one to leave something behind for `jarvis contradictions`
to read later.
"""
from __future__ import annotations

import json

from jarvis.cli import main
from jarvis.contradict import Conflict
from jarvis.models import Claim


def _write_claims_sidecar(path, claims):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "claims": [
            {"claim_id": c.claim_id, "text": c.text, "unit_id": c.unit_id, "quote": c.quote}
            for c in claims
        ]
    }), encoding="utf-8")


def test_contradictions_with_no_report_available_is_a_named_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    exit_code = main(["contradictions", "--project", "alpha"])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "report" in err.lower()


def test_contradictions_scans_claims_from_the_most_recent_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    claims = (Claim(claim_id="c-0", text="Widgets work well.", unit_id="p1:prose:1:0",
                    quote="widgets work well"),)
    reports_dir = tmp_path / "beta" / "reports"
    _write_claims_sidecar(reports_dir / "latest.json", claims)

    captured = {}

    def fake_scan_corpus(conn, claims_arg, nli, embedder, **kwargs):
        captured["claims"] = list(claims_arg)
        return []

    monkeypatch.setattr(cli_module, "scan_corpus", fake_scan_corpus)
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: object())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: object())

    exit_code = main(["contradictions", "--project", "beta"])
    assert exit_code == 0
    assert len(captured["claims"]) == 1
    assert captured["claims"][0].claim_id == "c-0"


def test_contradictions_from_report_flag_overrides_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    claims = (Claim(claim_id="c-0", text="Explicit report claim.", unit_id="p2:prose:1:0",
                    quote="explicit report claim"),)
    explicit_path = tmp_path / "elsewhere" / "custom.json"
    _write_claims_sidecar(explicit_path, claims)

    captured = {}

    def fake_scan_corpus(conn, claims_arg, nli, embedder, **kwargs):
        captured["claims"] = list(claims_arg)
        return []

    monkeypatch.setattr(cli_module, "scan_corpus", fake_scan_corpus)
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: object())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: object())

    exit_code = main(["contradictions", "--project", "gamma",
                      "--from-report", str(explicit_path)])
    assert exit_code == 0
    assert captured["claims"][0].text == "Explicit report claim."


def test_contradictions_writes_a_review_sheet(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis import cli as cli_module

    claims = (Claim(claim_id="c-0", text="X.", unit_id="p1:prose:1:0", quote="x"),)
    reports_dir = tmp_path / "delta" / "reports"
    _write_claims_sidecar(reports_dir / "latest.json", claims)

    conflict = Conflict(claim_id="c-0", claim_text="X.", claim_paper_id="p1",
                        unit_id="p2:prose:1:0", paper_id="p2", score=0.9,
                        evidence="not x")

    monkeypatch.setattr(cli_module, "scan_corpus", lambda *a, **kw: [conflict])
    monkeypatch.setattr(cli_module, "build_embedder", lambda config: object())
    monkeypatch.setattr(cli_module, "build_nli", lambda config: object())

    exit_code = main(["contradictions", "--project", "delta"])
    assert exit_code == 0

    sheet_path = tmp_path / "delta" / "reviews" / "contradictions.jsonl"
    assert sheet_path.is_file()
    lines = sheet_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["claim_id"] == "c-0"


def test_review_with_no_sheet_is_a_named_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    exit_code = main(["review", "--project", "epsilon", str(tmp_path / "nope.jsonl")])
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower() or "error" in err.lower()


def test_review_applies_verdicts_and_prints_precision(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis.store import open_store, save_contradictions

    db_path = tmp_path / "zeta" / "corpus.db"
    conn = open_store(db_path)
    save_contradictions(conn, [
        {"claim_id": "c-0", "unit_id": "u-1", "score": 0.9},
        {"claim_id": "c-1", "unit_id": "u-2", "score": 0.8},
    ], run_id="r1")
    conn.close()

    sheet = tmp_path / "sheet.jsonl"
    sheet.write_text(
        json.dumps({"claim_id": "c-0", "unit_id": "u-1", "verdict": True}) + "\n" +
        json.dumps({"claim_id": "c-1", "unit_id": "u-2", "verdict": False}) + "\n",
        encoding="utf-8",
    )

    exit_code = main(["review", "--project", "zeta", str(sheet)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0.5" in out or "50" in out


def test_review_reports_progress_on_a_partially_reviewed_sheet(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JARVIS_PROJECT_ROOT", str(tmp_path))
    from jarvis.store import open_store, save_contradictions

    db_path = tmp_path / "eta" / "corpus.db"
    conn = open_store(db_path)
    save_contradictions(conn, [
        {"claim_id": "c-0", "unit_id": "u-1", "score": 0.9},
        {"claim_id": "c-1", "unit_id": "u-2", "score": 0.8},
    ], run_id="r1")
    conn.close()

    sheet = tmp_path / "sheet.jsonl"
    sheet.write_text(
        json.dumps({"claim_id": "c-0", "unit_id": "u-1", "verdict": True}) + "\n" +
        json.dumps({"claim_id": "c-1", "unit_id": "u-2", "verdict": None}) + "\n",
        encoding="utf-8",
    )

    exit_code = main(["review", "--project", "eta", str(sheet)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "1" in out
