"""Seed labeling — the ground truth calibration and gate_recall are both measured against."""
import json

import pytest

from jarvis.gather import Candidate
from jarvis.label import label_progress, read_labels, sample_seed, write_label_sheet


def _candidates(n):
    return [Candidate(paper={"arxiv_id": f"p{i}", "title": f"Paper {i}",
                             "abstract": f"Abstract {i}"}) for i in range(n)]


def test_sampling_is_deterministic_for_a_given_seed():
    cands = _candidates(50)
    assert [c.pid for c in sample_seed(cands, size=10, seed=7)] == \
           [c.pid for c in sample_seed(cands, size=10, seed=7)]


def test_a_different_seed_gives_a_different_sample():
    cands = _candidates(50)
    assert [c.pid for c in sample_seed(cands, size=10, seed=1)] != \
           [c.pid for c in sample_seed(cands, size=10, seed=2)]


def test_sampling_more_than_exists_returns_everything():
    cands = _candidates(5)
    assert len(sample_seed(cands, size=100)) == 5


def test_the_default_seed_size_matches_the_spec():
    assert len(sample_seed(_candidates(500))) == 100


def test_the_sheet_is_one_json_object_per_line_with_a_null_label(tmp_path):
    path = tmp_path / "seed.jsonl"
    assert write_label_sheet(path, _candidates(3)) == 3

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["paper_id"] == "p0"
    assert rows[0]["title"] == "Paper 0"
    assert rows[0]["abstract"] == "Abstract 0"
    assert rows[0]["label"] is None


def test_reading_an_unlabelled_sheet_yields_nothing(tmp_path):
    path = tmp_path / "seed.jsonl"
    write_label_sheet(path, _candidates(3))
    assert read_labels(path) == {}


def test_reading_a_labelled_sheet_yields_booleans(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"paper_id": "p0", "label": true}\n'
        '{"paper_id": "p1", "label": false}\n'
        '{"paper_id": "p2", "label": null}\n',
        encoding="utf-8")
    assert read_labels(path) == {"p0": True, "p1": False}


def test_common_hand_typed_label_spellings_are_accepted(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"paper_id": "a", "label": "yes"}\n'
        '{"paper_id": "b", "label": "no"}\n'
        '{"paper_id": "c", "label": 1}\n'
        '{"paper_id": "d", "label": 0}\n'
        '{"paper_id": "e", "label": "Y"}\n',
        encoding="utf-8")
    assert read_labels(path) == {"a": True, "b": False, "c": True, "d": False, "e": True}


def test_a_malformed_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text('{"paper_id": "a", "label": true}\nnot json\n\n'
                    '{"label": true}\n', encoding="utf-8")
    assert read_labels(path) == {"a": True}


def test_progress_reports_how_much_is_left(tmp_path):
    path = tmp_path / "seed.jsonl"
    path.write_text(
        '{"paper_id": "a", "label": true}\n'
        '{"paper_id": "b", "label": null}\n'
        '{"paper_id": "c", "label": false}\n',
        encoding="utf-8")
    progress = label_progress(path)
    assert progress == {"total": 3, "labeled": 2, "relevant": 1, "remaining": 1}


def test_progress_on_a_missing_file_is_all_zeros(tmp_path):
    assert label_progress(tmp_path / "nope.jsonl")["total"] == 0


def test_writing_a_sheet_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "seed.jsonl"
    write_label_sheet(path, _candidates(1))
    assert path.is_file()


def test_labels_feed_straight_into_gate_recall(tmp_path):
    from jarvis.evaluate import gate_recall
    path = tmp_path / "seed.jsonl"
    path.write_text('{"paper_id": "a", "label": true}\n'
                    '{"paper_id": "b", "label": true}\n', encoding="utf-8")
    assert gate_recall({"a": "read_deep", "b": "defer"}, read_labels(path)) == pytest.approx(0.5)
