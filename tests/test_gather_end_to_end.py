"""The proof this plan exists to produce: a question becomes a measured corpus."""
import pytest

from jarvis.card import FakeCardExtractor, extract_and_verify
from jarvis.embed import FakeEmbedder
from jarvis.evaluate import gate_recall
from jarvis.gate import FakeVoter, calibrate, calibration_report, score_signals, screen
from jarvis.gather import SearchPlan, gather, save_candidates
from jarvis.ingest import failed, ingest_decided
from jarvis.label import read_labels, sample_seed, write_label_sheet
from jarvis.models import Block, Card, CardField, Paper
from jarvis.parse import FakeParser
from jarvis.retrieve import search
from jarvis.store import close_store, get_papers_by_depth, get_screen_signals, open_store

QUESTION = "how do quadrotors reject wind gusts?"

RELEVANT = [
    {"arxiv_id": "r1", "title": "Gust rejection for quadrotors",
     "abstract": "Wind gusts disturb quadrotors; we reject them.", "year": 2025},
    {"arxiv_id": "r2", "title": "Wind disturbance attenuation in UAVs",
     "abstract": "Quadrotors reject wind using adaptive control.", "year": 2024},
]
IRRELEVANT = [
    {"arxiv_id": "n1", "title": "Protein folding", "abstract": "We fold proteins.",
     "year": 2025},
    {"arxiv_id": "n2", "title": "Compiler optimization", "abstract": "We optimize loops.",
     "year": 2023},
]
CITED = {"arxiv_id": "r3", "title": "Gust tolerance benchmarks",
         "abstract": "Benchmarks for quadrotor gust tolerance.", "year": 2023}

BLOCKS = [
    Block(kind="heading", text="Results", page=2, section_path=("Results",)),
    Block(kind="paragraph",
          text="As shown in Table 1, the controller holds 94.2% tracking accuracy in gusts.",
          page=2, section_path=("Results",)),
    Block(kind="table", text="| method | acc |\n|---|---|\n| ours | 94.2 |",
          page=2, section_path=("Results",), label="Table 1"),
    Block(kind="caption", text="Table 1: Tracking accuracy under wind.", page=2,
          section_path=("Results",), label="Table 1"),
]

LABELS = {"r1": True, "r2": True, "r3": True, "n1": False, "n2": False}


def search_fn(query: str) -> list[dict]:
    return [dict(p) for p in RELEVANT + IRRELEVANT]


def neighbors():
    return (lambda pid: [dict(CITED)] if pid == "r1" else [], lambda pid: [])


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "corpus.db")
    yield conn
    close_store(conn)


@pytest.fixture
def gathered():
    return gather(QUESTION, SearchPlan(question=QUESTION, queries=(QUESTION,)), search_fn,
                  neighbors=neighbors(), score_fn=lambda p: 1.0, max_depth=1)


def test_gathering_finds_the_searched_and_the_cited_papers(gathered):
    assert {c.pid for c in gathered} == {"r1", "r2", "n1", "n2", "r3"}
    assert next(c for c in gathered if c.pid == "r3").graph_depth == 1


def test_the_gate_keeps_every_hand_labelled_relevant_paper(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0, "r2": 1.0, "r3": 1.0}), run_id="run1")
    assert gate_recall(decisions, LABELS) >= 0.95


def test_every_decision_carries_its_four_signals(corpus, gathered):
    save_candidates(corpus, gathered)
    screen(corpus, gathered, QUESTION, FakeEmbedder(), run_id="run1")
    logged = get_screen_signals(corpus, "run1")
    assert set(logged) == {c.pid for c in gathered}
    assert all(set(v) == {"embedding", "graph", "keyword", "llm_vote"}
               for v in logged.values())


def test_calibration_from_labels_meets_the_target(corpus, gathered):
    embedder = FakeEmbedder()
    qvec = embedder.encode([QUESTION])[0]
    rows = {c.pid: score_signals(c, QUESTION, qvec, embedder,
                                 FakeVoter({"r1": 1.0, "r2": 1.0, "r3": 1.0}))
            for c in gathered}
    thresholds = calibrate(rows, LABELS)
    assert calibration_report(rows, LABELS, thresholds)["recall"] >= 0.95


def test_the_label_sheet_round_trips_through_a_file(tmp_path, gathered):
    path = tmp_path / "seed.jsonl"
    written = write_label_sheet(path, sample_seed(gathered, size=3))
    assert written == 3
    assert read_labels(path) == {}, "a fresh sheet is unlabelled by construction"


def test_kept_papers_are_deep_read_and_deferred_ones_are_not(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0, "r2": 1.0, "r3": 1.0}), run_id="run1")
    results = ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    assert failed(results) == []
    deep = {p.paper_id for p in get_papers_by_depth(corpus, "deep")}
    assert {"r1", "r2", "r3"} <= deep


def test_no_paper_is_ever_removed_from_the_corpus(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(), run_id="run1")
    ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    everywhere = {p.paper_id for depth in ("metadata", "pending_deep", "deep")
                  for p in get_papers_by_depth(corpus, depth)}
    assert everywhere == {c.pid for c in gathered}, "defer is demotion, never deletion"


def test_the_ingested_corpus_is_retrievable_and_the_card_is_verified(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0}), run_id="run1")
    ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    hits = search(corpus, "tracking accuracy under wind", FakeEmbedder(), limit=5)
    assert any("94.2" in u.verbatim_text for u in hits)

    unit = next(u for u in hits if "94.2" in u.verbatim_text)
    card = Card(paper_id=unit.paper_id,
                metrics=(CardField("94.2", unit.unit_id, "| ours | 94.2 |"),))
    verified = extract_and_verify(corpus, Paper(paper_id=unit.paper_id, title="T"),
                                  FakeCardExtractor({unit.paper_id: card}))
    assert verified.metrics[0].binding_verified is True


def test_a_fabricated_card_binding_is_caught_without_consulting_a_model(corpus, gathered):
    save_candidates(corpus, gathered)
    decisions = screen(corpus, gathered, QUESTION, FakeEmbedder(),
                       voter=FakeVoter({"r1": 1.0}), run_id="run1")
    ingest_decided(corpus, decisions, gathered, FakeParser(BLOCKS), FakeEmbedder())

    hits = search(corpus, "tracking accuracy", FakeEmbedder(), limit=5)
    unit = next(u for u in hits if "94.2" in u.verbatim_text)
    card = Card(paper_id=unit.paper_id,
                metrics=(CardField("99.9", unit.unit_id, "| ours | 99.9 |"),))
    verified = extract_and_verify(corpus, Paper(paper_id=unit.paper_id, title="T"),
                                  FakeCardExtractor({unit.paper_id: card}))
    assert verified.metrics[0].binding_verified is False
