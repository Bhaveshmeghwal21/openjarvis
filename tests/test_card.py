"""Layer 2 — the paper card. An index over evidence, never a replacement for it."""
import pytest

from jarvis.card import (
    CardExtractor,
    FakeCardExtractor,
    LLMCardExtractor,
    extract_and_verify,
    unverified_fields,
    verify_card,
)
from jarvis.embed import FakeEmbedder
from jarvis.ingest import ingest_paper
from jarvis.models import Block, Card, CardField, Paper
from jarvis.parse import FakeParser
from jarvis.store import close_store, get_units, open_store

BLOCKS = [
    Block(kind="heading", text="Results", page=2, section_path=("Results",)),
    Block(kind="paragraph", text="Our controller reaches 94.2% accuracy on the KITTI set.",
          page=2, section_path=("Results",)),
]
PAPER = Paper(paper_id="p1", title="Gust-Robust Control", year=2025)


class _Router:
    def route(self, task: str) -> str:
        return "fake-model"


@pytest.fixture
def corpus(tmp_path):
    conn = open_store(tmp_path / "c.db")
    ingest_paper(conn, PAPER, "p.pdf", FakeParser(BLOCKS), FakeEmbedder())
    yield conn
    close_store(conn)


def test_fake_extractor_satisfies_the_protocol():
    assert isinstance(FakeCardExtractor({}), CardExtractor)


def test_a_real_quote_verifies_the_binding(corpus):
    unit = get_units(corpus, "p1")[0]
    card = Card(paper_id="p1",
                metrics=(CardField(value="94.2", unit_id=unit.unit_id,
                                   quote="reaches 94.2% accuracy"),))
    verified = verify_card(corpus, card)
    assert verified.metrics[0].binding_verified is True


def test_a_fabricated_quote_leaves_the_binding_unverified(corpus):
    unit = get_units(corpus, "p1")[0]
    card = Card(paper_id="p1",
                metrics=(CardField(value="99.9", unit_id=unit.unit_id,
                                   quote="reaches 99.9% accuracy"),))
    verified = verify_card(corpus, card)
    assert verified.metrics[0].binding_verified is False


def test_a_quote_pointing_at_a_nonexistent_unit_is_unverified(corpus):
    card = Card(paper_id="p1",
                metrics=(CardField(value="94.2", unit_id="nope", quote="94.2"),))
    assert verify_card(corpus, card).metrics[0].binding_verified is False


def test_every_field_kind_gets_verified_not_just_metrics(corpus):
    unit = get_units(corpus, "p1")[0]
    good = CardField(value="v", unit_id=unit.unit_id, quote="Our controller")
    bad = CardField(value="v", unit_id=unit.unit_id, quote="never written")
    card = Card(paper_id="p1", problem=good, method=bad, datasets=(good,), claims=(bad,))

    verified = verify_card(corpus, card)
    assert verified.problem.binding_verified is True
    assert verified.method.binding_verified is False
    assert verified.datasets[0].binding_verified is True
    assert verified.claims[0].binding_verified is False


def test_unverified_fields_are_surfaced_with_their_names(corpus):
    unit = get_units(corpus, "p1")[0]
    card = verify_card(corpus, Card(
        paper_id="p1",
        problem=CardField("a", unit.unit_id, "Our controller"),
        metrics=(CardField("b", unit.unit_id, "fabricated"),),
    ))
    names = [name for name, _ in unverified_fields(card)]
    assert names == ["metrics"]


def test_a_fully_verified_card_surfaces_nothing(corpus):
    unit = get_units(corpus, "p1")[0]
    card = verify_card(corpus, Card(paper_id="p1",
                                    problem=CardField("a", unit.unit_id, "Our controller")))
    assert unverified_fields(card) == []


def test_verification_never_deletes_an_unverified_field(corpus):
    unit = get_units(corpus, "p1")[0]
    card = verify_card(corpus, Card(
        paper_id="p1", metrics=(CardField("x", unit.unit_id, "fabricated"),)))
    assert len(card.metrics) == 1, "unverified is surfaced as such, never silently dropped"


def test_llm_extractor_builds_a_card_from_model_json(corpus):
    unit = get_units(corpus, "p1")[0]
    reply = {
        "problem": {"value": "gust rejection", "unit_id": unit.unit_id,
                    "quote": "Our controller"},
        "metrics": [{"value": "94.2", "unit_id": unit.unit_id,
                     "quote": "reaches 94.2% accuracy"}],
        "datasets": [{"value": "KITTI", "unit_id": unit.unit_id, "quote": "KITTI set"}],
    }
    card = LLMCardExtractor(_Router(), chat_fn=lambda *a, **k: reply).extract(
        PAPER, get_units(corpus, "p1"))
    assert card.problem.value == "gust rejection"
    assert card.metrics[0].value == "94.2"
    assert card.method is None


def test_llm_extractor_drops_fields_citing_a_unit_that_does_not_exist(corpus):
    reply = {"metrics": [{"value": "94.2", "unit_id": "hallucinated", "quote": "q"}]}
    card = LLMCardExtractor(_Router(), chat_fn=lambda *a, **k: reply).extract(
        PAPER, get_units(corpus, "p1"))
    assert card.metrics == ()


def test_llm_extractor_returns_an_empty_card_on_failure(corpus):
    def boom(*args, **kwargs):
        raise RuntimeError("no key")

    card = LLMCardExtractor(_Router(), chat_fn=boom).extract(PAPER, get_units(corpus, "p1"))
    assert card == Card(paper_id="p1")


def test_llm_extractor_routes_to_card_extraction(corpus):
    seen = {}

    def spy(router, task, prompt, **kwargs):
        seen["task"] = task
        return {}

    LLMCardExtractor(_Router(), chat_fn=spy).extract(PAPER, get_units(corpus, "p1"))
    assert seen["task"] == "card_extraction"


def test_extract_and_verify_persists_a_verified_card(corpus):
    from jarvis.store import get_card
    unit = get_units(corpus, "p1")[0]
    card = Card(paper_id="p1",
                metrics=(CardField("94.2", unit.unit_id, "reaches 94.2% accuracy"),))
    extract_and_verify(corpus, PAPER, FakeCardExtractor({"p1": card}))

    stored = get_card(corpus, "p1")
    assert stored.metrics[0].binding_verified is True
