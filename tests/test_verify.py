import pytest

from jarvis.models import Claim, Paper, Unit, UnitType, Verdict
from jarvis.store import close_store, open_store, save_paper, save_units
from jarvis.verify import FakeNLI, find_contradictions, quote_is_grounded, verify_claim

QUOTE = "our method reaches 94.2% on KITTI"
RAW = f"In Section 4 we show that {QUOTE} without extra supervision."


@pytest.fixture
def conn():
    c = open_store(":memory:")
    save_paper(c, Paper("p1", "T"), raw_text=RAW)
    save_units(c, [Unit(unit_id="u1", paper_id="p1", type=UnitType.PROSE, page=1,
                        section_path=("Results",), verbatim_text=RAW, ordinal=0)])
    yield c
    close_store(c)


def _claim(text="The method reaches 94.2% on KITTI.", quote=QUOTE):
    return Claim(claim_id="c1", text=text, unit_id="u1", quote=quote)


# --- stage 1: quote grounding ------------------------------------------------

def test_real_quote_is_grounded(conn):
    assert quote_is_grounded(conn, _claim()) is True


def test_fabricated_quote_is_not_grounded(conn):
    assert quote_is_grounded(conn, _claim(quote="reaches 99.9% on KITTI")) is False


def test_quote_grounded_despite_pdf_artifacts(conn):
    save_paper(conn, Paper("p2", "T"), raw_text="distur-\nbance   rejection works")
    save_units(conn, [Unit(unit_id="u2", paper_id="p2", type=UnitType.PROSE, page=1,
                           section_path=(), verbatim_text="x", ordinal=0)])
    claim = Claim(claim_id="c", text="t", unit_id="u2", quote="disturbance rejection")
    assert quote_is_grounded(conn, claim) is True


def test_quote_against_missing_unit_is_not_grounded(conn):
    assert quote_is_grounded(conn, Claim("c", "t", "nope", QUOTE)) is False


def test_empty_quote_is_not_grounded(conn):
    assert quote_is_grounded(conn, _claim(quote="")) is False


# --- stage 2: entailment -----------------------------------------------------

def test_supported_claim_gets_supported_verdict(conn):
    nli = FakeNLI({(QUOTE, _claim().text): {"entailment": 0.95, "neutral": 0.03,
                                            "contradiction": 0.02}})
    result = verify_claim(conn, _claim(), nli)
    assert result.verdict == Verdict.SUPPORTED
    assert result.quote_found is True
    assert result.entailment_score == pytest.approx(0.95)


def test_unsupported_claim_gets_neutral_verdict(conn):
    nli = FakeNLI(default={"entailment": 0.10, "neutral": 0.85, "contradiction": 0.05})
    assert verify_claim(conn, _claim(), nli).verdict == Verdict.NEUTRAL


def test_contradicted_claim_gets_contradicted_verdict(conn):
    nli = FakeNLI(default={"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85})
    result = verify_claim(conn, _claim(), nli)
    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradiction_score == pytest.approx(0.85)


def test_fabricated_quote_short_circuits_before_nli(conn):
    """A missing quote blocks the claim; the NLI model must never be consulted."""
    calls = []

    class SpyNLI:
        def predict(self, premise, hypothesis):
            calls.append((premise, hypothesis))
            return {"entailment": 1.0, "neutral": 0.0, "contradiction": 0.0}

    result = verify_claim(conn, _claim(quote="reaches 99.9%"), SpyNLI())
    assert result.verdict == Verdict.QUOTE_NOT_FOUND
    assert result.quote_found is False
    assert calls == []


def test_verification_carries_claim_and_unit_ids(conn):
    nli = FakeNLI(default={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05})
    result = verify_claim(conn, _claim(), nli)
    assert result.claim_id == "c1"
    assert result.unit_id == "u1"


def test_threshold_is_respected(conn):
    nli = FakeNLI(default={"entailment": 0.60, "neutral": 0.35, "contradiction": 0.05})
    assert verify_claim(conn, _claim(), nli, threshold=0.5).verdict == Verdict.SUPPORTED
    assert verify_claim(conn, _claim(), nli, threshold=0.9).verdict == Verdict.NEUTRAL


# --- contradiction detection (spec §8, same pass) ----------------------------

def test_find_contradictions_returns_conflicting_units(conn):
    other = Unit(unit_id="u9", paper_id="p1", type=UnitType.PROSE, page=2,
                 section_path=("Results",), verbatim_text="the method reaches 71% on KITTI",
                 ordinal=1)
    save_units(conn, [other])
    nli = FakeNLI(default={"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85})
    found = find_contradictions(conn, _claim(), [other], nli)
    assert found[0][0] == "u9"
    assert found[0][1] == pytest.approx(0.85)


def test_find_contradictions_ignores_agreeing_units(conn):
    other = Unit(unit_id="u9", paper_id="p1", type=UnitType.PROSE, page=2,
                 section_path=(), verbatim_text="also 94.2%", ordinal=1)
    save_units(conn, [other])
    nli = FakeNLI(default={"entailment": 0.90, "neutral": 0.05, "contradiction": 0.05})
    assert find_contradictions(conn, _claim(), [other], nli) == []


def test_find_contradictions_sorts_by_confidence(conn):
    a = Unit(unit_id="ua", paper_id="p1", type=UnitType.PROSE, page=2,
             section_path=(), verbatim_text="A", ordinal=1)
    b = Unit(unit_id="ub", paper_id="p1", type=UnitType.PROSE, page=3,
             section_path=(), verbatim_text="B", ordinal=2)
    save_units(conn, [a, b])
    nli = FakeNLI({
        ("A", _claim().text): {"entailment": 0.0, "neutral": 0.3, "contradiction": 0.7},
        ("B", _claim().text): {"entailment": 0.0, "neutral": 0.1, "contradiction": 0.9},
    })
    found = find_contradictions(conn, _claim(), [a, b], nli)
    assert [uid for uid, _ in found] == ["ub", "ua"]
