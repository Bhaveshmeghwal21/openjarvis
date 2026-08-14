"""Normalizers for the three sources spec §7A names alongside Crossref."""
from jarvis.sources import (
    PAPER_FIELDS,
    normalize_arxiv_entry,
    normalize_openalex,
    normalize_s2,
    openalex_abstract,
)

S2_ITEM = {
    "paperId": "abc123",
    "title": "Gust-Robust Quadrotor Control",
    "abstract": "We reject gusts.",
    "year": 2025,
    "venue": "ICRA",
    "citationCount": 42,
    "externalIds": {"ArXiv": "2501.00001", "DOI": "10.1/xyz"},
    "openAccessPdf": {"url": "https://example.org/p.pdf"},
    "fieldsOfStudy": ["Engineering"],
}

OPENALEX_ITEM = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/xyz",
    "title": "Gust-Robust Quadrotor Control",
    "publication_year": 2025,
    "cited_by_count": 42,
    "host_venue": {"display_name": "ICRA"},
    "authorships": [{"author": {"display_name": "A. Researcher"}}],
    "abstract_inverted_index": {"We": [0], "reject": [1], "gusts": [2]},
    "open_access": {"oa_url": "https://example.org/p.pdf"},
    "concepts": [{"display_name": "Control theory"}],
}

ARXIV_ENTRY = {
    "id": "http://arxiv.org/abs/2501.00001v2",
    "title": "Gust-Robust\n  Quadrotor Control",
    "summary": "We reject\n  gusts.",
    "published": "2025-01-03T00:00:00Z",
    "authors": ["A. Researcher", "B. Engineer"],
    "categories": ["cs.RO", "eess.SY"],
    "doi": "10.1/xyz",
}


def _has_contract(paper: dict) -> bool:
    return set(paper) == set(PAPER_FIELDS)


def test_every_normalizer_returns_exactly_the_common_contract():
    assert _has_contract(normalize_s2(S2_ITEM))
    assert _has_contract(normalize_openalex(OPENALEX_ITEM))
    assert _has_contract(normalize_arxiv_entry(ARXIV_ENTRY))


def test_s2_lifts_arxiv_and_doi_out_of_external_ids():
    p = normalize_s2(S2_ITEM)
    assert p["arxiv_id"] == "2501.00001"
    assert p["doi"] == "10.1/xyz"
    assert p["s2_id"] == "abc123"
    assert p["citation_count"] == 42


def test_s2_survives_a_record_with_nothing_but_a_title():
    p = normalize_s2({"title": "Bare"})
    assert p["title"] == "Bare"
    assert p["arxiv_id"] == ""
    assert p["year"] is None
    assert p["citation_count"] == 0


def test_s2_falls_back_to_an_arxiv_pdf_url_when_there_is_no_oa_pdf():
    p = normalize_s2({"title": "T", "externalIds": {"ArXiv": "2501.00001"}})
    assert p["pdf_url"] == "https://arxiv.org/pdf/2501.00001"


def test_openalex_abstract_is_rebuilt_from_the_inverted_index():
    assert openalex_abstract({"We": [0], "reject": [1], "gusts": [2]}) == "We reject gusts"
    assert openalex_abstract({"a": [0, 2], "b": [1]}) == "a b a"
    assert openalex_abstract(None) == ""


def test_openalex_strips_the_doi_url_prefix():
    assert normalize_openalex(OPENALEX_ITEM)["doi"] == "10.1/xyz"


def test_openalex_reads_authors_and_venue():
    p = normalize_openalex(OPENALEX_ITEM)
    assert p["authors"] == ["A. Researcher"]
    assert p["venue"] == "ICRA"
    assert p["abstract"] == "We reject gusts"


def test_arxiv_strips_the_version_suffix_and_collapses_wrapped_text():
    p = normalize_arxiv_entry(ARXIV_ENTRY)
    assert p["arxiv_id"] == "2501.00001"
    assert p["title"] == "Gust-Robust Quadrotor Control"
    assert p["abstract"] == "We reject gusts."
    assert p["year"] == 2025
    assert p["categories"] == ["cs.RO", "eess.SY"]
    assert p["pdf_url"] == "https://arxiv.org/pdf/2501.00001"


def test_normalized_records_dedup_against_each_other():
    from jarvis.sources import dedup_papers
    merged = dedup_papers([normalize_s2(S2_ITEM), normalize_arxiv_entry(ARXIV_ENTRY),
                           normalize_openalex(OPENALEX_ITEM)])
    assert len(merged) == 1, "the same paper from three APIs is one paper"
