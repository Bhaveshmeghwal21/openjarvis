"""Open-access PDF resolution (jarvis/sources.py).

Every test here encodes a failure measured against the real corpus, not a hypothetical:

- Of 177 real DOIs, 89 have a free PDF somewhere but only 50 are on a host that will
  actually serve it -- the other 39 are open access yet sit behind Akamai (MDPI) or
  Cloudflare (Hindawi) bot walls. Ranking repository copies ahead of publisher CDNs is
  what turns a known-OA paper into a fetched one.
- Unpaywall answers HTTP 422 ("Please use your real email address") for an empty *or*
  invented email, so that path had never once resolved a PDF. OpenAlex exposes the same
  OA data with no such wall, so it goes first.
"""
from __future__ import annotations

from jarvis.sources import openalex_oa_urls, rank_oa_urls


def test_repository_copies_outrank_bot_walled_publisher_cdns():
    # Measured: mdpi.com and ieeexplore both 403 a plain client, while the author's
    # deposited copy in an institutional repository serves the same paper unguarded.
    urls = [
        "https://www.mdpi.com/2504-446X/7/12/700/pdf",
        "https://ieeexplore.ieee.org/ielx7/6287639/9316650.pdf",
        "https://orbilu.uni.lu/bitstream/10993/47153/1/1823056.pdf",
        "https://arxiv.org/pdf/2603.19966",
    ]
    ranked = rank_oa_urls(urls)
    assert ranked[0] == "https://orbilu.uni.lu/bitstream/10993/47153/1/1823056.pdf"
    assert ranked[1] == "https://arxiv.org/pdf/2603.19966"
    assert all("mdpi.com" not in u and "ieeexplore" not in u for u in ranked[:2])


def test_ranking_is_stable_within_a_group():
    # Nothing else about the caller's ordering should be disturbed -- only the walled
    # hosts move, and they keep their relative order when they do.
    urls = ["https://repo.a.edu/1.pdf", "https://www.mdpi.com/x/pdf",
            "https://repo.b.edu/2.pdf", "https://ieeexplore.ieee.org/y.pdf"]
    assert rank_oa_urls(urls) == [
        "https://repo.a.edu/1.pdf", "https://repo.b.edu/2.pdf",
        "https://www.mdpi.com/x/pdf", "https://ieeexplore.ieee.org/y.pdf",
    ]


def test_ranking_drops_duplicates_keeping_the_first_occurrence():
    urls = ["https://repo.a.edu/1.pdf", "https://www.mdpi.com/x/pdf",
            "https://repo.a.edu/1.pdf"]
    assert rank_oa_urls(urls) == ["https://repo.a.edu/1.pdf", "https://www.mdpi.com/x/pdf"]


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("boom", request=None, response=self)


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, **kwargs):
        self.calls.append(url)
        return self._response


def test_every_location_is_collected_not_just_the_best_one(monkeypatch):
    # OpenAlex reports one `best_oa_location` plus a `locations` array. The best one is
    # frequently the publisher's own walled copy while a fetchable repository mirror sits
    # further down the array -- taking only the best throws that mirror away.
    payload = {
        "best_oa_location": {"pdf_url": "https://www.mdpi.com/walled/pdf"},
        "locations": [
            {"pdf_url": "https://www.mdpi.com/walled/pdf"},
            {"pdf_url": "https://eprints.qut.edu.au/1234/1/paper.pdf"},
            {"pdf_url": None},
        ],
    }
    client = _FakeClient(_FakeResponse(payload))
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    urls = openalex_oa_urls("10.3390/drones7120700", mailto="me@example.com")
    assert urls == [
        "https://eprints.qut.edu.au/1234/1/paper.pdf",
        "https://www.mdpi.com/walled/pdf",
    ]


def test_landing_pages_are_returned_when_locations_carry_no_pdf_url(monkeypatch):
    # Measured on a real gold-OA paper: six locations, exactly one with a `pdf_url` (the
    # publisher's bot-walled copy), while every reachable repository mirror appeared only
    # as a landing page. Reading `pdf_url` alone discards the copies that actually serve.
    payload = {
        "best_oa_location": {"pdf_url": "https://www.mdpi.com/walled/pdf"},
        "locations": [
            {"pdf_url": "https://www.mdpi.com/walled/pdf",
             "landing_page_url": "https://doi.org/10.3390/x"},
            {"pdf_url": None,
             "landing_page_url": "https://orbilu.uni.lu/handle/10993/47153"},
            {"pdf_url": None, "landing_page_url": "https://zenodo.org/record/2621428"},
        ],
    }
    client = _FakeClient(_FakeResponse(payload))
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    # Direct PDFs first (one request against a landing page's two), then repository
    # landing pages, and the doi.org resolver last since it redirects back to the wall.
    assert openalex_oa_urls("10.3390/x", mailto="me@example.com") == [
        "https://www.mdpi.com/walled/pdf",
        "https://orbilu.uni.lu/handle/10993/47153",
        "https://zenodo.org/record/2621428",
        "https://doi.org/10.3390/x",
    ]


def test_a_doi_resolver_ranks_behind_a_repository():
    # doi.org inherits whatever wall the publisher it redirects to puts up, so it must
    # never be preferred over a repository copy that serves the file directly.
    assert rank_oa_urls(["https://doi.org/10.1/x", "https://zenodo.org/record/1"]) == [
        "https://zenodo.org/record/1", "https://doi.org/10.1/x",
    ]


def test_an_openalex_outage_yields_no_urls_rather_than_raising(monkeypatch):
    # A source outage must cost the paper its OA fallback, never the run -- same contract
    # every other adapter in this module already honours.
    client = _FakeClient(_FakeResponse({}, status_code=500))
    monkeypatch.setattr("httpx.Client", lambda **kw: client)
    assert openalex_oa_urls("10.1/x", mailto="me@example.com") == []


def test_no_doi_means_no_network_call(monkeypatch):
    def boom(**kw):
        raise AssertionError("should not call OpenAlex without a DOI")

    monkeypatch.setattr("httpx.Client", boom)
    assert openalex_oa_urls("", mailto="me@example.com") == []
