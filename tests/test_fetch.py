"""Task 3: PDF fetch and cache (jarvis/fetch.py). Closes spec gap §5.1.

`fetch_pdf` never raises -- a bad PDF is per-paper data, exactly like `ingest_paper`'s own
contract for a bad parse. It returns a local path or `None`.
"""
from __future__ import annotations

from pathlib import Path

from jarvis.fetch import fetch_pdf

_REAL_PDF_BYTES = b"%PDF-1.4\n%fake but starts with the real magic bytes\n"
_HTML_BYTES = b"<html><body>please log in to view this article</body></html>"


class _FakeResponse:
    def __init__(self, content: bytes, content_type: str = "application/pdf",
                status_code: int = 200):
        self.content = content
        self.headers = {"content-type": content_type}
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("boom", request=None, response=self)


class _FakeClient:
    """A drop-in for httpx.Client that returns canned responses keyed by URL."""

    def __init__(self, responses: dict[str, _FakeResponse] | None = None,
                raise_on: set[str] | None = None):
        self._responses = responses or {}
        self._raise_on = raise_on or set()
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        if url in self._raise_on:
            import httpx
            raise httpx.ConnectError("boom")
        return self._responses.get(url, _FakeResponse(b"", status_code=404))


def test_a_cache_hit_performs_no_request(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    cache_dir.mkdir()
    cached = cache_dir / "p1.pdf"
    cached.write_bytes(_REAL_PDF_BYTES)

    def boom_if_called(*args, **kwargs):
        raise AssertionError("should not make a network request on a cache hit")

    monkeypatch.setattr("httpx.Client", boom_if_called)

    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path == str(cached)


def test_fetches_from_pdf_url_and_caches(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    client = _FakeClient({"http://example.com/x.pdf": _FakeResponse(_REAL_PDF_BYTES)})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path is not None
    assert Path(path).read_bytes() == _REAL_PDF_BYTES
    assert Path(path) == cache_dir / "p1.pdf"


def test_falls_back_to_url_field_when_no_pdf_url(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    client = _FakeClient({"http://example.com/page": _FakeResponse(_REAL_PDF_BYTES)})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"url": "http://example.com/page"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path is not None
    assert client.calls == ["http://example.com/page"]


def test_falls_back_to_unpaywall_by_doi_when_no_direct_url(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"

    def fake_unpaywall_pdf(email):
        assert email == "me@example.com"
        return lambda doi: "http://oa.example.com/free.pdf" if doi == "10.1/x" else None

    monkeypatch.setattr("jarvis.sources.make_unpaywall_pdf", fake_unpaywall_pdf)
    client = _FakeClient({"http://oa.example.com/free.pdf": _FakeResponse(_REAL_PDF_BYTES)})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"doi": "10.1/x"}, cache_dir, unpaywall_email="me@example.com",
                     paper_id="p1")
    assert path is not None
    assert client.calls == ["http://oa.example.com/free.pdf"]


def test_an_html_response_is_rejected_as_not_a_pdf(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    client = _FakeClient({"http://example.com/x.pdf":
                          _FakeResponse(_HTML_BYTES, content_type="text/html")})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path is None
    assert not (cache_dir / "p1.pdf").exists()


def test_bytes_lacking_the_pdf_magic_header_are_rejected_even_with_a_pdf_content_type(
    tmp_path, monkeypatch
):
    # A misconfigured server can claim application/pdf for an HTML error page -- the
    # magic-byte check must not trust the header alone.
    cache_dir = tmp_path / "pdfs"
    client = _FakeClient({"http://example.com/x.pdf":
                          _FakeResponse(_HTML_BYTES, content_type="application/pdf")})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path is None


def test_a_connection_failure_returns_none_never_raises(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    client = _FakeClient(raise_on={"http://example.com/x.pdf"})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path is None


def test_no_source_at_all_returns_none_without_a_network_call(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"

    def boom_if_called(*args, **kwargs):
        raise AssertionError("should not make a network request with nothing to fetch")

    monkeypatch.setattr("httpx.Client", boom_if_called)

    path = fetch_pdf({}, cache_dir, unpaywall_email="me@example.com", paper_id="p1")
    assert path is None


def test_a_retry_happens_once_after_a_transient_failure(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    attempts = {"n": 0}

    class FlakyClient(_FakeClient):
        def get(self, url, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                import httpx
                raise httpx.ConnectError("transient")
            return _FakeResponse(_REAL_PDF_BYTES)

    monkeypatch.setattr("httpx.Client", lambda **kw: FlakyClient())
    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="p1")
    assert path is not None
    assert attempts["n"] == 2


def test_paper_id_is_sanitized_so_it_cannot_escape_the_cache_directory(tmp_path, monkeypatch):
    cache_dir = tmp_path / "pdfs"
    client = _FakeClient({"http://example.com/x.pdf": _FakeResponse(_REAL_PDF_BYTES)})
    monkeypatch.setattr("httpx.Client", lambda **kw: client)

    path = fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
                     unpaywall_email="me@example.com", paper_id="../../evil")
    assert path is not None
    resolved = Path(path).resolve()
    assert cache_dir.resolve() in resolved.parents or resolved.parent == cache_dir.resolve()


def test_one_failed_fetch_is_per_paper_data_never_a_crash(tmp_path, monkeypatch):
    # fetch_pdf's own contract (never raises) is the mechanism; this test exercises the
    # same guarantee across a small batch, matching ingest_paper's established pattern.
    cache_dir = tmp_path / "pdfs"

    class MixedClient(_FakeClient):
        def get(self, url, **kwargs):
            self.calls.append(url)
            if "bad" in url:
                import httpx
                raise httpx.ConnectError("down")
            return _FakeResponse(_REAL_PDF_BYTES)

    monkeypatch.setattr("httpx.Client", lambda **kw: MixedClient())

    results = [
        fetch_pdf({"pdf_url": "http://example.com/good.pdf"}, cache_dir,
                 unpaywall_email="x@example.com", paper_id="good"),
        fetch_pdf({"pdf_url": "http://example.com/bad.pdf"}, cache_dir,
                 unpaywall_email="x@example.com", paper_id="bad"),
    ]
    assert results[0] is not None
    assert results[1] is None


def test_a_realistic_user_agent_is_sent_not_httpxs_default(tmp_path, monkeypatch):
    # Found live: httpx's default "python-httpx/x.y.z" User-Agent gets a bare 403 from
    # MDPI's Akamai bot detection on a paper that is otherwise open access -- the fetch
    # never reaches the magic-byte check at all. A browser-shaped UA is enough to be
    # served the same publicly-available PDF a browser would get.
    cache_dir = tmp_path / "pdfs"
    captured_kwargs = {}

    def spy_client(**kw):
        captured_kwargs.update(kw)
        return _FakeClient({"http://example.com/x.pdf": _FakeResponse(_REAL_PDF_BYTES)})

    monkeypatch.setattr("httpx.Client", spy_client)

    fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
             unpaywall_email="me@example.com", paper_id="p1")

    headers = captured_kwargs.get("headers", {})
    assert "python-httpx" not in headers.get("User-Agent", "")
    assert "Mozilla" in headers.get("User-Agent", "")


def test_redirects_are_followed(tmp_path, monkeypatch):
    # Several real sources (DOI resolvers, institutional repositories) reach the actual
    # PDF only after a redirect -- httpx.Client does not follow redirects by default.
    cache_dir = tmp_path / "pdfs"
    captured_kwargs = {}

    def spy_client(**kw):
        captured_kwargs.update(kw)
        return _FakeClient({"http://example.com/x.pdf": _FakeResponse(_REAL_PDF_BYTES)})

    monkeypatch.setattr("httpx.Client", spy_client)

    fetch_pdf({"pdf_url": "http://example.com/x.pdf"}, cache_dir,
             unpaywall_email="me@example.com", paper_id="p1")

    assert captured_kwargs.get("follow_redirects") is True
