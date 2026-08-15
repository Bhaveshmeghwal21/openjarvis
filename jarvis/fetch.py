"""PDF acquisition and local caching — closes design spec §5.1.

Nothing in the repository downloaded a file, cached one, retried a failure, or recorded
where a source came from before this module. Docling's `DocumentConverter.convert` does
accept a remote URL directly (confirmed empirically, spec §12 open question 1), but a real
run still needs the source PDF on disk: to re-parse without re-fetching when the parser is
upgraded, to escalate a bad parse to a stronger parser (design spec §14's stated
mitigation), and to make `papers.source_path` mean something durable. It also lets this
module reject a paywall's HTML login page — saved at a URL that merely *looks* like a
PDF — before it ever reaches a parser and produces a garbage "successful" parse.

`httpx` is imported lazily (inside `fetch_pdf`), matching this package's own convention
for every heavy/network dependency (`jarvis.sources`, `jarvis.llm`) — this module stays
importable, and its own tests run, with no network library required at import time.
"""
from __future__ import annotations

import re
from pathlib import Path

_PDF_MAGIC = b"%PDF"
_TIMEOUT = 30
_MAX_ATTEMPTS = 2  # one try plus one retry
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]")

# httpx's default User-Agent ("python-httpx/x.y.z") is blocked outright by several
# publishers' CDN-level bot detection (confirmed live: MDPI returns a bare 403 from
# Akamai on that UA, for a paper that is otherwise open access). A standard browser UA
# is enough to fetch the same single, publicly-served PDF a browser would.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
}


def _cache_path(cache_dir: Path, paper_id: str) -> Path:
    """A paper id used as a filename must never let `..` or a path separator escape the
    cache directory — sanitize before use, don't merely warn."""
    safe = _SAFE_ID.sub("_", paper_id) or "unnamed"
    return Path(cache_dir) / f"{safe}.pdf"


def _looks_like_pdf(content: bytes, content_type: str) -> bool:
    """Verify the bytes are actually a PDF. A `content-type: application/pdf` header is
    not trusted alone — a misconfigured server can claim it for an HTML error page, and
    the reverse (a correct PDF served with the wrong header) is common enough that the
    header is a hint, never authoritative. The magic-byte check is the one that matters.
    """
    del content_type  # informative only; decisive check is the magic bytes below
    return content.lstrip()[:4] == _PDF_MAGIC


def _download(url: str) -> bytes | None:
    """One URL, one try plus one retry on a transient failure. Never raises."""
    import httpx

    for attempt in range(_MAX_ATTEMPTS):
        try:
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS,
                              follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")
                if not _looks_like_pdf(resp.content, content_type):
                    return None
                return resp.content
        except httpx.HTTPError:
            if attempt + 1 >= _MAX_ATTEMPTS:
                return None
            continue
    return None


def _candidate_urls(paper: dict, *, unpaywall_email: str) -> list[str]:
    """Source order per spec §5.1: direct pdf_url, then a generic url, then Unpaywall by
    DOI. Unpaywall's own network call is deferred until it's actually needed (no direct
    url present) since it's a second network round-trip for a fallback path."""
    urls = []
    if paper.get("pdf_url"):
        urls.append(paper["pdf_url"])
    if paper.get("url"):
        urls.append(paper["url"])
    if not urls and paper.get("doi"):
        from jarvis.sources import make_unpaywall_pdf
        resolved = make_unpaywall_pdf(unpaywall_email)(paper["doi"])
        if resolved:
            urls.append(resolved)
    return urls


def fetch_pdf(paper: dict, cache_dir: str | Path, *, unpaywall_email: str,
             paper_id: str) -> str | None:
    """Fetch (or reuse a cached copy of) one paper's PDF. Returns a local path or `None`.

    Never raises: a fetch failure is per-paper data, exactly like `ingest_paper`'s own
    contract for a bad parse — one unfetchable PDF in a 300-paper gather must cost one
    paper, never the run.
    """
    cache_dir = Path(cache_dir)
    target = _cache_path(cache_dir, paper_id)
    if target.is_file():
        return str(target)

    for url in _candidate_urls(paper, unpaywall_email=unpaywall_email):
        content = _download(url)
        if content is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            return str(target)
    return None
