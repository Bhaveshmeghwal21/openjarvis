"""Multi-source literature search + dedup (spec §7 Stage A).

`combine_sources` merges several search functions (arXiv/S2/OpenAlex + Crossref/CORE) into
one deduped search_fn, degrading gracefully if a source fails. Normalizers map each API's
JSON to the common paper dict. Live adapters are lazy so this module imports offline.

Ported from NanoResearch/jarvis/tools/sources.py.
"""
from __future__ import annotations

import re
from typing import Callable

PAPER_FIELDS = (
    "doi", "arxiv_id", "s2_id", "title", "authors", "year", "venue",
    "abstract", "citation_count", "url", "pdf_url", "categories",
)


def dedup_papers(papers: list[dict]) -> list[dict]:
    """Dedup by arXiv id, then DOI, then title prefix."""
    seen_arxiv: set[str] = set()
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    out: list[dict] = []
    for p in papers:
        aid = (p.get("arxiv_id") or "").lower().strip()
        doi = (p.get("doi") or "").lower().strip()
        title = (p.get("title") or "").lower().strip()[:80]
        if (aid and aid in seen_arxiv) or (doi and doi in seen_doi) or (title and title in seen_title):
            continue
        if aid:
            seen_arxiv.add(aid)
        if doi:
            seen_doi.add(doi)
        if title:
            seen_title.add(title)
        out.append(p)
    return out


def combine_sources(*search_fns: Callable[[str], list[dict]]) -> Callable[[str], list[dict]]:
    """Aggregate multiple search_fns into one deduped search_fn (failing sources are skipped)."""
    def search(topic: str) -> list[dict]:
        papers: list[dict] = []
        for fn in search_fns:
            try:
                papers += fn(topic) or []
            except Exception:
                continue
        return dedup_papers(papers)
    return search


def normalize_crossref(item: dict) -> dict:
    dp = (item.get("issued", {}).get("date-parts") or [[None]])[0]
    return {
        "doi": item.get("DOI", ""),
        "arxiv_id": "",
        "s2_id": "",
        "title": (item.get("title") or [""])[0],
        "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])],
        "year": int(dp[0]) if dp and dp[0] else None,
        "venue": (item.get("container-title") or [""])[0],
        "abstract": re.sub(r"<[^>]+>", "", item.get("abstract", "") or "").strip(),
        "citation_count": item.get("is-referenced-by-count", 0) or 0,
        "url": item.get("URL", ""),
        "pdf_url": "",
        "categories": [],
    }


def make_crossref_search(rows: int = 20) -> Callable[[str], list[dict]]:
    """Live Crossref search (full citation graphs, DOI resolution)."""
    import httpx

    def search(topic: str) -> list[dict]:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get("https://api.crossref.org/works",
                                  params={"query": topic, "rows": rows})
                resp.raise_for_status()
                items = resp.json().get("message", {}).get("items", [])
        except Exception:
            return []
        return [normalize_crossref(i) for i in items]

    return search


def make_unpaywall_pdf(email: str) -> Callable[[str], str | None]:
    """Resolve a free PDF link for a DOI via Unpaywall (legal OA copies only)."""
    import httpx

    def pdf_for(doi: str) -> str | None:
        if not doi:
            return None
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"https://api.unpaywall.org/v2/{doi}", params={"email": email})
                resp.raise_for_status()
                loc = resp.json().get("best_oa_location") or {}
        except Exception:
            return None
        return loc.get("url_for_pdf") or None

    return pdf_for


def make_core_search(api_key: str, limit: int = 20) -> Callable[[str], list[dict]]:
    """Live CORE search (200M+ open-access full texts)."""
    import httpx

    def search(topic: str) -> list[dict]:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post("https://api.core.ac.uk/v3/search/works",
                                   headers={"Authorization": f"Bearer {api_key}"},
                                   json={"q": topic, "limit": limit})
                resp.raise_for_status()
                results = resp.json().get("results", [])
        except Exception:
            return []
        return [
            {
                "doi": r.get("doi", "") or "",
                "arxiv_id": r.get("arxivId", "") or "",
                "s2_id": "",
                "title": r.get("title", "") or "",
                "authors": [a.get("name", "") for a in (r.get("authors") or [])],
                "year": r.get("yearPublished"),
                "venue": r.get("publisher", "") or "",
                "abstract": r.get("abstract", "") or "",
                "citation_count": 0,
                "url": r.get("downloadUrl", "") or "",
                "pdf_url": r.get("downloadUrl", "") or "",
                "categories": [],
            }
            for r in results
        ]

    return search
