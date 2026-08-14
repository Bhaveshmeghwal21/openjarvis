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


def _arxiv_pdf(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""


def normalize_s2(item: dict) -> dict:
    """Semantic Scholar graph record -> the common paper dict."""
    ext = item.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv", "") or ""
    pdf = item.get("openAccessPdf") or {}
    return {
        "doi": ext.get("DOI", "") or "",
        "arxiv_id": arxiv_id,
        "s2_id": item.get("paperId", "") or "",
        "title": item.get("title", "") or "",
        "authors": [a.get("name", "") for a in (item.get("authors") or [])],
        "year": item.get("year"),
        "venue": item.get("venue", "") or "",
        "abstract": item.get("abstract", "") or "",
        "citation_count": item.get("citationCount", 0) or 0,
        "url": item.get("url", "") or "",
        "pdf_url": pdf.get("url", "") or _arxiv_pdf(arxiv_id),
        "categories": list(item.get("fieldsOfStudy") or []),
    }


def openalex_abstract(inverted: dict | None) -> str:
    """Rebuild prose from OpenAlex's inverted index ({word: [positions]})."""
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, where in inverted.items():
        positions += [(int(i), word) for i in (where or [])]
    return " ".join(word for _, word in sorted(positions))


def normalize_openalex(item: dict) -> dict:
    """OpenAlex work -> the common paper dict."""
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    venue = (item.get("host_venue") or {}).get("display_name", "") or ""
    oa_url = (item.get("open_access") or {}).get("oa_url", "") or ""
    return {
        "doi": doi,
        "arxiv_id": "",
        "s2_id": "",
        "title": item.get("title", "") or "",
        "authors": [(a.get("author") or {}).get("display_name", "")
                    for a in (item.get("authorships") or [])],
        "year": item.get("publication_year"),
        "venue": venue,
        "abstract": openalex_abstract(item.get("abstract_inverted_index")),
        "citation_count": item.get("cited_by_count", 0) or 0,
        "url": item.get("id", "") or "",
        "pdf_url": oa_url,
        "categories": [c.get("display_name", "") for c in (item.get("concepts") or [])],
    }


_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})")


def normalize_arxiv_entry(entry: dict) -> dict:
    """A pre-extracted arXiv Atom entry -> the common paper dict.

    Takes a plain dict rather than an XML node so the mapping is testable without a feed.
    """
    match = _ARXIV_ID.search(entry.get("id", "") or "")
    arxiv_id = match.group(1) if match else ""
    published = entry.get("published", "") or ""
    year = int(published[:4]) if published[:4].isdigit() else None
    return {
        "doi": entry.get("doi", "") or "",
        "arxiv_id": arxiv_id,
        "s2_id": "",
        "title": " ".join((entry.get("title") or "").split()),
        "authors": list(entry.get("authors") or []),
        "year": year,
        "venue": "arXiv",
        "abstract": " ".join((entry.get("summary") or "").split()),
        "citation_count": 0,
        "url": entry.get("id", "") or "",
        "pdf_url": _arxiv_pdf(arxiv_id),
        "categories": list(entry.get("categories") or []),
    }


def make_s2_search(limit: int = 20) -> Callable[[str], list[dict]]:
    """Live Semantic Scholar keyword search. Uses $S2_API_KEY when present."""
    import os

    import httpx

    fields = ("paperId,title,abstract,year,venue,citationCount,externalIds,"
              "openAccessPdf,authors,fieldsOfStudy,url")

    def search(topic: str) -> list[dict]:
        key = os.environ.get("S2_API_KEY", "")
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={"query": topic, "limit": limit, "fields": fields},
                    headers={"x-api-key": key} if key else {},
                )
                resp.raise_for_status()
                items = resp.json().get("data", [])
        except (httpx.HTTPError, ValueError):
            return []
        return [normalize_s2(i) for i in items]

    return search


def make_openalex_search(limit: int = 20, mailto: str = "") -> Callable[[str], list[dict]]:
    """Live OpenAlex search. `mailto` gets you into the polite pool."""
    import httpx

    def search(topic: str) -> list[dict]:
        params = {"search": topic, "per-page": limit}
        if mailto:
            params["mailto"] = mailto
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get("https://api.openalex.org/works", params=params)
                resp.raise_for_status()
                items = resp.json().get("results", [])
        except (httpx.HTTPError, ValueError):
            return []
        return [normalize_openalex(i) for i in items]

    return search


def make_arxiv_search(limit: int = 20) -> Callable[[str], list[dict]]:
    """Live arXiv Atom search, parsed with the stdlib XML parser."""
    import xml.etree.ElementTree as ET

    import httpx

    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def search(topic: str) -> list[dict]:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get("http://export.arxiv.org/api/query",
                                  params={"search_query": f"all:{topic}",
                                          "max_results": limit})
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
        except (httpx.HTTPError, ET.ParseError):
            return []
        out: list[dict] = []
        for node in root.findall("a:entry", ns):
            doi_node = node.find("arxiv:doi", ns)
            out.append(normalize_arxiv_entry({
                "id": (node.findtext("a:id", "", ns) or ""),
                "title": (node.findtext("a:title", "", ns) or ""),
                "summary": (node.findtext("a:summary", "", ns) or ""),
                "published": (node.findtext("a:published", "", ns) or ""),
                "authors": [n.findtext("a:name", "", ns) or ""
                            for n in node.findall("a:author", ns)],
                "categories": [c.get("term", "") for c in node.findall("a:category", ns)],
                "doi": (doi_node.text if doi_node is not None else "") or "",
            }))
        return out

    return search


_RETRACTION_MARKERS = ("retraction", "retracted")


def is_retracted_record(work: dict | None) -> bool:
    """True when a Crossref work is a retraction notice or is flagged as retracted."""
    if not work:
        return False
    for key in ("type", "subtype"):
        if any(m in str(work.get(key, "")).lower() for m in _RETRACTION_MARKERS):
            return True
    return any("retraction" in str(u.get("type", "")).lower()
               for u in (work.get("update-to") or []))


def make_retraction_check() -> Callable[[str], bool]:
    """Live Crossref retraction lookup by DOI. Spec §14: the cheapest failure to prevent."""
    import httpx

    def check(doi: str) -> bool:
        if not doi:
            return False
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"https://api.crossref.org/works/{doi}")
                resp.raise_for_status()
                work = resp.json().get("message", {})
        except (httpx.HTTPError, ValueError):
            return False
        return is_retracted_record(work)

    return check


def enrich_provenance(paper: dict,
                      retraction_check: Callable[[str], bool] | None = None) -> dict:
    """Return a copy of `paper` with a `retracted` flag resolved.

    A failed or absent lookup means *unknown*, which is recorded as not-retracted rather
    than blocking ingest — a paper is never dropped by this system, only flagged.
    """
    out = dict(paper)
    doi = (out.get("doi") or "").strip()
    retracted = False
    if doi and retraction_check is not None:
        try:
            retracted = bool(retraction_check(doi))
        except Exception:  # noqa: BLE001 - a source outage is not a retraction
            retracted = False
    out["retracted"] = retracted
    return out
