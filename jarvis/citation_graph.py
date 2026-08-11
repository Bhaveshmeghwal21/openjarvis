"""Citation-graph traversal for gather-stage recall (spec §7 Stage A).

Keyword search finds the obvious papers; real researchers follow citation chains. PaperQA2
found citation traversal materially improved retrieval recall, and recall correlated with
final answer accuracy — so this is a recall tool, not a retrieval substrate.

CitationWalker BFS-expands seed papers over their references (what they cite) and citations
(who cites them), bounded by depth, relevance threshold, and a budget. Fetch + score
functions are injected, so it is fully testable offline; `make_s2_neighbors` is the live
Semantic Scholar adapter.

Ported from NanoResearch/jarvis/agents/citation_graph.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable


def paper_id(paper: dict) -> str:
    return paper.get("arxiv_id") or paper.get("s2_id") or paper.get("title", "")[:120]


@dataclass
class CitationWalker:
    fetch_refs_fn: Callable[[str], list[dict]]        # references (what a paper cites)
    fetch_citations_fn: Callable[[str], list[dict]]   # citations (who cites a paper)
    score_fn: Callable[[dict], float]                 # relevance to the original query
    threshold: float = 0.5
    max_depth: int = 2
    budget: int = 500
    already_seen: set = field(default_factory=set)

    def walk(self, seeds: list[dict]) -> list[dict]:
        seen: set[str] = set(self.already_seen)
        results: list[dict] = []
        queue: deque[tuple[dict, int]] = deque()
        for s in seeds:
            sid = paper_id(s)
            if sid:
                seen.add(sid)                          # don't re-surface the seeds themselves
            queue.append((s, 0))

        while queue and len(results) < self.budget:
            paper, depth = queue.popleft()
            if depth >= self.max_depth:
                continue
            pid = paper_id(paper)
            if not pid:
                continue
            neighbours = (self.fetch_refs_fn(pid) or []) + (self.fetch_citations_fn(pid) or [])
            for nb in neighbours:
                nid = paper_id(nb)
                if not nid or nid in seen:
                    continue
                seen.add(nid)
                if self.score_fn(nb) >= self.threshold:
                    results.append(nb)
                    queue.append((nb, depth + 1))
                    if len(results) >= self.budget:
                        break
        return results


def _normalize(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv", "")
    pdf = p.get("openAccessPdf") or {}
    return {
        "s2_id": p.get("paperId", ""),
        "arxiv_id": arxiv_id,
        "doi": ext.get("DOI", "") or "",
        "title": p.get("title", "") or "",
        "abstract": p.get("abstract", "") or "",
        "year": p.get("year"),
        "venue": p.get("venue", "") or "",
        "citation_count": p.get("citationCount", 0) or 0,
        "pdf_url": pdf.get("url", "") or (f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""),
    }


def make_s2_neighbors(max_per: int = 50):
    """Live Semantic Scholar references/citations adapters: returns (fetch_refs, fetch_citations)."""
    import os

    import httpx

    base = "https://api.semanticscholar.org/graph/v1"
    fields = "paperId,title,abstract,year,venue,citationCount,externalIds,openAccessPdf"

    def _headers() -> dict:
        key = os.environ.get("S2_API_KEY", "")
        return {"x-api-key": key} if key else {}

    def _fetch(endpoint: str, pid: str) -> list[dict]:
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"{base}/paper/{pid}/{endpoint}",
                                  params={"fields": fields, "limit": max_per}, headers=_headers())
                resp.raise_for_status()
                data = resp.json().get("data", [])
        except Exception:
            return []
        key = "citedPaper" if endpoint == "references" else "citingPaper"
        return [_normalize(item[key]) for item in data if item.get(key)]

    return (lambda pid: _fetch("references", pid), lambda pid: _fetch("citations", pid))
