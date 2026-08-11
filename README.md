# Jarvis — a verifiable research corpus agent

Read hundreds of papers on a question, keep them, and answer from them — where **every claim
resolves to a verbatim span in a specific paper at a specific location**.

## Why

An LLM asked a research question consults its training data plus 10–20 live sources. Research
means reading hundreds of papers and retaining what they said. The gap is architecture, not
model quality:

| | |
|---|---|
| GPT-4o citation hallucination rate on scholarly queries | **78–90%** |
| Top models' citation support (ALCE) | **~50%** |
| LLM accuracy at picking the right paper to cite (CiteME) | **4–18%** |
| PaperQA2 (corpus-grounded) hallucinated citations found | **zero** |

Every general assistant re-researches from zero on each query, so none can afford deep
ingestion. A system that keeps a corpus can.

## The core idea

Gathering and compiling are separate stages with opposite objectives:

```
GATHER  ── optimize recall ──►  corpus  ── optimize precision ──►  COMPILE
 (paid once, amortized)                        (paid per question)
```

This dissolves the precision-vs-coverage trade-off that forces single-shot systems to choose
between casting wide (noise) and narrow (gaps). Nobody publishes it this way because
benchmarks are single-shot and don't reward a corpus you keep.

## Status

**Design complete, implementation at step 0 of 10.**

Read the spec first: [`docs/specs/2026-08-11-research-corpus-agent-design.md`](docs/specs/2026-08-11-research-corpus-agent-design.md).
Every non-obvious decision in it carries the measurement that drove it.

What exists so far — gather-stage primitives ported from the NanoResearch jarvis package:

| Module | Purpose |
|---|---|
| `jarvis/sources.py` | Multi-source search (Crossref, CORE, Unpaywall) + dedup |
| `jarvis/citation_graph.py` | Citation-graph traversal for recall |
| `jarvis/scoring.py` | Cosine, recency, citation weighting |
| `jarvis/router.py` | Task→tier model routing + measured cost logging |
| `jarvis/config.py` | Env/JSON config; never reads `.env` |
| `jarvis/llm.py` | Lazy OpenAI-compatible chat helper |

Not built yet: the corpus store, parsing, typed evidence units, retrieval, and verification —
build steps 1–5 in the spec. **Nothing is trustworthy until step 5 (the eval harness) exists.**

## Install

```bash
pip install -e ".[dev]"
```

Optional extras, installed as each build step lands: `llm`, `parse`, `index`, `verify`.

## Test

```bash
python -m pytest
```

All tests are offline — no network, no API keys, no model downloads.

## Configuration

Environment only; this project never reads `.env`.

| Variable | Purpose |
|---|---|
| `JARVIS_BASE_URL` / `JARVIS_API_KEY` | OpenAI-compatible endpoint |
| `JARVIS_MODEL_<TASK>` | Per-task model override (e.g. `JARVIS_MODEL_SYNTHESIS=gpt-4.1`) |
| `JARVIS_CONFIG` | Path to JSON config `{"models": {...}, "project_root": "..."}` |
| `JARVIS_PROJECT_ROOT` | Where corpora live (default `~/.jarvis/projects`) |
| `S2_API_KEY` | Semantic Scholar — raises rate limits, optional |
| `UNPAYWALL_EMAIL` | Required by the Unpaywall API |

## Design in one page

**Three layers over one immutable artifact:**

- **Layer 0** — the parsed paper, verbatim, never mutated. The only source of truth.
- **Layer 1** — typed evidence units (`prose` / `table` / `figure` / `equation`), parent-child,
  contextual prefix, hybrid BM25 + vector fused by RRF, cross-encoder reranked. The retrieval
  surface and the citation target.
- **Layer 2** — the paper card. Coverage ledger and comparison index. Every field anchored to a
  unit id and a verbatim quote. **Never the ground for a claim.**

**Verification is the differentiator.** Two failure modes, handled separately:

- *Citation hallucination* (reference or quote doesn't exist) → deterministic verbatim string
  match against Layer 0. Free, exact, target 100%.
- *Statement hallucination* (real paper, real quote, doesn't say that) → NLI entailment.
  **Not LLM-as-judge**, which correlates 0.101 with human judgement versus 0.638 for NLI.

Contradiction detection falls out of the same pass — NLI's third label — at no extra cost, and
is structurally impossible for any system that doesn't retain a corpus.

## Deliberately not built

Knowledge graph · semantic chunking · LLM-as-judge verification · multimodal embedding
retrieval · permanent cross-project library · connectors, podcasts, slide decks.

Each has a measurement behind the decision — see spec §11.

## License

TBD.
