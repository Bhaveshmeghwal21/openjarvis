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

**All ten build steps of the design spec are complete and merged to `main`.** Storage,
parsing, typed evidence units, hybrid retrieval, two-stage verification, multi-source
gather with a calibrated gate, cited Q&A, an MCP server, contradiction detection, and
long-form reports — each passed a final whole-branch adversarial review, several of which
found and fixed real Critical defects (see the `LEDGER-*.md` files for the full history of
each).

**The CLI (`docs/specs/2026-08-15-cli-and-operations.md`) is also complete, on this
branch**: `jarvis status/gather/ask/report/contradictions/review/calibrate/mcp` wire
every one of those ten build steps into one operator surface, closing three real pipeline
gaps found by reading the code rather than the plans — PDF acquisition (nothing
downloaded or cached a source file before this), card extraction (exported and tested but
never called by any production path), and measured cost (the column existed, nothing
wrote to it). See "Quickstart" below and `LEDGER-cli.md` for the full history.

Read the spec first: [`docs/specs/2026-08-11-research-corpus-agent-design.md`](docs/specs/2026-08-11-research-corpus-agent-design.md),
then [`docs/specs/2026-08-15-cli-and-operations.md`](docs/specs/2026-08-15-cli-and-operations.md).
Every non-obvious decision in either carries the measurement that drove it. Implementation
plans:
[steps 1–5](docs/plans/2026-08-11-verifiable-single-paper-core.md) ·
[step 6 — gather + gate](docs/plans/2026-08-14-gather-and-gate.md) ·
[step 7 — compile/Q&A](docs/plans/2026-08-14-compile-cited-qa.md) ·
[step 8 — MCP server](docs/plans/2026-08-14-mcp-server.md) ·
[step 9 — contradiction detection](docs/plans/2026-08-14-contradiction-detection.md) ·
[step 10 — long-form reports](docs/plans/2026-08-14-longform-reports.md) ·
[CLI and operations](docs/specs/2026-08-15-cli-and-operations.md).

| Module | Purpose |
|---|---|
| `jarvis/models.py` | Frozen domain types for Layers 0/1/2 and verification |
| `jarvis/text.py` | Text normalization and verbatim span-finding for quote verification |
| `jarvis/store.py` | SQLite persistence — the only module that writes SQL |
| `jarvis/parse.py` | Parser protocol, fake double, lazy Docling adapter |
| `jarvis/units.py` | Layer 0 → Layer 1: typed evidence units, binding rules, parent/child structure |
| `jarvis/context.py` | Contextual prefixes for child units (Anthropic Contextual Retrieval) |
| `jarvis/embed.py` | Embedder protocol, fake/BGE adapters, brute-force vector search |
| `jarvis/index.py` | FTS5 keyword index — the BM25 half of hybrid retrieval |
| `jarvis/retrieve.py` | Hybrid retrieval: RRF fusion, reranking, parent-unit expansion |
| `jarvis/verify.py` | Two-stage verification: deterministic quote grounding + NLI entailment |
| `jarvis/evaluate.py` | Evaluation metrics: quote fidelity, statement support, gate recall, coverage, contradiction precision |
| `jarvis/gather.py` | Stage A: search plan, multi-source fan-out, citation-graph expansion |
| `jarvis/gate.py` | Stage B: four independent signals, union decision, threshold calibration |
| `jarvis/label.py` | Seed-set sampling and hand-label round-trip for gate calibration |
| `jarvis/ingest.py` | Stage C: deep read into the corpus with per-paper failure isolation |
| `jarvis/card.py` | Layer 2 card extraction with mechanically verified quote bindings |
| `jarvis/evidence.py` | Hard cap on evidence units per synthesis call, primacy/recency ordering |
| `jarvis/retriever.py` | Iterative retrieval: a `Refiner` proposes follow-up queries, fused across rounds by RRF |
| `jarvis/writer.py` | Writer subagent: drafts an answer, emits explicit `(claim, unit_id, quote)` triples |
| `jarvis/answer.py` | `ask()` — wires retrieve → cap → write → verify; blocks fabrications, flags uncertainty |
| `jarvis/outline.py` | Report outline built from Layer 2 cards |
| `jarvis/report.py` | Section-by-section drafting, claim integration, coverage, markdown rendering |
| `jarvis/contradict.py` | Cross-paper contradiction scan, ranking, human review round-trip |
| `jarvis/tools.py` | The corpus as callable tools — a pure dispatcher, no protocol dependency |
| `jarvis/mcp_server.py` | stdio MCP adapter over `jarvis.tools` — translation only, no corpus logic |
| `jarvis/fetch.py` | PDF acquisition and local caching, with `%PDF` magic-byte verification |
| `jarvis/cli.py` | The `jarvis` command — the operator surface over every stage above |
| `jarvis/sources.py` | Multi-source search (arXiv, S2, OpenAlex, Crossref, CORE, Unpaywall) + dedup + retraction check |
| `jarvis/citation_graph.py` | Citation-graph traversal for recall |
| `jarvis/scoring.py` | Cosine, recency, citation weighting |
| `jarvis/router.py` | Task→tier model routing + measured cost logging |
| `jarvis/config.py` | Env/JSON config; never reads `.env` |
| `jarvis/llm.py` | Lazy OpenAI-compatible chat helper |

`citation_graph.py`, `scoring.py`, `router.py`, `config.py`, and `llm.py` are gather-stage
primitives ported from the NanoResearch jarvis package (no runtime dependency in either
direction — see spec §12). `sources.py` was ported and then substantially extended on this
branch with new source adapters and the retraction check. Everything else is new.

## Install

```bash
pip install -e ".[dev]"
```

Optional extras, needed as you use more of the pipeline: `llm` (writer/planner/voter/card
extractor — anything that calls a model), `parse` (Docling), `index` (the real embedder),
`verify` (the local NLI model), `mcp` (the MCP server). `jarvis <command>` names the exact
missing extra and environment variable if one is needed and absent — it never silently
substitutes a fake model.

## Test

```bash
python -m pytest
```

All tests are offline — no network, no API keys, no model downloads.

## Quickstart

```bash
pip install -e ".[llm,parse,index,verify]"
export JARVIS_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
export JARVIS_API_KEY=sk-...
export UNPAYWALL_EMAIL=you@example.com              # required by the Unpaywall fallback

# Build a corpus. Pauses after screening to show what would be deep-read before
# spending anything on parsing/embedding/card extraction -- pass --yes to proceed.
jarvis gather "does retrieval augmentation reduce hallucination in long-form QA?" \
  --project my-question

jarvis gather "..." --project my-question --yes   # proceeds through deep reads

jarvis status --project my-question               # paper counts by depth, cost so far

jarvis ask "what evaluation metrics did these papers use?" --project my-question

jarvis report "retrieval augmentation and hallucination" --project my-question --out report.md

jarvis contradictions --project my-question        # scans the report just written
jarvis review --project my-question reviews/contradictions.jsonl   # after hand-editing it

jarvis calibrate --project my-question seed --run-id <run-id>   # gate calibration
```

Every corpus is one SQLite file at `$JARVIS_PROJECT_ROOT/<project>/corpus.db` (default
`~/.jarvis/projects`). Re-running `jarvis gather` on the same project resumes rather than
re-searching: papers already screened and waiting for a deep read pick up right there.

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

## Use it from Claude Code

Install the extra and point a client at a project's corpus:

```bash
pip install -e ".[mcp]"
```

Then add to your MCP client config (`.mcp.json` for Claude Code):

```json
{
  "mcpServers": {
    "jarvis-corpus": {
      "command": "jarvis-mcp",
      "args": ["--db", "/absolute/path/to/corpus.db"]
    }
  }
}
```

(`jarvis mcp` works identically as a subcommand of the main CLI, for anyone who prefers
one entry point; `jarvis-mcp` keeps working unchanged for existing configs like the one
above.)

Six tools become available: `corpus_search`, `get_unit`, `get_paper`, `list_papers`,
`verify_quote`, and `ask`.

`verify_quote` is the one that matters. It is a deterministic string match against the
immutable parsed paper — no model, no cost — so an assistant can check a quotation before
asserting it. Add `--with-models` to enable `ask`, which additionally needs a writer model
and a local NLI model.

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
