# Research Corpus Agent — Design Spec

> **Status:** Approved design, ready for implementation planning
> **Date:** 2026-08-11
> **Repo:** standalone at `D:\LionXdrones\r&d\jarvis`, package `jarvis/`

## 1. Problem

An LLM asked a research question consults its training data plus 10–20 live sources. Real research means reading hundreds of papers and retaining what they said. The gap is not model quality — it is architecture:

- GPT-4o hallucinates citations **78–90%** of the time on scholarly queries.
- ALCE measures top models at **~50%** citation support.
- CiteME measures **4–18%** accuracy at identifying the correct paper to cite.
- PaperQA2, corpus-grounded, had **zero** hallucinated citations found.

Every general assistant re-researches from zero on each query, so none can afford deep ingestion. A system that keeps a corpus can.

## 2. Goal

Given a research question, produce a **durable, verifiable, per-project corpus** of a few hundred papers, and answer questions or generate reports from it where **every claim resolves to a verbatim span in a specific paper at a specific location**.

## 3. Scope

**In scope**
- Reasoning-driven gathering across academic sources, with citation-graph expansion
- Adaptive-depth ingestion: cheap screen for all, deep read for survivors
- Local, per-project corpus (~100–500 papers), one file, no server
- Cited Q&A over the corpus
- Structured multi-section report generation
- Cross-paper contradiction detection
- MCP server so external agents (Claude Code, Cursor) query the corpus natively

**Explicitly out of scope**
- Knowledge graph (see §11)
- Permanent cross-project library — each project is self-contained
- Non-academic connectors (Reddit, YouTube, Slack, Notion)
- Podcasts, slide decks, video overviews, real-time collaboration
- Web UI in v1 — CLI and MCP only
- Model weight training

## 4. Core architectural decision: decouple gather from compile

The precision-vs-coverage trade-off that separates PaperQA2 (high citation precision, low coverage) from OpenScholar (high coverage, expert-level precision) is a **retrieval-time artifact**. Both search *while* answering, forcing a choice between casting wide (noise) and narrow (gaps).

Splitting the stages dissolves it:

| Stage | Optimizes for | Cost |
|---|---|---|
| **Gather** | Recall. No answer is being written. | Paid once, amortized across every future query |
| **Compile** | Precision. Retrieval is now a local, solved problem. | Paid per question |

No published system does this because benchmarks are single-shot and do not reward a corpus you keep.

## 5. Data model — three layers over one immutable artifact

### Layer 0 — the parsed paper

Structured parse of the PDF, section hierarchy preserved, stored verbatim, **never mutated**. The only source of truth. Nothing else in the system may be the ground for a claim.

- Default parser: **Docling** (MIT, structured lossless output, CPU-capable)
- Escalation parser: **MinerU** for formula-dense or multi-column-hostile papers (leads OmniDocBench ~90.7)
- Rationale: Marker is faster at scale but its weights restrict commercial use and it collapses on degraded scans

### Layer 1 — typed evidence units

The retrieval surface **and** the citation target. Not plain text chunks.

Every unit carries: `paper_id`, `type`, `page`, `section_path`, `verbatim_text`, `context_prefix`, `parent_id`.

| Unit type | Composition |
|---|---|
| `prose` | Recursive ~512-token chunk on section boundaries |
| `table` | Linearized markdown **+ caption + body text that references it**, as one unit |
| `figure` | Caption + referring text + optional VLM description |
| `equation` | LaTeX + surrounding explanatory sentences |

**Why typed units, not passages.** SPIQA counts 117,707 tables and 152,487 figures across 25,859 papers — ~10 visual artifacts per paper. "Method X reached 94.2% on dataset Y" typically exists *only* in a table. A text-only citation target cannot ground it.

**Why not multimodal embedding retrieval.** A controlled evaluation (arXiv 2607.16604) found CLIP-style retrieval moves GPT-4o from 0.000 to 0.143 accuracy — scientific tables are topically similar while differing in one cell, so global image-text embeddings retrieve the right *kind* of figure, not the right figure. Typing and locating units instead raised table-question coverage **67% → 83%** with no retrieval change.

**Binding rule (non-negotiable).** A table without its headers is misleading; a figure without its caption is useless; "as shown in Figure 3" must stay linked to Figure 3. Units are never split across these boundaries.

**Parent/child structure.** Small child units are embedded and do the *matching*. Larger parent sections live in a key-value store and do the *generating*. Retrieve on children, expand to parents, dedupe when several children share a parent. Precision at match time, context at answer time.

**Contextual prefix.** Each child gets a 50–100 token LLM-generated description of its place in the paper, prepended before embedding (Anthropic Contextual Retrieval). Measured: 35% fewer retrieval failures alone, 49% with BM25, **67% with reranking**, at ~$1.02/M document tokens with prompt caching — negligible at this scale.

**Chunking.** Recursive 512-token on section boundaries. Not semantic chunking: measured **69% vs 54%** on academic papers, producing 43-token fragments, and 14x slower.

### Layer 2 — the paper card

Structured JSON per read paper. **Job: coverage bookkeeping and cross-paper comparison. Never the ground for a claim.**

```json
{
  "paper_id": "...",
  "problem": {"value": "...", "unit_id": "...", "quote": "..."},
  "method": {"value": "...", "unit_id": "...", "quote": "..."},
  "datasets": [{"value": "...", "unit_id": "...", "quote": "..."}],
  "metrics": [{"metric": "...", "value": "...", "unit_id": "...", "quote": "...", "binding_verified": true}],
  "claims": [{"text": "...", "unit_id": "...", "quote": "..."}],
  "limitations": [...]
}
```

**Every field carries a `unit_id` and a verbatim `quote`.** The card is an index over evidence, not a replacement for it.

**Why the card is demoted from my first proposal.** *Diagnosing Structural Failures in LLM-Based Evidence Extraction* (arXiv 2602.10881) shows LLMs handle isolated entity extraction well but fail at **preserving roles, methods, and effect-size attribution** — the relational binding. Precisely the fields that carry the value. Counter-evidence keeping cards alive: otto-SR reached **93.1%** extraction accuracy vs **79.7%** for dual human reviewers, with blinded humans siding with the machine ~70% of the time on disagreements — but that was a tight schema with explicit verification.

Therefore: fields requiring relational binding (`metrics`) carry `binding_verified` and go through the verification pass in §8. Un-verified bindings are surfaced as such.

### Provenance metadata

Attached at paper level, and injected into the search context of every unit:

- Citation count, venue, publication year (Semantic Scholar / OpenAlex)
- **Retraction status** (Crossref) — citing a retracted paper is the worst possible failure for this system and is a cheap lookup
- Version identity: arXiv v1/v2/vN, preprint→published mapping, DOI resolution

## 6. Storage

**One SQLite file per project.** `FTS5` provides BM25 natively; `sqlite-vec` provides vector search. Zero infrastructure, single artifact, trivially portable and backed up.

Tables: `papers` · `units` · `embeddings` · `cards` · `claims` · `verifications` · `screen_log` · `runs`

`papers`, `units`, and `embeddings` stay normalized rather than collapsed — required to re-embed with a new model, replace a stale source, or audit which units belonged to a removed paper.

## 7. Pipeline

### Stage A — Gather (recall-optimized)

1. **Question → search plan.** Decompose the question into sub-questions and generate multiple query formulations per sub-question.
2. **Multi-source search.** Semantic Scholar, OpenAlex, arXiv, Crossref. Reuse `jarvis/tools/sources.py` for aggregation and dedup.
3. **Citation-graph expansion.** Walk references and citations outward from high-scoring seeds. PaperQA2 found citation traversal materially improved recall, and recall correlated with final answer accuracy. Reuse `jarvis/agents/citation_graph.py`.
4. **Paper-level dedup** via SPECTER2 embeddings — free from the Semantic Scholar API as `embedding.specterv2`, no compute required. Resolve preprint↔published pairs.

### Stage B — The gate (the recall ceiling of the whole system)

This is the highest-risk component in the design.

**SESR-Eval** (arXiv 2507.19027) benchmarked 9 LLMs on title-abstract screening across 24 systematic reviews and 34,528 records **in software engineering** — the closest published domain to robotics/ML:

| Model | Recall |
|---|---|
| GPT-4o | 0.66 |
| Llama 4 Maverick | 0.61 |
| o3-mini | 0.52 |
| Claude 3.7 Sonnet | 0.46 |
| DeepSeek R1 | 0.42 |

Verdict: *"no LLM managed a high recall with reasonable precision."* Medical and environmental domains report >95% for the same task — **this is domain-dependent and our domain is the bad one.** A single-LLM gate could silently discard a third to half of the relevant literature while the system reports confident, well-cited answers.

Human baseline for calibration: single-reviewer screening misses **13%** of relevant studies (87% sensitivity, 97% dual-reviewer). Field standard for screening tools is **≥95% recall**, explicitly accepting poor precision.

**Design:**

- The gate is a **union of cheap signals**, never a single LLM judgment: embedding similarity to the question, citation-graph proximity to confirmed-relevant seeds, keyword/BM25 match, and an LLM vote. Union, not intersection — intersection loses whatever any single signal misses.
- **Calibrated per project** against a hand-labeled seed of ~100 papers, tuned to ≥95% recall.
- Three outcomes, not two: `read_deep` · `unsure` · `defer`. **`unsure` escalates to deep read.** There is no `exclude`.
- `defer` is **demotion, not deletion** — the paper stays in the corpus at metadata+abstract depth and is recoverable when the question shifts.
- Every decision written to `screen_log` with its per-signal scores, so the gate is auditable and re-runnable without re-fetching.

### Stage C — Deep read

For `read_deep` and `unsure` papers: parse (Layer 0) → build typed units (Layer 1) → contextual prefixes → embed → extract card (Layer 2) → verify bindings.

### Stage D — Compile (precision-optimized)

Subagents over the already-gathered corpus. Retrieval is local and solved; optimize purely for precision.

**Retrieval is agentic and iterative, not single-shot.** Search exposed as tools a subagent calls repeatedly with refined queries. This is the generalizable lesson from both Claude Code's agentic search and PaperQA2's tool decomposition, arrived at independently. (What does *not* generalize from Claude Code is grep-over-embeddings: that is a code-specific result driven by unique function names and exact import strings. Scientific prose has no such identifiers — "wind rejection," "disturbance attenuation," and "gust tolerance" are the same concept.)

**Retrieval stack:** BM25 (FTS5) + vector (sqlite-vec) → fused by **Reciprocal Rank Fusion, k=60** → cross-encoder rerank. RRF rather than score normalization because cosine and BM25 scores live on incompatible scales and RRF needs no tuning.

**Embedding models — two different jobs:**

| Job | Model | Why |
|---|---|---|
| Paper-level: discovery, dedup, find-similar | **SPECTER2** | Citation-informed, 6M triplets, 23 fields, SoTA on citation recommendation. Free via S2 API. |
| Unit-level: retrieval inside the corpus | **BGE-class general embedder** | A medical retrieval study found BGE, a small *general-domain* model, beat every domain-specific medical model. |

**Evidence budget is capped and ordered.** *Parsing and Evaluating Source Attribution in LLM Deep Research* (arXiv 2605.06635) found **increased search depth consistently degrades factual accuracy while surface-level citation metrics stay stable** — an information-overload effect that looks like improvement. Corroborated: order-preserving retrieval with 48K well-chosen tokens beats full-context 117K by **13 F1 points** at one-seventh the budget.

Therefore: hard cap on units per synthesis call; many small well-scoped calls rather than one large one; strongest evidence placed at the **beginning and end** of context to exploit primacy/recency and avoid lost-in-the-middle (20+ pp degradation).

**Long-form reports** follow AutoSurvey's decomposition: outline from paper-level cards → parallel subsection drafting with a bounded evidence set each → integration pass → verification pass.

## 8. Verification — the differentiator

This is what makes the system not-NotebookLM and not-SurfSense. Both ship "Perplexity-style cited answers," which is the artifact ALCE measures at ~50% support.

Two failure modes, handled separately:

**Citation hallucination** — the reference does not exist, or the quote is not in it.
→ **Deterministic verbatim string match** of the quote against Layer 0. Free, exact, catches fabrication with no model involved. Target: **100%**. Any failure blocks the claim.

**Statement hallucination** — the paper is real, the quote is real, but it does not say what is claimed. The dangerous one, because the link works and the source checks out.
→ **NLI entailment model** over (quote → claim).

**Do not use LLM-as-judge for this.** Correlation with human judgment on citation support:

| Method | Pearson |
|---|---|
| GPT-3.5 as judge (continuous) | 0.023 |
| GPT-3.5 as judge (discrete) | **0.101** |
| SummaC-Conv | 0.565 |
| AlignScore | 0.585 |
| **AutoAIS (NLI)** | **0.638** |

A model judging its own citation support is barely better than noise. Entailment models are ~6x better correlated, run locally, cost nothing per call, and are deterministic. AttributionBench found even *fine-tuned* GPT-3.5 reaches only ~80% macro-F1 on binary attribution — so this stage is a filter, not an oracle, and low-confidence results surface as flagged rather than silently passed.

### Contradiction detection — free from the same pass

NLI emits three labels: entailment, neutral, **contradiction**. The verification stage already computes them. Running claims from one paper against evidence from others yields cross-corpus conflicts at no additional cost.

PaperQA2's ContraCrow found **2.34 ± 1.99 contradictions per paper** across 93 biology papers, **70% validated by human experts.** This is a genuinely superhuman-shaped task — "one versus many" per claim, "many versus many" across a corpus — and it is structurally impossible for any system that does not retain a corpus.

Caveat: multi-document contradiction detection is hard for LLMs (arXiv 2504.00180) and for humans. **Ship as ranked candidates for human review, never as assertions.**

## 9. Subagent boundaries

Spawn only for context isolation, parallelization, or verification. Cap fan-out.

| Subagent | Input | Output | Parallel |
|---|---|---|---|
| `screener` | paper metadata batch | gate decisions + per-signal scores | yes, batched |
| `reader` | one paper | Layer 0 + Layer 1 units | yes, one per paper |
| `carder` | one paper's units | Layer 2 card | yes, one per paper |
| `retriever` | sub-question | ranked units, iteratively refined | yes, one per sub-question |
| `writer` | sub-question + bounded evidence | draft section + claims | yes, one per section |
| `verifier` | claims + cited units | pass/fail/flag per claim | yes, batched |

The **writer never verifies its own output** — separate subagent, separate context. This is the one multi-agent principle the literature is unambiguous about.

## 10. Evaluation harness

Built **before** capability expansion, not after. Without it, "the corpus is good" is unfalsifiable.

| Metric | Method | Target |
|---|---|---|
| Gate recall | Hand-labeled seed set, ~100 papers/project | **≥95%** |
| Quote fidelity | Deterministic string match | **100%** |
| Statement support | NLI entailment over (quote → claim) | ≥90% |
| Citation precision/recall | ALCE-style, NLI-based | tracked, no target v1 |
| Coverage | Fraction of `read_deep` corpus cited across a report | tracked |
| Contradiction precision | Human review of top-N candidates | ≥70% (ContraCrow parity) |
| Cost per project | Router logs, not estimates | tracked |

Note: **human citation lists are not ground truth** (arXiv 2605.29234). Do not score gather-recall against a paper's own bibliography.

## 11. Deliberate non-goals, with reasons

| Not building | Why |
|---|---|
| Knowledge graph | GraphRAG-Bench (ICLR 2026): *"GraphRAG frequently underperforms vanilla RAG on many real-world tasks."* Wins on multi-hop/global sensemaking, loses on fact lookup; dense narrative documents are its weak case. The citation graph stays — as a **gathering** tool for recall, not a retrieval substrate. |
| Semantic chunking | 69% vs 54% against recursive-512 on academic papers; 14x slower. |
| LLM-as-judge verification | 0.101 Pearson. See §8. |
| Multimodal embedding retrieval | 0.000 → 0.143 accuracy. Typed units are cheaper and work better. |
| Permanent cross-project library | Per-project scope chosen; removes the entire dedup/versioning/migration burden. |
| Connectors, podcasts, decks | The surface area that made SurfSense expensive to maintain, and none of it is the hard part. |

## 12. Relationship to NanoResearch

This is a **standalone repo**, not an extension of the NanoResearch `jarvis` package. Useful primitives were **copied and adapted**, not imported — there is no runtime dependency in either direction, and neither can break the other.

### Ported

| Source (NanoResearch/jarvis) | Now | Changes on port |
|---|---|---|
| `tools/sources.py` | `jarvis/sources.py` | Dropped `has_code`/`code_url` — repo-analysis fields, out of scope |
| `agents/citation_graph.py` | `jarvis/citation_graph.py` | Added `doi`/`venue` to the S2 normalizer; `already_indexed` → `already_seen` |
| `orchestrator/model_router.py` | `jarvis/router.py` | Retargeted at corpus tasks; routes by **tier**, not vendor. Verification deliberately has no route — §8 puts it on a local NLI model, and a test asserts no `verif*` task is routable to an LLM |
| `agents/llm.py` | `jarvis/llm.py` | Added system-message and `run_id` support; `JARVIS_*` env only |
| `config.py` | `jarvis/config.py` | `skills_root` → `project_root`; added `project_dir(name)` |
| `memory/entity_resolver.cosine` + `agents/relevancy.make_cosine_scorer` | `jarvis/scoring.py` | Extracted as dependency-free primitives |

### Deliberately not ported

- **`agents/relevancy.RelevancyGate`** — its 4-stage funnel uses *hard* cosine (≥0.72) and LLM (≥6/10) cutoffs, which is precisely the failure mode §7B exists to prevent. The gate is rebuilt from scratch as a calibrated union with no `exclude` outcome. Only `make_cosine_scorer` is kept, because the citation walker needs a scorer.
- **`memory/knowledge_graph.py`** and the entity-resolution stack — §11.
- **`self_learning/`** — the approval/promotion runtime solves a different problem.
- Gateway, skills, evolution, and web-app code — out of scope entirely.

`KnowledgeBaseIndex`-style "never read the same paper twice" behaviour is subsumed by the `papers` table in §6 and needs no separate store.

## 13. Build order

1. **Storage + data model.** SQLite schema, Layer 0/1/2 types, ingest one paper end to end.
2. **Parse + typed units.** Docling integration, unit typing, binding rules, contextual prefixes.
3. **Retrieval.** FTS5 + sqlite-vec + RRF + rerank, exposed as agent tools.
4. **Verification.** Verbatim matcher, then NLI entailment. *Before* any synthesis exists.
5. **Eval harness.** Seed labeling tool, the metrics in §10.
6. **Gather + gate.** Multi-source, citation expansion, calibrated union gate.
7. **Compile — Q&A.** Iterative retriever + writer + verifier subagents.
8. **MCP server.** Expose the corpus as tools; usable from Claude Code before any UI exists.
9. **Contradiction detection.** Reuse the NLI pass.
10. **Compile — long-form reports.** AutoSurvey-style decomposition.

Steps 1–5 produce a system that can prove it is correct on a single paper. Nothing before step 5 is trustworthy without it.

**This spec covers more than one implementation plan.** Steps 1–5 are the first plan — the verifiable single-paper core, buildable and testable end to end with hand-authored claims and no gathering. Steps 6–10 each warrant their own plan, written once the core measures correctly.

## 14. Risks

| Risk | Mitigation |
|---|---|
| **Gate silently drops relevant papers** — the top risk | Union of signals, ≥95% calibration, `unsure`→read, demotion not deletion, auditable log |
| Card binding errors (method×dataset×metric) | `binding_verified` flag, verification pass, surface unverified as such |
| Information overload degrades accuracy invisibly | Hard evidence caps, order-preserving placement, coverage tracked separately from citation count |
| Parsing failures on hard PDFs | Docling default, MinerU escalation, parse-failure log, never silently ingest an empty parse |
| Citing retracted work | Crossref retraction check at ingest and again at compile time |
| Cost overrun on deep reads | Router logs measured per run; adaptive depth is the control |
| Scope creep toward SurfSense's feature surface | §3 and §11 are the contract |

## 15. Open questions

1. **Which NLI model.** AlignScore vs a DeBERTa-v3-MNLI class model vs AutoAIS reimplementation. Decide empirically on the seed set in build step 4.
2. **VLM descriptions for figures** — worth the cost, or is caption + referring text sufficient? Measure at build step 2 before committing.
3. **Gate calibration transfer** — does a threshold calibrated on one project's seed set transfer to the next, or is per-project labeling required every time?
4. **Reranker**: local `bge-reranker-v2-m3` vs a hosted API. Latency and cost measurement in build step 3.

## 16. References

Ordered by how much they shaped this design.

- **SESR-Eval** (arXiv 2507.19027) — screening recall by domain; the gate risk
- **PaperQA2 / ContraCrow** (arXiv 2409.13740) — agentic tools, RCS, citation traversal, contradiction detection
- **OpenScholar** (Nature, arXiv 2411.14199) — coverage vs precision, self-feedback, ScholarQABench
- **Diagnosing Structural Failures in LLM-Based Evidence Extraction** (arXiv 2602.10881) — relational binding failure
- **Parsing and Evaluating Source Attribution in LLM Deep Research** (arXiv 2605.06635) — information-overload effect
- **Towards Fine-Grained Citation Evaluation** (CWI) — NLI vs LLM-judge correlations
- **ALCE** (arXiv 2305.14627), **AttributionBench** (arXiv 2402.15089), **CiteME/CiteGuard** — attribution ceilings
- **SPIQA** (arXiv 2407.09413) — table/figure prevalence and model performance by artifact type
- **When Do Multimodal and Graph-Augmented RAG Help?** (arXiv 2607.16604) — multimodal retrieval ceiling
- **GraphRAG-Bench** (ICLR 2026) — graph underperformance on narrative documents
- **Is Grep All You Need?** (arXiv 2605.15184) — harness effects, delivery format, lexical vs dense
- **Anthropic Contextual Retrieval** (Sept 2024) — 49–67% retrieval failure reduction
- **AutoSurvey** (NeurIPS 2024) — long-form decomposition
- **SPECTER2 / SciRepEval** (EMNLP 2023) — paper-level embeddings
- **otto-SR** (medRxiv 2025.06.13) — structured extraction accuracy vs humans
- **SurfSense** — convergent retrieval stack; the niche-abandonment signal
