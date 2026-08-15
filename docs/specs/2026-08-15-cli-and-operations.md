# CLI and operations — design spec

Status: draft, not yet implemented
Supersedes nothing. Extends `docs/specs/2026-08-11-research-corpus-agent-design.md`
(hereafter "the design spec"), which is complete through all ten of its build steps.

## 1. Problem

All ten build steps of the design spec exist, are tested, and are merged to `main`. The
system cannot be run.

`pyproject.toml` declares exactly one entry point — `jarvis-mcp` — and it is deliberately
read-only: gathering and screening were excluded from the MCP surface because they spend
money and rewrite the corpus (`LEDGER-mcp-server.md`, "Where this stops"). Every pipeline
stage exists as a library function. Nothing wires them together.

The consequence is concrete: **no corpus has ever been built.** `~/.jarvis/projects` does
not exist on the machine this was developed on. Starting `jarvis-mcp` today requires a
`--db` pointing at a `corpus.db` that has never been created. Every remaining open
question in the design spec — contradiction precision against the 70% target, gate
calibration transfer, reranker choice, cost per project — is blocked behind the same
missing thing.

This is not a defect in any plan. The design spec's build order (§13) never included an
operator surface. It is the gap between "the spec is fully built" and "the system is
usable."

## 2. Goal

A command surface that can take a research question from nothing to a queryable, citable
corpus, and then to answers, reports, and contradiction candidates — without anyone
writing a Python script.

Success is measured by one thing: **a real gather run completes end to end on a real
research question, and the resulting corpus answers a question with verified citations.**
Not a test fixture. Not a fake model.

## 3. Scope

**In scope**
- A single `jarvis` command with subcommands covering the full pipeline
- PDF acquisition and local caching — the missing stage between screening and parsing
- Card extraction wired into the ingest path
- Measured cost written into `runs.cost_usd`
- Resumable runs: a corpus half-built is a corpus, not a loss
- Human confirmation before irreversible spend

**Explicitly out of scope**
- Web UI — the design spec §3 rules this out for v1 ("CLI and MCP only", "no server"), and
  nothing learned since changes that. Revisit only with a real corpus to design against.
- HTTP/SSE MCP transport. A worthwhile later change (the adapter was built so protocol
  concerns live in one file with zero corpus logic), but it serves a corpus that does not
  yet exist. Sequence it after the first real gather, not before.
- Daemons, schedulers, queues, multi-user anything.
- Any new retrieval, verification, or synthesis logic. This spec adds an operator surface
  over existing functions and closes three specific pipeline gaps. Nothing else.

## 4. Two decisions made without a ruling

Both were raised and left unanswered; both are made here so the work is not blocked.
Either can be overruled cheaply before implementation starts.

**One command, subcommands.** `jarvis gather`, `jarvis ask`, `jarvis report`,
`jarvis contradictions`, `jarvis status`. Rationale: one entry point to install and
document, and every subcommand shares the same project resolution, config loading, and
store lifecycle — which is most of what a driver does. `jarvis-mcp` stays as a separate
declared script because it is already shipped and referenced in MCP client configuration,
and breaking it would break existing setups; it also becomes reachable as `jarvis mcp`.

**Gather pauses before deep reads.** The gate promotes both `read_deep` and `unsure` to
`pending_deep` (design spec §7B — `unsure` is read, never dropped). That is correct for
recall and expensive for a wallet: it is the point where a run stops costing search-API
calls and starts costing parse and embedding work on every survivor. `jarvis gather`
stops there by default, prints the decision counts, and waits. `--yes` proceeds
unattended. Rationale: the design spec's own risk table names cost overrun on deep reads
as a live risk whose control is adaptive depth — a confirmation point makes that control
reachable by a human on the first run, when nobody yet knows what a gather on this corpus
actually costs.

## 5. Three gaps this spec must close

These are not commands. They are missing pipeline links, found by reading the code rather
than the plans. Each blocks a real run.

### 5.1 There is no PDF acquisition step

`ingest_decided`'s default `path_for` resolves a candidate to `paper["pdf_url"]` or
`paper["url"]` — a URL. That string is handed to `Parser.parse(path, paper_id)`, and
`DoclingParser.parse` passes it straight to `DocumentConverter().convert(path)`.

Nothing in the repository downloads a file, caches one, retries a failure, or records
where a source came from. Whether Docling resolves a remote URL directly is **unverified
here** — no test exercises it, because every test uses `FakeParser`. Even if it does, a
real run needs the source PDF on disk: to re-parse without re-fetching when the parser is
upgraded, to escalate a bad parse to a stronger parser (design spec §14's stated
mitigation), and to make `papers.source_path` mean something durable.

**Required:** a fetch-and-cache step between screening and parsing. Local cache under the
project directory, keyed by paper id. `path_for` resolves to the cached local path. A
paper whose PDF cannot be fetched is a per-paper failure, never a run failure — matching
`ingest_paper`'s existing contract that a bad PDF is data, not a crash. Unpaywall is
already in `sources.py` for locating open-access PDFs and should be the fallback when a
candidate carries no direct `pdf_url`.

### 5.2 Card extraction is never called outside tests

`extract_and_verify` appears in `jarvis/__init__.py`'s exports, in `tests/test_card.py`,
and in `tests/test_gather_end_to_end.py`. It is called by no production code path.
`ingest_paper` does not extract cards.

This matters because `write_report` builds its outline from `corpus_cards(conn)`. On a
corpus built by a gather run as the code stands today, that returns nothing, and every
report comes out empty — the failure would look like a report bug and would not be one.

**Required:** card extraction runs as part of, or immediately after, deep ingestion, with
its own failure isolation. It needs a model, so it belongs on the same confirmation gate
as deep reads.

### 5.3 Nothing writes measured cost

`save_run(conn, run_id, question, started_at, cost_usd)` accepts a cost. `router.py` has
`CostTracker`/`ModelRouter`. The `runs.cost_usd` column exists. No code connects them —
a known loose end since the gather branch, and the design spec §10's one unclaimed metric.

**Required:** the gather command owns a `ModelRouter` for the whole run and writes
`router.cost.total_cost` into `save_run` when the run finishes, including when it finishes
badly. A cost number that only appears on success is not a cost control.

## 6. Command surface

| Command | Wires | Notes |
|---|---|---|
| `jarvis gather "<question>" --project <name>` | `LLMPlanner`/`TemplatePlanner` → `combine_sources(...)` → `gather()` → `save_candidates()` → `screen()` → **fetch** → `ingest_decided()` → **extract_and_verify** | The whole of stages A–C. Pauses before deep reads unless `--yes`. |
| `jarvis ask "<question>" --project <name>` | `ask()` → `render_answer()` | Needs writer + NLI. |
| `jarvis report "<topic>" --project <name>` | `write_report()` → `render_report()` | Needs cards to exist (§5.2). |
| `jarvis contradictions --project <name>` | `scan_corpus()` → `rank()` → `write_review_sheet()` / `render_conflicts()` | Claims come from stored answers/report drafts; see §9. |
| `jarvis review --project <name> <sheet>` | `read_reviews()` → `apply_reviews()` → `contradiction_precision` | Closes the human-review loop and produces the 70%-target number. |
| `jarvis status --project <name>` | `get_papers_by_depth()`, run rows | Paper counts by depth, last run, measured cost. |
| `jarvis calibrate --project <name>` | `label.py` round-trip → `calibrate()` → `calibration_report()` | Gate calibration against a hand-labeled seed set (design spec §7B). |
| `jarvis mcp --project <name>` | existing `mcp_server.main` | Alias; `jarvis-mcp` keeps working unchanged. |

Every command resolves its store through `Config.load().project_dir(name)`, one SQLite
file per project (design spec §6). `--db` remains available as an explicit override.

## 7. Model configuration

The pipeline's real (non-fake) implementations — `LLMPlanner`, `LLMVoter`,
`LLMCardExtractor`, `LLMRefiner`, `LLMWriter`, `LLMOutliner`, `BGEEmbedder`, `HFNLI`,
`DoclingParser` — have never run against a live endpoint. Every one is exercised only
against fakes or type signatures.

The CLI is where that changes, and it should change loudly rather than quietly: a
subcommand that needs a model it cannot construct must fail immediately with the name of
the missing variable, not proceed with a fake. Silent fallback to a fake model in an
operator tool would produce a corpus that looks real and is not — the exact failure class
this whole system exists to prevent.

Required environment is already defined by `Config.load()`: `JARVIS_BASE_URL`,
`JARVIS_API_KEY`, `UNPAYWALL_EMAIL`, optionally `S2_API_KEY`, `JARVIS_PROJECT_ROOT`,
`JARVIS_CONFIG`, `JARVIS_MODEL_<TASK>`. Extras: `llm`, `parse`, `index`, `verify`.

## 8. Failure, resumability, and honesty

- **Per-paper failures never end a run.** Already the contract in `ingest_decided`; the
  CLI must preserve it through fetch and card extraction too, and must report the count of
  failures at the end rather than burying it.
- **A partial gather is resumable.** Depth transitions (`metadata` → `pending_deep` →
  `deep`) already encode progress in the database. Re-running `gather` on an existing
  project must pick up `pending_deep` papers rather than re-searching from scratch.
- **A systemic failure must not read as a clean result.** `scan_corpus` was fixed on the
  contradiction-detection branch for exactly this — every claim failing looked identical
  to a clean corpus until a warning was added. The CLI has the same hazard at a larger
  scale: zero papers fetched, zero units indexed, and zero contradictions are all valid
  outputs and all suspicious. Report counts at every stage; a stage that processed nothing
  says so.

## 9. Where claims come from for a corpus scan

`scan_corpus(conn, claims, nli, embedder, ...)` takes claims as an argument. Answers and
report sections produce claims; nothing persists them as a corpus-wide set. The simplest
honest resolution: `jarvis contradictions` scans the claims from the most recent report
(or an explicit `--from-report`), because a report is the artifact that already spans the
corpus. Scanning a corpus with no report yet is a legitimate error, not an empty result.

This is a design decision worth revisiting once a real corpus exists — it is the one place
this spec is least confident, and the first real run will inform it more than reasoning
will.

## 10. Build order

1. **Project resolution, config, and `jarvis status`.** Smallest possible end-to-end slice:
   open a store, report what is in it. Establishes the shared plumbing every other
   subcommand uses.
2. **PDF fetch and cache** (§5.1), with per-paper failure isolation and Unpaywall fallback.
3. **`jarvis gather`**, including the confirmation gate, card extraction (§5.2), cost
   logging (§5.3), and resumability.
4. **`jarvis ask` and `jarvis report`.** Thin; the hard parts already exist.
5. **`jarvis contradictions` and `jarvis review`**, closing the loop to
   `contradiction_precision`.
6. **`jarvis calibrate`**, the gate's hand-label round trip.

Steps 1–3 are the ones that unblock everything. Steps 4–6 are thin wrappers over merged,
reviewed code.

## 11. Risks

| Risk | Mitigation |
|---|---|
| **The first live model run surfaces integration bugs the offline suite cannot catch** — the most likely outcome, given every `LLM*` class is fake-tested only | Expect it; treat the first gather as a debugging exercise, not a measurement. Run on a deliberately small budget first. |
| PDF fetching is hostile in practice — paywalls, redirects, rate limits, HTML-not-PDF | Per-paper isolation, cache, Unpaywall fallback, explicit failure counts. Accept a fetch success rate well below 100% on the first run. |
| Cost overrun on the first real gather | The §4 confirmation gate, a `--budget` cap carried into `gather()`'s existing `budget` parameter, and measured cost written even on failure. |
| Docling on real PDFs behaves unlike `FakeParser` in ways that corrupt units | `ingest_paper` already refuses an empty parse. Watch unit counts per paper on the first run; a paper yielding one giant unit is a parse failure wearing a success. |
| Scope creep from "CLI" into "product" | §3's out-of-scope list, and the design spec §11's own warning about surface area. |

## 12. Open questions

1. **Does Docling accept a URL directly?** Determines whether §5.1's fetch step is
   strictly required or merely correct. Answer empirically before building step 2.
2. **What is the real fetch success rate** on a typical 100–300 candidate gather? Nobody
   knows; it sets expectations for every downstream metric.
3. **Should card extraction be per-paper at ingest, or a batch pass afterward?** Batch is
   cheaper to retry and easier to skip; per-paper keeps a paper's lifecycle in one place.
4. **§9's claim-source decision** for corpus-wide contradiction scanning.
5. Everything still open in the design spec §15 — all five remain blocked on a real corpus,
   which is what this spec exists to make possible.
