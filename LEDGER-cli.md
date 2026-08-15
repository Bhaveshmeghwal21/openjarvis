# Ledger — cli-and-operations

Spec: `docs/specs/2026-08-15-cli-and-operations.md`
Branch: `cli-and-operations`, based on `main` at `997a1a4` (all ten spec build steps of
`docs/specs/2026-08-11-research-corpus-agent-design.md`)

## Architecture

The system's ten build steps existed as library functions with no operator surface —
`pyproject.toml` declared exactly one entry point (`jarvis-mcp`, deliberately read-only),
and no corpus had ever been built anywhere. This branch adds one command, `jarvis`, with
eight subcommands (`status`, `gather`, `ask`, `report`, `contradictions`, `review`,
`calibrate`, `mcp`), each a thin wrapper over already-merged, already-reviewed library
functions — plus three real pipeline gaps found by reading the code rather than the
plans, closed as part of this build:

- **§5.1 — no PDF acquisition.** `ingest_decided`'s default `path_for` resolved to a URL
  string, handed straight to a parser. New `jarvis/fetch.py` downloads and caches PDFs,
  rejecting anything that isn't actually a PDF by magic bytes rather than trusting a
  `content-type` header.
- **§5.2 — card extraction never called outside tests.** `extract_and_verify` was
  exported and tested but wired into no production path; `write_report`'s outline
  generation depends on it entirely. `cmd_gather` now calls it per successfully-ingested
  paper.
- **§5.3 — nothing wrote `runs.cost_usd`.** `cmd_gather` owns one `ModelRouter` for the
  whole run and writes its measured cost via `save_run` in a `finally`, so a run that
  crashes mid-gather still records what it spent.

A fourth gap, not named in the spec but discovered while building `jarvis contradictions`:
**there was no report-persistence layer anywhere.** `write_report` returns a `Report` held
only in memory; spec §9 says contradiction scanning reads claims from "the most recent
report," which is unimplementable without something to persist. `cmd_report` now writes a
small JSON sidecar (`claim_id`/`text`/`unit_id`/`quote`) next to its markdown, which
`cmd_contradictions` reads by default (or via `--from-report`).

## Tasks 1-8

All 8 plan tasks implemented directly, one at a time, strict TDD throughout:

```
06afbb4 feat: jarvis mcp alias and a real README quickstart              [Task 8]
7970caa feat: jarvis calibrate - the gate's hand-label round trip        [Task 7]
1adeb14 feat: jarvis contradictions and jarvis review                    [Task 6]
b2242dc feat: jarvis ask and jarvis report                              [Task 5]
0119563 feat: jarvis gather - wires stages A-C, closes spec gaps 5.2/5.3 [Task 4]
dbfe326 feat: PDF fetch and cache - closes spec gap 5.1                 [Task 3]
1d1efec feat: model construction with a fail-loud contract              [Task 2]
fcf77df feat: jarvis CLI skeleton - project resolution, jarvis status   [Task 1]
```

Pre-flight scans before every task (reading `Config`, `open_store`/`close_store`,
`get_papers_by_depth`, `save_run`, `ModelRouter`/`CostTracker`, `gather`/`save_candidates`/
`screen`/`ingest_decided`/`combine_sources`/`extract_and_verify`/`ask`, `write_report`/
`render_report`/`corpus_cards`, `scan_corpus`/`rank`/`write_review_sheet`/`render_conflicts`/
`read_reviews`/`apply_reviews`/`contradiction_precision`, `sample_seed`/`write_label_sheet`/
`read_labels`/`calibrate`/`calibration_report`/`label_progress`, the `LLM*` class
constructors, `mcp_server.main`) found zero interface conflicts — every signature this
build assumed matched exactly.

Real correctness issues found and fixed at implementation time (not by the final review —
found while building, verified before trusting):

- **Task 4**: `citation_graph.paper_id()` resolves a candidate's id from `arxiv_id`/`s2_id`
  before falling back to a title prefix. `_resumable_candidates`, which reconstructs
  `Candidate` objects from stored `pending_deep` papers for resumption, would otherwise
  reconstruct a *different* id than the paper's actual stored `paper_id` for any paper
  lacking both real id fields — silently breaking every subsequent `save_paper`/`set_depth`
  lookup keyed on the original id. Fixed by seeding the reconstructed `arxiv_id` with the
  stored `paper_id` itself when empty. Verified via a standalone `paper_id()` round-trip
  check before relying on it in the CLI, not assumed correct.
- **Task 8 (README verification)**: every command in the README's new Quickstart section
  was checked to actually parse against the real argparse tree before committing — caught
  a real ordering bug: `jarvis calibrate`'s `--project`/`--db` must precede the
  `seed`/`fit`/`progress` subcommand token (nested-subparser convention), not follow it as
  first drafted.

## Final state after Tasks 1-8

- **626 tests total** (569 pre-existing + 57 new across 8 new test files), all passing.
- **Whole-repo `ruff check .`: exactly 11 violations**, all pre-existing, unchanged
  baseline (5 `BLE001`, 3 `UP035`, 1 `I001`, 1 `S112`, 1 `UP037`).

## Final whole-branch adversarial review

Dispatched via subagent (2 retries needed due to transient network `DispatchFailure`
errors on the first two attempts; the third succeeded with a condensed prompt — no code
was affected, purely a dispatch-infrastructure issue). Framed explicitly around: (a) the
claim-id-collision defect class, checked a fourth time given its history on three prior
branches; (b) the fail-loud model-construction contract under adversarial, not just clean,
conditions; (c) whether the `--yes` confirmation gate in `cmd_gather` is genuinely
unbypassable, including via resumption; (d) cost-logging correctness under every failure
mode; (e) the claims-sidecar JSON round-trip under adversarial input; (f) path traversal
across every new file-writing code path; (g) a general robustness pass.

**Result: 0 Critical, 2 Important, 2 Minor findings** — every finding independently
reproduced live before being trusted, several areas explicitly confirmed clean rather than
left unchecked.

| # | Finding | Severity | Category | Disposition |
|---|---|---|---|---|
| 1 | The claim-id-collision defect class, checked a 4th time. | — | — | **Not found.** `Report.all_claims` can contain colliding ids across sections (pre-existing `report.py` behavior — `draft_section`'s dedup is scoped per section, not across the whole report), and the JSON sidecar faithfully round-trips that collision without creating a new one or silently fixing it. But `cmd_contradictions` hands loaded claims straight to `scan_corpus`, which applies `_dedupe_claim_ids` as its first statement — confirmed live: two claims sharing id `"c1"` became `["c1", "dedup-1"]` with texts preserved distinctly. `cmd_contradictions` correctly never assumes the sidecar is pre-deduped. |
| 2 | `_require_chat_credentials` accepted whitespace-only `JARVIS_BASE_URL`/`JARVIS_API_KEY` — Python truthiness on a bare `if not config.base_url`. A realistic shell copy-paste artifact (`JARVIS_API_KEY="   "`) would pass this check, reach `openai.OpenAI(...)` inside an `LLM*` class, fail there, and be silently swallowed by that class's own broad `except Exception` into an empty `Draft()` — indistinguishable from a genuinely empty corpus. | **Important** | 3 (uncovered gap — the check existed but didn't fully realize its own stated intent) | **Fixed.** Strip before checking on both variables. Independently reproduced before fixing (whitespace-only values passed the gate, `LLMWriter.write()` returned an empty `Draft`), reproduced-closed after fixing. Mutation-tested: reverting the `strip()` calls makes both new regression tests fail with the exact expected `DID NOT RAISE ModelBuildError`, restoring makes them pass again. |
| 3 | `resolve_db_path` passed `--project` straight into `Config.load().project_dir(name)`, a bare `Path.__truediv__` with no sanitization. `--project ../../../etc` or an absolute path resolved entirely outside `$JARVIS_PROJECT_ROOT`, and every subsequent subcommand's file I/O (corpus.db, PDF cache, report sidecars, review sheets, label sheets) would silently operate there. | **Important** | 3 (uncovered gap — `--db` was documented as the trusted, unsanitized override; `--project` was never meant to be, but wasn't actually checked) | **Fixed.** Resolve both the project root and the candidate path; require the candidate be the root itself or a descendant of it, else a named `SystemExit(2)`. `--db` remains an explicit, unsanitized override by design — only `--project`'s name-based resolution needed this. Independently reproduced before fixing (`../../../etc` and an absolute Windows path both resolved outside the root), reproduced-closed after. Mutation-tested: reverting the guard makes both new regression tests fail with the exact expected `DID NOT RAISE SystemExit`, restoring makes them pass again. |
| 4 | `_require_importable` only checks the top-level import succeeds; `BGEEmbedder`/`HFNLI`/`DoclingParser` construct lazily, so a partially-broken extra (importable but a sub-dependency fails at actual use time) fails later, outside any `ModelBuildError` handler. | Minor | non-issue | **No fix needed, verified directly rather than assumed.** Reproduced live: a simulated broken `SentenceTransformer` let `build_embedder` succeed, then `.encode()` raised a raw `OSError` — but `main()`'s own outer `except Exception` still catches it, prints a named error, and exits 1. Never a bare traceback. Confirmed by independently injecting a raising handler into `main()`'s dispatch and checking the output. |
| 5 | `_resumable_candidates`/`_screened_candidates`'s shared `arxiv_id or paper.paper_id` reconstruction trick could theoretically let two different stored papers collide to the same reconstructed candidate id. | Minor | non-issue | **No fix needed, verified directly.** The theoretical collision only occurs in a contrived scenario (two distinct stored papers whose `paper_id`s happen to coincide, which can't happen since `paper_id` is the store's own primary key). The realistic case — both papers genuinely lacking `arxiv_id`/`s2_id` — does not collide, since the fallback is a no-op that preserves each paper's already-distinct stored id. Confirmed both the contrived and the common case live. |

Zero findings, explicitly confirmed live rather than assumed: the `--yes` confirmation
gate — ran `jarvis gather` twice without `--yes` on the same project (first run screens
and leaves papers at `pending_deep`; second run resumes via `_resumable_candidates`, prints
"resuming...", but `if not args.yes: return 0` applies uniformly regardless of whether the
candidates came from a fresh search or resumption) — `deep` depth stayed empty after both
runs, confirming resumption cannot bypass the gate. Cost logging under every failure mode
— `save_run` with `router.cost.total_cost` fires in `cmd_gather`'s `finally` even when
`screen()` crashes before any real cost was logged (row written with `cost_usd=0.0`, exit
1), and `CostTracker.total_cost` genuinely accumulates across multiple `build_*` calls
sharing one `ModelRouter` instance (confirmed: cost after one logged call vs. after a
second call on the same router both reflected correctly, not reset). `_cache_path`'s
sanitizer in `jarvis/fetch.py` — confirmed live against 9 adversarial `paper_id` values
(`../../../etc/passwd`, `con`, embedded null bytes, a 300-character string, others) — every
one resolved confined under the cache directory. `cmd_review`'s precision print — no
division-by-zero risk in the f-string itself; `contradiction_precision({})` is documented
to return `0.0`, never raises. `main()`'s early `mcp` interception (before `_build_parser()`
is even called) — confirmed `jarvis --help` still lists `mcp` in the full command list and
exits 0, and `jarvis mcp --help` delegates correctly to `mcp_server.main`'s own help and
exits 0; the special-case does not break either. Atomicity of `write_label_sheet`/
`write_review_sheet`/`_save_claims_sidecar` — all three use a single `path.write_text(...)`
call; grep-confirmed this is the exact same pattern every one of them already uses, so this
branch is consistent with, not a regression from, the established convention (no prior
branch uses temp-file-plus-rename for these sheets either).

## Fix wave and re-review

Fixed in `b1448d3`: both Important findings (whitespace-credential stripping;
`--project` path-traversal guard). Both Minor findings required no fix, documented above
with the verification that closed them.

Given no Critical finding, a full independent subagent re-review is not strictly mandatory
per this repo's established convention — but both fixes touch security-relevant surface
(credential validation, path traversal), so a rigorous self-mutation-test was performed in
its place rather than a quick re-read: temporarily reverted each fix's specific lines via
scratch edits, confirmed the corresponding regression tests fail with the exact expected
assertion (`DID NOT RAISE ModelBuildError` / `DID NOT RAISE SystemExit`), restored via
`git checkout`, confirmed `git status --short` empty after each restoration. Final state
re-confirmed clean and passing after both mutation tests.

Final state after the fix wave: **631 tests passing** (626 + 5 new regression tests: 2
whitespace-credential, 3 project-path-traversal). `ruff check .` still exactly the
11-violation baseline.

## What this branch deliberately does not build

Per the spec's own explicit out-of-scope list (§3):

| Not built | Why |
|---|---|
| Web UI | Design spec §3 rules this out for v1 ("CLI and MCP only", "no server"). Revisit only with a real corpus to design against. |
| HTTP/SSE MCP transport | Worthwhile later, but would serve a corpus that doesn't exist yet. Sequenced after the first real gather. |
| Daemons, schedulers, queues, multi-user anything | Out of scope per §3. |
| Any new retrieval, verification, or synthesis logic | This branch is an operator surface over existing, already-reviewed functions plus three specific gap-closers. Nothing else. |

The claims-sidecar mechanism this branch had to invent (§9's "claims come from the most
recent report") is explicitly flagged by the spec itself as its least-confident decision —
worth revisiting once a real corpus and a real report exist to inform it, rather than
reasoning about it further in the abstract.

**The plan's own stated success criterion — a real gather run completing end to end on a
real research question, with the resulting corpus answering a question with verified
citations — is not yet demonstrated.** Every test in this branch's 62 new tests uses fakes
(`FakeEmbedder`, `FakeParser`, stub writers/NLI/planners). No `LLM*` class, `BGEEmbedder`,
`HFNLI`, or `DoclingParser` has been exercised against a real endpoint or a real PDF on
this branch, or on any branch before it. This is the next real milestone, and per the
spec's own risk table, expect it to be a debugging exercise, not a measurement.

## Branch-finish checklist

- [x] All 8 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed, explicitly framed
      around this repo's own claim-id-collision history (not found a 4th time) plus the
      spec's own specific concerns (confirmation gate, cost logging, fail-loud contract,
      path safety).
- [x] Fix wave for both Important findings; two Minor findings required no fix, verified
      directly and documented.
- [x] Rigorous self-mutation-test of both fixes in place of a full independent re-review,
      justified by the absence of a Critical finding but performed anyway given the
      security-relevant surface both fixes touch.
- [x] This ledger written as the closing record.
- [ ] Branch finish (merge to `main`) — not yet actioned, pending explicit go-ahead.
