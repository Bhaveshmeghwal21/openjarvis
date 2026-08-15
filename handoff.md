# Handoff — jarvis

Updated 2026-08-15 (eleventh update). Supersedes the prior version, which described the
CLI as complete but unmerged. It is now merged to `main` and pushed. **The system can be
run for the first time in this project's history.**

**State in one line:** all ten spec build steps plus the CLI
(`docs/specs/2026-08-15-cli-and-operations.md`) are on `main` and pushed. There is no
remaining implementation work. The next milestone is the first real gather run — see "What
comes next" below.

## Orient yourself in 60 seconds

```
main:  19c6209 (all 10 spec steps + CLI), pushed, in sync with origin
```

| What | Where |
|---|---|
| The CLI spec | `docs/specs/2026-08-15-cli-and-operations.md` |
| The original design spec | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Record of the CLI's build + review | `LEDGER-cli.md` (on `main`) |
| Records of all 10 spec build steps' build + review | `LEDGER.md`, `LEDGER-gather-and-gate.md`, `LEDGER-compile-cited-qa.md`, `LEDGER-mcp-server.md`, `LEDGER-contradiction-detection.md`, `LEDGER-longform-reports.md` (all on `main`) |

```
main:
  HEAD:   19c6209  merge: CLI and operations (docs/specs/2026-08-15-cli-and-operations.md)
  tests:  631 passing
  ruff:   11 pre-existing violations, zero new
  remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync
```

## What changed since the last handoff

**`cli-and-operations` was merged into `main` and pushed.** Verified independently rather
than taken on trust before merging: full suite re-run on the branch (631 passing, exit 0)
and again on `main` itself after the merge (631 passing, exit 0, no conflicts), `ruff
check .` at the 11-violation baseline, and the ledger's specific claims spot-checked
directly against the actual code — the `%PDF` magic-byte check really is in `fetch.py`,
`cost_usd` really is written via `router.cost.total_cost` in a `finally`, `extract_and_verify`
really is called from `cmd_gather`, both Important fixes (whitespace-credential stripping,
`--project` path-traversal containment) really are in place, and `scan_corpus`'s existing
`_dedupe_claim_ids` guard really does cover claims loaded from the new JSON sidecar. The
merged worktree and local branch were removed afterward.

**The system can now be operated end to end from one command, `jarvis`**:
`jarvis status/gather/ask/report/contradictions/review/calibrate/mcp`. Three real pipeline
gaps found by reading the code (not the plans) were closed as part of this build:

1. **No PDF acquisition step existed.** New `jarvis/fetch.py` downloads and caches PDFs,
   verifying `%PDF` magic bytes rather than trusting a `content-type` header or a URL
   shape — a paywall's HTML login page saved at a PDF-looking URL is now a fetch failure,
   not a garbage parse.
2. **Card extraction was never called outside tests.** `cmd_gather` now calls
   `extract_and_verify` per successfully-ingested paper. Without this, `write_report`'s
   outline generation (which reads `corpus_cards(conn)`) would produce an empty report on
   any real gathered corpus, and it would look like a report bug.
3. **Nothing wrote `runs.cost_usd`.** `cmd_gather` owns one `ModelRouter` for the whole
   run and writes its measured cost via `save_run` in a `finally`, so a run that crashes
   mid-gather still records what it spent.

A fourth gap, not named in the spec but found while building `jarvis contradictions`:
**there was no report-persistence layer anywhere** — `write_report` returns a `Report`
held only in memory, and spec §9's own resolution ("scan claims from the most recent
report") is unimplementable without something to persist. `cmd_report` now writes a small
JSON claims sidecar next to its markdown for `cmd_contradictions` to read later.

## THE CLAIM-ID-COLLISION PATTERN — STILL THE MOST IMPORTANT THING IN THIS FILE

Checked a **fourth time** on this branch (`cmd_ask`/`cmd_report`/`cmd_contradictions` all
handle `Claim` objects, and `cmd_report`'s new JSON sidecar round-trips them for later
reuse). **Not found this time** — `scan_corpus`'s established `_dedupe_claim_ids` guard
correctly neutralizes a colliding-claim-id sidecar round-trip before any lookup happens,
confirmed live. This is now the pattern's fourth clean check-and-confirm after three real
Critical findings on three separate prior branches, which is exactly what having it as a
mandatory pre-flight/review checklist item is for.

**Restated for whoever reads this next**: if a change introduces a new function that
constructs, aggregates, ranks, or looks up claims (or objects derived from claims) by an
id, explicitly verify `jarvis.answer._dedupe_claim_ids` is applied before that id is used
as a lookup/grouping key. Do not rely on a review to catch it by accident a fifth time —
check it directly, the way a pre-flight scan checks every other interface assumption.

## What is built

All ten of the original design spec's build steps, plus the CLI operator surface over all
of them. **All on `main`.**

| Component | Status | Modules |
|---|---|---|
| Spec build steps 1-10 | done, `main` | see prior handoff versions / `LEDGER*.md` files on `main` |
| CLI (spec `2026-08-15-cli-and-operations.md`) | **done, `main`** | `cli.py`, `fetch.py` |

## The CLI's adversarial review — worth reading in full before touching cli.py or fetch.py

`LEDGER-cli.md` has the complete record. The final whole-branch review checked the
claim-id-collision pattern a fourth time (not found — see above) and found 0 Critical, 2
Important, 2 Minor findings, all independently reproduced live:

- **Important, fixed**: `_require_chat_credentials` accepted whitespace-only
  `JARVIS_BASE_URL`/`JARVIS_API_KEY` (Python truthiness on a bare `if not`) — a realistic
  shell copy-paste artifact that would silently degrade to "looks like an empty corpus"
  instead of failing loud. Fixed by stripping before checking.
- **Important, fixed**: `--project` flowed unsanitized into `Config.load().project_dir(name)`
  — `--project ../../../etc` or an absolute path resolved entirely outside
  `$JARVIS_PROJECT_ROOT`. Fixed by requiring the resolved candidate be the project root
  itself or a descendant of it. `--db` remains an explicit, unsanitized override by
  design — this only affected `--project`'s name-based resolution.
- **Minor, no fix needed (verified directly)**: a partially-broken optional extra (e.g. a
  sub-dependency of `sentence-transformers` failing at actual model-load time, not at the
  top-level import `_require_importable` checks) still produces a named error and exit 1
  via `main()`'s own outer exception handler — never a bare traceback.
- **Minor, no fix needed (verified directly)**: the id-reconstruction trick shared by
  `_resumable_candidates`/`_screened_candidates` (`arxiv_id or paper.paper_id`) only
  theoretically collides in a contrived scenario; the realistic case (a paper genuinely
  lacking both real id fields) does not collide.
- **Explicitly confirmed clean, not just unchecked**: the `--yes` confirmation gate cannot
  be bypassed via resumption (ran `gather` twice without `--yes` on the same project,
  confirmed `deep` depth stayed empty after both runs); cost logging fires correctly under
  every tested failure mode including a mid-run crash before any real cost was logged;
  `jarvis/fetch.py`'s path sanitizer correctly confines 9 adversarial `paper_id` values;
  `main()`'s early `mcp` interception doesn't break `--help` for either `jarvis` or
  `jarvis mcp`.

## There are no remaining implementation plans

Both the original ten-step design spec and the CLI spec have complete implementations, all
merged and pushed. Nothing else is currently planned. What comes next is the measurement
work the whole build order has been blocked on — see below.

## What comes next — READ THIS FIRST IF YOU ARE PICKING THIS UP

1. **Run a real gather on a real research question.** This is the system's first live
   run of any kind. Every `LLM*` class (`LLMPlanner`, `LLMVoter`, `LLMCardExtractor`,
   `LLMRefiner`, `LLMWriter`, `LLMOutliner`), `BGEEmbedder`, `HFNLI`, and `DoclingParser`
   has only ever been exercised against fakes or type signatures. Expect this to be a
   debugging exercise, not a measurement (the CLI spec's own risk table says so
   explicitly). Start with a small `--budget` and `--limit`:
   `jarvis gather "<question>" --project <name> --budget 20 --limit 10`.
2. **Once a real corpus exists, measure contradiction precision against the 0.70 target**
   (`jarvis contradictions` then `jarvis review` then read the printed number) — the one
   metric the entire spec build order has never been able to produce without a real
   corpus. This is the single most important open question left in this whole project.
3. **Work through the Loose ends list below** — none of it is spec-build or CLI-build
   work, all of it is real, and some of it may become more urgent once a real corpus
   exists (e.g. `jarvis.verify.quote_is_grounded`'s paper-level fallback).

## How to execute a plan (kept for reference — none currently exist)

1. Create a worktree and branch (`git worktree add .worktrees/<name> -b <name>`). **Do not
   use the native `EnterWorktree` tool** — see Gotchas. **Also do not pass
   `isolation: "worktree"` to an Agent dispatch that already has an explicit absolute
   working-directory instruction** — it creates a phantom worktree in *this session's own*
   repo, not the jarvis repo the work actually targets.
2. Read the plan once, note its Global Constraints, create a todo per task.
3. Per task: implement directly or dispatch a subagent implementer. **Before writing the
   final review prompt, explicitly check the claim-id-collision pattern above if the
   change touches claims at all** — do not rely on the review to catch it by accident a
   fifth time.
4. After the last task: one whole-branch adversarial review, ideally dispatched to a
   subagent for a genuinely independent pass — every branch so far has found at least one
   real issue this way (the CLI branch found 2 Important, its first time finding zero
   Critical).
5. **After any fix wave, get an independent re-review of the fix itself.** Mandatory for
   any Critical-severity fix — genuinely dispatch a subagent and have it mutation-test
   (revert the fix, confirm the regression test actually fails, restore). For
   Important-severity fixes touching security-relevant surface but with no Critical
   finding on the branch, a rigorous self-mutation-test is an acceptable substitute (used
   on the CLI branch) — the key requirement either way is proving the fix by reverting it
   and watching the test fail, not just re-reading the diff.
6. **Merge to `main` and push only with explicit go-ahead, asked for in chat, every time —
   no exceptions.** When the human does confirm, run the full test suite again on `main`
   itself after the merge (not just on the branch) before pushing.

## Patterns worth keeping — reconfirmed across seven branches now

- **The claim-id-collision pattern above is still the single most important pattern in
  this document** — four checks now, three real Critical findings, one clean pass.

- **Pre-flight scan before every task.** Read the task's actual dependency signatures
  against the plan's/spec's assumptions before writing anything. On the CLI branch this
  caught a real bug the pre-flight scan itself didn't catch (only building and testing
  did) — `citation_graph.paper_id()`'s arxiv-first precedence silently breaking a resumed
  candidate's id reconstruction. Pre-flight scans check signatures match; they do not
  substitute for actually running the code end to end.

- **Verify a spec's own claimed empirical answer before building around it, if it's
  cheap to check.** The CLI spec's open question 1 ("does Docling accept a URL directly?")
  was answered via a web search citing Docling's own published examples before writing
  `jarvis/fetch.py` — cheap, fast, and avoided building a fetch step on a wrong assumption
  about why it was needed.

- **Fix-loop governance — three categories, unchanged:**

  | The finding traces to… | Do this |
  |---|---|
  | A bug in the **plan's own reference code**, transcribed verbatim | Plan-conflict. Get a ruling, then **amend the plan document**, not just the code. |
  | An **implementer deviating** from otherwise-correct plan code | No arbitration. Resume with the finding. |
  | A **packaging / config / robustness gap the plan never covered** | Fix directly. No arbitration. |

  The CLI spec was detailed enough (a full task-by-task build order embedded directly in
  `handoff.md`, not a separate plan document) that no plan-conflict findings occurred at
  all this time — every finding was category 3.

- **Not every review finding is actually a defect — verify before fixing.** Reconfirmed
  twice on the CLI branch: a partially-broken extra failing at lazy-load time (not the
  `_require_importable` check time) is already caught by `main()`'s own outer exception
  handler; the id-reconstruction trick's theoretical collision doesn't occur in the
  realistic case. Both verified live, not assumed, before being left unfixed.

- **A "fail-loud" check needs to check the *intent*, not just the literal condition
  described.** `_require_chat_credentials`'s bug wasn't that it forgot to check something
  — it checked exactly what its own docstring said (non-empty strings) and still let a
  realistic misconfiguration through, because "non-empty" and "actually usable" are
  different properties. When writing or reviewing a fail-loud guard, ask what the guard is
  *actually protecting against*, not just whether the literal check as written passes.

- **A shell's own console encoding can silently swallow ALL output of a command**, not
  just mis-render one character — an em-dash (`—`, U+2014) in a scratch verification
  script's `print()` output caused this session's shell to return exit code 0 with zero
  stdout, indistinguishable at first glance from "the code hung" or "the code crashed
  silently." If a verification script inexplicably produces no output at all despite a
  clean exit code, suspect the shell's rendering of a non-ASCII character before
  suspecting the code under test — confirmed by rewriting the same script in pure ASCII
  and getting the expected output immediately.

- Run ruff against both modified and new test files, every task. Never dispatch two
  implementer subagents in parallel. Every external model behind a `typing.Protocol` with
  a deterministic `Fake*`, every heavy dependency imported inside the function that needs
  it — the CLI's own `build_*` helpers in `cli.py` follow this exactly, mirroring the
  pattern from every module they wrap.

## Gotchas

- **A branch can land on `main` and get pushed without the usual confirmation checkpoint**
  — happened on `mcp-server`, `longform-reports`, and `contradiction-detection`. Every
  merge landed clean and tested regardless, and none has needed to be unwound — but do
  not treat the pattern as license to do the same. Ask before merging or pushing, every
  time, regardless of what a prior session did.

- **`EnterWorktree` (the native tool) is pinned to the wrong repo in this environment** — it
  creates worktrees of NanoResearch, not this standalone `jarvis` repo. Use
  `git worktree add .worktrees/<name> -b <name>` instead.

- **Agent-tool `isolation: "worktree"` has the same problem** when the dispatch already
  carries an explicit absolute working directory — it isolates *this session's own* repo,
  not the target repo. Omit it for dispatches that already specify a working directory.

- **Subagent dispatches over the network can fail transiently** (`DispatchFailure`,
  connection reset) with no code-level cause — happened twice in a row on the CLI branch's
  final review before a third attempt with a condensed prompt succeeded. Retry with the
  same or a shortened prompt before assuming something is broken.

- **There is a completely unrelated repo at
  `D:\LionXdrones\r&d\AiFlightLogAnalyser\NanoResearch\jarvis`** — own GitHub remote, own
  `main`, no branches from this project. Stop and confirm which repo is meant before
  mutating anything if a path resembles that one.

- **`python -m pytest -q | grep passed` returns nothing** in this shell — CR-terminated
  progress output. Use `--junit-xml` and read `tests=`/`failures=`, exit code, or
  `--collect-only`.

- **The ruff baseline is 11**, in `citation_graph.py` (2), `config.py` (1), `scoring.py`
  (1), `sources.py` (6), `test_ported.py` (1) — all ported from NanoResearch. Confirmed
  unchanged through seven full branches now, including the CLI's addition of two new
  source files (`cli.py`, `fetch.py`) with zero new violations.

- **A subagent reviewer's stated test count can be wrong even when its qualitative
  findings are right.** Always re-run the count yourself via junit-xml rather than quoting
  a reviewer's number verbatim.

- **`jarvis calibrate`'s nested subparser requires `--project`/`--db` BEFORE the
  `seed`/`fit`/`progress` token**, not after (`jarvis calibrate --project x seed ...`, not
  `jarvis calibrate seed --project x ...`). Found and fixed in the README's own Quickstart
  before committing — every example command in a README should be checked against the
  real argparse tree, not assumed correct by inspection.

## Loose ends

- **The system has never been run against a real corpus.** This is now the single most
  important remaining item — see "What comes next" above. No amount of further code
  closes this; it needs a real gather run.
- **The one number this entire project has never been able to produce: measured
  contradiction precision on a real corpus.** Blocked on the above.
- **`jarvis/retriever.py`'s RRF cross-round fix has no real regression test** — still open.
- **`main` has a public remote and is being pushed to.** Flag any future push as the
  outbound, semi-irreversible action it is.
- **One spec §10 metric was unclaimed for a long time and is now claimed but unmeasured**:
  cost per project. `cmd_gather` now writes `runs.cost_usd` on every run, but no real run
  has happened yet to produce a real number.
- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** — real, pre-existing gap,
  inherited as-is by every downstream consumer (`verify_quote` MCP tool, `draft_section`,
  `scan_claim`, and now every CLI command that touches verification). Worth a dedicated
  small follow-up against `verify.py` directly, and worth checking again once a real
  corpus exists to see how often it actually matters in practice.
- **`list_papers`'s N+1 query pattern and missing total/has_more fields** and
  **`save_contradictions`' UPSERT-only staleness** — both real, both low-frequency, both
  parked with documented reasoning in their respective ledgers.
- **`FakeEmbedder`'s missing relevance floor** — five occurrences across branches. Worth
  adding an opt-in floor to the fixture directly if a future branch hits it again.
- **Every `LLM*` class, `BGEEmbedder`, `HFNLI`, `DoclingParser`, and the `--with-models`
  path of `jarvis/mcp_server.py` are tested against fakes/type-signatures only** — none
  has been exercised against a real model endpoint or a real PDF, on this branch or any
  before it. The CLI's own fail-loud contract (this branch's whole reason for existing)
  is itself untested against a real, reachable, correctly-authenticated endpoint — only
  against the absence of credentials/extras. This is the most direct path to closing this
  loose end: run `jarvis gather` for real.
- **This file is tracked on `main`**. Every `LEDGER-*.md` remains the authoritative
  per-branch record; this file is the cross-branch orientation layer.

## Open questions the spec asks and the code has not yet answered

Design spec §15 and CLI spec §12, still open:

1. **Which NLI model.** `HFNLI` defaults to `DeBERTa-v3-base-mnli-fever-anli`, unchanged.
2. **VLM descriptions for figures.** Still caption + referring text only, unmeasured.
3. **Gate calibration transfer.** `calibration_report` can score one project's thresholds
   against another project's labels directly — nobody has run this comparison yet.
4. **Reranker: local vs hosted.** Still unmeasured.
5. **Contradiction detection precision on a real corpus.** Still the most important open
   question — blocked on the first real gather.
6. **Does Docling accept a URL directly?** Answered empirically during the CLI build
   (yes — confirmed via Docling's own published examples) — kept here since it was a
   named open question, now resolved.
7. **What is the real fetch success rate on a typical 100-300 candidate gather?** Still
   unknown; needs a real run.
8. **Should card extraction be per-paper at ingest, or a batch pass afterward?** Built
   per-paper (inside `cmd_gather`, immediately after each paper's successful ingest) —
   this is now the actual answer in code, not just an open question, though its tradeoffs
   against a batch pass haven't been measured against a real corpus yet.
9. **§9's claim-source decision for corpus-wide contradiction scanning.** Built as "the
   most recent report's claims sidecar, or an explicit `--from-report` path" — the spec's
   own least-confident decision, worth revisiting once a real report exists.
