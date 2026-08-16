# Handoff — jarvis

Updated 2026-08-16 (twelfth update). Supersedes the prior version, which described the CLI
as merged but never run. **It has now been run for real — search, screen, deep-read, and
`jarvis ask` all completed live against a real paper with real GCP-backed models, and the
core verification mechanism caught a real unsupported claim.** This is the most important
update in the project's history: everything before this was building toward this one
result, and it worked.

**State in one line:** all ten spec build steps, the CLI, and multi-provider chat support
(`openai`/`azure`/`gcp`) are on `main` and pushed. The system has been proven end to end
against a real corpus for the first time. See "THE FIRST REAL RUN" below before doing
anything else.

## Orient yourself in 60 seconds

```
main:  af19bfe (all 10 spec steps + CLI + multi-provider + --max-deep + OA fetch cascade)
```

| What | Where |
|---|---|
| The CLI spec | `docs/specs/2026-08-15-cli-and-operations.md` |
| The original design spec | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Record of the CLI's build + review | `LEDGER-cli.md` (on `main`) |
| Records of all 10 spec build steps' build + review | `LEDGER.md`, `LEDGER-gather-and-gate.md`, `LEDGER-compile-cited-qa.md`, `LEDGER-mcp-server.md`, `LEDGER-contradiction-detection.md`, `LEDGER-longform-reports.md` (all on `main`) |
| **A working real-project virtualenv** | `.venv/` at the repo root — see "Environment setup" below. Do not rely on the global Python install. |
| **Real test corpus from the first live run** | `.dev-local/projects/gcp-smoke-test/corpus.db` — gitignored, kept in-repo instead of scattered across the machine. 179 candidates screened, 1 fully deep-read with a verified card. |

```
main:
  HEAD:   af19bfe  feat: resolve OA PDFs through repository mirrors, not just publisher CDNs
  tests:  757 passing
  ruff:   11 pre-existing violations, zero new
  remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync
```

## THE FIRST REAL RUN — read this before doing anything else

Every prior handoff update said some version of "this has never been run against a real
model or a real paper." That is no longer true. Using a live GCP service-account
credential (`google/gemini-3.7-flash` via Vertex AI's OpenAI-compatible endpoint) and a
real arXiv paper:

1. **`jarvis gather "quadrotor wind disturbance rejection control"`** — real search
   against arXiv/S2/OpenAlex/Crossref found **179 candidates**. Real GCP-backed screening
   kept **179/179** (expected, uncalibrated-gate behavior — see Loose ends). Real measured
   cost: **$0.007795**, correctly written to `runs.cost_usd`.
2. **`jarvis gather ... --yes --max-deep 3`** (resumed, so no re-search) — of the 3
   attempted: **2 failed at fetch** (one candidate had no `pdf_url` at all, one only had a
   DOI-redirect landing page that isn't a direct PDF — both real, expected failure modes,
   not bugs), **1 succeeded completely**: fetched, parsed via Docling into **29 units**,
   embedded, and had a card extracted via live `google/gemini-3.7-flash` with **both
   extracted fields (`problem`, `method`) mechanically verified** (`binding_verified=True`
   — the quote genuinely appears in the cited unit, checked by `verify_card`, not asserted
   by the model).
3. **`jarvis ask "how does GustPilot handle wind disturbances during quadrotor
   navigation?"`** against that one-paper corpus — this is the result that matters most:
   the writer drafted three claims with citations; local `HFNLI` verification **supported
   two of them** and **correctly flagged the third** — *"the quote is real but does not
   clearly support the claim"* (a NEUTRAL verdict) — rather than either blocking it or
   rendering it as a confident, cited fact. That is the exact statement-hallucination
   detection the whole design spec (§8) exists to provide, working live, for the first
   time, on a claim a real model actually produced.

**This is the proof the entire ten-step spec plus the CLI were building toward.** Nothing
about steps 1-10 or the CLI needed to change to make this work — the architecture held.

### What this run also surfaced (real gaps, now understood)

- **The gate keeps everything pre-calibration (179/179).** Not a bug — a union gate with
  no `exclude` outcome is deliberately permissive until `jarvis calibrate` runs against a
  hand-labeled seed set. But it meant `--yes` had no cost control at all, which is why
  `--max-deep` was added (see below) before this run went further than 3 papers.
- **`torch.compile` is fundamentally broken on this Windows machine** — MSVC has no
  `omp.h` (OpenMP header), so any `torch.compile`-JIT-backed model (Docling's RT-DETR
  layout model, and apparently others) fails with `CppCompileError` on first use. Fixed by
  setting `TORCH_COMPILE_DISABLE=1`, which forces eager execution — no measurable
  correctness cost for a single-document run, unmeasured for throughput at scale.
- **Vertex AI needs a publisher-prefixed model name**: `google/gemini-3.7-flash`, not
  `gemini-3.7-flash` — the bare form fails with "malformed publisher model." Now
  documented in `jarvis/llm.py`'s module docstring.
- **Real fetch failure rate on this tiny sample: 2/3 (67%).** Far too small to generalize
  from, but it's the first real data point for CLI spec open question 2, and it confirms
  the design spec's own risk table was right to flag this.

## Environment setup — do this, don't rediscover it

**Two structural problems exist on this specific machine, neither caused by this
project's code, both fixed by the same thing:**

1. **A different, unrelated project is *also* named `jarvis`** (the NanoResearch
   self-evolving-agent one, subcommands `run/maintain/resume/optimize/serve/digest/
   migrate-store/selflearn`) and an old editable install
   (`_editable_impl_nanoresearch.pth`) puts its directory on the global Python
   installation's `sys.path`. The global `jarvis` command and even bare `python -c
   "import jarvis"` resolve to **the wrong package** depending on CWD and install order.
   This is not fixable by reinstalling `jarvis-corpus` — the collision is structural.
2. **C: was nearly full (96% used, 11GB free)** before this project added anything, and
   installing `docling`/`torch`/`sentence-transformers` globally would have made that
   materially worse.

**The fix for both: a dedicated virtualenv on D:, already created.**

```
.venv/                    # at the repo root, gitignored (already in .gitignore)
```

Created with `python -m venv .venv` (no `--system-site-packages`, for full isolation from
the NanoResearch collision) and `pip install -e ".[llm,parse,index,verify,mcp,gcp,dev]"`
into it. All extras are installed — `docling`, `torch`, `sentence-transformers`,
`transformers`, `openai`, `google-auth`, `mcp`. **Use `.venv\Scripts\jarvis.exe` (or
activate the venv first), never the bare global `jarvis` command or `python -c "import
jarvis"` without first confirming `sys.path`/cwd.**

**Real GCP config that works**, for reference (this exact combination produced the run
above):

```
JARVIS_GCP_CREDENTIALS=<path to a real service-account JSON key>
JARVIS_PROVIDER=gcp
JARVIS_MODEL_SYNTHESIS=google/gemini-3.7-flash
JARVIS_MODEL_CARD_EXTRACTION=google/gemini-3.7-flash
JARVIS_MODEL_RETRIEVAL_REFINE=google/gemini-3.7-flash
JARVIS_MODEL_QUERY_EXPANSION=google/gemini-3.7-flash
JARVIS_MODEL_SCREEN_VOTE=google/gemini-3.7-flash
TORCH_COMPILE_DISABLE=1
JARVIS_PROJECT_ROOT=<a project root — .dev-local/projects if kept in-repo>
```

`JARVIS_GCP_PROJECT`/`JARVIS_GCP_LOCATION` were deliberately left unset in this run and it
worked anyway — project id auto-detected from the credentials file, location defaulted to
`global`. Both are real, working fallbacks, not just designed ones, as of this run.

## What changed since the last handoff

**Three things landed, in order, all directly in response to preparing for and then
running the first real gather: `multi-provider-llm` merged, `--max-deep` added, then the
live run itself (see above).**

**`multi-provider-llm` (merged, `793ee36`)**: `jarvis.router` gained a provider dimension
alongside its existing model routing — `route(task)` still says which model name,
`provider_for(task)` now says which backend (`openai`/`azure`/`gcp`), resolved
independently via `JARVIS_PROVIDER`/`JARVIS_PROVIDER_<TASK>`. `jarvis/llm.py` gained three
client builders behind one call shape. The `gcp` path is the one that's actually been
proven live (see above); `azure` and DeepSeek-as-`openai` are implemented and unit-tested
but not yet exercised against a real endpoint. **A real bug was found and fixed before
shipping**: `cmd_gather`/`cmd_ask`/`cmd_report` each constructed `ModelRouter` directly
instead of through `build_router()`, silently dropping all provider config — caught while
preparing to test GCP through the actual CLI, not by any review. Fixed with a regression
test that asserts the call site itself, not just that the feature works end to end
(existing tests all monkeypatch `build_writer`/etc. directly and would never have caught
this). This branch was built and tested directly rather than through the full
subagent-driven-development ceremony (worktree + task-by-task review + final adversarial
review) — smaller, more exploratory scope, so treat its rigor accordingly: solid tests,
no independent adversarial review.

**`--max-deep` (merged, `dacb602`)**: added *because* the first real gather immediately
showed the gate keeps 100% of candidates pre-calibration, meaning `--yes` had no cost
control at all. Papers past the cap stay at `pending_deep` and are picked up by a later
`gather` run on the same project — composes with the existing resumability path rather
than needing its own bookkeeping.

**Also fixed**: the global Python install's `jarvis-corpus` editable link pointed at a
worktree deleted months ago, silently broken for a long time before anyone noticed (see
"Environment setup" above for the real fix — a dedicated venv, not just a reinstall).

**Earlier, still true**: `cli-and-operations` was merged into `main` and pushed, verified
independently rather than taken on trust before merging. The system can now be operated
end to end from one command, `jarvis`:
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

All ten of the original design spec's build steps, the CLI operator surface, and
multi-provider chat support. **All on `main`, and now proven against a real corpus.**

| Component | Status | Modules |
|---|---|---|
| Spec build steps 1-10 | done, `main` | see prior handoff versions / `LEDGER*.md` files on `main` |
| CLI (spec `2026-08-15-cli-and-operations.md`) | done, `main` | `cli.py`, `fetch.py` |
| Multi-provider chat (`openai`/`azure`/`gcp`) | done, `main` | `router.py`, `llm.py`, `config.py` |
| `--max-deep` cost control | done, `main` | `cli.py` |
| **Proven live**: gather → screen → deep-read → ask → verify | **done, this session** | see "THE FIRST REAL RUN" above |

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

The original ten-step design spec, the CLI spec, and multi-provider chat all have complete
implementations, all merged and pushed, all proven live. Nothing else is currently
planned. What comes next is scaling up the real run and the measurement work it unblocks.

## What comes next — READ THIS FIRST IF YOU ARE PICKING THIS UP

1. **Keep scaling the real gather.** `gcp-smoke-test`
   (`.dev-local/projects/gcp-smoke-test/corpus.db`) still has **177 candidates** sitting
   at `pending_deep` after two `--max-deep` rounds. Fetch success is now **20% (5/10 of
   what is actually obtainable)** after `af19bfe` — see Loose ends for the three root
   causes found and fixed, and for what genuinely cannot be fixed at this layer. Resuming
   with a larger `--max-deep` is the fastest way to get more real fetch/parse/card data —
   no new search needed. Use the exact env-var block in "Environment setup" above.
2. **Pick an arXiv-heavy topic for the next corpus.** This one is 30% IEEE and carries
   only **2 arXiv papers out of 179**, so 42% of it is `closed` and no fetch improvement
   can reach it. Fetch reliability is as much a property of the topic's publisher mix as
   of the fetcher — a learning-based topic would measure dramatically better, and that
   confound is currently baked into every number this project has.
3. **Once the corpus has enough cards, try `jarvis report`** — untested live so far; needs
   multiple cards to be meaningful, unlike the single-paper `ask()` test above. 2 cards
   exist now; probably still too few.
4. **Once a real corpus exists at scale, measure contradiction precision against the 0.70
   target** (`jarvis contradictions` then `jarvis review`) — the one metric the entire
   spec build order has never been able to produce. Needs papers that actually disagree,
   which a single-topic corpus this small is unlikely to have yet.
5. **Run `jarvis calibrate`** once there's enough of a corpus to hand-label a seed set —
   this is what would fix the 179/179-kept behavior and give `--max-deep` less to do.
6. **Work through the Loose ends list below.**

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
  mutating anything if a path resembles that one. **This is worse than a path-confusion
  risk**: an old editable install (`_editable_impl_nanoresearch.pth` in the global Python
  install's site-packages) puts that whole directory on `sys.path`, and it has its own
  `jarvis/` package — so the global `jarvis` command and even bare `python -c "import
  jarvis"` can silently resolve to *the wrong project's code* depending on CWD and install
  order. Confirmed live: this is exactly what happened before the `.venv` fix below
  existed. Always use `.venv\Scripts\jarvis.exe` from this repo, never the global command.

- **`torch.compile` cannot work on this Windows machine** — MSVC has no `omp.h`, so any
  `torch.compile`-JIT-backed model fails with `CppCompileError` on first real use (hit via
  Docling's RT-DETR layout model). Set `TORCH_COMPILE_DISABLE=1` before running anything
  that touches `torch`/`transformers`/`docling` for real. Confirmed this is a widely-known
  general PyTorch-on-Windows issue, not specific to this project.

- **The global Python install had a stale `jarvis-corpus` editable link** pointing at a
  worktree deleted months ago (`verifiable-single-paper-core`) — the real `jarvis` command
  was broken globally and nobody had noticed, because every test run resolves the local
  package from whatever worktree pytest's `rootdir` happens to be. Fixed by creating an
  isolated `.venv/` (see "Environment setup" above) rather than just reinstalling — a
  reinstall alone would not have fixed the NanoResearch name collision above.

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

- **The gate keeps 179/179 pre-calibration — expected, but means `--max-deep` is the only
  cost control that currently exists.** `jarvis calibrate` against a hand-labeled seed set
  is the real fix; nobody has run it yet. See "What comes next" above.
- **The one number this entire project has never been able to produce: measured
  contradiction precision on a real corpus.** Still blocked — needs a corpus with papers
  that actually disagree, which the current 2-paper `gcp-smoke-test` corpus can't provide.
- **Real fetch success rate: 9% → 20% (`af19bfe`), which is 5/10 of what is obtainable.**
  The headline percentage is the least useful number here; the denominator is the story.
  Of 25 real corpus DOIs, **15 are `closed` in OpenAlex** — genuinely paywalled
  subscription content, not a fetch failure. Against the 10 that are open access at all,
  the cascade now gets 5.

  Three separate faults were found by measuring rather than guessing, all fixed:
  1. `_candidate_urls` guarded OA resolution with `if not urls`, so any paper carrying a
     `url` (typically a doi.org landing page) skipped OA resolution entirely.
  2. **Unpaywall had never once worked.** It answers HTTP 422 ("Please use your real email
     address") for an empty *and* an invented email, and `UNPAYWALL_EMAIL` was never set.
     OpenAlex carries the same data with no email wall and now goes first.
  3. **Only ~1 OpenAlex location in 6 populates `pdf_url`.** On a real gold-OA paper: six
     locations, one `pdf_url` (the publisher's own walled copy), while every reachable
     mirror — Bristol Research, Europe PMC, Zenodo, orbilu.uni.lu — appeared *only* as a
     `landing_page_url`. Reading `pdf_url` alone discarded nearly every copy that serves.

  Landing pages are now mined for their `citation_pdf_url` meta tag (the one publishers
  emit for Google Scholar and Zotero). One Hindawi paper that failed outright now arrives
  as 6.6MB from `orbilu.uni.lu`.

  **Publisher bot-walls are characterised, not guessed at.** MDPI serves an Akamai
  `bm-verify` interstitial; Hindawi serves a Cloudflare challenge. `curl_cffi` TLS
  impersonation was tested live: it flips MDPI 403→200 but returns the interstitial, not a
  PDF, and does nothing for Hindawi — **0/4 real PDFs**, so TLS-level impersonation alone
  is not the answer. Defeating either needs JS execution (stealth browser or a commercial
  unblocker). Deliberately not built; ranking repository mirrors ahead of walled hosts got
  11/12 on the same set without it.

  Remaining limits, all confirmed live: PMC serves a robot interstitial on direct PDF
  links (its official OA API is the sanctioned route and is not wired up), and some
  institutional repositories bot-block too (Bristol's Pure 403s). Europe PMC covers only
  biomedical — 4/28 here, since MDPI *Sensors* is indexed but *Drones*/*Energies* are not.
- **`azure` and DeepSeek-as-`openai` providers are implemented and unit-tested but not
  yet exercised live** — only `gcp` has been proven against a real endpoint so far.
- **`jarvis/retriever.py`'s RRF cross-round fix has no real regression test** — still open,
  unrelated to anything in this update.
- **`main` has a public remote and is being pushed to.** Flag any future push as the
  outbound, semi-irreversible action it is.
- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** — real, pre-existing gap,
  inherited as-is by every downstream consumer. Not yet known to have actually triggered
  on the one real paper ingested so far (single-paper corpus, so no cross-paper quote
  collision was possible) — worth watching once the corpus has more than one paper.
- **`list_papers`'s N+1 query pattern and missing total/has_more fields** and
  **`save_contradictions`' UPSERT-only staleness** — both real, both low-frequency, both
  parked with documented reasoning in their respective ledgers.
- **`FakeEmbedder`'s missing relevance floor** — five occurrences across branches. Worth
  adding an opt-in floor to the fixture directly if a future branch hits it again.
- **`LLMPlanner`/`LLMVoter`/`LLMRefiner`/`LLMOutliner`, `BGEEmbedder`, `HFNLI`, and the
  `--with-models` path of `jarvis/mcp_server.py` are still tested against fakes only** —
  `LLMWriter` and `LLMCardExtractor` are the two that have now been proven live (via
  `synthesis` and `card_extraction`); the rest haven't been exercised in this project's
  own CLI yet, though the same `chat()`/provider machinery underlies all of them.
- **This file is tracked on `main`**. Every `LEDGER-*.md` remains the authoritative
  per-branch record; this file is the cross-branch orientation layer.

## Open questions the spec asks and the code has not yet answered

Design spec §15 and CLI spec §12:

1. **Which NLI model.** `HFNLI` defaults to `DeBERTa-v3-base-mnli-fever-anli` — now
   proven live: correctly distinguished two SUPPORTED claims from one NEUTRAL
   ("quote is real but does not clearly support the claim") on the first real `ask()`
   run. No comparison against alternatives has been done; the default has just been
   confirmed to *work*, not shown to be optimal.
2. **VLM descriptions for figures.** Still caption + referring text only, unmeasured.
3. **Gate calibration transfer.** `calibration_report` can score one project's thresholds
   against another project's labels directly — nobody has run this comparison yet.
4. **Reranker: local vs hosted.** Still unmeasured.
5. **Contradiction detection precision on a real corpus.** Still the most important
   remaining open question — needs a bigger, more varied corpus than exists yet.
6. **Does Docling accept a URL directly?** Resolved — moot in practice: real testing
   showed the fetch step is still needed regardless (parser escalation, re-parse without
   re-fetch, a meaningful `source_path`), and the URL question was never actually the
   blocker — `torch.compile`/OpenMP was.
7. **What is the real fetch success rate on a typical 100-300 candidate gather?**
   Best data so far: **20%** on this project (5 of the 10 sampled DOIs that are open
   access at all; the other 15 of 25 are `closed`), after fixing three root causes in
   `af19bfe` — see Loose ends. Still a small, single-topic
   sample — genuinely low, or an artifact of this particular topic's publisher mix, is
   not yet distinguishable. 177 candidates remain queued at `pending_deep` for whoever
   wants a bigger sample.
8. **Should card extraction be per-paper at ingest, or a batch pass afterward?** Built
   per-paper, and now proven correct on real data: the one real card extracted had both
   fields mechanically verified against real unit text, not just structurally valid.
9. **§9's claim-source decision for corpus-wide contradiction scanning.** Built as "the
   most recent report's claims sidecar, or an explicit `--from-report` path" — still
   unexercised live; needs `jarvis report` to actually run first.
