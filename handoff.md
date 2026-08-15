# Handoff — jarvis

Updated 2026-08-15 (fourth update). Supersedes the prior version, which described
`compile-cited-qa` as not started. It is now complete on its own branch, reviewed, fixed,
and not yet merged.

**State in one line:** spec build steps 1–6 are on `main` and pushed. Step 7 (compile —
cited Q&A) is complete on branch `compile-cited-qa`, reviewed and fixed, not yet merged.
Steps 8–10 each have a written plan; none has been started.

## Orient yourself in 60 seconds

```
main:               76fe424 (steps 1-6), pushed, in sync with origin
compile-cited-qa:   branched from main at e847ca7, HEAD at 09a5bf8, NOT YET MERGED
```

| What | Where |
|---|---|
| The design spec — read this first | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Completed plan, steps 1-5 | `docs/plans/2026-08-11-verifiable-single-paper-core.md` |
| Completed plan, step 6 | `docs/plans/2026-08-14-gather-and-gate.md` |
| Completed plan, step 7 (on `compile-cited-qa`, unmerged) | `docs/plans/2026-08-14-compile-cited-qa.md` |
| **Written but not started: steps 8, 9, 10** | `docs/plans/2026-08-14-{mcp-server,contradiction-detection,longform-reports}.md` |
| Record of steps 1-5's build + review | `LEDGER.md` (on `main`) |
| Record of step 6's build + two-round review | `LEDGER-gather-and-gate.md` (on `main`) |
| Record of step 7's build + review | `LEDGER-compile-cited-qa.md` (on `compile-cited-qa`, unmerged) |

```
main:
  HEAD:   76fe424  merge: gather and gate (spec build step 6)
  tests:  329 passing
  ruff:   11 pre-existing violations
  remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync

compile-cited-qa (unmerged):
  HEAD:   09a5bf8  docs: final ledger and README update for compile-cited-qa (spec step 7)
  tests:  403 passing (329 baseline + 74 new)
  ruff:   still exactly 11 pre-existing violations, zero new
```

## What changed since the last handoff

**The compile-cited-qa plan (spec build step 7) was fully executed**, on a new worktree
and branch, in one continuous session: all 6 plan tasks via subagent-driven development
(fresh haiku implementer per task, sonnet task reviewer), a final whole-branch adversarial
review on opus, a consolidated fix wave, and a re-review the controller performed directly
after the fixing agent hit a session usage limit mid-report (the commit had already
landed — verified via `git log`/`git status` before treating anything as failed). Not yet
merged.

## What is built

Spec §13 lists ten build steps. **1 through 7 are done.** 1-6 are on `main`; 7 is on
`compile-cited-qa`, unmerged.

| Step | Status | Modules |
|---|---|---|
| 1. Storage + data model | done, `main` | `models.py`, `store.py` |
| 2. Parse + typed units | done, `main` | `parse.py`, `units.py`, `context.py`, `text.py` |
| 3. Retrieval | done, `main`, two caveats below | `index.py`, `embed.py`, `retrieve.py` |
| 4. Verification | done, `main`, one caveat below | `verify.py` |
| 5. Eval harness | done, `main`, one gap below | `evaluate.py` |
| 6. Gather + gate | done, `main` | `gather.py`, `gate.py`, `label.py`, `ingest.py`, `card.py` + extensions to `store.py`/`sources.py` |
| 7. Compile — cited Q&A | **done, `compile-cited-qa`, unmerged** | `evidence.py`, `retriever.py`, `writer.py`, `answer.py` |
| 8-10 | **not started** | plans written, see table above |

Caveats carried forward, still true:

- **Step 3's "exposed as agent tools"** is not done — that's step 8, the MCP server plan.
- **Step 3's `sqlite-vec`** was deliberately replaced with brute-force numpy cosine.
- **Step 4's `quote_is_grounded`** has a real, pre-existing (unchanged since the
  single-paper-core plan) gap: it falls back to matching a quote anywhere in the paper's
  raw text, not just the cited unit, so a quote that only exists in a *different* unit of
  the same paper can still ground a claim citing the *wrong* unit. Harmless while nothing
  renders `[unit_id]` to a human — but step 7 is the first code that does exactly that.
  Flagged in `LEDGER-compile-cited-qa.md` as a real follow-up against `verify.py` itself,
  deliberately not bundled into step 7's branch since the root cause predates it entirely.
- **Step 5's "seed labeling tool"** is built (`jarvis/label.py`). Citation precision/recall
  now exist too (step 7, `jarvis/evaluate.py`). Still missing: contradiction precision
  (step 9); cost-per-project is unclaimed by any plan — see Loose ends.

Step 7 in detail: iterative retrieval where a `Refiner` proposes follow-up queries fused
across rounds by Reciprocal Rank Fusion (reusing `jarvis.retrieve.rrf`), a hard evidence
cap with primacy/recency reordering, a writer subagent emitting explicit
`(claim, unit_id, quote)` triples with two mechanical rejection rules, and `ask()` wiring
retrieve → cap → write → verify with claims mechanically deduped and re-checked against
the bounded evidence set before verification runs. Three outcomes: SUPPORTED (kept,
cited), NEUTRAL/CONTRADICTED (flagged with a warning), QUOTE_NOT_FOUND (blocked, removed
from the rendered answer entirely).

## The compile-cited-qa adversarial review — worth reading before touching answer.py, writer.py, or retriever.py

`LEDGER-compile-cited-qa.md` has the complete record. The final whole-branch review found
1 Critical + 5 Important findings, all tracing to the plan's own reference code:

- **`Answer.claim_for`'s first-match id lookup (Critical, fixed).** If two claims in one
  `Draft` ever share a `claim_id`, a verdict for one can get attached to the other via the
  lookup — demonstrated directly: a blocked (fabricated) claim's text rendered under a
  citation while the footer simultaneously said it was removed. Not reachable by any
  writer shipped in this branch (ids are always unique in practice), but `Writer` is an
  untrusted `typing.Protocol` boundary with nothing enforcing that structurally. Fixed by
  guaranteeing unique ids in `ask()` before verification runs.
- **`ask()` never re-checked a citation against its own bounded evidence set (Important,
  fixed).** The "no citation outside evidence" rule lived only inside one `Writer`
  implementation, not enforced on the consuming side.
- **Primacy/recency ordering inside `ask()` was unverified (Important, fixed).** Swapping
  `order_for_context()` for `reversed()` left the whole suite green.
- **Two "no retrievable evidence" tests were vacuous (Important, fixed) — the fourth
  instance of a pattern already fixed twice during Tasks 4 and 6.** `FakeEmbedder`'s
  vector search has no relevance floor, so a "nonsense query" against the shared fixture
  still returned every unit; the tests only passed via the writer's own empty-draft
  fallback, never via genuinely empty retrieval.
- **A dead LLM call reads to the user as "empty corpus," silently (Important, fixed).**
  `LLMWriter`/`LLMRefiner` swallowed every model failure with no logging.
- **Task 2's own RRF fusion fix has no real regression test (Important, parked, human
  declined to fix).** Confirmed directly: reverting `jarvis/retriever.py` to its pre-fix
  round-block-concatenation version leaves all 14 `test_retriever.py` tests passing — the
  "corrected" dedup test only asserts id-uniqueness and round count, never order.

Every fix was independently re-verified — for the two highest-stakes ones (the Critical
id-collision fix and the ordering fix), the controller **temporarily reverted each fix and
confirmed its own regression test genuinely fails without it**, then restored and
reconfirmed green, rather than trusting review by inspection alone.

**The pattern worth internalizing from this branch specifically:** a test asserting
`X <= N` or `X >= 0` is not evidence that X is being computed correctly — it can pass
whether the computation runs or not. This happened four separate times across one plan
(Tasks 4, 6, and two more caught only by the final review). The fix every time was the
same: recompute the expected value independently using the same primitives the code under
test uses, and compare — or, when checking a fix specifically, temporarily revert the fix
and confirm the test actually fails.

## The remaining plans

| # | Plan | Spec step | Tasks | Status |
|---|---|---|---|---|
| 1 | `2026-08-14-gather-and-gate.md` | 6 | 13 | **done, merged, pushed** |
| 2 | `2026-08-14-compile-cited-qa.md` | 7 | 6 | **done, reviewed, fixed — unmerged** |
| 3 | `2026-08-14-mcp-server.md` | 8 | 5 | not started; depends on plan 2 (done, on its branch) |
| 4 | `2026-08-14-contradiction-detection.md` | 9 | 5 | not started; *useful* only once a real gathered corpus exists |
| 5 | `2026-08-14-longform-reports.md` | 10 | 6 | not started; depends on plan 2 (done, on its branch) |

**Recommended next: merge `compile-cited-qa`, then MCP server, then contradiction
detection and long-form reports in either order.** MCP and long-form reports both depend
on `jarvis.answer`/`jarvis.writer`/`jarvis.retriever`/`jarvis.evidence`, which only exist
on the unmerged `compile-cited-qa` branch right now — either merge it first, or branch the
next plan's worktree from `compile-cited-qa` instead of `main` if working in parallel.

Each plan is self-contained: goal, architecture, global constraints, file structure, and
per-task TDD steps with the actual code. Hand one to a fresh session and it needs nothing
else from this document or any conversation history.

## How to execute a plan

1. Create a worktree and branch (`git worktree add .worktrees/<name> -b <name>`). **Do not
   use the native `EnterWorktree` tool** — see Gotchas. **Also do not pass
   `isolation: "worktree"` to an Agent dispatch that already has an explicit absolute
   working-directory instruction** — it creates a phantom worktree in *this session's own*
   repo (currently NanoResearch), not the jarvis repo the work actually targets. Harmless
   if caught (nothing gets written there), but worth cleaning up immediately if it happens
   (`git worktree remove --force <path>` then `git branch -D <phantom-branch>`).
2. Read the plan once, note its Global Constraints, create a todo per task.
3. Per task: implement directly or dispatch a subagent implementer — both have worked
   well. Whichever is used, the pre-flight-scan and TDD discipline below must still be
   followed exactly.
4. After the last task: one whole-branch adversarial review on the most capable model.
5. **After any fix wave, get an independent re-review of the fix itself** — not just of
   the original finding. If the dispatching/re-reviewing agent hits a session usage limit
   mid-task, check `git log`/`git status` before assuming anything failed — the commit
   frequently already landed. It's fine for the controller to perform the re-review
   directly rather than re-dispatching, especially for a small fix — but for a
   safety-critical fix (this branch had one), **mutation-test it yourself**: temporarily
   revert the fix, confirm its regression test actually fails, then restore.
6. Merge to `main` and push only with explicit go-ahead — every branch so far has waited
   for that, and `main` has a public remote, which raises the stakes of an unreviewed push.

## Patterns worth keeping — reconfirmed across four branches now

- **Pre-flight scan before every task.** Read the task's actual dependency signatures
  against the plan's assumptions before writing anything.

- **Fix-loop governance — three categories, unchanged:**

  | The finding traces to… | Do this |
  |---|---|
  | A bug in the **plan's own reference code**, transcribed verbatim | Plan-conflict. Get a ruling, then **amend the plan document**, not just the code. |
  | An **implementer deviating** from otherwise-correct plan code | No arbitration. Resume with the finding. |
  | A **packaging / config / robustness gap the plan never covered** | Fix directly. No arbitration. |

  A fix's own regression is categorized and re-reviewed exactly like an original finding.

- **A test asserting `<=`, `>=`, or "count didn't go up" is a red flag, not proof.** Four
  separate vacuous-test findings across `compile-cited-qa` alone, all sharing one root
  cause: a fixture too small, too uniform, or too permissive (`FakeEmbedder` has no
  relevance floor) for the assertion to ever have a chance of failing. The fix is always
  the same: recompute the expected value independently with the same primitives the code
  under test uses, and compare — or, when checking whether a *fix* is real, temporarily
  revert it and confirm the test actually fails without it.

- **A session-limit failure on a dispatched agent is not necessarily a task failure.**
  Check `git log`/`git status` in the target worktree before treating it as one — the
  agent may have completed and committed the actual work before dying while writing its
  own report file.

- Run ruff against both modified and new test files, every task. Resolve stale counts in a
  plan's prose in favor of the literal code. Never dispatch two implementer subagents in
  parallel. Every external model behind a `typing.Protocol` with a deterministic `Fake*`,
  every heavy dependency imported inside the function that needs it — this is why 403
  tests run offline in about thirteen seconds.

## Gotchas

- **`EnterWorktree` (the native tool) is pinned to the wrong repo in this environment** — it
  creates worktrees of NanoResearch, not this standalone `jarvis` repo. Use
  `git worktree add .worktrees/<name> -b <name>` instead.

- **Agent-tool `isolation: "worktree"` has the same problem** when the dispatch already
  carries an explicit absolute working directory — it isolates *this session's own* repo,
  not the target repo. Omit it for dispatches that already specify a working directory.

- **There is a completely unrelated repo at
  `D:\LionXdrones\r&d\AiFlightLogAnalyser\NanoResearch\jarvis`** — own GitHub remote
  (`OpenRaiser/NanoResearch`), own `main`, no branches from this project. If asked to run
  git commands "from" a path resembling that one, stop and confirm which repo is meant
  before mutating anything.

- **`python -m pytest -q | grep passed` returns nothing** in this shell — CR-terminated
  progress output. Use `--junit-xml` and read `tests=`/`failures=`, or `--collect-only`.

- **The ruff baseline is 11**, in `citation_graph.py` (2), `config.py` (1), `scoring.py`
  (1), `sources.py` (6), `test_ported.py` (1) — all files ported from NanoResearch, not
  written for this project. Confirmed unchanged through three full plans now. Do not fix
  as part of feature work; do not add to it.

- **A "rescue floor" for a degenerate computed value is hard to get right on the first
  try.** An unconditional floor fixes the degenerate case but silently regresses any value
  with real separation below the floor. Budget an extra review round for that boundary.

## Loose ends

- **`compile-cited-qa` is not merged.** Branch-finish (merge to `main`, worktree/branch
  cleanup, push) has not been proposed as an action yet this session — same posture as
  every prior branch: propose mechanics and wait for explicit confirmation.
- **`main` has a public remote and is being pushed to.** Flag any future push as the
  outbound, semi-irreversible action it is.
- **One spec §10 metric is unclaimed by any remaining plan: cost per project.**
  `router.py`'s `CostTracker`/`ModelRouter` and the `runs.cost_usd` column both exist;
  nothing writes measured usage into a run. The gather plan is the natural home for this
  (wiring `ModelRouter.cost.total_cost` into `save_run` at the end of a gather run) —
  small follow-up, not urgent.
- **`jarvis.verify.quote_is_grounded`'s paper-level fallback (see caveats above) is a real,
  pre-existing gap now more visible because of step 7.** Worth a dedicated small follow-up
  plan/task against `verify.py` directly — not bundled into any existing plan.
- **`LLMPlanner`, `LLMVoter`, `LLMCardExtractor`, `LLMRefiner`, and `LLMWriter`'s live
  paths are tested against fakes only** — none has been exercised against a real model
  endpoint yet. Deliberate deferral, not a gap, but worth knowing.
- **This file is untracked**, deliberately. `LEDGER.md`, `LEDGER-gather-and-gate.md`, and
  `LEDGER-compile-cited-qa.md` are the tracked records.

## Open questions the spec asks and the code has not yet answered

Spec §15, still open:

1. **Which NLI model.** `HFNLI` defaults to `DeBERTa-v3-base-mnli-fever-anli`, unchanged.
2. **VLM descriptions for figures.** Still caption + referring text only, unmeasured.
3. **Gate calibration transfer.** `calibration_report` can score one project's thresholds
   against another project's labels directly — nobody has run this comparison yet.
4. **Reranker: local vs hosted.** Still unmeasured.
