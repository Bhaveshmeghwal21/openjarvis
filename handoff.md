# Handoff — jarvis

Updated 2026-08-14 (third update same day). Supersedes the prior version, which described
`gather-and-gate` as an unmerged branch. It has since been merged into `main` and pushed to
the public remote. The `verifiable-single-paper-core` and `gather-and-gate` worktrees and
local branches have been removed — both were fully merged and safe to delete.

**State in one line:** spec build steps 1–6 are on `main` and pushed. Steps 7–10 each have a
written plan; none has been started.

## Orient yourself in 60 seconds

```
main:  508f4e2 (steps 1-5 + 5 plan docs) -> 76fe424 (merge: gather and gate, step 6)
       pushed to origin/main, in sync (0 ahead, 0 behind)
```

| What | Where |
|---|---|
| The design spec — read this first | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Completed plan, steps 1-5 | `docs/plans/2026-08-11-verifiable-single-paper-core.md` |
| Completed plan, step 6 | `docs/plans/2026-08-14-gather-and-gate.md` |
| **Written but not started: steps 7, 8, 9, 10** | `docs/plans/2026-08-14-{compile-cited-qa,mcp-server,contradiction-detection,longform-reports}.md` |
| Record of steps 1-5's build + review | `LEDGER.md` |
| Record of step 6's build + two-round review | `LEDGER-gather-and-gate.md` |

```
HEAD:   76fe424  merge: gather and gate (spec build step 6)
tests:  329 passing (python -m pytest, from repo root)
ruff:   exactly 11 pre-existing violations, zero in any file this project added
remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync
```

## What is built

Spec §13 lists ten build steps. **1 through 6 are done and on `main`.**

| Step | Status | Modules |
|---|---|---|
| 1. Storage + data model | done | `models.py`, `store.py` |
| 2. Parse + typed units | done | `parse.py`, `units.py`, `context.py`, `text.py` |
| 3. Retrieval | done, two caveats below | `index.py`, `embed.py`, `retrieve.py` |
| 4. Verification | done | `verify.py` |
| 5. Eval harness | done, one gap below | `evaluate.py` |
| 6. Gather + gate | done | `gather.py`, `gate.py`, `label.py`, `ingest.py`, `card.py` + extensions to `store.py`/`sources.py` |
| 7-10 | **not started** | plans written, see table above |

Three caveats carried forward from steps 1-5 (still true, nothing here changed them):

- **Step 3's "exposed as agent tools"** is not done — that's step 8, the MCP server plan.
- **Step 3's `sqlite-vec`** was deliberately replaced with brute-force numpy cosine.
- **Step 5's "seed labeling tool"** — this is now actually built (`jarvis/label.py`, Task
  10 of the gather-and-gate plan). What's still missing: citation precision/recall land in
  the Q&A plan (step 7); contradiction precision lands in the contradiction plan (step 9);
  cost-per-project is unclaimed by any plan — see Loose ends.

Step 6 in detail: Stage A (search-plan generation, multi-source fan-out across arXiv/S2/
OpenAlex/Crossref/CORE/Unpaywall, citation-graph expansion with recorded hop depth), Stage B
(the gate — four independent signals: embedding similarity, citation-graph proximity, keyword
overlap, one LLM vote; union decision with three outcomes and no `exclude`; per-project
threshold calibration against a hand-labeled seed), Stage C (deep read into the existing
corpus pipeline with per-paper failure isolation), and Layer 2 card extraction with every
field's quote mechanically re-verified against Layer 0.

## The gather-and-gate adversarial review — worth reading before touching gate.py or store.py

`LEDGER-gather-and-gate.md` has the complete record; independently re-verified by a
separate session (test run, ruff run, diffs of all three fix commits read directly) before
the merge, not just accepted on the executing session's word. Two real findings, both fixed
and independently re-reviewed:

- **`calibrate()`'s default floor** (category-1, plan-conflict, traced to the plan's own
  Task 9 reference code): a signal with zero variance among labeled-relevant papers would
  calibrate to threshold `0.0`, which `decide()`'s inclusive `>=` then admits everything on.
  First fix (a flat `0.05` floor) passed its own test but introduced a real regression,
  caught by independent re-review: it also clamped up signals with genuine fine-grained
  separation clustered below `0.05`, dropping their recall for no reason. Second fix made
  the default floor conditional — it only rescues a signal whose raw quantile-fit is within
  `1e-9` of zero (genuinely degenerate); an *explicitly passed* floor still applies
  unconditionally. Read `calibrate()` in `jarvis/gate.py` directly if touching this function
  — the docstring explains the two modes.
- **`set_depth`** (category-2): was a bare `UPDATE` with no rowcount check, silently
  no-oping on an unsaved `paper_id`. Now raises `ValueError` naming the missing id.

The lesson worth carrying into every future fix loop: **a fix passing its own regression
test is not sufficient evidence of correctness.** The first calibrate() fix did, and still
had a real, constructible regression elsewhere. Always dispatch an independent re-review of
a fix itself, not just of the original finding.

Four minor findings were parked, not fixed (full one-line records in
`LEDGER-gather-and-gate.md`): an incomplete `decisions` map indistinguishable from explicit
`defer`; `expand_citations()` risking a duplicate (not lost) candidate when called without
`already_seen` outside `gather()`; no `PRAGMA busy_timeout` for concurrent multi-process
screening; `LLMCardExtractor`'s unit-id validation scope being wider than the model's actual
prompt (caught downstream by mechanical verification regardless).

## The five plans, and what's left

| # | Plan | Spec step | Tasks | Status |
|---|---|---|---|---|
| 1 | `2026-08-14-gather-and-gate.md` | 6 | 13 | **done, merged, pushed** |
| 2 | `2026-08-14-compile-cited-qa.md` | 7 | 6 | not started |
| 3 | `2026-08-14-contradiction-detection.md` | 9 | 5 | not started; *useful* only once a real gathered corpus exists |
| 4 | `2026-08-14-mcp-server.md` | 8 | 5 | not started; depends on plans 2 |
| 5 | `2026-08-14-longform-reports.md` | 10 | 6 | not started; depends on plans 2 |

**Recommended order for what's left: 2 → 4 → 3 → 5.** Plan 2 (Q&A) is next — it only needs
the core, which is on `main`. Plan 4 (MCP) comes next because it makes everything usable
from Claude Code, which is worth a lot for the remaining work. Plans 3 and 5 are last
because both are measurement-gated: contradiction detection's own definition-of-done names
a 70% precision target that decides whether the feature ships at all, and long-form reports
depend on plan 2's writer/retriever/evidence machinery directly.

Each plan is self-contained: goal, architecture, global constraints, file structure, and
per-task TDD steps with the actual code. Hand one to a fresh session and it needs nothing
else from this document or any conversation history.

## How to execute a plan

1. Create a worktree and branch (`git worktree add .worktrees/<name> -b <name>`). **Do not
   use the native `EnterWorktree` tool** — see Gotchas.
2. Read the plan once, note its Global Constraints, create a todo per task.
3. Per task: implement directly or dispatch a subagent implementer — both have worked well
   on this project. Whichever is used, the pre-flight-scan and TDD discipline below must
   still be followed exactly.
4. After the last task: one whole-branch adversarial review on the most capable model.
5. **After any fix wave, dispatch an independent re-review of the fix itself** — not just of
   the original finding. `gather-and-gate` is the concrete, now twice-verified proof that a
   fix can pass its own regression test and still be wrong elsewhere.
6. Merge to `main` and push only with explicit go-ahead — both prior branches waited for
   that, and `main` now has a public remote, which raises the stakes of an unreviewed push.

## Patterns worth keeping — reconfirmed across three branches now

- **Pre-flight scan before every task.** Read the task's actual dependency signatures
  against the plan's assumptions before writing anything. Caught real defects in reference
  code multiple times across both completed plans.

- **Fix-loop governance — three categories, unchanged:**

  | The finding traces to… | Do this |
  |---|---|
  | A bug in the **plan's own reference code**, transcribed verbatim | Plan-conflict. Get a ruling, then **amend the plan document**, not just the code. |
  | An **implementer deviating** from otherwise-correct plan code | No arbitration. Resume with the finding. |
  | A **packaging / config / robustness gap the plan never covered** | Fix directly. No arbitration. |

  A fix's own regression is categorized and re-reviewed exactly like an original finding —
  don't just patch it silently.

- **A plan's own *test* code can assume something false about a shared fixture.**
  `gather-and-gate`'s Task 8 found the plan's reference test assumed `FakeEmbedder` gives
  near-zero similarity for unrelated text; its hash-bucket design actually has a ~0.2-0.3
  noise floor for any two short texts. Fixed at the test-fixture level (a purpose-built
  double for that one test), not by changing the shared fixture everything else depends on.

- Run ruff against both modified and new test files, every task. Resolve stale counts in a
  plan's prose in favor of the literal code. Never dispatch two implementer subagents in
  parallel. Every external model behind a `typing.Protocol` with a deterministic `Fake*`,
  every heavy dependency imported inside the function that needs it — this is why 329 tests
  run offline in about ten seconds.

## Gotchas

- **`EnterWorktree` (the native tool) is pinned to the wrong repo in this environment** — it
  creates worktrees of NanoResearch, not this standalone `jarvis` repo. Use
  `git worktree add .worktrees/<name> -b <name>` instead.

- **There is a completely unrelated repo at
  `D:\LionXdrones\r&d\AiFlightLogAnalyser\NanoResearch\jarvis`** — own GitHub remote
  (`OpenRaiser/NanoResearch`), own `main`, no branches from this project. If asked to run
  git commands "from" a path resembling that one, stop and confirm which repo is meant
  before mutating anything.

- **`python -m pytest -q | grep passed` returns nothing** in this shell — CR-terminated
  progress output. Use `--junit-xml` and read `tests=`/`failures=`, or `--collect-only`.

- **The ruff baseline is 11**, in `citation_graph.py` (2), `config.py` (1), `scoring.py`
  (1), `sources.py` (6), `test_ported.py` (1) — all files ported from NanoResearch, not
  written for this project. Confirmed unchanged through two full plans now. Do not fix as
  part of feature work; do not add to it.

- **A "rescue floor" for a degenerate computed value is hard to get right on the first
  try.** An unconditional floor fixes the degenerate case but silently regresses any value
  with real separation below the floor. If a future plan needs similar logic, budget for at
  least one extra review round specifically probing that boundary.

## Loose ends

- **`main` has a public remote and is being pushed to.** Any future push should be flagged
  as the outbound, semi-irreversible action it is — same treatment as any push, just worth
  restating now that it's a routine possibility rather than a one-time event.
- **One spec §10 metric is unclaimed by any of the four remaining plans: cost per
  project.** `router.py`'s `CostTracker`/`ModelRouter` and the `runs.cost_usd` column both
  exist; nothing writes measured usage into a run. The gather plan is where cost is
  actually incurred, so wiring `ModelRouter.cost.total_cost` into `save_run` at the end of
  a gather run is the natural home — worth a small follow-up task whenever convenient, not
  urgent.
- **`LLMPlanner`, `LLMVoter`, and `LLMCardExtractor`'s live paths are tested against fakes
  only** — none has been exercised against a real model endpoint yet. Deliberate deferral
  (protocol + fake + lazy adapter is exactly what makes it safe to defer), not a gap, but
  worth knowing before assuming gather-and-gate is validated end-to-end against a real LLM.
- **This file is untracked**, deliberately — a working document about state, not part of
  the tracked history. `LEDGER.md` and `LEDGER-gather-and-gate.md` are the tracked records.

## Open questions the spec asks and the code has not yet answered

Spec §15, still open:

1. **Which NLI model.** `HFNLI` defaults to `DeBERTa-v3-base-mnli-fever-anli`, unchanged.
2. **VLM descriptions for figures.** Still caption + referring text only, unmeasured.
3. **Gate calibration transfer.** `calibration_report` can score one project's thresholds
   against another project's labels directly — nobody has run this comparison yet since
   there is only one project's seed data available so far.
4. **Reranker: local vs hosted.** Still unmeasured.
