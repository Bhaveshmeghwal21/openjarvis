# Handoff — jarvis

Updated 2026-08-15 (sixth update). Supersedes the prior version, which described spec
build steps 1-7 as complete and merged, with step 8 (MCP server) not yet started.

**State in one line:** spec build steps 1–7 are on `main` and pushed. Step 8 (MCP server)
is complete on branch `mcp-server`, tested, adversarially reviewed, fixed, not yet merged.
Steps 9–10 each have a written plan; neither has been started.

## Orient yourself in 60 seconds

```
main:        4be7458 (steps 1-7), pushed, in sync with origin
mcp-server:  branched from 4be7458, HEAD at 629ee0f, NOT YET MERGED
```

| What | Where |
|---|---|
| The design spec — read this first | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Completed plan, steps 1-5 | `docs/plans/2026-08-11-verifiable-single-paper-core.md` |
| Completed plan, step 6 | `docs/plans/2026-08-14-gather-and-gate.md` |
| Completed plan, step 7 | `docs/plans/2026-08-14-compile-cited-qa.md` |
| Completed plan, step 8 (on `mcp-server`, unmerged) | `docs/plans/2026-08-14-mcp-server.md` |
| **Written but not started: steps 9, 10** | `docs/plans/2026-08-14-{contradiction-detection,longform-reports}.md` |
| Record of steps 1-5's build + review | `LEDGER.md` (on `main`) |
| Record of step 6's build + two-round review | `LEDGER-gather-and-gate.md` (on `main`) |
| Record of step 7's build + review | `LEDGER-compile-cited-qa.md` (on `main`) |
| Record of step 8's build + review | `LEDGER-mcp-server.md` (only on `mcp-server` until merged) |

```
main:
  HEAD:   4be7458  docs: update handoff for compile-cited-qa merge
  tests:  403 passing
  ruff:   11 pre-existing violations, zero new
  remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync

mcp-server (unmerged):
  HEAD:   629ee0f  docs: ledger for mcp-server (spec build step 8)
  tests:  449 passing (403 baseline + 46 new)
  ruff:   still exactly 11 pre-existing violations, zero new
```

## What changed since the last handoff

**The MCP server plan (spec build step 8) was fully executed**, on a new worktree and
branch (`mcp-server`), in one continuous session: all 5 plan tasks implemented directly, a
final whole-branch adversarial review, a fix wave, and a ledger. Not yet merged.

The corpus is now exposed as 6 tools over a pure dispatcher with no protocol dependency
(`jarvis/tools.py`): `corpus_search`, `get_unit`, `get_paper`, `list_papers`,
`verify_quote`, `ask`. A thin stdio adapter (`jarvis/mcp_server.py`) translates the MCP
protocol to and from that dispatcher and contains zero corpus logic — confirmed both by a
targeted test and by the final review reading the file directly. `mcp` is imported lazily
and never at module scope in either file; the whole package (including `jarvis.tools`)
still imports cleanly with `mcp` absent.

## What is built

Spec §13 lists ten build steps. **1 through 8 are done.** 1–7 are on `main`; 8 is complete
on `mcp-server`, unmerged.

| Step | Status | Modules |
|---|---|---|
| 1. Storage + data model | done, `main` | `models.py`, `store.py` |
| 2. Parse + typed units | done, `main` | `parse.py`, `units.py`, `context.py`, `text.py` |
| 3. Retrieval | done, `main`, caveats below | `index.py`, `embed.py`, `retrieve.py` |
| 4. Verification | done, `main`, caveat below | `verify.py` |
| 5. Eval harness | done, `main` | `evaluate.py` |
| 6. Gather + gate | done, `main` | `gather.py`, `gate.py`, `label.py`, `ingest.py`, `card.py` + extensions to `store.py`/`sources.py` |
| 7. Compile — cited Q&A | done, `main` | `evidence.py`, `retriever.py`, `writer.py`, `answer.py` |
| 8. MCP server | **done, `mcp-server`, unmerged** | `tools.py`, `mcp_server.py` |
| 9-10 | **not started** | plans written, see table below |

Caveats carried forward from before, still true:

- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** (pre-existing since the
  single-paper-core plan, unchanged by every branch since) lets a quote that exists only
  in a *different* unit of the same paper still ground a claim citing the *wrong* unit.
  `verify_quote` (the new MCP tool) inherits this exactly as-is — it calls
  `quote_is_grounded` directly. Still flagged as a real follow-up against `verify.py`
  itself in `LEDGER-compile-cited-qa.md`; still not bundled into any branch, including
  this one, since the root cause predates all of them.
- **`sqlite-vec`** was deliberately replaced with brute-force numpy cosine (step 3).
- **Reranker: local vs hosted** — still unmeasured.

## The mcp-server adversarial review — worth reading before touching tools.py or mcp_server.py

`LEDGER-mcp-server.md` has the complete record. The final whole-branch review found 0
Critical, 0 Important, 6 Minor findings — the cleanest result of any branch on this repo so
far, and it explicitly did **not** repeat the Critical claim-id-collision defect class
found on `compile-cited-qa`, confirmed by testing duplicate `claim_id` collisions in three
orderings against the `ask` tool specifically.

Two findings were worth fixing and were fixed: `clamp_limit`'s `except` clause missed
`OverflowError` (caught by the dispatcher's own outer defense-in-depth regardless, but
worth closing at the source); `ask`'s tool description omitted mentioning its `nli`
dependency despite requiring it. One finding (`list_papers`'s `DEPTHS` including
`"abstract"`) was investigated and found on closer inspection **not to be a defect at
all** — `store.py`'s own schema comment documents `abstract` as part of the `depth`
column's vocabulary, reserved for a future ingest stage. The remaining findings (an N+1
query pattern in `list_papers`, a test-methodology gap in how the "no `mcp` import" rule is
asserted, and an already-resolved import-style note) were parked with documented
reasoning.

The fix wave's re-review was done directly by the controller rather than re-dispatched —
both fixes were mutation-tested (the pre-fix code path was reconstructed and confirmed to
genuinely fail, not just re-run against the existing suite) rather than merely inspected.

**The pattern worth internalizing from this branch specifically:** the same
`FakeEmbedder`-has-no-relevance-floor vacuous-test pattern that hit `compile-cited-qa` four
times hit this branch too, on the very first task that touched retrieval
(`corpus_search`'s empty-hits test). The fix was identical to the established precedent:
use a genuinely empty store, not a "nonsense enough" query string, because against a
nonempty store no query is nonsense enough for RRF fusion to return zero hits. This is now
five occurrences of the same root cause across two branches — worth hardening
`FakeEmbedder` itself with an opt-in relevance floor if a sixth branch hits it.

## The remaining plans

| # | Plan | Spec step | Tasks | Status |
|---|---|---|---|---|
| 1 | `2026-08-14-gather-and-gate.md` | 6 | 13 | **done, merged, pushed** |
| 2 | `2026-08-14-compile-cited-qa.md` | 7 | 6 | **done, merged, pushed** |
| 3 | `2026-08-14-mcp-server.md` | 8 | 5 | **done, unmerged** (branch `mcp-server`) |
| 4 | `2026-08-14-contradiction-detection.md` | 9 | 5 | not started; *useful* only once a real gathered corpus exists |
| 5 | `2026-08-14-longform-reports.md` | 10 | 6 | not started; unblocked (depends on plan 2, on `main`) |

**Recommended next, once `mcp-server` is merged: long-form reports**, then contradiction
detection, or either order — both are now fully unblocked. Contradiction detection is
listed second only because it is most useful once a real (not just test-fixture) gathered
corpus exists to run it over; if a real corpus already exists by the time this is read,
either order is fine.

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
4. After the last task: one whole-branch adversarial review, ideally dispatched to a
   subagent for a genuinely independent pass — every branch so far has found at least one
   real issue this way, even when (as with `mcp-server`) the result was 0 Critical/0
   Important.
5. **After any fix wave, get an independent re-review of the fix itself** — not just of
   the original finding. If the dispatching/re-reviewing agent hits a session usage limit
   mid-task, check `git log`/`git status` before assuming anything failed — the commit
   frequently already landed. It's fine for the controller to perform the re-review
   directly rather than re-dispatching for a small, non-safety-critical fix wave (as with
   `mcp-server`'s two Minor fixes) — but for a safety-critical fix, **mutation-test it
   yourself regardless of who does the re-review**: temporarily revert the fix, confirm
   its regression test actually fails, then restore.
6. Merge to `main` and push only with explicit go-ahead — every branch so far has waited
   for that, and `main` has a public remote, which raises the stakes of an unreviewed push.
   When the human does confirm, run the full test suite again on `main` itself after the
   merge (not just on the branch) before pushing — the merge itself is worth verifying.

## Patterns worth keeping — reconfirmed across five branches now

- **Pre-flight scan before every task.** Read the task's actual dependency signatures
  against the plan's assumptions before writing anything.

- **Fix-loop governance — three categories, unchanged:**

  | The finding traces to… | Do this |
  |---|---|
  | A bug in the **plan's own reference code**, transcribed verbatim | Plan-conflict. Get a ruling, then **amend the plan document**, not just the code. |
  | An **implementer deviating** from otherwise-correct plan code | No arbitration. Resume with the finding. |
  | A **packaging / config / robustness gap the plan never covered** | Fix directly. No arbitration. |

  A fix's own regression is categorized and re-reviewed exactly like an original finding.

- **Not every review finding is actually a defect — verify before fixing, not just before
  trusting a claim of resolution.** `mcp-server`'s Finding 1 (`"abstract"` as a phantom
  depth value) looked real until `store.py`'s own schema comment was checked directly,
  which showed it documented as part of the depth vocabulary all along. The fix-wave
  discipline of "confirm independently before acting" applies to *deciding whether to fix
  something* just as much as it applies to *confirming a fix worked*.

- **`FakeEmbedder` has no relevance floor — this has now caused the identical vacuous-test
  failure five times across two branches.** Any new test asserting "a nonsense query
  returns zero hits" against a *nonempty* store will fail this way; use a genuinely empty
  store instead. If a sixth branch hits this, it's worth adding an opt-in relevance floor
  to `FakeEmbedder` directly rather than fixing test-by-test forever.

- **A test asserting `<=`, `>=`, or "count didn't go up" is a red flag, not proof.** Recompute
  the expected value independently with the same primitives the code under test uses, and
  compare — or, when checking whether a *fix* is real, temporarily revert it and confirm
  the test actually fails without it.

- **A session-limit failure on a dispatched agent is not necessarily a task failure.**
  Check `git log`/`git status` in the target worktree before treating it as one.

- Run ruff against both modified and new test files, every task. Resolve stale counts in a
  plan's prose in favor of the literal code. Never dispatch two implementer subagents in
  parallel. Every external model behind a `typing.Protocol` with a deterministic `Fake*`,
  every heavy dependency imported inside the function that needs it — this is why 449
  tests run offline in under twenty seconds, `mcp` included, without it installed.

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
  progress output. Use `--junit-xml` and read `tests=`/`failures=`, exit code (`0` = all
  passed), or `--collect-only`.

- **The ruff baseline is 11**, in `citation_graph.py` (2), `config.py` (1), `scoring.py`
  (1), `sources.py` (6), `test_ported.py` (1) — all files ported from NanoResearch, not
  written for this project. Confirmed unchanged through five full plans now. Do not fix
  as part of feature work; do not add to it.

- **A subagent reviewer's stated test count can be wrong even when its qualitative
  findings are right** — `mcp-server`'s final review reported 569 tests; the controller's
  own `--junit-xml` run on the same commit said 448. The findings were still independently
  verified and were real (where real); the raw count was simply disregarded as unreliable.
  Always re-run the count yourself rather than quoting a reviewer's number verbatim.

- **A "rescue floor" for a degenerate computed value is hard to get right on the first
  try.** An unconditional floor fixes the degenerate case but silently regresses any value
  with real separation below the floor. Budget an extra review round for that boundary —
  this was `gather-and-gate`'s `calibrate()` saga, not repeated on any branch since.

## Loose ends

- **`jarvis/retriever.py`'s RRF cross-round fix has no real regression test** (from
  `compile-cited-qa`'s review) — still open. Worth a small dedicated task before the next
  change to that file: assert on fused *order*, not just on id-uniqueness or round count.
- **`main` has a public remote and is being pushed to.** Flag any future push as the
  outbound, semi-irreversible action it is.
- **One spec §10 metric is unclaimed by any remaining plan: cost per project.**
  `router.py`'s `CostTracker`/`ModelRouter` and the `runs.cost_usd` column both exist;
  nothing writes measured usage into a run. Small follow-up, not urgent.
- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** (see caveats above) is a
  real, pre-existing gap, now inherited as-is by the new `verify_quote` MCP tool too.
  Worth a dedicated small follow-up plan/task against `verify.py` directly.
- **`list_papers`'s N+1 query pattern and missing total/has_more fields** (`mcp-server`'s
  Finding 2, parked) — not urgent at current corpus sizes, worth revisiting if a corpus
  large enough for either to matter in practice shows up.
- **`FakeEmbedder`'s missing relevance floor** (see patterns above) — a fifth occurrence of
  the same vacuous-test root cause. Worth adding an opt-in floor to the fixture directly if
  a sixth branch hits it, rather than continuing to fix it test-by-test.
- **`LLMPlanner`, `LLMVoter`, `LLMCardExtractor`, `LLMRefiner`, `LLMWriter`, and now the
  `--with-models` path of `jarvis/mcp_server.py`'s `build_context`/`serve` are tested
  against fakes/type-signatures only** — none has been exercised against a real model
  endpoint, and `serve()` itself has never actually been run (the `mcp` package is not
  installed in this environment). Deliberate deferral, not a gap, but worth knowing.
- **This file is tracked on `main`**. `LEDGER.md`, `LEDGER-gather-and-gate.md`,
  `LEDGER-compile-cited-qa.md`, and `LEDGER-mcp-server.md` (once merged) remain the
  authoritative per-branch records; this file is the cross-branch orientation layer, kept
  current at the end of each branch's work.

## Open questions the spec asks and the code has not yet answered

Spec §15, still open:

1. **Which NLI model.** `HFNLI` defaults to `DeBERTa-v3-base-mnli-fever-anli`, unchanged.
2. **VLM descriptions for figures.** Still caption + referring text only, unmeasured.
3. **Gate calibration transfer.** `calibration_report` can score one project's thresholds
   against another project's labels directly — nobody has run this comparison yet.
4. **Reranker: local vs hosted.** Still unmeasured.
