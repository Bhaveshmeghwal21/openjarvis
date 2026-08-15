# Handoff — jarvis

Updated 2026-08-15 (eighth update). Supersedes the prior version, which described steps
1-8 and 10 as complete on `main`, step 9 not started.

**State in one line:** spec build steps 1-8 and 10 are on `main` and pushed. Step 9
(contradiction detection) is complete on branch `contradiction-detection`, tested,
adversarially reviewed, fixed, not yet merged. **Once this merges, all ten spec build
steps exist.**

## Orient yourself in 60 seconds

```
main:                    e9e1d11 (steps 1-8, 10), pushed, in sync with origin
contradiction-detection: branched from e9e1d11, HEAD at fa7c75f, NOT YET MERGED
```

| What | Where |
|---|---|
| The design spec — read this first | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Completed plan, steps 1-5 | `docs/plans/2026-08-11-verifiable-single-paper-core.md` |
| Completed plan, step 6 | `docs/plans/2026-08-14-gather-and-gate.md` |
| Completed plan, step 7 | `docs/plans/2026-08-14-compile-cited-qa.md` |
| Completed plan, step 8 | `docs/plans/2026-08-14-mcp-server.md` |
| Completed plan, step 9 (on `contradiction-detection`, unmerged) | `docs/plans/2026-08-14-contradiction-detection.md` |
| Completed plan, step 10 | `docs/plans/2026-08-14-longform-reports.md` |
| Record of steps 1-5's build + review | `LEDGER.md` (on `main`) |
| Record of step 6's build + two-round review | `LEDGER-gather-and-gate.md` (on `main`) |
| Record of step 7's build + review | `LEDGER-compile-cited-qa.md` (on `main`) |
| Record of step 8's build + review | `LEDGER-mcp-server.md` (on `main`) |
| Record of step 9's build + review | `LEDGER-contradiction-detection.md` (only on `contradiction-detection` until merged) |
| Record of step 10's build + review | `LEDGER-longform-reports.md` (on `main`) |

```
main:
  HEAD:   e9e1d11  merge: long-form reports (spec build step 10)
  tests:  516 passing
  ruff:   11 pre-existing violations, zero new
  remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync

contradiction-detection (unmerged):
  HEAD:   fa7c75f  docs: ledger for contradiction-detection (spec build step 9)
  tests:  569 passing (516 baseline + 53 new)
  ruff:   still exactly 11 pre-existing violations, zero new
```

## What changed since the last handoff

**Long-form reports (spec build step 10) was merged into `main` and pushed.** The corpus
can now generate a structured multi-section report — outline built from Layer 2 cards,
each section drafted against its own bounded evidence set, claims deduplicated across
sections, coverage measured honestly, rendered to markdown with references. Its final
review found a real Critical defect (see "The claim-id-collision pattern" below).

**Contradiction detection (spec build step 9) — the last of the ten spec build steps —
was fully executed** on a new worktree and branch, in one continuous session: all 5 plan
tasks implemented directly, a final whole-branch adversarial review, a fix wave, an
independent re-review, and a ledger. Not yet merged.

The corpus can now surface where one paper contradicts another, as ranked candidates for
a human to review — never as assertions. `jarvis/contradict.py` retrieves cross-paper
evidence topically close to a claim, reuses the verification pass's own NLI model to score
disagreement (free — NLI already computes the contradiction label, just previously
unread), ranks candidates, and runs a full human-review round-trip a precision metric
(`jarvis.evaluate.contradiction_precision`, target ≥0.70, ContraCrow parity) is measured
against.

## THE CLAIM-ID-COLLISION PATTERN — READ THIS BEFORE TOUCHING ANY CODE THAT AGGREGATES CLAIMS

**This is now a three-time-confirmed structural hazard in this codebase, found Critical on
three separate branches:**

1. `compile-cited-qa` — `Answer.claim_for`'s first-match lookup let a blocked claim's
   fabricated text render as if it were the supported claim sharing its id.
2. `longform-reports` — `draft_section` omitted the same defense `ask()` applies,
   reproducing the identical exploit at section scope.
3. `contradiction-detection` — `rank()`'s `(claim_id, unit_id)` dedup key silently
   misattributed which claim a piece of cross-paper evidence actually contradicts, when
   two distinct claims shared an id.

**The root cause is the same every time**: any function that looks up or aggregates
`Claim`/claim-derived objects by `claim_id` implicitly assumes the id is unique. `Writer`
(and any future claim-producing boundary) is explicitly documented as untrusted —
`claims_from_json` enforces uniqueness internally, but nothing on the *consuming* side
enforces it structurally, and it is easy to add a new function that touches claims without
remembering this.

**The established, working fix, applied identically all three times**: import and apply
`jarvis.answer._dedupe_claim_ids` (and, where citations are checked against a bounded
evidence set, also `_drop_citations_outside_evidence`) at the entry point of any new
function that receives a `Sequence[Claim]` from a caller, before doing anything with
`claim_id` as a lookup or grouping key. Do not reimplement this logic — import the same
two functions from `jarvis.answer` every time, so a future fix to either applies
everywhere at once.

**Mandatory checklist item for any future branch, before writing a final review prompt**:
if the plan's own tasks introduce a new function that constructs, aggregates, ranks, or
looks up claims (or objects derived from claims, like `Conflict`) by an id, explicitly
verify `_dedupe_claim_ids` is applied before that id is used as a lookup/grouping key. Do
not wait for the adversarial review to find this by accident — check it directly first,
the way the pre-flight scan already checks every other interface assumption.

## What is built

Spec §13 lists ten build steps. **1 through 8 and 10 are done, all on `main`. 9 is done,
unmerged.**

| Step | Status | Modules |
|---|---|---|
| 1. Storage + data model | done, `main` | `models.py`, `store.py` |
| 2. Parse + typed units | done, `main` | `parse.py`, `units.py`, `context.py`, `text.py` |
| 3. Retrieval | done, `main`, caveats below | `index.py`, `embed.py`, `retrieve.py` |
| 4. Verification | done, `main`, caveat below | `verify.py` |
| 5. Eval harness | done, `main` | `evaluate.py` |
| 6. Gather + gate | done, `main` | `gather.py`, `gate.py`, `label.py`, `ingest.py`, `card.py` |
| 7. Compile — cited Q&A | done, `main` | `evidence.py`, `retriever.py`, `writer.py`, `answer.py` |
| 8. MCP server | done, `main` | `tools.py`, `mcp_server.py` |
| 9. Contradiction detection | **done, `contradiction-detection`, unmerged** | `contradict.py` + extensions to `store.py`/`evaluate.py` |
| 10. Long-form reports | done, `main` | `outline.py`, `report.py` |

Caveats carried forward, still true:

- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** (pre-existing since the
  single-paper-core plan) lets a quote that exists only in a *different* unit of the same
  paper still ground a claim citing the *wrong* unit. Still a real follow-up against
  `verify.py` itself; still not bundled into any branch.
- **`sqlite-vec`** was deliberately replaced with brute-force numpy cosine (step 3).
- **Reranker: local vs hosted** — still unmeasured.

## The contradiction-detection adversarial review — worth reading in full before touching contradict.py

`LEDGER-contradiction-detection.md` has the complete record. The final whole-branch review
was deliberately framed around this repo's own history of the claim-id-collision defect,
and found it a third time (see above), plus two Important robustness gaps: `read_reviews`
crashed on a single non-UTF-8 byte anywhere in a hand-edited review file, losing every
other genuinely valid review in that file to one uncaught `UnicodeDecodeError`; and
`scan_corpus`'s per-claim exception handling gave zero signal when a systemic failure (a
broken NLI model, not a data problem) made every claim fail — indistinguishable from a
genuinely clean corpus. Both fixed; the second one also added `_LOGGER.warning` to
`contradict.py`, which previously had no logging at all — a real gap against the
established convention in `writer.py`/`retriever.py`.

One Minor finding fixed (a multi-line evidence quote could visually forge a fake queue
entry in `render_conflicts`'s output — fixed by whitespace-collapsing rendered text). Two
Minor findings parked with documented reasoning: `save_contradictions` never removes a
stale row for a candidate a rescan no longer produces (parallels `mcp-server`'s own parked
N+1 finding); `report()`'s `{}`-vs-`None` handling for `contradiction_reviews` matches the
plan's own locked-in test, almost certainly intentional.

**The plan's own stated success criterion — contradiction precision on the first real
corpus, target ≥0.70 (ContraCrow parity) — is not yet measured.** No real (non-test-fixture)
gathered corpus exists anywhere on this machine as of this branch's completion. This is
the single most important open question this branch leaves behind, and it requires a real
corpus plus real human review time to answer — no amount of further code closes it.

## The remaining plans

**There are no remaining plans.** All ten spec build steps have a complete implementation.

| # | Plan | Spec step | Tasks | Status |
|---|---|---|---|---|
| 1 | `2026-08-14-gather-and-gate.md` | 6 | 13 | done, merged, pushed |
| 2 | `2026-08-14-compile-cited-qa.md` | 7 | 6 | done, merged, pushed |
| 3 | `2026-08-14-mcp-server.md` | 8 | 5 | done, merged, pushed |
| 4 | `2026-08-14-longform-reports.md` | 10 | 6 | done, merged, pushed |
| 5 | `2026-08-14-contradiction-detection.md` | 9 | 5 | **done, unmerged** (branch `contradiction-detection`) |

**Recommended next, for whichever session picks this up**: merge `contradiction-detection`
(with explicit go-ahead — see "How to execute a plan" below), then there is no more spec
build work defined by `docs/specs/2026-08-11-research-corpus-agent-design.md`. What comes
after is measurement and hardening, not new build steps:

1. **Run a real gather on a real research question** and see whether the whole pipeline
   holds up past test fixtures — this is the first time any of steps 6-10 will run against
   more than a handful of hand-built papers.
2. **Measure contradiction precision on that real corpus** against the 70% target — the
   one number this entire spec build order has been unable to produce without a real
   corpus to run it over.
3. Work through the **Loose ends** list below — none of it is spec-build work, all of it
   is real.

## How to execute a plan

(Kept for the next time a plan exists — none currently do.)

1. Create a worktree and branch (`git worktree add .worktrees/<name> -b <name>`). **Do not
   use the native `EnterWorktree` tool** — see Gotchas. **Also do not pass
   `isolation: "worktree"` to an Agent dispatch that already has an explicit absolute
   working-directory instruction** — it creates a phantom worktree in *this session's own*
   repo, not the jarvis repo the work actually targets.
2. Read the plan once, note its Global Constraints, create a todo per task.
3. Per task: implement directly or dispatch a subagent implementer — both have worked
   well. **Before writing the final review prompt, explicitly check the claim-id-collision
   pattern above if the plan touches claims at all** — do not rely on the review to catch
   it by accident a fourth time.
4. After the last task: one whole-branch adversarial review, ideally dispatched to a
   subagent for a genuinely independent pass — every branch so far has found at least one
   real issue this way.
5. **After any fix wave, get an independent re-review of the fix itself.** For a
   Critical-severity fix, genuinely dispatch a subagent and have it mutation-test (revert
   the fix, confirm the regression test actually fails, restore) — this has now been done
   three times for the same defect class and caught it cleanly every time.
6. **Merge to `main` and push only with explicit go-ahead, asked for in chat, every time —
   no exceptions.** When the human does confirm, run the full test suite again on `main`
   itself after the merge (not just on the branch) before pushing.

## Patterns worth keeping — reconfirmed across six branches now

- **The claim-id-collision pattern above is now the single most important pattern in this
  document.** Everything else below is secondary to it for any branch touching claims.

- **Pre-flight scan before every task.** Read the task's actual dependency signatures
  against the plan's assumptions before writing anything.

- **Fix-loop governance — three categories, unchanged:**

  | The finding traces to… | Do this |
  |---|---|
  | A bug in the **plan's own reference code**, transcribed verbatim | Plan-conflict. Get a ruling, then **amend the plan document**, not just the code. |
  | An **implementer deviating** from otherwise-correct plan code | No arbitration. Resume with the finding. |
  | A **packaging / config / robustness gap the plan never covered** | Fix directly. No arbitration. |

- **A test's own reference numbers can be wrong even when the code under test is right.**
  `contradiction-detection`'s Task 4 caught a genuine arithmetic error in the plan's own
  test data (2/3 ≈ 0.667 does not clear a 0.70 target the test claimed to demonstrate
  meeting) — verified independently (`python -c "print(2/3, 2/3 >= 0.70)"`) before fixing
  the test, not the metric function, which was correct.

- **`FakeEmbedder` has no relevance floor** — five confirmed occurrences across three
  branches now. Any test asserting "a nonsense query returns zero hits" against a
  *nonempty* store will fail this way; use a genuinely empty store instead.

- **Not every review finding is actually a defect — verify before fixing.**
  `mcp-server`'s Finding 1 (`"abstract"` as a phantom depth value) looked real until
  `store.py`'s own schema comment was checked directly, which showed it documented as part
  of the depth vocabulary all along.

- **A "silent except" is a red flag on any function that scans/iterates over many items.**
  `contradiction-detection`'s `scan_corpus` masked a systemic failure identically to a
  clean result until a warning log was added. If a function's own per-item error handling
  exists "so one bad item doesn't abort the whole run," it needs a log line too, or a
  genuinely broken pipeline reads as "nothing found."

- **A single `read_text(encoding="utf-8")` on a whole file is a single point of failure**
  for any format meant to tolerate partial corruption (JSONL, line-oriented formats in
  general). Decode per line if the contract is "one bad line doesn't take down the rest."

- Run ruff against both modified and new test files, every task. Never dispatch two
  implementer subagents in parallel. Every external model behind a `typing.Protocol` with
  a deterministic `Fake*`, every heavy dependency imported inside the function that needs
  it.

## Gotchas

- **A branch can land on `main` and get pushed without the usual confirmation checkpoint**
  — it happened once, on `mcp-server`. Ask before merging or pushing, every time,
  regardless of what a prior session did.

- **`EnterWorktree` (the native tool) is pinned to the wrong repo in this environment** — it
  creates worktrees of NanoResearch, not this standalone `jarvis` repo. Use
  `git worktree add .worktrees/<name> -b <name>` instead.

- **Agent-tool `isolation: "worktree"` has the same problem** when the dispatch already
  carries an explicit absolute working directory — it isolates *this session's own* repo,
  not the target repo. Omit it for dispatches that already specify a working directory.

- **There is a completely unrelated repo at
  `D:\LionXdrones\r&d\AiFlightLogAnalyser\NanoResearch\jarvis`** — own GitHub remote, own
  `main`, no branches from this project. Stop and confirm which repo is meant before
  mutating anything if a path resembles that one.

- **`python -m pytest -q | grep passed` returns nothing** in this shell — CR-terminated
  progress output. Use `--junit-xml` and read `tests=`/`failures=`, exit code, or
  `--collect-only`.

- **The ruff baseline is 11**, in `citation_graph.py` (2), `config.py` (1), `scoring.py`
  (1), `sources.py` (6), `test_ported.py` (1) — all ported from NanoResearch. Confirmed
  unchanged through six full plans now.

- **A subagent reviewer's stated test count can be wrong even when its qualitative
  findings are right.** Always re-run the count yourself via junit-xml rather than quoting
  a reviewer's number verbatim.

## Loose ends

- **The one number this entire spec build order has never been able to produce: measured
  contradiction precision on a real corpus.** See "The contradiction-detection adversarial
  review" above. No amount of further code closes this — it needs a real gather run and
  real human review time.
- **`jarvis/retriever.py`'s RRF cross-round fix has no real regression test** — still open.
- **`main` has a public remote and is being pushed to.** Flag any future push as the
  outbound, semi-irreversible action it is.
- **One spec §10 metric is unclaimed: cost per project.** `router.py`'s
  `CostTracker`/`ModelRouter` and the `runs.cost_usd` column both exist; nothing writes
  measured usage into a run.
- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** — real, pre-existing gap,
  inherited as-is by every downstream consumer (`verify_quote` MCP tool, `draft_section`,
  now `scan_claim` too). Worth a dedicated small follow-up against `verify.py` directly.
- **`list_papers`'s N+1 query pattern and missing total/has_more fields** (`mcp-server`'s
  Finding 2, parked) and **`save_contradictions`' UPSERT-only staleness**
  (`contradiction-detection`'s Finding 5, parked) — both real, both low-frequency, both
  parked with the same reasoning shape.
- **`FakeEmbedder`'s missing relevance floor** — five occurrences now. Worth adding an
  opt-in floor to the fixture directly if a future branch hits it again.
- **`LLMPlanner`, `LLMVoter`, `LLMCardExtractor`, `LLMRefiner`, `LLMWriter`, `LLMOutliner`,
  and the `--with-models` path of `jarvis/mcp_server.py` are tested against
  fakes/type-signatures only** — none has been exercised against a real model endpoint.
  Deliberate deferral, not a gap, but worth knowing before the "run a real gather" step
  above.
- **This file is tracked on `main`**. Every `LEDGER-*.md` remains the authoritative
  per-branch record; this file is the cross-branch orientation layer.

## Open questions the spec asks and the code has not yet answered

Spec §15, still open:

1. **Which NLI model.** `HFNLI` defaults to `DeBERTa-v3-base-mnli-fever-anli`, unchanged.
2. **VLM descriptions for figures.** Still caption + referring text only, unmeasured.
3. **Gate calibration transfer.** `calibration_report` can score one project's thresholds
   against another project's labels directly — nobody has run this comparison yet.
4. **Reranker: local vs hosted.** Still unmeasured.
5. **Contradiction detection precision on a real corpus.** See Loose ends above — the
   newest open question, and now the most important one, since it's the last piece of the
   spec's own build order that genuinely cannot be answered without real-world data.
