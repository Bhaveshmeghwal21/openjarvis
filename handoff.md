# Handoff — jarvis

Updated 2026-08-15 (ninth update). Supersedes the prior version, which described step 9
(contradiction detection) as complete but unmerged. It is now merged to `main` and pushed.
**All ten spec build steps now exist on `main`.**

**State in one line:** spec build steps 1–10 are all on `main` and pushed. There are no
remaining implementation plans. What's left is measurement and hardening — see "What comes
next" below.

## Orient yourself in 60 seconds

```
main:  4edbb3f (steps 1-10), pushed, in sync with origin
```

| What | Where |
|---|---|
| **The next work: CLI + operations spec** — no plan written yet | `docs/specs/2026-08-15-cli-and-operations.md` |
| The original design spec — read this first for context | `docs/specs/2026-08-11-research-corpus-agent-design.md` |
| Completed plan, steps 1-5 | `docs/plans/2026-08-11-verifiable-single-paper-core.md` |
| Completed plan, step 6 | `docs/plans/2026-08-14-gather-and-gate.md` |
| Completed plan, step 7 | `docs/plans/2026-08-14-compile-cited-qa.md` |
| Completed plan, step 8 | `docs/plans/2026-08-14-mcp-server.md` |
| Completed plan, step 9 | `docs/plans/2026-08-14-contradiction-detection.md` |
| Completed plan, step 10 | `docs/plans/2026-08-14-longform-reports.md` |
| Record of steps 1-5's build + review | `LEDGER.md` (on `main`) |
| Record of step 6's build + two-round review | `LEDGER-gather-and-gate.md` (on `main`) |
| Record of step 7's build + review | `LEDGER-compile-cited-qa.md` (on `main`) |
| Record of step 8's build + review | `LEDGER-mcp-server.md` (on `main`) |
| Record of step 9's build + review | `LEDGER-contradiction-detection.md` (on `main`) |
| Record of step 10's build + review | `LEDGER-longform-reports.md` (on `main`) |

```
main:
  HEAD:   4edbb3f  merge: contradiction detection (spec build step 9)
  tests:  569 passing
  ruff:   11 pre-existing violations, zero new
  remote: origin -> https://github.com/Bhaveshmeghwal21/openjarvis (public), pushed, in sync
```

## What changed since the last handoff

**Contradiction detection (spec build step 9) — the last of the ten spec build steps —
was merged into `main` and pushed.** Verified independently rather than taken on trust:
full suite re-run on `main` post-merge (569 passing, exit 0), `ruff check .` at the
11-violation baseline, and the specific claim in the branch's own ledger — that
`jarvis/contradict.py`'s `rank()` reuses `jarvis.answer._dedupe_claim_ids` rather than
reinventing a fix — confirmed directly by grep: `jarvis/contradict.py:26` imports it,
`jarvis/contradict.py:136` calls it. Same for `jarvis/report.py` (long-form reports,
step 10), which imports both `_dedupe_claim_ids` and `_drop_citations_outside_evidence`
from `jarvis.answer` at line 18. Both now-merged worktrees and local branches were removed.

The corpus can now surface where one paper contradicts another, as ranked candidates for a
human to review — never as assertions. `jarvis/contradict.py` retrieves cross-paper
evidence topically close to a claim, reuses the verification pass's own NLI model to score
disagreement (free — NLI already computes the contradiction label), ranks candidates, and
runs a full human-review round-trip a precision metric
(`jarvis.evaluate.contradiction_precision`, target ≥0.70, ContraCrow parity) is measured
against — not yet measured on a real corpus; see Loose ends.

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

**The established, working fix, applied identically all three times, and confirmed still
in place after this handoff's own verification pass**: import and apply
`jarvis.answer._dedupe_claim_ids` (and, where citations are checked against a bounded
evidence set, also `_drop_citations_outside_evidence`) at the entry point of any new
function that receives a `Sequence[Claim]` from a caller, before doing anything with
`claim_id` as a lookup or grouping key. Do not reimplement this logic — import the same
two functions from `jarvis.answer` every time.

**Mandatory checklist item for any future work on this codebase**: if a change introduces a
new function that constructs, aggregates, ranks, or looks up claims (or objects derived
from claims, like `Conflict`) by an id, explicitly verify `_dedupe_claim_ids` is applied
before that id is used as a lookup/grouping key. Do not wait for a review to find this by
accident a fourth time — check it directly, the way a pre-flight scan checks every other
interface assumption. There is no more plan-execution work left where this would come up
as a task-by-task discipline, but it applies equally to any future ad hoc change.

## What is built

Spec §13 lists ten build steps. **All ten are done, all on `main`.**

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
| 9. Contradiction detection | done, `main` | `contradict.py` + extensions to `store.py`/`evaluate.py` |
| 10. Long-form reports | done, `main` | `outline.py`, `report.py` |

Caveats carried forward, still true:

- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** (pre-existing since the
  single-paper-core plan) lets a quote that exists only in a *different* unit of the same
  paper still ground a claim citing the *wrong* unit. Inherited as-is by every downstream
  consumer (`verify_quote` MCP tool, `draft_section`, `scan_claim`). Still a real follow-up
  against `verify.py` itself; still not bundled into any branch.
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
genuinely clean corpus. Both fixed; the second one also added logging to
`contradict.py`, which previously had none.

One Minor finding fixed (a multi-line evidence quote could visually forge a fake queue
entry in `render_conflicts`'s output). Two Minor findings parked with documented reasoning:
`save_contradictions` never removes a stale row for a candidate a rescan no longer
produces; `report()`'s `{}`-vs-`None` handling for `contradiction_reviews` matches the
plan's own locked-in test.

**The plan's own stated success criterion — contradiction precision on the first real
corpus, target ≥0.70 (ContraCrow parity) — is not yet measured.** No real
(non-test-fixture) gathered corpus exists anywhere on this machine. This is the single
most important open question the whole spec build leaves behind, and it requires a real
corpus plus real human review time to answer — no amount of further code closes it.

## There are no remaining implementation plans (but there is a new spec)

All ten of the *original* design spec's build steps have a complete implementation,
merged, pushed, and independently verified. `docs/specs/2026-08-15-cli-and-operations.md`
defines the next body of work and has **no implementation plan yet** — writing one is the
first task for whoever picks this up.

| # | Plan | Spec step | Tasks | Status |
|---|---|---|---|---|
| 1 | `2026-08-14-gather-and-gate.md` | 6 | 13 | done, merged, pushed |
| 2 | `2026-08-14-compile-cited-qa.md` | 7 | 6 | done, merged, pushed |
| 3 | `2026-08-14-mcp-server.md` | 8 | 5 | done, merged, pushed |
| 4 | `2026-08-14-longform-reports.md` | 10 | 6 | done, merged, pushed |
| 5 | `2026-08-14-contradiction-detection.md` | 9 | 5 | done, merged, pushed |

## What comes next — READ THIS FIRST IF YOU ARE PICKING THIS UP

Nothing in `docs/specs/2026-08-11-research-corpus-agent-design.md`'s build order (§13)
remains unbuilt. **But the system cannot be run**, and that is now the top priority.

**There is a new spec for the next body of work:
[`docs/specs/2026-08-15-cli-and-operations.md`](docs/specs/2026-08-15-cli-and-operations.md).**
Read it before doing anything else. It is a design spec, not an implementation plan — the
plan still needs writing (see "How to execute a plan" below, and
`superpowers:writing-plans`).

The short version of why it exists: `pyproject.toml` declares exactly one entry point,
`jarvis-mcp`, and it is deliberately read-only. Every pipeline stage exists as a library
function; nothing wires them together. **No corpus has ever been built** —
`~/.jarvis/projects` does not exist on this machine, so `jarvis-mcp` has no `--db` to
point at. Every remaining open question (contradiction precision vs. the 70% target, gate
calibration transfer, reranker choice, cost per project) is blocked behind that.

**Three real pipeline gaps were found by reading the code while writing that spec** — none
of them appear in any plan or ledger, and each blocks a real run:

1. **No PDF acquisition step exists.** `ingest_decided`'s default `path_for` resolves to
   `paper["pdf_url"]` — a URL — and hands it to `DoclingParser.parse`, which passes it to
   `DocumentConverter().convert(path)`. Nothing downloads, caches, or retries. Whether
   Docling resolves a remote URL is *unverified* — every test uses `FakeParser`.
2. **Card extraction is never called outside tests.** `extract_and_verify` is exported and
   tested but wired into no production path, while `write_report` builds its outline from
   `corpus_cards(conn)`. On a real gathered corpus, every report would come out empty, and
   it would look like a report bug.
3. **Nothing writes `runs.cost_usd`.** `save_run` accepts it, `ModelRouter` tracks it, no
   code connects them.

Recommended order, per that spec's §10: project resolution + `jarvis status` → PDF fetch
and cache → `jarvis gather` (with the confirmation gate, card extraction, and cost
logging) → `ask`/`report` → `contradictions`/`review` → `calibrate`. The first three
unblock everything; the rest are thin wrappers over merged, reviewed code.

**Two decisions were made without a ruling** and are documented in that spec's §4 — one
`jarvis` command with subcommands, and `gather` pausing for confirmation before deep reads.
Both are cheap to overrule before implementation starts; neither should be discovered
silently mid-build.

**Deliberately sequenced *after* the first real gather, not before:** HTTP/SSE MCP
transport (worthwhile, but it would serve a corpus that doesn't exist), and any web UI
(the design spec §3 rules it out for v1 — "CLI and MCP only", "no server").

After that, the measurement work the whole build order has been blocked on: run a real
gather, then measure contradiction precision against the 70% target. Expect the first live
run to be a debugging exercise rather than a measurement — every `LLM*` class has only
ever run against fakes.

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
   fourth time.
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
  document.** Everything else below is secondary to it for any future work touching claims.

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
  meeting) — verified independently before fixing the test, not the metric function, which
  was correct.

- **`FakeEmbedder` has no relevance floor** — five confirmed occurrences across three
  branches now. Any test asserting "a nonsense query returns zero hits" against a
  *nonempty* store will fail this way; use a genuinely empty store instead.

- **Not every review finding is actually a defect — verify before fixing.**
  `mcp-server`'s Finding 1 (`"abstract"` as a phantom depth value) looked real until
  `store.py`'s own schema comment was checked directly, which showed it documented as part
  of the depth vocabulary all along.

- **A "silent except" is a red flag on any function that scans/iterates over many items.**
  `contradiction-detection`'s `scan_corpus` masked a systemic failure identically to a
  clean result until a warning log was added.

- **A single `read_text(encoding="utf-8")` on a whole file is a single point of failure**
  for any format meant to tolerate partial corruption (JSONL, line-oriented formats in
  general). Decode per line if the contract is "one bad line doesn't take down the rest."

- Run ruff against both modified and new test files, every task. Never dispatch two
  implementer subagents in parallel. Every external model behind a `typing.Protocol` with
  a deterministic `Fake*`, every heavy dependency imported inside the function that needs
  it.

## Gotchas

- **A branch can land on `main` and get pushed without the usual confirmation checkpoint**
  — it happened on `mcp-server`, `longform-reports`, and `contradiction-detection`: in each
  case a "not yet merged, pending explicit go-ahead" note in the branch's own ledger was
  followed within minutes by the merge commit itself, with no visible confirmation step in
  between in git history. Every merge landed on `main` clean and tested regardless, and
  none has needed to be unwound — but if you are a session picking this up, do not treat
  the pattern as license to do the same. Ask before merging or pushing, every time,
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

- **The system has no operator surface — nothing can be run.** Now covered by
  `docs/specs/2026-08-15-cli-and-operations.md`; see "What comes next" above. Includes
  three previously-unrecorded pipeline gaps (no PDF acquisition, card extraction never
  called outside tests, nothing writes `runs.cost_usd`).
- **The one number this entire spec build order has never been able to produce: measured
  contradiction precision on a real corpus.** See "The contradiction-detection adversarial
  review" above. No amount of further code closes this — it needs a real gather run and
  real human review time, both blocked on the CLI above.
- **`jarvis/retriever.py`'s RRF cross-round fix has no real regression test** — still open.
- **`main` has a public remote and is being pushed to.** Flag any future push as the
  outbound, semi-irreversible action it is — see the Gotchas entry about confirmation.
- **One spec §10 metric is unclaimed: cost per project.** `router.py`'s
  `CostTracker`/`ModelRouter` and the `runs.cost_usd` column both exist; nothing writes
  measured usage into a run.
- **`jarvis.verify.quote_is_grounded`'s paper-level fallback** — real, pre-existing gap,
  inherited as-is by every downstream consumer. Worth a dedicated small follow-up against
  `verify.py` directly.
- **`list_papers`'s N+1 query pattern and missing total/has_more fields** and
  **`save_contradictions`' UPSERT-only staleness** — both real, both low-frequency, both
  parked with documented reasoning in their respective ledgers.
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
5. **Contradiction detection precision on a real corpus.** The newest open question, and
   now the most important one, since it's the last piece of the spec's own build order
   that genuinely cannot be answered without real-world data.
