# Ledger — contradiction-detection (spec build step 9)

Plan: `docs/plans/2026-08-14-contradiction-detection.md`
Spec: `docs/specs/2026-08-11-research-corpus-agent-design.md` (build step 9, §8)
Branch: `contradiction-detection`, based on `main` at `e9e1d11` (includes spec build steps
1-8 and 10, merged)

## Architecture

Free from the verification pass, per spec §8: NLI emits entailment, neutral, and
contradiction, and the verifier has been computing all three and reading two since the
single-paper-core plan. For each claim, retrieve topically-close evidence from *other
papers* (`opposing_units`), run the same NLI model over (evidence → claim), and keep the
ones scoring high on `contradiction` (`scan_claim`/`scan_corpus`). Candidates are persisted
with their scores (`jarvis/store.py`), rendered as a review sheet
(`write_review_sheet`/`render_conflicts`), and scored for precision against a human's
adjudication (`jarvis.evaluate.contradiction_precision`). **Output is candidates, never
assertions** — this constraint is load-bearing throughout, not decorative.

## Tasks 1-5

All 5 plan tasks implemented directly, one at a time, strict TDD throughout:

```
19e2159 test: end-to-end contradiction scan, review, and precision       [Task 5]
2832737 feat: contradiction review round-trip and precision metric       [Task 4]
669329c feat: corpus-wide contradiction scan with ranking and budget     [Task 3]
478381c feat: cross-paper opposing-evidence retrieval                    [Task 2]
9c03c50 feat: contradiction candidate storage with human review state    [Task 1]
```

Pre-flight scans before each task (reading `find_contradictions`, `NLIModel`, `search`,
`get_unit`, `all_units`'s cross-paper exclusion pattern, `Claim`/`Verdict`/`Verification`,
`EvalReport`/`report()` against the plan's assumptions) found zero interface conflicts —
every signature matched exactly, including `jarvis.verify.find_contradictions`, which had
been written during the single-paper-core plan and sat unused ever since (there was no
multi-paper corpus to run it over until `gather-and-gate` landed).

Four ruff violations found in the plan's own reference code, fixed at implementation time,
all category-1 (plan-conflict) or category-3 (packaging robustness), none touching
production behavior:

- **Task 2**: `I001` (import sort order) — the plan's reference test imports
  `jarvis.contradict` before `jarvis.context` alphabetically, reversed from correct order.
- **Task 3**: `S112` (try-except-continue without logging) — new code, not the ported
  baseline `sources.py` already carries this exact pattern 5×, but the plan's own
  zero-new-violations constraint applies to new code regardless. Suppressed via `noqa`
  alongside the existing `BLE001` suppression at the time; later removed once Task 4's fix
  wave added real logging to that block (see below).
- **Task 4**: one genuine **arithmetic bug** in the plan's own reference test data —
  `test_the_report_flags_whether_the_target_is_met`'s "good" example used 2 correct out of
  3 reviewed candidates (2/3 ≈ 0.667), which does **not** clear the 0.70 target the test
  is meant to demonstrate meeting. Verified the arithmetic independently
  (`python -c "print(2/3, 2/3 >= 0.70)"` → `0.667 False`) before changing the test data to
  3-of-4 (0.75), which does clear it. One `ISC004` (implicit string concatenation) in
  `render_conflicts`'s reference code, fixed by parenthesizing.
- **Task 5**: one more `I001` in this task's own new e2e test file.

## Final state after Tasks 1-5

- **565 tests total** (516 pre-existing + 49 new: 10 in `test_store_contradictions.py`, 33
  in `test_contradict.py`, 6 in `test_contradict_end_to_end.py`), all passing, confirmed via
  junit-xml.
- **`ruff check .`: exactly 11 violations**, all pre-existing, unchanged baseline.
- Definition of done (plan's own criteria), confirmed before the review: offline, no
  network/API keys/model downloads ✓ · cross-paper only, enforced by test rather than
  convention (`test_the_scan_never_reports_the_paper_disagreeing_with_itself`) ✓ · rendered
  output is a review queue, never assertions
  (`test_candidates_are_never_presented_as_facts`) ✓ · human review survives a rescan
  (`test_a_scan_is_rerunnable_without_losing_earlier_judgments`) ✓ · `jarvis.evaluate.report`
  returns a real precision number against the 70% ContraCrow-parity target ✓ · `ruff check .`
  at exactly the 11-violation baseline ✓.

## Final whole-branch adversarial review

Dispatched via subagent after Task 5, framed explicitly around this repo's own history —
by this point, the same claim-id-collision defect class had already been found Critical on
two prior branches (`compile-cited-qa`, then `longform-reports`), so the review was
deliberately pointed at whether any function here constructs or aggregates claims the same
vulnerable way, in addition to the plan's own specific constraints (candidates-never-
assertions, review-round-trip correctness, budget/limit edge cases, cross-paper exclusion
airtightness).

**Result: 1 Critical, 2 Important, 3 Minor findings** — independently reproduced before
fixing, not taken on the reviewer's word.

| # | Finding | Severity | Category | Disposition |
|---|---|---|---|---|
| 1 | `rank()`'s dedup key `(claim_id, unit_id)` assumes `claim_id` uniquely identifies one logical claim — nothing in `scan_corpus` enforced that. **This is the third occurrence of the claim-id-collision defect class across three branches** (`compile-cited-qa`'s Critical Finding 1, `longform-reports`' Critical finding, now this one). Live exploit: two distinct `Claim` objects sharing one `claim_id`, both anchored to units in the same paper, each disputed by a *different* other paper. `rank()` silently collapsed the two conflicts sharing a colliding unit down to one, and the surviving `Conflict` for the second paper carried the **first claim's** `claim_text` — misattributing which claim that paper's evidence actually disputes. | **Critical** | 3 (uncovered gap — `scan_corpus` never imports or applies `_dedupe_claim_ids`, unlike `ask()` and `draft_section()`, both of which apply it specifically because a `Claim` source is an untrusted boundary) | **Fixed** in `f1d7e10`. Reproduced the exploit independently before fixing. Fixed by applying `jarvis.answer._dedupe_claim_ids` to the incoming `claims` sequence at `scan_corpus`'s entry point — the same fix pattern already established twice before, applied a third time rather than reinvented. Lower severity than the two prior occurrences: no fabricated text ever renders as verified here (this is misattribution/silent data loss, not fabrication-passes-as-fact), but the root cause and the fix are identical. Independently re-reviewed via a genuinely dispatched subagent given the severity — mutation-tested, additionally verified against a novel 3-way (not just 2-way) collision variant, and against the persistence/review round-trip after dedup. |
| 2 | `read_reviews` crashed on a single non-UTF-8 byte anywhere in a review file — a realistic hand-editing artifact (e.g. a pasted smart quote) — losing **every other genuinely valid review in that file** to one uncaught `UnicodeDecodeError`, contradicting the module's own established "skip one bad line" contract (`test_a_malformed_review_line_is_skipped` already established that a malformed *JSON* line should be skipped, not fatal — this crashed one level earlier, at decode time, before JSON parsing ever runs). | Important | 3 (robustness gap — `read_text(encoding="utf-8")` decodes the whole file atomically, so one bad byte anywhere takes down everything) | **Fixed** in `f1d7e10`. Changed to reading raw bytes and decoding per line, so a corrupted line is skipped exactly like a malformed-JSON line. Regression test added and mutation-tested by the independent re-review (reverting to whole-file decode reproduces the crash). |
| 3 | `scan_corpus`'s per-claim `except Exception: continue` gave **zero signal** when a systemic failure (a broken NLI model, a broken embedder — not a data problem, a configuration/infrastructure problem) made *every* claim fail. The return value (`[]`) was byte-for-byte identical to a genuinely clean corpus with zero real contradictions — an operator running a 300-paper scan against a misconfigured model would see "0 candidates found" with no indication anything was wrong. | Important | 3 (robustness gap — this module had no `logging` import at all, unlike `jarvis/writer.py` and `jarvis/retriever.py`, both of which already log a warning on their own model-call failures for exactly this reason) | **Fixed** in `f1d7e10`. Added a `_LOGGER.warning` when any claims failed to scan, mirroring `writer.py`'s own established convention verbatim. Removed the now-unused `S112` noqa suppression on that `except` block, since it now logs and no longer matches the rule the suppression was silencing. Regression test added and mutation-tested. |
| 4 | `render_conflicts`'s free-text embedding of `evidence` (a verbatim quote, which routinely contains its own embedded newlines) had no delimiter marking where a quote ends — a multi-line quote could visually forge what looks like a second, fake numbered candidate entry in the rendered queue, potentially misleading a human skimming a long list. Both the real and forged entries still say "candidate" (this does *not* let the system assert a contradiction as fact — the candidates-never-assertions constraint held throughout), but the visual forgery is still a real UX/trust hazard in the review artifact. | Minor | 3 (uncovered rendering-safety gap) | **Fixed** in `f1d7e10`. Whitespace-collapsed `claim_text` and `evidence` before rendering, so an embedded newline can no longer produce a line that looks like a new numbered entry. `write_review_sheet`'s JSONL output was already unaffected (proper JSON string escaping keeps embedded content safely contained regardless). Regression test added and mutation-tested. |
| 5 | `save_contradictions` is UPSERT-only — it never removes a row for a candidate a rescan no longer produces. A stale row (with its old score and its old human review) persists indefinitely, read back indistinguishably from a currently-live candidate. | Minor | 3 (uncovered scaling/UX gap) | **Parked**, documented reasoning in the fix-wave commit message. A stale `invalid` row is harmless; a stale `valid` row could misleadingly look current, but this parallels `mcp-server`'s own parked N+1/missing-total-field finding — a real but low-frequency gap with no established branch convention for table-row TTL/expiry, out of scope for a first version. |
| 6 | `report()`'s `if contradiction_reviews else None` guard treats an explicitly-empty `{}` identically to passing nothing at all, diverging from `contradiction_precision({})`'s own documented `0.0` return. | Minor | non-issue | **Parked, not applicable.** Matches the plan's own locked-in test (`test_the_report_omits_the_metric_when_nothing_was_reviewed`) exactly — almost certainly intentional (an empty-but-present review dict and a genuinely-absent one are treated the same: "nothing has been reviewed yet," reported as `None` rather than a `0.0` that would read as "measured at zero precision"). Worth a one-line docstring clarification, not a functional fix. |

Zero findings, explicitly confirmed live rather than assumed: budget/limit enforcement
(`budget=0`, `budget=-1`, `limit=0`, negative limits all correctly yield zero results with
no exceptions or hangs; a 10-claim scan with `budget=3` correctly kept exactly the first 3,
both in memory and persisted); cross-paper exclusion airtightness even against
byte-identical text (two papers sharing a verbatim-identical benchmark description still
correctly include each other's copy while excluding 100% of the claim's own paper — the
exclusion is by `paper_id` column match, never by content similarity;
`expand_parents=False` structurally closes the one plausible parent-expansion leak path);
SQL injection safety (`claim_id`/`unit_id` values containing `'; DROP TABLE ...` round-trip
safely through every function — fully parameterized); the three-state `reviewed` column at
scale (150 candidates, exact 50/50/50 split, `get_contradiction_reviews` matched
expectations exactly); `apply_reviews`/`set_contradiction_review`'s commit granularity
(traced directly: commits per row, not batched — a simulated interruption after 3 of 5 rows
left exactly 3 durably committed, no partial-row corruption); duplicate JSON keys
(last-key-wins, per spec); deeply nested JSON (5000 levels, no `RecursionError`, correctly
rejected by the `isinstance(row, dict)` check since the top level isn't a dict);
last-line-wins semantics for repeated `(claim_id, unit_id)` verdicts across multiple lines
in one review sheet.

**This branch's Critical finding is the same defect class found twice before, on two
different branches, in two different functions — reproduced for a third time despite the
final review being explicitly framed to look for it.** This confirms the pattern is not a
one-off implementer mistake but a structural hazard of this codebase's design (any function
that aggregates or looks up `Claim`/claim-derived objects by `claim_id` needs this guard,
and the guard is easy to omit because it is not enforced by any type or interface — only by
convention and by remembering three prior incidents). This is now documented in `handoff.md`
as a **mandatory checklist item** for any future branch touching claims.

## Fix wave and re-review

Fixed in `f1d7e10`: the Critical finding (applied `_dedupe_claim_ids` at `scan_corpus`'s
entry point), both Important findings (per-line byte decode in `read_reviews`; a warning
log on systemic scan failure in `scan_corpus`), and the Minor rendering-injection finding
(whitespace-collapse in `render_conflicts`). Two Minor findings parked with the reasoning
above, documented in the fix-wave commit message.

Given the Critical severity, the re-review was **genuinely independently dispatched** to a
subagent. The re-review:

- **Mutation-tested all four fixes**: disabled each fix's specific change in turn (the
  dedup call, the per-line decode, the warning log, the whitespace-collapse), reran each
  fix's own regression test, confirmed every one fails without its fix and passes with it
  restored. Confirmed `git status --short`/`git diff HEAD` empty after each restoration.
- **Tried a novel bypass variant on the Critical fix**: a three-way (not just two-way)
  `claim_id` collision, three claims each anchored to a different unit in the same paper,
  each disputed by a different other paper. `_dedupe_claim_ids` renamed the second and
  third occurrences positionally (`dedup-1`, `dedup-2`), and each resulting claim's
  conflicts carried exactly one correct `claim_text` with zero cross-bleed — the fix
  generalizes to N-way collisions, not just the pairwise case tested directly.
- **Verified the fix's interaction with persistence**: confirmed `save_contradictions`
  persists rows keyed by the post-dedup ids, and `get_contradiction_reviews` round-trips
  reviews correctly per distinct `(claim_id, unit_id)` key even when the originally-colliding
  claims share overlapping opposing units — no cross-key bleed.
- **Independently re-ran the full suite and ruff** on the final committed state: 569 tests
  passing, 0 failures, 0 errors (via junit-xml parsing); `ruff check .` at exactly 11
  violations, and specifically confirmed the `S112` suppression removal was precisely
  scoped to `contradict.py` only (the identical pattern in `sources.py`'s own pre-existing
  baseline is untouched).
- **Confirmed the worktree was left completely clean** after all mutation-testing scratch
  edits and two standalone scratch scripts were reverted/deleted.

Final state after the fix wave: **569 tests passing** (565 + 4 new regression tests), 0
failures, 0 errors. `ruff check .` still exactly the 11-violation baseline.

## What this plan deliberately does not build

Per the plan's own explicit "Where this stops, and the one number to watch" section:

| Not built | Why |
|---|---|
| An LLM summary of each candidate for review | `jarvis/router.py` already routes `contradiction_review` to the frontier tier for exactly this. Deferred because a summary layer over a detector whose real-world precision is unmeasured would make noise more persuasive, not less — build it after the precision number exists against a real corpus. |
| An MCP `find_contradictions` tool | Worth adding to `jarvis/tools.py`'s registry once this branch lands and precision holds against a real corpus. |

The plan's own stated success criterion — contradiction precision on the **first real
corpus** — is not yet measured, because no real (non-test-fixture) gathered corpus exists
anywhere on this machine as of this branch's completion (confirmed:
`~/.jarvis/projects` doesn't exist). Spec §10 targets ≥70%, ContraCrow parity. This is the
single most important open question left by this branch, and it cannot be answered by more
code — it requires a real corpus and real human review time.

## Branch-finish checklist

- [x] All 5 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed, explicitly framed
      around this repo's own claim-id-collision history.
- [x] Fix wave for the Critical and both Important findings, plus one Minor rendering
      fix; two Minor findings parked with documented reasoning.
- [x] Independent re-review of the fix wave, genuinely dispatched given the Critical
      severity — mutation-tested all four fixes, tried a novel 3-way collision variant,
      verified the persistence round-trip, worktree confirmed clean.
- [x] This ledger written as the closing record.
- [ ] Branch finish (merge to `main`) — not yet actioned, pending explicit go-ahead.
