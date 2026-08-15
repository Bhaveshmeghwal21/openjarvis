# Ledger — longform-reports (spec build step 10)

Plan: `docs/plans/2026-08-14-longform-reports.md`
Spec: `docs/specs/2026-08-11-research-corpus-agent-design.md` (build step 10, §7 Stage D)
Branch: `longform-reports`, based on `main` at `8d9ca14` (includes spec build steps 1-8,
merged)

## Architecture

AutoSurvey's decomposition, in four passes. **Outline** from Layer 2 cards
(`jarvis/outline.py`) — the card finally earns its one documented job (spec §5: "coverage
bookkeeping and cross-paper comparison"). **Section drafting** (`jarvis/report.py`), each
section getting its own bounded evidence set and its own retrieval call — never one giant
context. **Integration**, deduplicating claims several independently-drafted sections both
happened to make. **Assembly and verification**, run over the whole report exactly as
`jarvis.answer` runs it for one question, plus coverage — the fraction of the deep-read
corpus actually cited, tracked and reported honestly whether or not it flatters the report.

## Tasks 1-6

All 6 plan tasks implemented directly, one at a time, strict TDD throughout:

```
91f59dc test: end-to-end multi-section report over a real 3-paper corpus   [Task 6]
58e3d57 feat: markdown report rendering with references and coverage      [Task 5]
bbfd8a2 feat: report assembly with corpus coverage measurement            [Task 4]
ad768b1 feat: integration pass deduplicating claims across sections       [Task 3]
061b1ca feat: per-section drafting with its own evidence budget          [Task 2]
9613079 feat: report outlines built from layer 2 cards                   [Task 1]
```

Pre-flight scans before each task (reading `Card`/`CardField`, `get_papers_by_depth`,
`get_card`, `all_units`, `retrieve_iteratively`/`Refiner`, `cap`/`order_for_context`,
`Writer`, `verify_claim`, `report`/`EvalReport`/`coverage`, `Paper`/`get_paper` against the
plan's assumptions) found zero interface conflicts — every signature the plan assumed
matched the actual code exactly, carried over unmodified from the four prior merged
branches.

Six ruff violations found in the plan's own reference code and fixed at implementation
time, all category-1 (plan-conflict, traces to the plan's own reference code/tests) or
category-3 (packaging robustness the plan never covered), none touching production
behavior:

- **Task 1**: `B023` (loop variable not bound in a lambda closure) and `B017` ×2 (blind
  `Exception` assertion) in the plan's own reference test for `test_llm_outliner_falls_back_on_junk`
  and `test_outline_types_are_frozen`. Fixed using a default-argument closure capture for
  the first, and `FrozenInstanceError` for the second two — the latter is the
  cross-branch convention confirmed via grep across seven other test files
  (`test_answer.py`, `test_evidence.py`, `test_writer.py`, `test_gather.py`,
  `test_ingest.py`, `test_retriever.py`, `test_models.py`), none of which ever use
  `pytest.raises(Exception)`.
- **Task 2/3/4**: three `F401` (unused-import) violations from the plan's own reference
  code importing symbols one task before they were actually used (`Sequence` in `report.py`,
  `SectionDraft` and later `Report` in `test_report.py`) — each removed when introduced,
  re-added exactly when the task that needed it landed.
- **Task 5**: two `ISC004` (unparenthesized implicit string concatenation inside a list
  literal — ruff's own "did you forget a comma?" ambiguity warning) in `render_report`'s
  reference code — fixed by wrapping each multi-line f-string in parentheses.

## Final state after Tasks 1-6

- **513 tests total** (449 pre-existing + 64 new: 17 in `test_outline.py`, 40 in
  `test_report.py`, 7 in `test_report_end_to_end.py`), all passing, confirmed via
  junit-xml.
- **`ruff check .`: exactly 11 violations**, all pre-existing, unchanged from every prior
  branch's baseline.
- Definition of done (plan's own criteria), confirmed before the review: offline, no
  network/API keys/model downloads ✓ · one bounded retrieval+cap call per section, never
  one call for the whole report (`test_every_section_is_retrieved_and_budgeted_separately`)
  ✓ · quote fidelity not relaxed for a long report (`test_quote_fidelity_is_not_relaxed_for_a_long_report`,
  a 10-claim report held to 1.0 fidelity) ✓ · a fabricated claim never reaches the rendered
  report (`test_a_fabricated_claim_never_reaches_the_rendered_report`) ✓ · low coverage
  reported honestly (`test_the_report_reports_low_coverage_honestly`) ✓ · `ruff check .`
  at exactly the 11-violation baseline ✓.

## Final whole-branch adversarial review

Dispatched via subagent after Task 6, framed around 8 areas: per-section evidence
budgeting (verified live, not just by test), claim integration correctness (unit_id
uniqueness, normalization edge cases), coverage calculation honesty against
`jarvis.evaluate.coverage`'s actual semantics, rendering safety against markdown/fabrication
injection, `cited_paper_ids`'s resolution strategy versus `citation_graph.paper_id`'s
documented colon-in-title fallback, `TemplateOutliner`/`LLMOutliner` robustness against
hostile replies, `write_report`'s outliner-or-outline duck-typing, and a general adversarial
pass.

**Result: 1 Critical, 1 Important, 2 Minor findings** — independently reproduced before
fixing, not taken on the reviewer's word.

| # | Finding | Severity | Category | Disposition |
|---|---|---|---|---|
| 1 | `draft_section()` omitted the two claim-side defenses `jarvis.answer.ask` applies before verification — `_dedupe_claim_ids` and `_drop_citations_outside_evidence` — reintroducing the exact defect class that was **Critical Finding 1 on `compile-cited-qa`**. Live exploit: a `Writer` emitting two claims sharing one `claim_id` within a single section (fabricated first, real second) let `SectionDraft.claim_for`'s first-match lookup resolve the SUPPORTED verification back to the fabricated claim's text, while the render simultaneously reported "1 claim(s) removed: quote not found" — false confidence that fabrication was caught, when it was actually displayed as verified. A second, independent gap from the same missing safeguard: a claim citing a unit_id outside the section's own capped evidence verified and rendered as if it came from evidence the section actually bounded. | **Critical** | 3 (uncovered gap — the plan's Task 2 explicitly required "the retrieval, budgeting, and verification logic is identical and must stay identical" between `draft_section` and `ask`, but its own reference code did not literally apply `ask`'s two claim-side defenses) | **Fixed** in `9f36cfa`. Reproduced the exploit myself independently before fixing (confirmed the exact rendered output showing fabricated text as supported). Fixed by importing and applying `_dedupe_claim_ids`/`_drop_citations_outside_evidence` from `jarvis.answer`, mirroring `ask()` exactly rather than reimplementing the logic. Re-verified both exploits closed via the same reproduction script. Added two permanent regression tests. Independently re-reviewed via a dispatched subagent (see below) given the severity — mutation-tested, three novel bypass variants tried, none found a new hole. |
| 2 | `LLMOutliner`'s `try/except` in `outline()` wrapped only the `chat_fn()` call itself, not the post-processing of its return value. A hostile reply object whose own `.get()` raises (or whose title/question value's `__str__` raises) propagated uncaught instead of falling back to the template, contradicting the module's own docstring ("Falls back to the template on failure"). Live-verified severity in practice is low: every JSON-realistic malformation a real `chat()`→`json.loads()` pipeline could actually produce was already handled safely by the existing `isinstance` checks — the crash was only reachable via a non-JSON custom Python test double. | Important | 3 (robustness gap — reachable only via a hand-crafted `chat_fn` double, not the real pipeline documented in `jarvis/llm.py`, but a genuine gap against the stated fallback contract) | **Fixed** in `9f36cfa`. Moved all hostile-input-reachable post-processing into a `_parsed_outline` helper wrapped by the same `try/except` as the `chat_fn()` call. Added a regression test using a `dict` subclass whose `.get()` raises. Independently re-verified: the test fails without the fix (confirmed by the re-review subagent reverting the split and re-running). |
| 3 | No length cap on `LLMOutliner`'s parsed title/question strings — live-tested a single 10,000,000-character title passed through untruncated with no crash, 0.004s. | Minor | 3 (uncovered hardening gap) | **Parked.** A rendering-bloat/memory surface only, not a correctness bug, and only reachable via truly adversarial model output — not the real `chat()`→`json.loads()` pipeline. |
| 4 | `jarvis.text.normalize()` does not strip zero-width spaces (U+200B), so two claim texts differing only by an embedded zero-width space are not deduplicated by `integrate()`'s `_claim_key`. | Minor | 3 (uncovered edge case in a shared, pre-existing module) | **Parked.** An under-merge (a little repetition in rendered output), not a data-loss risk — and the plan's own docstring for `_claim_key` already explicitly accepts under-merging as the correct tradeoff over over-merging. Fixing would mean changing `jarvis.text.normalize`, a module shared by every other branch — out of scope for this fix wave. |

Zero findings, explicitly confirmed live rather than assumed: per-section evidence
budgeting (a 6-section corpus test confirmed exactly 6 separate `retrieve_iteratively`/`cap`
calls, never batched, each section's writer receiving exactly its own capped unit count);
claim integration correctness (`unit_id` format `f"{paper_id}:{type}:{page}:{ordinal}"`
confirmed globally unique by construction, so cross-paper collision is structurally
impossible; dropped duplicates correctly drop their verification together; a claim sharing
a unit but different text survives; German ß-vs-ss correctly treated as non-duplicate);
coverage calculation honesty (a unit backed only by a blocked claim contributes exactly 0.0
coverage, cross-checked byte-for-byte against a direct call to `jarvis.evaluate.coverage`);
`cited_paper_ids`'s resolve-through-units strategy (reproduced the exact scenario the code's
own comment cites — a paper titled "Attention: All You Need for Gust Rejection" with no
arxiv/s2 id — confirmed `citation_graph.paper_id()` really does fall back to the
colon-containing title, validating that resolving through `Unit` objects rather than
parsing `unit_id` strings is necessary, not just defensive); `write_report`'s
outliner-or-outline duck-typing (every malformed input — bare `Section`, `None`, a dict
shaped like `Outline`'s fields, a wrong-signature duck type — raises immediately and
clearly, never silently produces a wrong report); general robustness (empty corpus/outline
handled cleanly with no division-by-zero, exact percentages format correctly with no
floating-point drift, `Report`/`Section` confirmed frozen and hashable).

**This branch's Critical finding is the same defect class found once before, on a
different branch, in a different function — and would have shipped again had the final
review not specifically been framed to check for it.** The pattern is now documented in
`handoff.md` as a standing item to check on any future branch that adds a second
claim-rendering surface.

## Fix wave and re-review

Fixed in `9f36cfa`: the Critical finding (imported and applied `jarvis.answer`'s two
claim-side defenses in `draft_section`, exactly mirroring `ask()`) and the Important finding
(widened `LLMOutliner`'s exception handling to cover the whole hostile-input-reachable
parse, not just the `chat_fn()` call). Both Minor findings parked with the reasoning above,
documented in the fix-wave commit message.

Given the Critical severity, the re-review was **genuinely independently dispatched** to a
subagent rather than self-checked — per this repo's own standing rule that a Critical fix
requires real independent verification, not a quick self-check. The re-review:

- **Mutation-tested both new regression tests**: disabled the fix's two new lines directly,
  reran the tests, confirmed both fail without the fix (`AssertionError` on the fabricated
  text rendering as supported; `AssertionError` on the out-of-budget claim surviving).
  Restored and confirmed `git status --short`/`git diff HEAD` both empty afterward.
- **Tried three novel bypass variants** beyond the original two-claim reproduction: a
  three-way collision (two fabricated + one real), a reversed-order collision (real claim
  first, fabricated second — the mirror image of the original exploit), and a combined
  collision-plus-out-of-budget exploit. All three were correctly handled by the fix; no new
  hole found.
- **Confirmed cross-section claim_id reuse is safe**: two different sections both using
  `claim_id="c-0"` for their own distinct real claims do not collide with each other —
  `_dedupe_claim_ids` operates within one `Writer.write()` call's claims, exactly as
  required.
- **Mutation-tested the Important fix** the same way: reverted the `outline()`/
  `_parsed_outline` split back to the narrower try/except, confirmed the specific new
  regression test (`test_llm_outliner_falls_back_when_the_replys_own_get_raises`) fails
  without it.
- **Independently re-ran the full suite and ruff** on the final committed state: 516 tests
  passing, 0 failures, 0 errors (via junit-xml parsing); `ruff check .` at exactly 11
  violations.
- **Confirmed the worktree was left completely clean** (`git status --short` and
  `git diff HEAD` both empty) after the reviewer's own mutation-testing edits were reverted.

Final state after the fix wave: **516 tests passing** (513 + 3 new regression tests), 0
failures, 0 errors. `ruff check .` still exactly the 11-violation baseline.

## What this plan deliberately does not build

Per the plan's own explicit "Where this stops" section:

| Not built | Why |
|---|---|
| Parallel section drafting | `draft_section` is independent per section by construction, so threading/async is mechanical — deferred because it adds a concurrency surface this codebase currently has none of, and an eight-section report is not slow enough yet to justify it. Measure first. |
| A revision pass | AutoSurvey iterates; this plan drafts once. Add only if a measured quote-fidelity/coverage number says the first draft is not good enough — a second model pass over an already-verified report risks introducing ungrounded prose as much as improving it. |
| Contradiction surfacing inside the report | The plan explicitly calls this out as the single highest-value addition once `docs/plans/2026-08-14-contradiction-detection.md` (spec step 9) lands and its precision is measured against the 70% target. |

## Branch-finish checklist

- [x] All 6 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed.
- [x] Fix wave for the Critical and Important findings; two Minor findings parked with
      documented reasoning.
- [x] Independent re-review of the fix wave, genuinely dispatched (not self-checked) given
      the Critical severity — mutation-tested, three novel bypass variants tried, worktree
      confirmed clean.
- [x] This ledger written as the closing record.
- [ ] Branch finish (merge to `main`) — not yet actioned, pending explicit go-ahead.
