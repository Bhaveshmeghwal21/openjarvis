# Ledger — compile-cited-qa (final, consolidated)

Plan: `docs/plans/2026-08-14-compile-cited-qa.md`
Spec: `docs/specs/2026-08-11-research-corpus-agent-design.md` (build step 7)
Branch: `compile-cited-qa`, forked from `main` at `e847ca7` (tip of `main` after the
gather-and-gate merge, before this branch started)

## Tasks 1-6

All six complete, each via subagent-driven development (haiku implementer, sonnet
task reviewer), matching the process this project has used since the single-paper-core
branch.

```
f06b520 test: end-to-end cited question answering                        [Task 6]
25aebda feat: alce-style citation precision and recall metrics           [Task 5]
d4b9dc2 fix: strengthen evidence-cap test coverage with a multi-unit fixture
61b55b3 docs: amend Task 4 reference test fixture for the evidence-cap coverage gap
3306725 feat: answer assembly with claim blocking and flagging           [Task 4]
9204205 feat: writer subagent emitting validated claim triples           [Task 3]
74c2846 fix: fuse retrieval rounds by RRF instead of round-block order
3570c4a docs: amend Task 2 reference code for the cross-round ranking fix
9ce6a9b feat: iterative retrieval with query refinement                  [Task 2]
fe4cf68 feat: capped and order-preserving evidence budget                [Task 1]
```

Two plan-conflicts found and fixed during task review:

- **Task 2 (Important, real correctness bug):** `retrieve_iteratively`'s reference code
  accumulated each round's hits by simple round-block concatenation, despite
  `Retrieval.units`'s own docstring claiming "ranked best-first across all rounds." A
  later round's most relevant unit could be silently truncated by `cap()` behind an
  earlier round's weakest hit — undermining the whole reason a `Refiner` exists. Traced
  verbatim to the plan's own Task 2 reference code → plan-conflict, human ruling "fix via
  cross-round RRF fusion" (reusing `jarvis.retrieve.rrf`, the same function `search()`
  already uses to fuse BM25 and vector rankings). Fixed in `74c2846`. Plan amended in
  `3570c4a`.
- **Tasks 4 and 6 (Important, same root cause twice):** the reference test fixtures had
  exactly as many retrievable units as `max_units`, so the tests meant to prove evidence
  capping was real passed identically whether or not `cap()`/`order_for_context()` ran at
  all. Production code was correct both times — pure test-coverage gaps. Human ruling
  "fix now, test-only" both times: Task 4 expanded the shared fixture to 3 units (`61b55b3`
  → `d4b9dc2`); Task 6 gave one test its own 5-unit corpus scoped to that test alone,
  leaving the shared fixture and six sibling tests untouched (`1d473ec` → `63b06bb`).

Every fix round in Tasks 1-6 was independently re-reviewed (scoped re-review, not
self-verification), and the Task 4/6 re-reviews went beyond transcription-checking:
Task 4's re-reviewer monkeypatched `cap()` to a no-op and confirmed the strengthened
assertion actually catches that exact regression, while the old assertion would have
silently passed it.

## Final state (after Tasks 1-6, before the final review)

- **399 tests, all passing**, zero network, no API keys, no model downloads.
- **`ruff check .`: exactly 11 pre-existing violations**, same five files as every prior
  branch's baseline (`citation_graph.py`, `config.py`, `scoring.py`, `sources.py`,
  `test_ported.py`). Zero in any file this plan's Tasks 1-6 touched.

## Final whole-branch adversarial review

Dispatched on the most capable available model (opus) after Task 6, per the plan's own
closing instruction that this review should not be cost-optimized — framed adversarially
against the plan's own binding constraints (writer never verifies itself, no LLM routed to
verification, a blocked claim is removed not annotated, no cap-bypassing path) and
specifically instructed to probe for more instances of the vacuous-test pattern already
found twice in Tasks 4 and 6.

Found 1 Critical + 5 Important findings, all tracing to this plan's own reference code —
not implementer deviation. Two of the reviewer's claims were independently re-verified by
the controller *before* presenting anything to the human partner, following this branch's
own established discipline of not trusting a review report on its word alone:

- The **Critical** finding was reproduced directly against the real store (a two-claim
  draft with colliding `claim_id`s, one fabricated and one grounded, rendered the
  fabricated claim's text under a citation while the footer simultaneously reported it as
  removed).
- The claim that **Task 2's own fix has no real regression test** was confirmed by
  reverting `jarvis/retriever.py` to its pre-fix round-block-concatenation version
  (`git show 9ce6a9b:jarvis/retriever.py`) and observing all 14 `test_retriever.py` tests
  still pass — the corrected dedup test only asserts id-uniqueness and round count, never
  order, so it cannot distinguish RRF fusion from the bug it was written to catch.

| # | Finding | Severity | Ruling | Disposition |
|---|---|---|---|---|
| 1 | `Answer.claim_for` resolves by first-match `claim_id` lookup; two claims sharing an id let a later claim's verdict attach to an unrelated earlier claim's text — a blocked claim's fabricated text could render under a citation while the footer reported it removed. Not reachable by any writer shipped in this branch (`claims_from_json` produces unique ids), but `Writer` is an untrusted `typing.Protocol` boundary with nothing on the consuming side enforcing uniqueness | **Critical** | **Fix now** (category-1 plan-conflict: `claim_for` traced verbatim to Task 4's own reference code) | Fixed in `db3b022`: `_dedupe_claim_ids` reassigns any colliding id before verification runs. Regression test added; independently mutation-tested by the controller — reverted the fix, confirmed the test genuinely fails with exactly the reviewer's original reproduction, restored, reconfirmed green. |
| 2 | `ask()` never re-checks that a returned claim's `unit_id` is actually within the bounded evidence set it handed the writer — the "no citation outside evidence" rule lived only inside one `Writer` implementation (`claims_from_json`), not enforced on the consuming side | Important | **Fix now** (category-1 plan-conflict: `ask()` traced verbatim to Task 4's own reference code) | Fixed in `db3b022`: `_drop_citations_outside_evidence` filters any claim citing a `unit_id` outside `evidence` before verification. Regression test added, read directly against the amendment — matches exactly. |
| 3 | Primacy/recency ordering inside `ask()` was unit-tested in isolation (`jarvis.evidence.order_for_context`) but never verified as actually applied by `ask()` itself — swapping it for `reversed()` left the full suite green | Important | **Fix now** (category-1 plan-conflict: the weak test traced verbatim to Task 4's own reference test) | Fixed in `db3b022`: replaced test independently recomputes the expected order via the same primitives `ask()` calls and compares against what the writer actually received. Independently mutation-tested by the controller — bypassed `order_for_context()` in `ask()`, confirmed the test genuinely fails, restored. |
| 4 | The two "no retrievable evidence" tests queried the shared 3-unit fixture with a nonsense string; `FakeEmbedder`'s vector search has no relevance floor and always ranks every candidate, so both tests only passed via `FakeWriter({})`'s empty-draft fallback, never via genuinely empty retrieval — the fourth instance of the pattern already fixed twice in Tasks 4 and 6 | Important | **Fix now** (category-1 plan-conflict: traced verbatim to Task 4's own reference test) | Fixed in `db3b022`: both tests now use a genuinely empty store (`open_store(tmp_path / "empty.db")`), exercising the real empty-retrieval path. Read directly against the amendment — matches exactly. |
| 5 | `LLMWriter.write`/`LLMRefiner.refine` swallow every model failure (network error, expired key, malformed JSON) with no logging anywhere — an operator with a dead model gets back a confident, false "no evidence in this corpus" message | Important | **Fix now** — category 3 (packaging/robustness gap the plan never covered, no arbitration needed; bundled into the fix wave for locality) | Fixed in `db3b022`: module-level `logging.getLogger(__name__)` added to both modules, `_LOGGER.warning(..., exc_info=True)` on failure. Regression tests via `caplog`, read directly against the amendment — match exactly. |
| 6 | Task 2's own corrected dedup test (`test_units_are_deduped_across_rounds`) doesn't actually prove RRF fusion over round-block concatenation — see the controller's independent revert-and-test confirmation above | Important | **Parked** — human declined to include in this fix wave | Not fixed. Documented in the plan's final amendment section as a real, acknowledged gap: a genuinely discriminating test would need a `FakeReranker` forcing round 1 to rank a weak unit last and round 2 to rank a strong unit first, then asserting the round-2 unit precedes the round-1 tail unit. Left as a follow-up. |
| 7 | `jarvis.verify.quote_is_grounded`'s pre-existing fallback to a paper-wide `find_span` (unchanged by this branch, inherited from the single-paper-core plan) lets a quote that exists only in a *different* unit of the same paper still ground a claim citing the *wrong* unit — a real gap, but this branch is the first code to render `[unit_id]` to a human as a location claim, which is what turns the pre-existing looseness into a user-visible risk | Important | **Out of scope for this branch** — root cause predates this plan entirely and the fallback exists for a real reason (parent/child unit boundaries) | Not fixed, not part of this fix wave. Documented in the plan as worth a dedicated follow-up task against `jarvis/verify.py` directly. |

Six minor/observational findings were also raised (an existing metric being numerically
identical to another in the current pipeline, `Answer.text` retaining the unverified draft
under a misleadingly-safe field name, ordering-not-documented on `Answer.units`, export
asymmetry in `evaluate`'s metric functions, overpromising test names, a stray untracked
`test_output.txt`). All either pre-existing/out of this plan's scope or purely cosmetic;
recorded here per this branch's own convention (documented, never silently dropped, never
escalated), not fixed. The stray file was confirmed absent from the worktree.

## Fix wave and re-review

Fix wave dispatched as **one** consolidated fixer (not one per finding, per the skill),
covering findings 1-5. Commit `db3b022`. The dispatching agent hit a session usage limit
while writing its own report file, *after* the commit had already landed — verified
directly (`git log`, `git status`) before treating anything as failed, per the documented
gotcha ("session-limit failures aren't real task failures — check git state first").

Rather than re-dispatch a scoped-re-review subagent and risk the same limit, the
controller performed the re-review directly:

- Confirmed the diff touched exactly the six intended files
  (`jarvis/answer.py`, `jarvis/writer.py`, `jarvis/retriever.py`, `tests/test_answer.py`,
  `tests/test_writer.py`, `tests/test_retriever.py`), nothing else.
- Ran the full suite directly: **403 tests, 0 errors, 0 failures**.
- Ran `ruff check .` directly: exactly 11 pre-existing violations, zero new.
- Confirmed `test_output.txt` absent from the worktree.
- **Independently mutation-tested the two highest-stakes fixes**, not just reviewed them
  by inspection: temporarily removed `_dedupe_claim_ids`/`_drop_citations_outside_evidence`
  from `ask()` and confirmed `test_two_claims_sharing_an_id_never_let_one_render_as_the_other`
  genuinely fails, reproducing the reviewer's exact original finding; separately bypassed
  `order_for_context()` in `ask()` and confirmed
  `test_the_writer_only_ever_sees_capped_ordered_evidence` genuinely fails. Restored the
  fixed file after each probe and reconfirmed the working tree was byte-identical to HEAD
  (`git status --short` empty) before moving on.
- Read Fix 2 (out-of-evidence citation drop), Fix 4 (genuinely empty store), and Fix 5
  (module-level loggers) directly against the plan amendment's exact specified code and
  confirmed each matched.

## Final commit history, `e847ca7..HEAD`, newest first

```
db3b022 fix: close the final whole-branch review's findings (claim-id collisions,
        out-of-evidence citations, unverified ordering, vacuous empty-retrieval tests,
        silent llm failures)                                    [final review fix wave]
1aac01d docs: amend plan with final whole-branch review's fix wave
63b06bb fix: give the evidence-cap end-to-end test a corpus large enough to cap
1d473ec docs: amend Task 6 reference test for the second vacuous evidence-cap gap
f06b520 test: end-to-end cited question answering                        [Task 6]
25aebda feat: alce-style citation precision and recall metrics           [Task 5]
d4b9dc2 fix: strengthen evidence-cap test coverage with a multi-unit fixture
61b55b3 docs: amend Task 4 reference test fixture for the evidence-cap coverage gap
3306725 feat: answer assembly with claim blocking and flagging           [Task 4]
9204205 feat: writer subagent emitting validated claim triples           [Task 3]
74c2846 fix: fuse retrieval rounds by RRF instead of round-block order
3570c4a docs: amend Task 2 reference code for the cross-round ranking fix
9ce6a9b feat: iterative retrieval with query refinement                  [Task 2]
fe4cf68 feat: capped and order-preserving evidence budget                [Task 1]
```

## Final state

- **403 tests, all passing**, zero network, no API keys, no model downloads.
- **`ruff check .`: exactly 11 pre-existing violations**, same five files, zero new —
  confirmed by the controller directly at every checkpoint, including after every restore
  during the mutation-testing re-review.
- Definition of done (plan's own criteria): `test_a_fabricated_number_never_reaches_the_reader`
  ✓ · `test_fabrication_is_caught_even_when_the_model_is_certain` ✓ ·
  `test_the_evidence_reaching_the_writer_is_never_the_whole_corpus` ✓ (now genuinely
  discriminating, per the final review's Fix 3 and Task 6's own fix) ·
  `jarvis.evaluate.report` returns citation precision and recall ✓ · `ruff check .` clean
  against the baseline ✓.

## What this plan does not build

Spec build steps 8-10, each with its own written plan:

| Step | Why it waits |
|---|---|
| 8. MCP server | Pure surface over `ask()`, which must work first — it now does. |
| 9. Contradiction detection | Reuses the NLI pass; useful once a multi-paper corpus exists via the gather-and-gate plan. |
| 10. Long-form reports | Needs `ask()`'s retriever/writer/evidence machinery directly — the plan document says so explicitly. |

Also explicitly out of scope, documented above: `verify.py`'s pre-existing paper-level
quote fallback (Finding 7) — a real but pre-existing gap this branch made more visible by
being the first code to render `[unit_id]` as a location claim to a human.

## Branch-finish checklist

- [x] All 6 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed.
- [x] All real findings adjudicated: 5 fixed (1 Critical, 4 Important), 1 parked with a
      ruling (Task 2 test coverage), 1 documented as out of scope (verify.py fallback).
- [x] Plan document amended for every category-1 (plan-conflict) fix, including the final
      review's fix wave.
- [x] Re-review of the fix wave completed by the controller directly (session-limit
      workaround), including independent mutation-testing of the two highest-stakes fixes.
- [x] Minor/observational findings parked with one-line records (above), never dropped.
- [x] This ledger written as the closing record.
- [ ] SDD workspace cleanup.
- [ ] Branch finish (merge to `main`, worktree/branch cleanup).
