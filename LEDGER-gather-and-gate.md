# Ledger — gather-and-gate (spec build step 6)

Plan: `docs/plans/2026-08-14-gather-and-gate.md`
Spec: `docs/specs/2026-08-11-research-corpus-agent-design.md` (build step 6, §7 Stages A/B/C)
Branch: `gather-and-gate`, based on `main` at `508f4e2` (which already includes the merged
`verifiable-single-paper-core` work at `d7f8672`, plus the five committed step 6-10 plan
documents)

## Tasks 1-13

All 13 plan tasks implemented directly, one at a time, following strict TDD (write failing
test, verify fails, implement, verify passes, ruff, commit):

```
0dd4b16 test: end-to-end gather, gate, deep read, and card verification    [Task 13]
63975a8 feat: layer 2 card extraction with verified quote bindings        [Task 12]
bd23b96 feat: stage c deep read with parse-failure isolation              [Task 11]
11a5713 feat: seed-set sampling and hand-label round-trip                 [Task 10]
4db1eac feat: per-project gate calibration against a labeled seed set     [Task 9]
fdcec99 feat: union gate decision with three outcomes and audit log       [Task 8]
ee764d9 feat: four independent gate signals with per-signal failure isolation [Task 7]
49247cc feat: citation-graph expansion with recorded hop depth            [Task 6]
d12a532 feat: multi-source candidate fan-out with dedup and persistence   [Task 5]
accc266 feat: crossref retraction check and provenance enrichment        [Task 4]
a765438 feat: arxiv, semantic scholar, and openalex source adapters       [Task 3]
b4c9203 feat: search plan generation for gather-stage fan-out             [Task 2]
031bb2f feat: screening log, card, and run persistence                    [Task 1]
```

Pre-flight scans (reading each task's actual dependency signatures before writing code)
found zero plan-conflicts in the reference code's interface assumptions — every consumed
function (`paper_id`, `CitationWalker`, `cosine`, `paper_text`, `apply_prefixes`,
`TemplatePrefix`, `index_units`, `index_units_fts`, `build_units`, `quote_is_grounded`,
`KEPT_DECISIONS`, `GATE_RECALL_TARGET`) matched exactly what the plan's briefs assumed,
carried over unmodified from the completed single-paper-core branch.

One test-fixture-level plan-conflict found and fixed during Task 8 (not a production-code
defect): the plan's own reference test `test_a_deferred_paper_stays_in_the_corpus_at_metadata_depth`
assumed `FakeEmbedder` (built in the prior branch, unchanged) would give near-zero cosine
similarity for an unrelated paper. In practice `FakeEmbedder`'s hash-bucket design has a
noise floor of ~0.2-0.3 cosine similarity between *any* two short texts, including random
gibberish — this always clears `Thresholds.unsure_ratio`'s default band, so no paper's
embedding signal can register as fully absent using that fixture. Fixed by injecting a
minimal orthogonal-vector embedder double for that one test, isolating the signal under
test rather than relying on `FakeEmbedder`'s approximate behavior. No production code
changed; `decide()`/`screen()`/`Thresholds` match the plan's reference code exactly.

## Final state after Tasks 1-13

- **323 tests total** (185 pre-existing + 138 new), all passing, confirmed via junit-xml
  (`tests="323" failures="0" errors="0"`), zero network access.
- **`ruff check .`: exactly 11 violations**, all pre-existing, in `citation_graph.py` (2),
  `config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1) — the same
  five files and same count as the single-paper-core branch's final baseline. Zero new
  violations in any file this plan touched, confirmed by per-file ruff runs at every task
  checkpoint.
- Definition of done (plan's own criteria): offline, no network/API keys/model downloads
  ✓ · gate keeps every hand-labelled relevant paper at ≥95% recall, proven by calibration
  against real labels rather than asserted ✓ · no paper is ever removed from the corpus,
  `defer` demotes to metadata depth ✓ · a fabricated card binding is caught mechanically,
  without consulting a model ✓ · every gate decision logs all four per-signal scores ✓ ·
  `ruff check .` at the pre-existing baseline only ✓.

## Final whole-branch adversarial review

Dispatched via subagent after Task 13, framed to specifically try to break the eight
safety properties spec §7B/§14 and the plan's own "Global Constraints" name as the reason
this plan exists (gate never excludes, defer is demotion not deletion, no LLM routed to
verification, card bindings mechanically verified, citation-graph expansion respects
budget/depth/dedup, ingest isolates per-paper failures, no plan-reference-code conflicts,
general adversarial pass on races/floats/degenerate calibration).

Eight findings, adjudicated:

| # | Finding | Severity | Category | Disposition |
|---|---|---|---|---|
| 1a | Incomplete `decisions` map passed to `ingest_decided` treated identically to explicit `defer` | Minor | 3 | Parked — paper row still persists at `metadata`, no invariant violated, requires a caller to pass a partial map that the plan's own `screen()` never produces. |
| 1b | `set_depth` was a bare `UPDATE` with no rowcount check, silently no-oping on an unsaved `paper_id` | Moderate | 2 | **Fixed** — see below. |
| 2d / 7a | `calibrate()`'s default `floor=0.0` (verbatim from the plan's own Task 9 reference code) let a zero-variance signal calibrate to a threshold of `0.0`, which `decide()`'s inclusive `>=` then admits every candidate on, degenerating the gate to keep-everything for that signal | Moderate | **1 (plan-conflict)** | **Fixed**, in two passes — see below. Plan amended. |
| 5d | `expand_citations` called directly (bypassing `gather()`) without an explicit `already_seen` can re-surface a search-phase hit as a duplicate citation-origin candidate | Minor | 3 | Parked — over-inclusion (duplicate), not paper loss; `gather()`'s own reference code always passes `already_seen` correctly. |
| 7d | Plan's Task 11 reference code lists an unused `set_depth` import; the real implementation correctly omitted it | Zero | 1 (harmless errata) | Not a defect. Noted for completeness. |
| 8a | No `PRAGMA busy_timeout` configured; a genuine concurrent multi-process `screen()` collision on the same `(paper_id, run_id)` would raise `sqlite3.OperationalError` rather than corrupt data | Moderate | 3 | Parked — fail-loud not fail-silent, outside this plan's single-process scope; every one of the 138 new tests exercises single-process use. |
| 8e | `LLMCardExtractor`'s `_to_field` validates a hallucinated `unit_id` against the *full* unit set passed to `.extract()`, not the truncated subset actually shown in the prompt (`max_units=40`) | Minor | 2 | Parked — downstream `verify_card`'s mechanical quote check independently catches any resulting fabrication regardless (confirmed by direct trace of `get_unit`/`quote_is_grounded` returning `False` for any unverifiable quote), so the stated "never silently accepted as real" property holds even with this gap. |

Categories with **zero findings**, confirmed and stated explicitly by the reviewer:
citation-graph budget/depth/cycle handling (Finding 5 overall, aside from 5d); no LLM ever
routed to verification (traced every call path through `card.py`/`router.py`, confirmed
`DEFAULT_ROUTING` has no `verif*` key and `test_verification_is_not_routed_to_an_llm`
still holds); hallucinated `unit_id` acceptance (two independent layers of defense, both
confirmed); ingest failure isolation (matches plan's reference code exactly, confirmed by
trace and by the specified tests).

## Fix wave — two rounds, both independently re-reviewed

**Round 1** (`d485c93`): fixed both Finding 1b and Finding 2d/7a.
- Finding 1b: `set_depth` now checks `cursor.rowcount` and raises `ValueError` naming the
  missing `paper_id` instead of silently affecting zero rows.
- Finding 2d/7a (first attempt): changed `calibrate()`'s default `floor` from `0.0` to a
  new module constant `MIN_CALIBRATED_THRESHOLD = 0.05`, applied unconditionally via
  `max(floor, raw)`.

Plan amended in `5c626e3` (Task 9 section, "Amended post-implementation" block).

**Independent re-review of round 1** confirmed Finding 1b fully resolved with no
regression (only production call site, `screen()`, always calls `save_candidates` first;
confirmed by call-site audit). But it found a **genuine residual regression** in the
Finding 2d/7a fix: the flat `MIN_CALIBRATED_THRESHOLD` floor clamped up *any* signal
scoring below `0.05`, including one with real fine-grained separation among relevant
papers clustered below that value — dropping that signal's own recall for no reason.
Confirmed by construction: relevant-paper scores of `0.01`–`0.048` on one signal, recall
fell from `1.0` (old `floor=0.0` behavior) to `0.5` under the flat-floor fix.

**Round 2** (`4518817`): corrected the design. `calibrate()`'s `floor` parameter became
`float | None = None`. Left at its default, the floor is applied *only* to a signal whose
raw quantile-fit value is genuinely degenerate — never to one with any real separation,
however small. Passed explicitly (any value, including `0.0`), the floor instead behaves
as the original unconditional minimum, preserving the plan's own Task 9 test
(`test_calibration_respects_a_floor_so_a_signal_never_admits_everything`, which passes
`floor=0.2` against a deliberately nonzero raw fit and expects it to win regardless).
Plan amendment updated in the same commit to narrate both the original bug and the
flat-floor regression.

**Second independent re-review of round 2** confirmed the corrected design genuinely
resolves both the original degenerate-signal bug and the flat-floor regression, verified
against constructed scenarios beyond the existing tests (including a mixed case with one
degenerate and one real-separation signal in the same calibration call, confirming
strictly per-signal treatment with no cross-signal leakage). It surfaced two minor,
non-blocking polish items: the docstring's "every labeled-relevant paper scores 0.0"
phrasing over-claimed relative to the actual quantile-value check, and the bit-exact
`raw == 0.0` comparison would not rescue a signal that is degenerate in every practical
sense but lands on a tiny nonzero float (e.g. `1e-17`) due to upstream floating-point
noise.

**Polish commit** (`8276631`): replaced the bit-exact check with `_DEGENERATE_EPSILON =
1e-9` tolerance; corrected the docstring wording to describe the raw quantile-fit value
rather than "every paper."

Full fix-wave commit sequence:
```
8276631 fix: calibrate() degenerate-signal check tolerates floating-point noise
4518817 fix: calibrate()'s default floor now rescues only genuinely degenerate signals
5c626e3 docs: amend Task 9 reference code for the calibrate() degenerate-floor fix
d485c93 fix: calibrate() no longer defaults to a degenerate zero floor; set_depth raises
        on an unsaved paper_id
```

This required two full review-fix cycles rather than one — a genuinely more thorough
fix-loop than any single round in the prior single-paper-core branch, because the first
re-review caught a real regression in the first fix attempt rather than rubber-stamping it.

## Final state after the fix wave

- **329 tests, all passing** (323 + 3 new regression tests for Finding 1b/2d-7a round 1
  + 2 new regression tests for the round-2 correction and epsilon polish, net of the one
  test-fixture change during Task 8). Confirmed via junit-xml at every checkpoint.
- **`ruff check .`: still exactly 11 pre-existing violations**, same five files, zero new
  — confirmed after every commit in the fix wave, including the two re-review rounds.
- Regression tests added: `test_set_depth_on_a_nonexistent_paper_raises_instead_of_silently_doing_nothing`
  (`tests/test_store_screen.py`); `test_calibration_never_produces_a_threshold_that_admits_a_zero_scoring_signal`,
  `test_the_default_floor_does_not_sacrifice_recall_on_a_signal_with_real_separation`,
  `test_an_explicit_floor_is_unconditional_even_when_the_raw_fit_is_nonzero`,
  `test_a_near_zero_float_from_floating_point_noise_still_counts_as_degenerate`
  (all in `tests/test_gate.py`).

## What this plan deliberately does not build

Per the plan's own scope and the spec's build order — spec build steps 7-10, each
warranting its own plan (already written, per `docs/plans/2026-08-14-*.md`):

| Step | Why it waits |
|---|---|
| 7. Compile — Q&A | Needs a real multi-paper corpus to retrieve/write/verify over, which this plan now produces. |
| 8. MCP server | Pure surface over a core that must work first; benefits from steps 6+7 being usable from Claude Code early. |
| 9. Contradiction detection | `find_contradictions` (from the single-paper-core branch) is only useful once there is a multi-paper corpus to run it over — this plan produces that corpus but does not itself run cross-paper contradiction search. |
| 10. Long-form reports | Furthest from the verifiable core; depends on steps 6 and 7. |

Also unbuilt within this plan's own scope: `LLMPlanner`, `LLMVoter`, and
`LLMCardExtractor`'s live model calls are all implemented and tested against fakes, but
none has been run against a real LLM endpoint — that is a runtime/deployment concern, not
a code-completeness gap, and the plan's own design (protocol + fake + lazy live adapter)
is what makes that safe to defer.

## Branch-finish checklist

- [x] All 13 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed.
- [x] All real findings fixed (1 category-1 plan-conflict, requiring two fix rounds due to
      a caught regression, + 1 category-2 packaging gap).
- [x] Plan document amended for the category-1 fix, twice (once per fix round).
- [x] Two rounds of independent scoped re-review, the second catching what the first
      missed — both confirmed resolved.
- [x] Minor findings (1a, 5d, 7d, 8a, 8e) parked with documented one-line records above.
- [x] This ledger written as the closing record.
- [ ] SDD workspace cleanup (none created for this plan — no `.superpowers/sdd/` scratch
      directory was used since implementation was done directly, not via subagent dispatch
      per task).
- [ ] Branch finish (merge to `main` or equivalent) — not yet actioned, pending the same
      kind of explicit go-ahead the single-paper-core branch required before its merge.
