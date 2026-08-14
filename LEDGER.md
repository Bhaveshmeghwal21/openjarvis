# Ledger — verifiable-single-paper-core (final, consolidated)

Plan: `docs/plans/2026-08-11-verifiable-single-paper-core.md`
Spec: `docs/specs/2026-08-11-research-corpus-agent-design.md` (build steps 1-5)
Branch: `verifiable-single-paper-core`, merge-base with `main` at `b1b7f1e`

**Correction to an earlier assumption made during this session:** the SDD ledger at
`.superpowers/sdd/2026-08-11-verifiable-single-paper-core/progress.md` (plus per-task briefs,
reports, and review diffs in the same directory) **does exist** and covers Tasks 1-10 in
detail — an earlier glob search for it during this session missed it and incorrectly
concluded no ledger existed. `handoff.md` (present in the parent repo, not this worktree) was
a *summary* of that ledger for handoff purposes, not a replacement for it. This document
appends Tasks 11-15 and the final review to that history rather than superseding it; the
original `progress.md` is left untouched as the authoritative Task 1-10 record.

## Tasks 1-10 (from `.superpowers/sdd/.../progress.md` — summarized, not altered)

All ten complete and reviewed. Two category-1 plan-conflicts required a human ruling and a
plan amendment: Task 2 (`find_span` unanchored substring match), Task 4 (`save_units`
dropping `section_path` on re-save), and Task 7 (three separate defects: decimal-numbered
label false-positive, silently-dropped textless artifacts, and an Eq./Equation abbreviation
mismatch — the last one initially mis-logged as minor, self-corrected to Important). One
category-2 finding (Task 9: implementer narrowed a verbatim-brief exception handler, no
plan-conflict, resumed directly). One category-3 finding (Task 10: `numpy` used by
`vector_search()` but never declared in `pyproject.toml`; also removed a stale unused
`sqlite-vec` entry). A running list of minor findings was parked throughout (full text in
`progress.md`) and triaged below alongside the final review's own findings.

**Ruff baseline note, corrected:** `progress.md` records "12 pre-existing violations... in
jarvis/citation_graph.py, sources.py, config.py, scoring.py, __init__.py, test_ported.py"
as of Task 1, reconfirmed unchanged through Task 10. That count silently dropped to **11**
partway through Task 15 of *this* continuation, not at the final review as originally
assumed: Task 15's plan-directed rewrite of `jarvis/__init__.py` (extending its exports) was
a wholesale replacement of that file's import block, and running `ruff --fix` on the result
incidentally corrected the one pre-existing `I001` import-order violation `__init__.py` had
carried since Task 1, as a side effect, not a deliberate fix. Verified directly: checking out
`__init__.py` as it stood at `c2b3967` (Task 10's HEAD, before Task 15 touched it) and running
`ruff check` against it under this project's actual config reproduces exactly that one
violation. From Task 15 onward the correct baseline is **11**, in `citation_graph.py` (2),
`config.py` (1), `scoring.py` (1), `sources.py` (6), `test_ported.py` (1) — confirmed present
and unchanged at every subsequent checkpoint through the final review and fix wave.

## Tasks 11-15 (this continuation)

```
627158e test: end-to-end single-paper ingest, retrieve, and verify        [Task 15]
56733e2 feat: evaluation metrics for quote fidelity, support, gate recall, coverage [Task 14]
387da50 feat: two-stage verification with quote matching and nli entailment [Task 13]
35faebe feat: hybrid retrieval with rrf fusion and parent expansion       [Task 12]
ce0dbe8 feat: fts5 keyword index with safe query escaping                [Task 11]
```

No plan-conflicts in Tasks 11-15's own reference code (pre-flight scans against the actual
current state of `store.py`, `context.py`, `embed.py`, `units.py`, and `models.py` found every
consumed interface matched exactly what the plan's briefs assumed). Two stale test-count
expectations in the plan's own prose were found and resolved in favor of the literal code
(Task 12: prose said 14, code defines 13; Task 13: prose said 15, code defines 14 — both are
the plan miscounting its own Step-1 code block, nothing was dropped, consistent with the
same category of inconsistency already logged for Task 10 in `progress.md`). Four cosmetic
ruff findings in the plan's reference code (deprecated `typing` imports vs. `collections.abc`,
import sort order, `__all__` sort order) were fixed directly as no-arbitration style/packaging
gaps, consistent with Task 10's precedent for non-plan-conflict findings.

## Final state (after Tasks 1-15 and the fix wave below)

- **185 tests, all passing** (`python -m pytest -q`, zero network, no API keys, no model
  downloads — every test uses a `Fake*` double). Verified by direct collection count
  (`--collect-only` summed across all 16 test files), not estimated.
- **`ruff check .`: exactly 11 violations**, all pre-existing per the corrected baseline
  above, in `citation_graph.py` (2), `config.py` (1), `scoring.py` (1), `sources.py` (6),
  `test_ported.py` (1). Zero violations in any file this branch's Tasks 11-15 or fix wave
  touched. Confirmed via `git diff b1b7f1e..HEAD --name-only` that none of those five files
  were modified by this branch at all.
- Definition of done (plan's own criteria, verbatim): "no network access, no API keys, no
  model downloads" ✓ · "the system catches fabrication mechanically, not by asking a model
  to be honest" ✓ (`test_a_fabricated_number_is_caught_without_consulting_the_model`,
  `tests/test_end_to_end.py`) · "`jarvis.evaluate.report` returns real numbers for quote
  fidelity and statement support" ✓ · "`ruff check .` is clean" — clean for every file this
  plan added; the 11 pre-existing violations predate Task 1 and are documented, not silently
  ignored.

## Final whole-branch adversarial review

Dispatched after Task 15, per the plan's own closing instruction ("dispatch the final
whole-branch code review on the most capable available model... this is the one review in
the whole process that should NOT be cost-optimized"). Full original report was written to
an untracked scratch file (`FINAL_REVIEW.md`, deleted before branch-finish — content
preserved here and in the fix-wave/amendment commits).

Four findings, adjudicated:

| # | Finding | Severity | Ruling | Disposition |
|---|---|---|---|---|
| 1 | `units.py`: caption/reference binding was a document-wide exact-label scan; two artifacts sharing a label (e.g. appendix renumbering) would each absorb the other's caption, letting a misattributed-but-verbatim quote ground against the wrong unit | Critical-adjacent | **Fix now** (category-1 plan-conflict: traced verbatim to Task 7's own reference code) | Fixed in `01d4044`, scoped to nearest-same-labeled-artifact binding. Plan Task 7 amended in `dffb7b6`. Independently re-reviewed and confirmed resolved across 3-artifact, non-adjacent, and tie orderings. |
| 2 | `verify.py`: `verify_claim` resolved an exact/near tie between entailment and contradiction scores to `SUPPORTED` rather than `NEUTRAL` | Moderate | **Fix now** (category-1 plan-conflict: traced verbatim to Task 13's own reference code) | Fixed in `01d4044`, ties above threshold now route to `NEUTRAL`. Plan Task 13 amended in `dffb7b6`. Independently re-reviewed against a 14-point boundary matrix, confirmed resolved with no regression to ordinary cases. |
| 3 | `embed.py`/`retrieve.py`: `vector_search` raised an opaque `numpy.ValueError` on embedder dimension drift (stale index after a model swap), not reachable via adversarial query text | Moderate | **Fix now** (category-3: packaging/robustness gap, no plan-text contradiction) — **note:** this rediscovers a finding Task 10 already parked as minor ("`vector_search`'s dimension-mismatch behavior... verified adversarially... undocumented in its docstring — caller-discipline gap, not a functional defect," `progress.md`). The final review independently re-derived it and called it moderate rather than minor; adjudicated as worth fixing regardless of which severity label is "correct," since an explicit named-dimension error is strictly better than an implicit one either way. | Fixed in `01d4044`: explicit dimension check, clear `ValueError` naming expected vs. actual dims, now also documented in the docstring (closing Task 10's originally-parked documentation gap in the same fix). No plan amendment needed (not a plan-conflict). Independently re-reviewed, error message and propagation through `retrieve.search()` confirmed correct. |
| 4 | `units.py`: `find_references`'s `\b` word-boundary check cannot match a lettered subtable variant ("Table 3a"), so referencing prose for a subtable is silently dropped from the artifact unit's context (though it still exists in the corpus via ordinary prose chunking — a completeness gap in artifact binding, not a corpus-wide loss) | Minor | **Park** (per this branch's own established convention: documented, never silently dropped, never escalated for arbitration) | Not fixed. No cross-contamination confirmed — strictly a completeness note. Documented here as the parked record. Revisit if/when lettered-subtable references are observed in a real corpus. |

Fix wave: `01d4044` (all 3 real findings + 16 regression tests across
`test_units_artifacts.py`, `test_verify.py`, `test_embed.py`, `test_retrieve.py`).
Plan amendment: `dffb7b6` (Task 7 and Task 13 sections, "Amended post-implementation" blocks).

Scoped re-review of the fix wave (independent dispatch, not self-verification) confirmed all
three fixes genuinely resolve their findings under adversarial probing beyond the original
test cases (three-artifact same-label scenarios, a 14-point NLI boundary matrix, direct
triggering of the dimension-drift error through the public `retrieve.search()` entrypoint),
with zero new issues introduced and zero drift between the plan's amended text and the real
committed code.

## Full commit history, `b1b7f1e..HEAD`, newest first

```
dffb7b6 docs: amend Task 7 and Task 13 reference code for the post-review fixes
01d4044 fix: scope artifact binding to nearest label, resolve NLI ties to neutral,
        guard vector search dimension drift            [final review fix wave]
627158e test: end-to-end single-paper ingest, retrieve, and verify        [Task 15]
56733e2 feat: evaluation metrics for quote fidelity, support, gate recall, coverage [Task 14]
387da50 feat: two-stage verification with quote matching and nli entailment [Task 13]
35faebe feat: hybrid retrieval with rrf fusion and parent expansion       [Task 12]
ce0dbe8 feat: fts5 keyword index with safe query escaping                [Task 11]
c2b3967 fix: declare numpy as an index-extra dependency, drop unused sqlite-vec
f007ca7 feat: embedder protocol and brute-force vector search            [Task 10]
6684fae fix: revert to catch-all exception handling for LLMPrefix fallback
4c7cc0d feat: contextual prefixes for child units                        [Task 9]
aec459e feat: parent-child unit structure and full layer 1 builder       [Task 8]
1d499c1 fix: reject decimal-numbered labels, handle Eq. abbreviation, never drop artifacts
6a993b2 docs: fix Eq./Equation abbreviation mismatch in Task 7 reference code
a84f8fc docs: fix two binding-rule defects in Task 7 reference code
3dab14a feat: table/figure/equation units with caption and reference binding [Task 7]
9bf65ab feat: recursive prose chunking on section boundaries              [Task 6]
e3b4a89 feat: parser protocol with fake and lazy docling adapter          [Task 5]
085957a fix: refresh section_path on unit re-save
c2cfab3 docs: fix stale section_path on unit re-save in reference code
5a289b8 feat: paper and unit persistence                                  [Task 4]
5f51705 feat: sqlite schema for corpus store                             [Task 3]
03bbbf7 fix: boundary-anchor find_span to reject partial-token matches
7086a69 docs: fix unanchored substring match in find_span reference code
5089c60 feat: text normalization and span finding for quote verification [Task 2]
cc9d8a0 fix: sort imports in test_models.py
9425c12 feat: domain types for layers 0/1/2 and verification              [Task 1]
```

## What this plan deliberately does not build

Per the plan's own closing section — spec build steps 6-10, each warranting its own plan:

| Step | Why it waits |
|---|---|
| 6. Gather + gate | Needs the eval harness (Task 14) to calibrate against. |
| 7. Compile — Q&A | Needs retrieval and verification proven first (both now are). |
| 8. MCP server | Pure surface over a core that must work first. |
| 9. Contradiction detection | `find_contradictions` exists (Task 13) but is unused until there is a multi-paper corpus to run it over. |
| 10. Long-form reports | Furthest from the verifiable core. |

Also unbuilt: the Layer 2 card extractor. `Card`/`CardField` types exist (Task 1) and
`binding_verified` is wired, but LLM card extraction belongs with the gather plan.

## Branch-finish checklist

- [x] All 15 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed.
- [x] All real findings fixed (2 category-1 plan-conflicts + 1 category-3 packaging gap,
      the latter also closing a documentation gap Task 10 had already parked).
- [x] Plan document amended for both category-1 fixes.
- [x] Scoped re-review of the fix wave, independently confirming resolution.
- [x] Minor finding (Finding 4) parked with a documented one-line record (above).
- [x] This ledger written as the closing record, appending Tasks 11-15 and the final review
      to the pre-existing Task 1-10 ledger at `.superpowers/sdd/.../progress.md` rather than
      replacing it.
- [ ] SDD workspace cleanup and `FINAL_REVIEW.md` scratch file removal.
- [ ] Branch finish (merge to `main` or equivalent).
