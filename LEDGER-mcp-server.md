# Ledger — mcp-server (spec build step 8)

Plan: `docs/plans/2026-08-14-mcp-server.md`
Spec: `docs/specs/2026-08-11-research-corpus-agent-design.md` (build step 8, §3)
Branch: `mcp-server`, based on `main` at `4be7458` (which already includes the merged
`gather-and-gate` and `compile-cited-qa` branches — spec build steps 6 and 7)

## Architecture

Two layers, per the plan's own explicit framing: `jarvis/tools.py` is a pure dispatcher —
a registry of tool specs with JSON schemas, argument validation, and handlers that take a
context and return plain dicts. It imports nothing from `mcp` and is fully testable
offline. `jarvis/mcp_server.py` is a thin stdio adapter that imports `mcp` lazily and does
nothing but translate between the protocol and the dispatcher. Every behaviour worth
testing lives in the layer with no protocol dependency.

## Tasks 1-5

All 5 plan tasks implemented directly, one at a time, following strict TDD (write failing
test, verify fails, implement, verify passes, ruff, commit):

```
c61b51d test: mcp client loop, hostile-argument coverage, and client docs   [Task 5]
ff2b5e5 feat: stdio mcp server and jarvis-mcp entry point                  [Task 4]
4a31591 feat: deterministic quote verification and cited ask tools         [Task 3]
d2407df feat: corpus search, unit, paper, and listing tools                [Task 2]
288b4ab feat: tool registry and no-raise dispatcher                       [Task 1]
```

Pre-flight scans (reading each task's actual dependency signatures before writing code)
found the interfaces this plan consumes — `search`, `get_unit`, `get_paper`, `get_units`,
`get_papers_by_depth`, `quote_is_grounded`, `ask`, `render_answer`, `Answer.claim_for`,
`FakeWriter`, `FakeNLI`, `BGEEmbedder`, `ModelRouter`, `HFNLI`, `LLMWriter` — matched the
plan's assumptions exactly, carried over unmodified from the two prior merged branches.

One test-fixture-level plan-conflict found and fixed during Task 2 (not a production-code
defect, and the fourth occurrence of this exact pattern across this repo's three feature
branches): the plan's own reference test for `corpus_search`'s empty-hits case queried a
nonempty store with a "nonsense" string and asserted zero results. `FakeEmbedder`'s vector
search has no relevance floor — RRF fusion always ranks *some* candidate regardless of
query content once the store is nonempty, so no query string is "nonsense enough." Fixed
by using a genuinely empty store instead (`open_store(tmp_path / "empty.db")`), the exact
fix already established as precedent by `LEDGER-compile-cited-qa.md` Finding 4. No
production code changed.

One new ruff finding caught during Task 4 against the plan's own reference code:
`import mcp.types as types` triggers `PLR0402` (manual-from-import). Changed to
`from mcp import types` — functionally identical, holds the 11-violation baseline exactly.

One new ruff finding caught during Task 5 (`RUF022`, `__all__` not sorted) after extending
`jarvis/__init__.py`'s exports — resolved with `ruff check --fix`.

## Final state after Tasks 1-5

- **448 tests total** (403 pre-existing + 45 new: 10 in `test_tools_dispatch.py`, 27 in
  `test_tools.py`, 8 in `test_mcp_end_to_end.py`), all passing, confirmed via junit-xml.
- **`ruff check .`: exactly 11 violations**, all pre-existing, same five files as every
  prior branch (`citation_graph.py` 2, `config.py` 1, `scoring.py` 1, `sources.py` 6,
  `test_ported.py` 1). Zero new violations left in any file this plan touched.
- Definition of done (plan's own criteria): offline, no network/API keys/model downloads,
  and the whole suite passes without the `mcp` package installed ✓ (confirmed via
  `python -c "import jarvis; import jarvis.tools"` succeeding with `mcp` absent) ·
  `test_no_tool_call_can_raise` passes against hostile arguments on every registered tool
  ✓ · `test_the_server_module_contains_no_corpus_logic` passes ✓ ·
  `test_the_dishonest_path_is_refused_at_the_last_step` passes — a plausible fabricated
  quote is caught deterministically with no model involved ✓ · `ruff check .` at exactly
  the 11-violation baseline ✓.

## Final whole-branch adversarial review

Dispatched via subagent after Task 5, framed around 8 specific areas: the no-raise
contract under hostile arguments, `jarvis/tools.py`'s isolation from `mcp` (both the
implementation and the test methodology asserting it), `verify_quote`'s determinism and
freedom from any model routing, the `ask` tool's resistance to the exact claim-id-collision
defect class that was Critical on `compile-cited-qa`, type-confusion robustness in
`clamp_limit`/`requires`, `jarvis/mcp_server.py`'s freedom from corpus logic, tool
description accuracy against actual behavior, and a general adversarial pass (races,
N+1 queries, silent truncation, plan-vs-implementation conflicts).

Result: **0 Critical, 0 Important, 6 Minor findings**, no code changes made by the
reviewer (read-only audit as instructed). Independently re-verified before trusting: the
reviewer's stated test count (569) did not match this session's own confirmed count (448)
and was disregarded as likely miscounted or cross-contaminated with another environment;
every other claim was checked directly.

| # | Finding | Severity | Category | Disposition |
|---|---|---|---|---|
| 1 | `list_papers`'s `DEPTHS` tuple includes `"abstract"`, but no ingestion path in this repo currently writes `depth="abstract"` | Minor | Initially flagged as 1 (plan-conflict) | **Not a defect on closer inspection.** `store.py`'s own schema comment (from the `gather-and-gate` branch) documents the `depth` column's full vocabulary as `metadata \| abstract \| pending_deep \| deep` — `abstract` is part of the documented depth model, reserved for a future ingest stage, not a phantom value invented by this plan. `list_papers` correctly supports querying for it. Left unchanged; correction recorded in the fix-wave commit message rather than in a code change. |
| 2 | `list_papers` has an N+1 query pattern (one `get_units` call per paper for the count — confirmed at 301 SQL statements for 300 papers via `sqlite3.set_trace_callback`), and no response carrying a `count` also carries a `total`/`has_more`, so a client cannot tell "10 of 10" from "10 of 300" once `limit` clamps the result | Minor | 3 (uncovered gap) | **Parked.** Not yet pathologically slow (8ms at N=300 in the reviewer's own measurement) and out of scope for a read-only surface whose corpora, per every prior branch's own test fixtures, are single-paper to low-hundreds in size. Worth a follow-up if `list_papers` is ever pointed at a corpus large enough for either the N+1 cost or the truncation-visibility gap to matter in practice. |
| 3 | `clamp_limit`'s `except (TypeError, ValueError)` missed `OverflowError` — `int(float("inf"))` raises it directly | Minor | 3 (robustness gap the plan's own reference code didn't cover) | **Fixed** in `e90bcd2`. `except (TypeError, ValueError, OverflowError)` added; regression test added. The no-raise contract held end-to-end regardless (`call_tool`'s own outer `try/except` — explicit defense-in-depth per the plan's Global Constraints — already caught it and returned a clean error dict), but `clamp_limit`'s own docstring claims to "never trust the other side," so this closes the gap at the source. |
| 4 | `ask`'s tool description said "Requires a configured writer model" but never mentioned `nli`, despite `requires=("writer", "nli")` | Minor | 1 (plan-conflict — traces verbatim to the plan's own Task 3 reference text) | **Fixed** in `e90bcd2`. Description now names both dependencies. Purely cosmetic: the dispatcher's actual runtime error already names whichever dependency is genuinely missing, so no client-visible behavior was ever wrong — only the static tool listing undersold what `ask` needs. |
| 5 | The test asserting `jarvis/tools.py` never imports `mcp` (`test_tools_module_does_not_import_mcp`) does so via a source-text grep, which is evadable in principle (e.g. `importlib.import_module("mcp")` would violate the constraint's spirit while passing the grep) | Minor | 3 (test-methodology gap) | **Parked.** The reviewer independently AST-walked the actual `jarvis/tools.py` and confirmed it is clean of any `mcp` reference in any form — this is a latent weakness in how the constraint is *tested*, not a live defect in what's *shipped*. Worth hardening the test to an AST-based check if this module is touched again, but not urgent. |
| 6 | `jarvis/mcp_server.py` uses `from mcp import types` where the plan's own reference code used `import mcp.types as types` | None (non-issue) | 2 (already-resolved deviation) | **Not applicable.** This was already a deliberate, already-documented deviation made during Task 4 (commit `ff2b5e5`) specifically to avoid a new `PLR0402` ruff violation — functionally identical, no behavioral difference, already recorded. |

Zero findings, explicitly confirmed by the reviewer through direct execution rather than
inspection: the no-raise contract (survived unicode edge cases, deeply nested structures,
SQL-injection-shaped strings in every string argument, non-JSON-serializable objects, and
concurrent `REGISTRY` mutation across six threads with zero exceptions crossing the
boundary and zero database damage); `jarvis/tools.py`'s actual isolation from `mcp`
(AST-confirmed clean); `verify_quote`'s determinism (full call path traced through
`quote_is_grounded`, zero model or network involvement, works with `ctx.nli`/`ctx.writer`
both `None`); and the `ask` tool's resistance to the Critical claim-id-collision defect
class from `compile-cited-qa` (tested duplicate `claim_id` collisions in three orderings
including a three-way collision; `answer.py`'s `_dedupe_claim_ids` fix from that branch's
own remediation generalizes correctly here, and `jarvis/tools.py`'s `_ask` handler adds no
new exposure since it only ever iterates `answer.supported`/`answer.flagged`, both of which
only ever contain verified claims).

**This branch does not repeat the Critical defect class found on `compile-cited-qa`.**

## Fix wave and re-review

Fixed in `e90bcd2`: Finding 3 (`clamp_limit` now catches `OverflowError`, regression test
added) and Finding 4 (`ask`'s description now names both required dependencies). Findings
1, 2, 5, 6 parked with the reasoning above, documented in the fix-wave commit message —
never silently dropped, per this repo's established convention.

Independent re-review performed directly by the controller rather than re-dispatched to a
subagent, consistent with `handoff.md`'s own guidance that this is acceptable for a small,
non-safety-critical fix wave (contrast with `gather-and-gate`'s `calibrate()` fix, which
needed two full dispatch-and-re-review cycles because the first fix attempt introduced a
real regression). Both fixes were mutation-tested rather than merely re-run:

- **`clamp_limit`**: the pre-fix `except (TypeError, ValueError)` clause was reconstructed
  inline and confirmed to genuinely raise `OverflowError` on `float("inf")`; the actual
  current fixed code was confirmed not to.
- **`ask`'s description**: directly inspected — `REGISTRY["ask"].description` now mentions
  both `"writer"` and `"nli"`, matching `requires=("writer", "nli")` exactly.

Final state after the fix wave: **449 tests passing** (448 + 1 new regression test for the
`clamp_limit` fix), 0 failures, 0 errors. `ruff check .` still exactly the 11-violation
baseline. No second fix-and-re-review cycle was needed.

## What this plan deliberately does not build

Per the plan's own explicit "Where this stops" section:

| Not exposed | Why |
|---|---|
| Gathering / screening | Spends money and rewrites the corpus. A read-only surface can be pointed at a shared project safely; a write surface cannot. |
| Card extraction | Same reason, plus it needs a model the search-only server does not load. |
| Contradiction scanning | `docs/plans/2026-08-14-contradiction-detection.md` (spec step 9) builds it; expose it here afterward if it proves useful interactively. |
| Long-form report generation | Minutes-long and multi-call. Wrong shape for a synchronous tool call. |

Also unbuilt within this plan's own scope: the `--with-models` path in
`build_context` (real `BGEEmbedder`, `HFNLI`, `LLMWriter`, routed through `ModelRouter`)
is implemented and type-checked against the actual constructors of each, but has not been
exercised end-to-end against a live model endpoint or the real `mcp` package (which is not
installed in this environment) — `serve()` itself was never executed, only confirmed to
import correctly when `mcp` is absent. This is a deployment-time verification gap, not a
code-completeness gap: the plan's own design (protocol imports lazy, inside the function
that needs them) is exactly what makes that safe to defer.

## Branch-finish checklist

- [x] All 5 plan tasks complete, tested, committed.
- [x] Final whole-branch adversarial review dispatched and completed.
- [x] Fix wave for the two findings worth fixing; three parked with documented reasoning
      (one of the three findings the reviewer raised, Finding 1, was found on
      investigation not to be a defect at all — also documented).
- [x] Independent re-review of the fix wave, performed directly with mutation-testing on
      both fixes rather than merely re-running the existing suite.
- [x] This ledger written as the closing record.
- [ ] Branch finish (merge to `main`) — not yet actioned, pending the same kind of
      explicit go-ahead every merge on this repo has required so far.
