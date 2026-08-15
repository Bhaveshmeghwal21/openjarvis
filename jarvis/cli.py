"""`jarvis` — the operator surface over the corpus pipeline.

Every subcommand shares the same project resolution, config loading, and store lifecycle
(this module owns all three); each subcommand's own logic is a thin wrapper over already
merged, reviewed library functions (design spec docs/specs/2026-08-15-cli-and-operations.md
§3 — "no new retrieval, verification, or synthesis logic").

Heavy dependencies are never imported at module scope, matching the rest of this package
(`jarvis.llm`, `jarvis.embed`, `jarvis.verify`, `jarvis.parse` all import lazily) — this
file must stay importable, and its own tests offline, without any optional extra
installed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from jarvis.answer import ask, render_answer
from jarvis.card import extract_and_verify
from jarvis.config import Config
from jarvis.contradict import (
    apply_reviews,
    rank,
    read_reviews,
    render_conflicts,
    scan_corpus,
    write_review_sheet,
)
from jarvis.evaluate import contradiction_precision
from jarvis.fetch import fetch_pdf
from jarvis.gate import Signals, calibrate, calibration_report, screen
from jarvis.gather import Candidate, gather, save_candidates
from jarvis.ingest import failed, ingest_decided
from jarvis.label import label_progress, read_labels, sample_seed, write_label_sheet
from jarvis.models import Claim
from jarvis.report import corpus_cards, render_report, write_report
from jarvis.router import ModelRouter
from jarvis.sources import (
    combine_sources,
    make_arxiv_search,
    make_crossref_search,
    make_openalex_search,
    make_s2_search,
)
from jarvis.store import (
    close_store,
    get_paper,
    get_papers_by_depth,
    get_screen_signals,
    open_store,
    save_run,
)

DEPTHS = ("deep", "pending_deep", "metadata", "abstract")


class ModelBuildError(RuntimeError):
    """A model this subcommand needs could not be constructed.

    Always names the exact missing environment variable or optional extra. Raised
    instead of ever letting a subcommand fall through to a `Fake*` double -- silently
    substituting a fake model in an operator tool would produce a corpus that looks real
    and is not, the precise failure class this whole system exists to prevent (design
    spec docs/specs/2026-08-15-cli-and-operations.md §7).
    """


def _require_chat_credentials(config: Config) -> None:
    """`jarvis.llm.chat` reads `JARVIS_BASE_URL`/`JARVIS_API_KEY` from the environment
    directly, not from a `Config` object -- checking `config`'s own copy of the same
    values here (populated from the same environment by `Config.load()`) fails loud
    before any `LLM*` class's own `try/except` around `chat_fn()` would otherwise
    swallow the resulting `openai` error and quietly fall back to an empty/neutral
    result that looks like "no evidence" rather than "misconfigured".

    Strips before checking: a whitespace-only value (`JARVIS_API_KEY="   "`, a realistic
    copy-paste artifact from a shell export) is truthy in Python and would otherwise pass
    this check, reach `openai.OpenAI(...)`, fail there with a connection/auth error inside
    an `LLM*` class's own broad `except Exception`, and silently produce the exact
    "empty draft, looks like no evidence" outcome this check exists to prevent —
    contradicting the fail-loud contract this function is named for.
    """
    if not (config.base_url or "").strip():
        raise ModelBuildError(
            "JARVIS_BASE_URL is not set — required to construct a real model client"
        )
    if not (config.api_key or "").strip():
        raise ModelBuildError(
            "JARVIS_API_KEY is not set — required to construct a real model client"
        )


def _require_importable(module: str, extra: str) -> None:
    """Fail naming the exact optional extra, not a bare `ModuleNotFoundError`."""
    try:
        __import__(module)
    except ModuleNotFoundError as exc:
        raise ModelBuildError(
            f"{module!r} is not installed — run `pip install -e \".[{extra}]\"` "
            f"to enable this command"
        ) from exc


def build_router(config: Config):
    """Always constructible: no model call happens until `.route()`/`chat()` is used."""
    from jarvis.router import ModelRouter
    return ModelRouter(overrides=config.model_overrides)


def build_writer(config: Config, router):
    _require_chat_credentials(config)
    from jarvis.writer import LLMWriter
    return LLMWriter(router)


def build_planner(config: Config, router):
    _require_chat_credentials(config)
    from jarvis.gather import LLMPlanner
    return LLMPlanner(router)


def build_voter(config: Config, router):
    _require_chat_credentials(config)
    from jarvis.gate import LLMVoter
    return LLMVoter(router)


def build_card_extractor(config: Config, router):
    _require_chat_credentials(config)
    from jarvis.card import LLMCardExtractor
    return LLMCardExtractor(router)


def build_refiner(config: Config, router):
    _require_chat_credentials(config)
    from jarvis.retriever import LLMRefiner
    return LLMRefiner(router)


def build_outliner(config: Config, router):
    _require_chat_credentials(config)
    from jarvis.outline import LLMOutliner
    return LLMOutliner(router)


def build_embedder(config: Config):
    """`config` is accepted for a signature uniform with the other `build_*` helpers,
    even though a `BGEEmbedder` needs no credentials -- only the extra installed."""
    del config
    _require_importable("sentence_transformers", "index")
    from jarvis.embed import BGEEmbedder
    return BGEEmbedder()


def build_nli(config: Config):
    del config
    _require_importable("transformers", "verify")
    from jarvis.verify import HFNLI
    return HFNLI()


def build_parser(config: Config):
    del config
    _require_importable("docling", "parse")
    from jarvis.parse import DoclingParser
    return DoclingParser()


def resolve_db_path(*, project: str | None, db: str | None) -> Path:
    """Resolve the corpus db path for a subcommand. `--db` is an explicit override.

    Exits with a named error (never a bare traceback) when neither is given — every
    subcommand needs to know which corpus it is operating on, and guessing would be
    worse than asking.

    `--project` is a name, not a path, and is never sanitized by `Config.project_dir`
    itself (plain `Path.__truediv__`, which accepts `..` and absolute overrides
    unconditionally) — an operator running this on a shared or automated system could
    otherwise point `--project ../../../etc` (or an absolute path) at an arbitrary
    location on disk, entirely outside `$JARVIS_PROJECT_ROOT`, and every subsequent
    subcommand would read/write corpus data, PDFs, and review sheets there without any
    indication anything unusual happened. Resolved and checked here, once, for every
    subcommand that reaches this function — rather than requiring each of the 8
    subcommands' own file-writing code (fetch cache, report sidecars, review sheets,
    label sheets) to defend against it separately.
    """
    if db:
        return Path(db)
    if project:
        root = Config.load().project_root.resolve()
        candidate = (root / project).resolve()
        if candidate != root and root not in candidate.parents:
            print(f"error: --project {project!r} resolves outside the project root "
                  f"({root}) — project names must not contain '..' or be absolute paths",
                  file=sys.stderr)
            raise SystemExit(2)
        return candidate / "corpus.db"
    print("error: one of --project or --db is required", file=sys.stderr)
    raise SystemExit(2)


def _open(args: argparse.Namespace):
    """Resolve, then open. A path that cannot be created (blocked by a file, permissions)
    becomes a named error, not a traceback."""
    path = resolve_db_path(project=args.project, db=args.db)
    try:
        return open_store(path)
    except OSError as exc:
        print(f"error: could not open corpus at {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def cmd_status(conn, args: argparse.Namespace) -> int:
    """Paper counts by depth, total units, and the most recent run's cost (spec §6)."""
    print("papers by depth:")
    for depth in DEPTHS:
        count = len(get_papers_by_depth(conn, depth))
        print(f"  {depth:<13} {count}")

    total_units = conn.execute("SELECT COUNT(*) FROM units").fetchone()[0]
    print(f"units: {total_units}")

    last_run = conn.execute(
        "SELECT run_id, question, started_at, cost_usd FROM runs "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if last_run is None:
        print("runs: none yet")
    else:
        print(f"last run: {last_run['run_id']} ({last_run['started_at']}) "
              f"cost=${last_run['cost_usd']:.4f}")
    return 0


def _resumable_candidates(conn) -> list[Candidate]:
    """Papers already at `pending_deep` from an earlier run's screen. Re-running `gather`
    must pick up these rather than re-searching from scratch (spec §8) -- the depth
    transition already encodes progress; only the fields `Candidate`/`to_paper` read are
    reconstructed, which is exactly the subset `ingest_decided`'s `path_for` and
    `to_paper` actually consume.

    `citation_graph.paper_id()` resolves a candidate dict's id from `arxiv_id` first,
    then `s2_id`, falling back to a title prefix only when both are empty -- a paper
    stored with neither field populated (real for many sources) would otherwise
    reconstruct to a *different* id than `papers.paper_id` already on file, silently
    breaking every subsequent `save_paper`/`set_depth` lookup keyed on the original id.
    Putting the stored `paper_id` itself into `arxiv_id` when it is empty keeps
    `paper_id(candidate.paper) == paper.paper_id` exactly (`paper_id()` checks `arxiv_id`
    before `s2_id`, so this is decisive regardless of whether `s2_id` is also set).
    """
    out = []
    for paper in get_papers_by_depth(conn, "pending_deep"):
        out.append(Candidate(paper={
            "id": paper.paper_id, "title": paper.title, "authors": list(paper.authors),
            "year": paper.year, "venue": paper.venue, "doi": paper.doi,
            "arxiv_id": paper.arxiv_id or paper.paper_id, "s2_id": paper.s2_id,
            "abstract": paper.abstract, "citation_count": paper.citation_count,
            "retracted": paper.retracted, "pdf_url": paper.source_path,
        }))
    return out


def cmd_gather(conn, args: argparse.Namespace) -> int:
    """Stages A-C end to end: search -> screen -> [confirm] -> fetch -> ingest -> cards.

    Owns one `ModelRouter` for the whole run and writes its measured cost via `save_run`
    in a `finally`, so a run that fails midway still records what it spent (spec §5.3) --
    a cost number that only appears on success is not a cost control.
    """
    run_id = uuid.uuid4().hex[:12]
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    config = Config.load()
    router = ModelRouter(overrides=config.model_overrides)

    try:
        return _run_gather(conn, args, config, router, run_id)
    finally:
        save_run(conn, run_id, question=args.question, started_at=started_at,
                 cost_usd=router.cost.total_cost)


def _run_gather(conn, args: argparse.Namespace, config: Config, router,
                run_id: str) -> int:
    resumed = _resumable_candidates(conn)
    if resumed:
        print(f"resuming {len(resumed)} paper(s) already screened as pending_deep "
              f"— skipping search and screen")
        candidates = resumed
        decisions = {c.pid: "read_deep" for c in candidates}
    else:
        try:
            planner = build_planner(config, router)
        except ModelBuildError:
            from jarvis.gather import TemplatePlanner
            planner = TemplatePlanner()

        search_fn = combine_sources(
            make_arxiv_search(limit=args.limit),
            make_s2_search(limit=args.limit),
            make_openalex_search(limit=args.limit, mailto=config.unpaywall_email or ""),
            make_crossref_search(rows=args.limit),
        )
        candidates = gather(args.question, planner, search_fn, budget=args.budget)
        save_candidates(conn, candidates)
        print(f"found {len(candidates)} candidate(s)")
        if not candidates:
            print("nothing found — stopping before screen")
            return 0

        try:
            embedder = build_embedder(config)
        except ModelBuildError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            voter = build_voter(config, router)
        except ModelBuildError:
            voter = None

        decisions = screen(conn, candidates, args.question, embedder, voter=voter,
                           run_id=run_id)

    from jarvis.gate import KEPT
    kept_count = sum(1 for d in decisions.values() if d in KEPT)
    print(f"screen: {kept_count}/{len(decisions)} kept for deep read")

    if not args.yes:
        print("pausing before deep reads — pass --yes to fetch, parse, embed, and "
              "extract cards for the kept papers (this is the point a run starts "
              "costing more than search API calls)")
        return 0

    try:
        parser = build_parser(config)
        embedder = build_embedder(config)
    except ModelBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cache_dir = Path(resolve_db_path(project=args.project, db=args.db)).parent / "pdfs"

    def path_for(candidate: Candidate) -> str:
        return fetch_pdf(candidate.paper, cache_dir, unpaywall_email=config.unpaywall_email or "",
                        paper_id=candidate.pid) or ""

    results = ingest_decided(conn, decisions, candidates, parser, embedder,
                             path_for=path_for)
    ok = [r for r in results if r.ok]
    bad = failed(results)
    print(f"ingested {len(ok)} paper(s), {len(bad)} failed")
    for r in bad:
        print(f"  failed: {r.paper_id} — {r.error}")

    if ok:
        try:
            extractor = build_card_extractor(config, router)
        except ModelBuildError as exc:
            print(f"error: cannot extract cards: {exc}", file=sys.stderr)
            return 1
        extracted = 0
        for result in ok:
            paper = next((c for c in candidates if c.pid == result.paper_id), None)
            if paper is None:
                continue
            from jarvis.gather import to_paper
            try:
                extract_and_verify(conn, to_paper(paper), extractor)
                extracted += 1
            except Exception as exc:  # noqa: BLE001 - one card failing is not the run failing
                print(f"  card extraction failed: {result.paper_id} — {exc}",
                      file=sys.stderr)
        print(f"extracted {extracted}/{len(ok)} card(s)")

    return 0


def cmd_ask(conn, args: argparse.Namespace) -> int:
    """Thin wrapper over `ask()` -> `render_answer()`. Needs a writer and an NLI model."""
    config = Config.load()
    router = ModelRouter(overrides=config.model_overrides)
    try:
        embedder = build_embedder(config)
        writer = build_writer(config, router)
        nli = build_nli(config)
    except ModelBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    answer = ask(conn, args.question, embedder, writer, nli)
    print(render_answer(answer))

    if args.out:
        Path(args.out).write_text(render_answer(answer), encoding="utf-8")
    return 0


def cmd_report(conn, args: argparse.Namespace) -> int:
    """Thin wrapper over `write_report()` -> `render_report()`.

    Fails loud when `corpus_cards(conn)` is empty rather than emitting an empty report
    (design spec §5.2) — without this, a corpus that never had card extraction wired in
    (the exact gap this whole spec closes for `jarvis gather`) would produce a report
    that looks broken rather than a clear, named reason why.
    """
    cards = corpus_cards(conn)
    if not cards:
        print("error: no Layer 2 cards in this corpus — run `jarvis gather --yes` "
              "first, or this project's deep-read papers were ingested before card "
              "extraction was wired in and need a re-gather", file=sys.stderr)
        return 1

    config = Config.load()
    router = ModelRouter(overrides=config.model_overrides)
    try:
        embedder = build_embedder(config)
        writer = build_writer(config, router)
        nli = build_nli(config)
    except ModelBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        outliner = build_outliner(config, router)
    except ModelBuildError:
        from jarvis.outline import TemplateOutliner
        outliner = TemplateOutliner()

    report = write_report(conn, args.topic, outliner, embedder, writer, nli)
    rendered = render_report(conn, report)
    print(rendered)

    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")

    _save_claims_sidecar(_reports_dir(args) / "latest.json", report.all_claims)
    return 0


def _reports_dir(args: argparse.Namespace) -> Path:
    return Path(resolve_db_path(project=args.project, db=args.db)).parent / "reports"


def _save_claims_sidecar(path: Path, claims) -> None:
    """The claims a report's sections actually cited, as JSON next to its markdown.

    There is no report-persistence layer anywhere else in this codebase --
    `write_report` returns a `Report` held only in memory. Spec §9's own resolution
    ("scan the claims from the most recent report") is unimplementable without this:
    something has to leave a report's claims somewhere `jarvis contradictions` can find
    them later. JSON, not markdown-parsing, because a claim's `unit_id`/`quote` need to
    round-trip exactly, and rendered markdown deliberately drops verification detail for
    flagged/blocked claims.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"claims": [
        {"claim_id": c.claim_id, "text": c.text, "unit_id": c.unit_id, "quote": c.quote}
        for c in claims
    ]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_claims_sidecar(path: Path) -> list[Claim]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id", "") or "")
        text = str(row.get("text", "") or "")
        unit_id = str(row.get("unit_id", "") or "")
        quote = str(row.get("quote", "") or "")
        if claim_id and text and unit_id and quote:
            out.append(Claim(claim_id=claim_id, text=text, unit_id=unit_id, quote=quote))
    return out


def cmd_contradictions(conn, args: argparse.Namespace) -> int:
    """`scan_corpus()` -> `rank()` -> a review sheet and a human-readable queue.

    Claims come from the most recent report, or `--from-report` (design spec §9 — the
    spec's own least-confident decision). Scanning with no report available is a named
    error, not an empty result — an empty scan and "there was nothing to scan" must never
    look the same.
    """
    sidecar = Path(args.from_report) if args.from_report else _reports_dir(args) / "latest.json"
    if not sidecar.is_file():
        print(f"error: no report found at {sidecar} — run `jarvis report` first, or "
              f"pass --from-report <path> to an existing one", file=sys.stderr)
        return 1
    claims = _load_claims_sidecar(sidecar)

    config = Config.load()
    try:
        embedder = build_embedder(config)
        nli = build_nli(config)
    except ModelBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    run_id = uuid.uuid4().hex[:12]
    conflicts = scan_corpus(conn, claims, nli, embedder, run_id=run_id,
                            budget=args.budget)
    print(render_conflicts(conflicts))

    sheet_path = Path(resolve_db_path(project=args.project, db=args.db)).parent \
        / "reviews" / "contradictions.jsonl"
    rows_written = write_review_sheet(sheet_path, rank(conflicts))
    print(f"\nwrote {rows_written} candidate(s) to {sheet_path}")
    return 0


def cmd_review(conn, args: argparse.Namespace) -> int:
    """`read_reviews()` -> `apply_reviews()` -> `contradiction_precision`, printed
    against the 0.70 target. This is the command that finally produces the number the
    entire build order has been blocked on (design spec §5, §6)."""
    sheet = Path(args.sheet)
    if not sheet.is_file():
        print(f"error: review sheet not found: {sheet}", file=sys.stderr)
        return 1

    reviews = read_reviews(sheet)
    applied = apply_reviews(conn, reviews)
    print(f"applied {applied} review(s)")

    precision = contradiction_precision(reviews)
    print(f"contradiction precision: {precision:.1%} "
          f"({'meets' if precision >= 0.70 else 'below'} the 0.70 target) "
          f"over {len(reviews)} reviewed candidate(s)")
    return 0


def _screened_candidates(conn, run_id: str) -> tuple[list[Candidate], dict[str, Signals]]:
    """Every paper `screen()` scored under `run_id`, as `Candidate`s (for
    `sample_seed`/`write_label_sheet`, which read `.paper` dict fields) plus their raw
    `Signals` (for `calibrate()`). Reuses the same id-preserving reconstruction as
    `_resumable_candidates` in `cmd_gather` -- `Candidate.pid` must resolve back to the
    exact stored `paper_id`, or every downstream lookup silently keys on the wrong id."""
    raw_signals = get_screen_signals(conn, run_id)
    candidates: list[Candidate] = []
    signal_rows: dict[str, Signals] = {}
    for paper_id, scores in raw_signals.items():
        paper = get_paper(conn, paper_id)
        if paper is None:
            continue
        candidates.append(Candidate(paper={
            "id": paper.paper_id, "title": paper.title, "abstract": paper.abstract,
            "year": paper.year, "arxiv_id": paper.arxiv_id or paper.paper_id,
            "s2_id": paper.s2_id,
        }))
        signal_rows[paper_id] = Signals(**{k: v for k, v in scores.items()
                                          if k in ("embedding", "graph", "keyword",
                                                   "llm_vote")})
    return candidates, signal_rows


def cmd_calibrate(conn, args: argparse.Namespace) -> int:
    """The gate's hand-label round trip (design spec §7B): `sample_seed` ->
    `write_label_sheet` -> [a human edits it] -> `read_labels` -> `calibrate` ->
    `calibration_report`, plus `label_progress` for a sheet still in progress."""
    labels_dir = Path(resolve_db_path(project=args.project, db=args.db)).parent / "labels"

    if args.subcommand == "seed":
        candidates, _ = _screened_candidates(conn, args.run_id)
        if not candidates:
            print(f"error: no screened papers found for run_id={args.run_id!r} — "
                  f"run `jarvis gather` first", file=sys.stderr)
            return 1
        seed = sample_seed(candidates, size=args.size)
        sheet_path = labels_dir / "seed.jsonl"
        n = write_label_sheet(sheet_path, seed)
        print(f"wrote {n} paper(s) to label at {sheet_path}")
        return 0

    if args.subcommand == "progress":
        progress = label_progress(args.sheet)
        print(f"labeled {progress['labeled']}/{progress['total']} "
              f"({progress['relevant']} relevant, {progress['remaining']} remaining)")
        return 0

    if args.subcommand == "fit":
        _, signal_rows = _screened_candidates(conn, args.run_id)
        labels = read_labels(args.sheet)
        relevant = [pid for pid, is_relevant in labels.items()
                   if is_relevant and pid in signal_rows]
        if not relevant:
            print("error: no labeled-relevant paper in this sheet matches a screened "
                  "paper for this run_id — calibration needs at least one true positive",
                  file=sys.stderr)
            return 1

        thresholds = calibrate(signal_rows, labels)
        report = calibration_report(signal_rows, labels, thresholds)
        print(f"recall: {report['recall']:.1%}  precision: {report['precision']:.1%}")
        print(f"kept {report['kept']}/{report['labeled']} labeled paper(s), "
              f"{report['relevant_kept']}/{report['relevant']} relevant kept")
        print(f"thresholds: {report['thresholds']}")
        return 0

    print(f"error: unknown calibrate subcommand {args.subcommand!r}", file=sys.stderr)
    return 2


COMMANDS = {
    "status": cmd_status,
    "gather": cmd_gather,
    "ask": cmd_ask,
    "report": cmd_report,
    "contradictions": cmd_contradictions,
    "review": cmd_review,
    "calibrate": cmd_calibrate,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis")
    sub = parser.add_subparsers(dest="command", required=True)

    status_p = sub.add_parser("status")
    _add_store_args(status_p)

    gather_p = sub.add_parser("gather")
    _add_store_args(gather_p)
    gather_p.add_argument("question", help="the research question to gather papers on")
    gather_p.add_argument("--yes", action="store_true",
                          help="proceed through deep reads and card extraction without "
                               "pausing for confirmation")
    gather_p.add_argument("--budget", type=int, default=200,
                          help="max citation-graph-expanded candidates (gather()'s own "
                               "budget parameter)")
    gather_p.add_argument("--limit", type=int, default=20,
                          help="max results per search query, per source")

    ask_p = sub.add_parser("ask")
    _add_store_args(ask_p)
    ask_p.add_argument("question", help="the question to answer from the corpus")
    ask_p.add_argument("--out", default=None, help="also write the rendered answer here")

    report_p = sub.add_parser("report")
    _add_store_args(report_p)
    report_p.add_argument("topic", help="the report topic")
    report_p.add_argument("--out", default=None, help="also write the rendered report here")

    contradictions_p = sub.add_parser("contradictions")
    _add_store_args(contradictions_p)
    contradictions_p.add_argument("--from-report", default=None,
                                  help="explicit path to a report's claims sidecar JSON "
                                       "(default: this project's most recent report)")
    contradictions_p.add_argument("--budget", type=int, default=500,
                                  help="max candidates returned (scan_corpus()'s own "
                                       "budget parameter)")

    review_p = sub.add_parser("review")
    _add_store_args(review_p)
    review_p.add_argument("sheet", help="path to a reviewed contradictions.jsonl sheet")

    calibrate_p = sub.add_parser("calibrate")
    _add_store_args(calibrate_p)
    calibrate_sub = calibrate_p.add_subparsers(dest="subcommand", required=True)

    seed_p = calibrate_sub.add_parser("seed")
    seed_p.add_argument("--run-id", required=True,
                        help="the gather run_id whose screened papers to sample from")
    seed_p.add_argument("--size", type=int, default=100,
                        help="seed set size (sample_seed()'s own size parameter)")

    fit_p = calibrate_sub.add_parser("fit")
    fit_p.add_argument("--run-id", required=True,
                       help="the gather run_id whose screened papers to calibrate against")
    fit_p.add_argument("sheet", help="path to a hand-labeled seed.jsonl sheet")

    progress_p = calibrate_sub.add_parser("progress")
    progress_p.add_argument("sheet", help="path to a label sheet in progress")

    sub.add_parser(
        "mcp", help="alias for jarvis-mcp — every remaining argument is passed through "
                    "to it unchanged (handled before this parser sees them; see main())"
    )

    return parser


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None,
                        help="project name, resolved under $JARVIS_PROJECT_ROOT")
    parser.add_argument("--db", default=None,
                        help="explicit corpus db path, overrides --project")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)

    if argv and argv[0] == "mcp":
        # Delegates entirely to mcp_server.main, which parses its own --db/--with-models
        # and manages its own store lifecycle (build_context/serve) -- this command never
        # opens a store itself, unlike every other subcommand. Handled before argparse
        # ever sees the remaining args: argparse.REMAINDER on a subparser does not
        # reliably capture leading `--flag`-shaped tokens (confirmed directly -- it
        # raised "unrecognized arguments: --db" here), so mcp's own args must never
        # reach this module's parser at all.
        from jarvis.mcp_server import main as mcp_main
        return mcp_main(argv[1:])

    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = COMMANDS.get(args.command)
    if handler is None:
        print(f"error: unknown command {args.command!r}", file=sys.stderr)
        return 2

    try:
        conn = _open(args)
    except SystemExit as exc:
        return int(exc.code or 1)

    try:
        return handler(conn, args)
    except Exception as exc:  # noqa: BLE001 - a subcommand's failure is a named error
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        close_store(conn)


if __name__ == "__main__":
    raise SystemExit(main())
