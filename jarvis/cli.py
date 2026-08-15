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
import sys
from pathlib import Path

from jarvis.config import Config
from jarvis.store import close_store, get_papers_by_depth, open_store

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
    result that looks like "no evidence" rather than "misconfigured"."""
    if not config.base_url:
        raise ModelBuildError(
            "JARVIS_BASE_URL is not set — required to construct a real model client"
        )
    if not config.api_key:
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
    """
    if db:
        return Path(db)
    if project:
        return Config.load().project_dir(project) / "corpus.db"
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


COMMANDS = {
    "status": cmd_status,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("--project", default=None,
                       help="project name, resolved under $JARVIS_PROJECT_ROOT")
        p.add_argument("--db", default=None,
                       help="explicit corpus db path, overrides --project")

    return parser


def main(argv: list[str] | None = None) -> int:
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
