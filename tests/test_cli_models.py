"""Task 2: model construction and the fail-loud contract.

A subcommand that needs a model it cannot construct must fail immediately with the name
of the missing variable or extra -- never proceed with a `Fake*`. A corpus built with a
fake model looks real and is not, which is the exact failure class this whole system
exists to prevent (design spec §7).
"""
from __future__ import annotations

import pytest

from jarvis.cli import (
    ModelBuildError,
    build_card_extractor,
    build_embedder,
    build_nli,
    build_parser,
    build_planner,
    build_router,
    build_voter,
    build_writer,
)
from jarvis.config import Config


def _cfg(**overrides) -> Config:
    base = {"project_root": "/tmp/does-not-matter", "model_overrides": {}, "base_url": None,
            "api_key": None, "unpaywall_email": None}
    base.update(overrides)
    return Config(**base)


def test_build_router_is_always_constructible_no_model_needed():
    router = build_router(_cfg())
    assert router is not None


def test_build_writer_fails_naming_the_missing_base_url():
    with pytest.raises(ModelBuildError, match="JARVIS_BASE_URL"):
        build_writer(_cfg(api_key="x"), build_router(_cfg()))


def test_build_writer_fails_naming_the_missing_api_key():
    with pytest.raises(ModelBuildError, match="JARVIS_API_KEY"):
        build_writer(_cfg(base_url="http://x"), build_router(_cfg()))


def test_build_writer_rejects_whitespace_only_base_url():
    # A whitespace-only value is truthy in Python and would otherwise pass a bare
    # `if not config.base_url` check, reach openai.OpenAI(...), and fail there inside
    # an LLM* class's own broad except -- producing the exact "looks like no evidence"
    # silent outcome this fail-loud contract exists to prevent. Found by a final
    # whole-branch adversarial review; reproduced independently before fixing.
    with pytest.raises(ModelBuildError, match="JARVIS_BASE_URL"):
        build_writer(_cfg(base_url="   ", api_key="k"), build_router(_cfg()))


def test_build_writer_rejects_whitespace_only_api_key():
    with pytest.raises(ModelBuildError, match="JARVIS_API_KEY"):
        build_writer(_cfg(base_url="http://x", api_key="   "), build_router(_cfg()))


def test_build_writer_succeeds_when_both_present():
    writer = build_writer(_cfg(base_url="http://x", api_key="k"), build_router(_cfg()))
    assert writer is not None


def test_build_planner_fails_naming_missing_credentials():
    with pytest.raises(ModelBuildError, match="JARVIS_BASE_URL"):
        build_planner(_cfg(), build_router(_cfg()))


def test_build_voter_fails_naming_missing_credentials():
    with pytest.raises(ModelBuildError, match="JARVIS_BASE_URL"):
        build_voter(_cfg(), build_router(_cfg()))


def test_build_card_extractor_fails_naming_missing_credentials():
    with pytest.raises(ModelBuildError, match="JARVIS_BASE_URL"):
        build_card_extractor(_cfg(), build_router(_cfg()))


def test_build_embedder_fails_naming_the_missing_extra_when_sentence_transformers_absent(
    monkeypatch,
):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ModelBuildError, match="index"):
        build_embedder(_cfg())


def test_build_nli_fails_naming_the_missing_extra_when_transformers_absent(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ModelBuildError, match="verify"):
        build_nli(_cfg())


def test_build_parser_fails_naming_the_missing_extra_when_docling_absent(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "docling" or name.startswith("docling."):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ModelBuildError, match="parse"):
        build_parser(_cfg())


def test_no_missing_model_path_ever_returns_a_fake_instead_of_raising():
    # A ModelBuildError is the only acceptable outcome of a missing dependency -- never a
    # silently-substituted Fake*. This is the contract's single most important guarantee,
    # so assert it directly rather than only by absence of a Fake* in the return type.
    with pytest.raises(ModelBuildError):
        build_writer(_cfg(), build_router(_cfg()))
    with pytest.raises(ModelBuildError):
        build_planner(_cfg(), build_router(_cfg()))
    with pytest.raises(ModelBuildError):
        build_voter(_cfg(), build_router(_cfg()))
    with pytest.raises(ModelBuildError):
        build_card_extractor(_cfg(), build_router(_cfg()))
