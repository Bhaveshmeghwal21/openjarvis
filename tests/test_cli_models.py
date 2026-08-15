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


# --- provider-aware credential checks -----------------------------------------------
# A task routed to `azure` or `gcp` needs different environment variables than the
# `openai` default -- the fail-loud check has to know which provider a task actually
# uses, not assume every task needs JARVIS_BASE_URL/JARVIS_API_KEY.

def test_azure_provider_requires_api_version_in_addition_to_base_url_and_key():
    cfg = _cfg(base_url="https://x.openai.azure.com/", api_key="k", provider="azure")
    with pytest.raises(ModelBuildError, match="JARVIS_API_VERSION"):
        build_writer(cfg, build_router(cfg))


def test_azure_provider_still_requires_base_url_and_key():
    cfg = _cfg(api_version="2024-08-01-preview", provider="azure")
    with pytest.raises(ModelBuildError, match="JARVIS_BASE_URL"):
        build_writer(cfg, build_router(cfg))


def test_azure_provider_succeeds_with_all_three_present():
    cfg = _cfg(base_url="https://x.openai.azure.com/", api_key="k",
              api_version="2024-08-01-preview", provider="azure")
    assert build_writer(cfg, build_router(cfg)) is not None


def test_gcp_provider_requires_credentials_path_not_an_api_key():
    cfg = _cfg(gcp_project="my-project", provider="gcp")
    with pytest.raises(ModelBuildError, match="JARVIS_GCP_CREDENTIALS"):
        build_writer(cfg, build_router(cfg))


def test_gcp_provider_does_not_require_a_project_set_explicitly():
    # jarvis.llm._gcp_client falls back to the project id embedded in the service-account
    # file itself when JARVIS_GCP_PROJECT is unset -- this check must not require it, or
    # it would block a configuration that actually works at runtime.
    cfg = _cfg(gcp_credentials_path="/secrets/sa.json", provider="gcp")
    assert build_writer(cfg, build_router(cfg)) is not None


def test_gcp_provider_does_not_require_base_url_or_api_key():
    # GCP auth is a service-account file exchanged for a token, not a static API key --
    # JARVIS_BASE_URL/JARVIS_API_KEY being unset must not block this provider.
    cfg = _cfg(gcp_credentials_path="/secrets/sa.json", gcp_project="my-project",
              provider="gcp")
    assert build_writer(cfg, build_router(cfg)) is not None


def test_a_task_specific_provider_override_is_checked_against_that_provider():
    # `build_writer` always checks the "synthesis" task -- overriding just that task to
    # gcp must apply gcp's requirements even though the config's default provider is
    # still openai.
    cfg = _cfg(provider="openai", provider_overrides={"synthesis": "gcp"},
              gcp_credentials_path="/secrets/sa.json", gcp_project="my-project")
    assert build_writer(cfg, build_router(cfg)) is not None


def test_build_router_carries_provider_settings_from_config():
    cfg = _cfg(provider="azure", provider_overrides={"synthesis": "gcp"})
    router = build_router(cfg)
    assert router.provider_for("outline") == "azure"
    assert router.provider_for("synthesis") == "gcp"
