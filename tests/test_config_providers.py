"""Config's provider resolution -- JARVIS_PROVIDER / JARVIS_PROVIDER_<TASK>, mirroring the
existing JARVIS_MODEL_<TASK> pattern one level up (which backend, not which model name)."""
from __future__ import annotations

from pathlib import Path

from jarvis.config import Config


def _cfg(**overrides) -> Config:
    base = {"project_root": Path("/tmp/does-not-matter")}
    base.update(overrides)
    return Config(**base)


def test_default_provider_is_openai():
    assert _cfg().provider_for("synthesis") == "openai"


def test_load_reads_jarvis_provider_env_var(monkeypatch):
    monkeypatch.setenv("JARVIS_PROVIDER", "azure")
    cfg = Config.load()
    assert cfg.provider == "azure"
    assert cfg.provider_for("synthesis") == "azure"


def test_load_reads_jarvis_provider_env_var_lowercased(monkeypatch):
    monkeypatch.setenv("JARVIS_PROVIDER", "AZURE")
    assert Config.load().provider == "azure"


def test_load_reads_per_task_provider_override(monkeypatch):
    monkeypatch.setenv("JARVIS_PROVIDER", "azure")
    monkeypatch.setenv("JARVIS_PROVIDER_SYNTHESIS", "gcp")
    cfg = Config.load()
    assert cfg.provider_for("synthesis") == "gcp"
    assert cfg.provider_for("screen_vote") == "azure"


def test_per_task_override_via_config_object_wins_over_default():
    cfg = _cfg(provider="azure", provider_overrides={"synthesis": "gcp"})
    assert cfg.provider_for("synthesis") == "gcp"
    assert cfg.provider_for("screen_vote") == "azure"


def test_load_reads_api_version_and_gcp_settings(monkeypatch):
    monkeypatch.setenv("JARVIS_API_VERSION", "2024-08-01-preview")
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.setenv("JARVIS_GCP_PROJECT", "my-project")
    monkeypatch.setenv("JARVIS_GCP_LOCATION", "europe-west4")
    cfg = Config.load()
    assert cfg.api_version == "2024-08-01-preview"
    assert cfg.gcp_credentials_path == "/secrets/sa.json"
    assert cfg.gcp_project == "my-project"
    assert cfg.gcp_location == "europe-west4"


def test_load_leaves_provider_settings_unset_when_no_env_present(monkeypatch):
    for var in ("JARVIS_PROVIDER", "JARVIS_API_VERSION", "JARVIS_GCP_CREDENTIALS",
               "JARVIS_GCP_PROJECT", "JARVIS_GCP_LOCATION"):
        monkeypatch.delenv(var, raising=False)
    cfg = Config.load()
    assert cfg.provider == "openai"
    assert cfg.api_version is None
    assert cfg.gcp_credentials_path is None
