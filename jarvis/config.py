"""Config — environment variables + optional JSON. Never reads `.env`.

Resolution order for model overrides (later wins):
  1. JSON file at $JARVIS_CONFIG  ->  {"models": {"<task>": "<model>"}}
  2. env vars  JARVIS_MODEL_<TASK>=<model>   (task lowercased)

Provider selection follows the identical pattern, one level up: which backend a task's
chat call goes through, independent of which model name is requested on that backend.
`JARVIS_PROVIDER` sets the default for every task; `JARVIS_PROVIDER_<TASK>` overrides one
task. Three providers: `openai` (any OpenAI-compatible endpoint — the public API, a proxy,
or a vendor like DeepSeek that speaks the same wire format, the default), `azure` (Azure
OpenAI, a genuinely different client shape — deployment-based routing, required
`api-version`, `AzureOpenAI` rather than `OpenAI`), `gcp` (Vertex AI's OpenAI-compatible
endpoint, authenticated via a service-account JSON key exchanged for a short-lived bearer
token rather than a static API key).

Ported from NanoResearch/jarvis/config.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_TASK_ENV_PREFIX = "JARVIS_MODEL_"
_PROVIDER_TASK_ENV_PREFIX = "JARVIS_PROVIDER_"
DEFAULT_PROJECT_ROOT = Path.home() / ".jarvis" / "projects"
PROVIDERS = ("openai", "azure", "gcp")


@dataclass
class Config:
    project_root: Path
    model_overrides: dict[str, str] = field(default_factory=dict)
    base_url: str | None = None
    api_key: str | None = None
    unpaywall_email: str | None = None
    provider: str = "openai"
    provider_overrides: dict[str, str] = field(default_factory=dict)
    api_version: str | None = None
    gcp_credentials_path: str | None = None
    gcp_project: str | None = None
    gcp_location: str | None = None

    @classmethod
    def load(cls) -> "Config":
        cfg: dict = {}
        cfg_path = os.environ.get("JARVIS_CONFIG")
        if cfg_path and Path(cfg_path).is_file():
            cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))

        overrides: dict[str, str] = {str(k).lower(): str(v) for k, v in cfg.get("models", {}).items()}
        for key, val in os.environ.items():
            if key.startswith(_TASK_ENV_PREFIX) and val:
                overrides[key[len(_TASK_ENV_PREFIX):].lower()] = val

        provider_overrides: dict[str, str] = {
            str(k).lower(): str(v) for k, v in cfg.get("provider_overrides", {}).items()
        }
        for key, val in os.environ.items():
            if key.startswith(_PROVIDER_TASK_ENV_PREFIX) and val:
                provider_overrides[key[len(_PROVIDER_TASK_ENV_PREFIX):].lower()] = val.lower()

        project_root = Path(
            cfg.get("project_root")
            or os.environ.get("JARVIS_PROJECT_ROOT")
            or DEFAULT_PROJECT_ROOT
        )
        return cls(
            project_root=project_root,
            model_overrides=overrides,
            base_url=cfg.get("base_url") or os.environ.get("JARVIS_BASE_URL"),
            api_key=cfg.get("api_key") or os.environ.get("JARVIS_API_KEY"),
            unpaywall_email=cfg.get("unpaywall_email") or os.environ.get("UNPAYWALL_EMAIL"),
            provider=str(cfg.get("provider") or os.environ.get("JARVIS_PROVIDER") or "openai").lower(),
            provider_overrides=provider_overrides,
            api_version=cfg.get("api_version") or os.environ.get("JARVIS_API_VERSION"),
            gcp_credentials_path=cfg.get("gcp_credentials_path")
            or os.environ.get("JARVIS_GCP_CREDENTIALS"),
            gcp_project=cfg.get("gcp_project") or os.environ.get("JARVIS_GCP_PROJECT"),
            gcp_location=cfg.get("gcp_location") or os.environ.get("JARVIS_GCP_LOCATION"),
        )

    def project_dir(self, name: str) -> Path:
        """Directory for one project's corpus. One SQLite file lives here (spec §6)."""
        return self.project_root / name

    def provider_for(self, task: str) -> str:
        """Which provider backs one task's chat calls. Per-task override wins."""
        return self.provider_overrides.get(task.lower(), self.provider)
