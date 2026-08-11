"""Config — environment variables + optional JSON. Never reads `.env`.

Resolution order for model overrides (later wins):
  1. JSON file at $JARVIS_CONFIG  ->  {"models": {"<task>": "<model>"}}
  2. env vars  JARVIS_MODEL_<TASK>=<model>   (task lowercased)

Ported from NanoResearch/jarvis/config.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

_TASK_ENV_PREFIX = "JARVIS_MODEL_"
DEFAULT_PROJECT_ROOT = Path.home() / ".jarvis" / "projects"


@dataclass
class Config:
    project_root: Path
    model_overrides: dict[str, str] = field(default_factory=dict)
    base_url: str | None = None
    api_key: str | None = None
    unpaywall_email: str | None = None

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
        )

    def project_dir(self, name: str) -> Path:
        """Directory for one project's corpus. One SQLite file lives here (spec §6)."""
        return self.project_root / name
