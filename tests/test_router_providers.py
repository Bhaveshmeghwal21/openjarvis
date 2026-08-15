"""ModelRouter's provider dimension -- which backend a task's chat call goes through,
resolved independently of which model name it asks for."""
from __future__ import annotations

from jarvis.router import ModelRouter


def test_default_provider_is_openai_when_unset():
    assert ModelRouter().provider_for("synthesis") == "openai"


def test_explicit_default_provider_applies_to_every_task():
    r = ModelRouter(provider="azure")
    assert r.provider_for("synthesis") == "azure"
    assert r.provider_for("screen_vote") == "azure"


def test_per_task_override_wins_over_the_default_provider():
    r = ModelRouter(provider="azure", provider_overrides={"synthesis": "gcp"})
    assert r.provider_for("synthesis") == "gcp"
    assert r.provider_for("screen_vote") == "azure"


def test_provider_names_are_case_insensitive():
    r = ModelRouter(provider="Azure", provider_overrides={"Synthesis": "GCP"})
    assert r.provider_for("SYNTHESIS") == "gcp"
    assert r.provider_for("screen_vote") == "azure"


def test_provider_routing_is_independent_of_model_routing():
    """A task can move to a different vendor without its model name changing, and vice
    versa -- the two dimensions never interact."""
    r = ModelRouter(overrides={"synthesis": "my-local-model"}, provider="azure")
    assert r.route("synthesis") == "my-local-model"
    assert r.provider_for("synthesis") == "azure"
