"""jarvis.llm — provider dispatch (openai / azure / gcp), all offline.

`openai` and `google-auth` are optional extras; nothing here may require them actually
installed. Every provider path is exercised against a fake module injected into
`sys.modules`, matching this module's own contract that heavy dependencies are imported
lazily inside the function that needs them.
"""
from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from jarvis.llm import chat


class _RecordingCompletions:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _RecordingChat:
    def __init__(self, response):
        self.completions = _RecordingCompletions(response)


class _RecordingClient:
    """Shared by the fake `OpenAI` and `AzureOpenAI` classes below -- records every
    constructor kwarg so a test can assert on exactly what each provider built."""

    instances: ClassVar[list[_RecordingClient]] = []

    def __init__(self, response, **kwargs):
        self.kwargs = kwargs
        self.chat = _RecordingChat(response)
        _RecordingClient.instances.append(self)


def _canned_response(content='{"ok": true}', prompt_tokens=10, completion_tokens=5):
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens)
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def fake_openai(monkeypatch):
    """Injects a fake `openai` module exposing recording `OpenAI`/`AzureOpenAI` classes."""
    _RecordingClient.instances = []
    response = _canned_response()

    def make(**kwargs):
        return _RecordingClient(response, **kwargs)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = make
    fake_module.AzureOpenAI = make
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return response


@pytest.fixture
def fake_google_auth(monkeypatch):
    """Injects a fake `google.auth`/`google.oauth2` chain. The whole dotted path has to be
    present in `sys.modules`, not just the leaves -- `from a.b.c import D` still resolves
    parent packages, so a partially-faked chain raises `ModuleNotFoundError` on `google.b`
    even when the leaf module is cached."""
    calls = {"from_file": [], "refreshed": False}

    class FakeRequest:
        pass

    class FakeCredentials:
        token = "fake-bearer-token"
        project_id = "project-from-credentials-file"

        @classmethod
        def from_service_account_file(cls, path, scopes=None):
            calls["from_file"].append((path, scopes))
            return cls()

        def refresh(self, request):
            calls["refreshed"] = True

    for name in ("google", "google.auth", "google.auth.transport",
                "google.auth.transport.requests", "google.oauth2",
                "google.oauth2.service_account"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    for name in ("google", "google.auth", "google.auth.transport", "google.oauth2"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)

    requests_mod = types.ModuleType("google.auth.transport.requests")
    requests_mod.Request = FakeRequest
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_mod)

    sa_mod = types.ModuleType("google.oauth2.service_account")
    sa_mod.Credentials = FakeCredentials
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", sa_mod)

    return calls


class _Router:
    def __init__(self, provider="openai"):
        self._provider = provider
        self.logged = []

    def route(self, task: str) -> str:
        return "fake-model"

    def provider_for(self, task: str) -> str:
        return self._provider

    def log_usage(self, task, input_tokens, output_tokens, run_id="", model=None):
        self.logged.append((task, input_tokens, output_tokens, run_id, model))


def test_openai_provider_builds_a_plain_client_from_base_url_and_key(
    fake_openai, monkeypatch,
):
    monkeypatch.setenv("JARVIS_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("JARVIS_API_KEY", "sk-test")
    chat(_Router("openai"), "synthesis", "hello")

    assert len(_RecordingClient.instances) == 1
    built = _RecordingClient.instances[0]
    assert built.kwargs == {"base_url": "https://api.deepseek.com/v1", "api_key": "sk-test"}


def test_azure_provider_builds_azureopenai_with_endpoint_key_and_api_version(
    fake_openai, monkeypatch,
):
    monkeypatch.setenv("JARVIS_BASE_URL", "https://my-resource.openai.azure.com/")
    monkeypatch.setenv("JARVIS_API_KEY", "azure-key")
    monkeypatch.setenv("JARVIS_API_VERSION", "2024-08-01-preview")
    chat(_Router("azure"), "synthesis", "hello")

    built = _RecordingClient.instances[0]
    assert built.kwargs == {
        "azure_endpoint": "https://my-resource.openai.azure.com/",
        "api_key": "azure-key",
        "api_version": "2024-08-01-preview",
    }


def test_gcp_provider_exchanges_the_service_account_file_for_a_bearer_token(
    fake_openai, fake_google_auth, monkeypatch,
):
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.setenv("JARVIS_GCP_PROJECT", "my-project")
    monkeypatch.setenv("JARVIS_GCP_LOCATION", "europe-west4")
    chat(_Router("gcp"), "synthesis", "hello")

    assert fake_google_auth["from_file"] == [
        ("/secrets/sa.json", ["https://www.googleapis.com/auth/cloud-platform"])
    ]
    assert fake_google_auth["refreshed"] is True

    built = _RecordingClient.instances[0]
    assert built.kwargs["api_key"] == "fake-bearer-token"
    assert built.kwargs["base_url"] == (
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/my-project"
        "/locations/europe-west4/endpoints/openapi"
    )


def test_gcp_endpoint_defaults_location_to_global_when_unset(
    fake_openai, fake_google_auth, monkeypatch,
):
    # `global` is the default deliberately -- no region needs to be known ahead of time.
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.setenv("JARVIS_GCP_PROJECT", "my-project")
    monkeypatch.delenv("JARVIS_GCP_LOCATION", raising=False)
    chat(_Router("gcp"), "synthesis", "hello")

    built = _RecordingClient.instances[0]
    assert built.kwargs["base_url"] == (
        "https://aiplatform.googleapis.com/v1/projects/my-project"
        "/locations/global/endpoints/openapi"
    )


def test_gcp_global_location_uses_the_unprefixed_host_not_global_dash_aiplatform(
    fake_openai, fake_google_auth, monkeypatch,
):
    # `global` is a real, documented exception to every other location's
    # `{location}-aiplatform.googleapis.com` shape -- there is no `global-aiplatform...`
    # host. Asserted as its own test since getting this wrong silently produces a DNS
    # failure that looks nothing like a credentials or model problem.
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.setenv("JARVIS_GCP_PROJECT", "my-project")
    monkeypatch.setenv("JARVIS_GCP_LOCATION", "global")
    chat(_Router("gcp"), "synthesis", "hello")

    built = _RecordingClient.instances[0]
    assert "global-aiplatform" not in built.kwargs["base_url"]
    assert built.kwargs["base_url"].startswith("https://aiplatform.googleapis.com/")


def test_gcp_project_falls_back_to_the_credentials_files_own_project_id_when_unset(
    fake_openai, fake_google_auth, monkeypatch,
):
    # A service-account key already carries its project id -- requiring the operator to
    # also set JARVIS_GCP_PROJECT is redundant friction when it isn't set explicitly.
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.delenv("JARVIS_GCP_PROJECT", raising=False)
    chat(_Router("gcp"), "synthesis", "hello")

    built = _RecordingClient.instances[0]
    assert "/projects/project-from-credentials-file/" in built.kwargs["base_url"]


def test_gcp_project_env_var_wins_over_the_credentials_files_project_id(
    fake_openai, fake_google_auth, monkeypatch,
):
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.setenv("JARVIS_GCP_PROJECT", "explicit-project")
    chat(_Router("gcp"), "synthesis", "hello")

    built = _RecordingClient.instances[0]
    assert "/projects/explicit-project/" in built.kwargs["base_url"]


def test_chat_fetches_a_fresh_token_on_every_call_rather_than_caching(
    fake_openai, fake_google_auth, monkeypatch,
):
    monkeypatch.setenv("JARVIS_GCP_CREDENTIALS", "/secrets/sa.json")
    monkeypatch.setenv("JARVIS_GCP_PROJECT", "my-project")
    router = _Router("gcp")
    chat(router, "synthesis", "one")
    chat(router, "synthesis", "two")

    assert len(fake_google_auth["from_file"]) == 2


def test_chat_sends_the_routed_model_and_the_prompt_as_a_user_message(
    fake_openai, monkeypatch,
):
    monkeypatch.setenv("JARVIS_BASE_URL", "http://x")
    monkeypatch.setenv("JARVIS_API_KEY", "k")
    chat(_Router("openai"), "synthesis", "what is the answer?")

    call = _RecordingClient.instances[0].chat.completions.calls[0]
    assert call["model"] == "fake-model"
    assert call["messages"] == [{"role": "user", "content": "what is the answer?"}]
    assert "response_format" not in call


def test_chat_prepends_a_system_message_when_given_one(fake_openai, monkeypatch):
    monkeypatch.setenv("JARVIS_BASE_URL", "http://x")
    monkeypatch.setenv("JARVIS_API_KEY", "k")
    chat(_Router("openai"), "synthesis", "q", system="be terse")

    call = _RecordingClient.instances[0].chat.completions.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "be terse"}


def test_json_mode_requests_a_json_object_and_parses_the_reply(fake_openai, monkeypatch):
    monkeypatch.setenv("JARVIS_BASE_URL", "http://x")
    monkeypatch.setenv("JARVIS_API_KEY", "k")
    result = chat(_Router("openai"), "synthesis", "q", json_mode=True)

    call = _RecordingClient.instances[0].chat.completions.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert result == {"ok": True}


def test_non_json_mode_returns_the_raw_text(fake_openai, monkeypatch):
    monkeypatch.setenv("JARVIS_BASE_URL", "http://x")
    monkeypatch.setenv("JARVIS_API_KEY", "k")
    result = chat(_Router("openai"), "synthesis", "q")
    assert result == '{"ok": true}'


def test_measured_usage_is_logged_against_the_router_not_estimated(
    fake_openai, monkeypatch,
):
    monkeypatch.setenv("JARVIS_BASE_URL", "http://x")
    monkeypatch.setenv("JARVIS_API_KEY", "k")
    router = _Router("openai")
    chat(router, "synthesis", "q", run_id="run-1")

    assert router.logged == [("synthesis", 10, 5, "run-1", "fake-model")]
