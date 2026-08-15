"""Shared OpenAI-compatible chat helper (lazy import, routed, cost-logged).

Imported lazily by callers so the package stays importable offline and tests run with fakes.
Ported from NanoResearch/jarvis/agents/llm.py.

Three providers, one call shape (`client.chat.completions.create(...)`) for all of them:

  * `openai`  — any endpoint that speaks the OpenAI wire format as-is: the public API, a
    proxy, or a vendor like DeepSeek. `JARVIS_BASE_URL` + `JARVIS_API_KEY`. The default.
  * `azure`   — Azure OpenAI. A genuinely different client (`AzureOpenAI`, not `OpenAI`):
    deployment-based routing, a required `api-version`, `api-key` auth rather than Bearer.
    `JARVIS_BASE_URL` (the Azure endpoint), `JARVIS_API_KEY`, `JARVIS_API_VERSION`. `model`
    must be the deployment name, not a model family name.
  * `gcp`     — Vertex AI's OpenAI-compatible endpoint. Auth is a service-account JSON key
    exchanged for a short-lived bearer token, not a static string — there is no long-lived
    "GCP API key" for this path. `JARVIS_GCP_CREDENTIALS` (path to the key file) is the
    only required setting; `JARVIS_GCP_PROJECT` falls back to the project id embedded in
    the key file itself when unset, and `JARVIS_GCP_LOCATION` defaults to `global` — no
    region needs to be known ahead of time, and Vertex's `global` location genuinely uses
    a different, unprefixed host than every other region. Requires the `gcp` extra
    (`google-auth`). **The model name must carry a publisher prefix — `google/gemini-3.7-flash`,
    not `gemini-3.7-flash`.** The bare form fails with a 400 naming it a "malformed
    publisher model"; this is easy to miss since it's the one provider where the model
    string alone isn't enough. Confirmed end to end against a real project: real
    service-account auth, the `global` endpoint, and a real `google/gemini-3.7-flash`
    reply — not just design-time reasoning, unlike the other two providers' live paths.
"""
from __future__ import annotations

import json
import os


def _client_for(provider: str):
    """Build the right client for one provider. All three end up as an object exposing
    `.chat.completions.create(...)` — Vertex's OpenAI-compatible surface makes this
    possible for `gcp` too, so nothing downstream of this function needs to know which
    provider it's talking to."""
    if provider == "azure":
        return _azure_client()
    if provider == "gcp":
        return _gcp_client()
    from openai import OpenAI
    return OpenAI(base_url=os.environ.get("JARVIS_BASE_URL"),
                  api_key=os.environ.get("JARVIS_API_KEY"))


def _azure_client():
    from openai import AzureOpenAI
    return AzureOpenAI(
        azure_endpoint=os.environ.get("JARVIS_BASE_URL"),
        api_key=os.environ.get("JARVIS_API_KEY"),
        api_version=os.environ.get("JARVIS_API_VERSION"),
    )


def _gcp_client():
    from openai import OpenAI
    credentials = _gcp_credentials()
    project = os.environ.get("JARVIS_GCP_PROJECT") or credentials.project_id
    return OpenAI(base_url=_gcp_endpoint(project), api_key=_gcp_access_token(credentials))


def _gcp_credentials():
    from google.oauth2 import service_account
    return service_account.Credentials.from_service_account_file(
        os.environ.get("JARVIS_GCP_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )


def _gcp_access_token(credentials) -> str:
    """Exchange the service-account JSON key for a short-lived bearer token. Fetched fresh
    on every call rather than cached — simpler than tracking expiry, and cheap relative to
    an LLM call."""
    from google.auth.transport.requests import Request
    credentials.refresh(Request())
    return credentials.token


def _gcp_endpoint(project: str) -> str:
    """`global` is the default deliberately: it needs no region to be known ahead of time
    and uses a plain host with no region prefix (`aiplatform.googleapis.com`, not
    `global-aiplatform.googleapis.com` — a real, easy-to-get-wrong exception to every
    other location's `{location}-aiplatform.googleapis.com` shape). A pinned region is
    still honored when set, for anyone who wants regional latency or data residency."""
    location = os.environ.get("JARVIS_GCP_LOCATION") or "global"
    host = "aiplatform.googleapis.com" if location == "global" \
        else f"{location}-aiplatform.googleapis.com"
    return (f"https://{host}/v1/projects/{project}"
            f"/locations/{location}/endpoints/openapi")


def chat(router, task: str, prompt: str, *, json_mode: bool = False,
         temperature: float = 0.0, run_id: str = "", system: str | None = None):
    """Route `task` to a model and a provider, call it, log measured usage, return text or
    parsed JSON."""
    model = router.route(task)
    provider = router.provider_for(task)
    client = _client_for(provider)

    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=model, temperature=temperature, messages=messages, **kwargs
    )
    u = resp.usage
    router.log_usage(task, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0),
                     run_id=run_id, model=model)
    content = resp.choices[0].message.content
    return json.loads(content) if json_mode else content
