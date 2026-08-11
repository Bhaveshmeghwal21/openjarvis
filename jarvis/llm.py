"""Shared OpenAI-compatible chat helper (lazy import, routed, cost-logged).

Imported lazily by callers so the package stays importable offline and tests run with fakes.
Ported from NanoResearch/jarvis/agents/llm.py.
"""
from __future__ import annotations

import json
import os


def chat(router, task: str, prompt: str, *, json_mode: bool = False,
         temperature: float = 0.0, run_id: str = "", system: str | None = None):
    """Route `task` to a model, call it, log measured usage, return text or parsed JSON."""
    from openai import OpenAI

    base_url = os.environ.get("JARVIS_BASE_URL")
    api_key = os.environ.get("JARVIS_API_KEY")
    model = router.route(task)
    client = OpenAI(base_url=base_url, api_key=api_key)

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
