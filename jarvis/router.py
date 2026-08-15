"""Task -> model-tier routing + cost tracking (spec §7, §10).

Routing is by *tier*, not vendor, so models can be swapped without touching call sites.
Every model call logs measured tokens — spec §10 requires cost figures to come from logs,
never from estimates typed into a document.

Ported from NanoResearch/jarvis/orchestrator/model_router.py, retargeted at corpus tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Tier -> default model. Override per-task via config (see config.py).
TIERS: dict[str, str] = {
    "cheap_structured": "gemini-flash-lite",
    "long_context_reader": "gemini-flash",
    "code_reasoning": "kimi-k2",
    "frontier_judgment": "gpt-4.1",
}

# Corpus task -> tier. Verification is deliberately absent: spec §8 puts it on a local NLI
# model, not an LLM (LLM-as-judge correlates 0.101 with human judgement vs 0.638 for NLI).
DEFAULT_ROUTING: dict[str, str] = {
    "screen_vote": "cheap_structured",       # gate signal, one of several (spec §7B)
    "query_expansion": "cheap_structured",
    "contextual_prefix": "cheap_structured",  # per-chunk 50-100 token locator
    "card_extraction": "long_context_reader",
    "unit_typing": "cheap_structured",
    "figure_describe": "long_context_reader",
    "retrieval_refine": "cheap_structured",
    "outline": "frontier_judgment",
    "synthesis": "frontier_judgment",
    "contradiction_review": "frontier_judgment",
}
DEFAULT_TIER = "cheap_structured"

# Approximate USD per 1M tokens (input, output). ESTIMATES for pre-run budgeting only;
# reported costs must come from logged usage.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-flash-lite": (0.10, 0.40),
    "gemini-flash": (0.15, 0.60),
    "kimi-k2": (0.60, 2.50),
    "gpt-4.1": (2.00, 8.00),
}
_DEFAULT_PRICE = (1.00, 3.00)


def _price_for(model: str) -> tuple[float, float]:
    """Match an arbitrary model name to a price family (handles config overrides)."""
    m = model.lower()
    if "lite" in m:
        return PRICES["gemini-flash-lite"]
    if "flash" in m or "gemini" in m:
        return PRICES["gemini-flash"]
    if "kimi" in m or "deepseek" in m:
        return PRICES["kimi-k2"]
    if "gpt-4" in m or "gpt4" in m or "gpt-5" in m or "opus" in m or "sonnet" in m:
        return PRICES["gpt-4.1"]
    return _DEFAULT_PRICE


@dataclass
class CostEntry:
    task: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    run_id: str = ""


@dataclass
class CostTracker:
    entries: list[CostEntry] = field(default_factory=list)

    @staticmethod
    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pin, pout = _price_for(model)
        return (input_tokens / 1_000_000) * pin + (output_tokens / 1_000_000) * pout

    def log(self, task: str, model: str, input_tokens: int,
            output_tokens: int, run_id: str = "") -> float:
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        self.entries.append(CostEntry(task, model, input_tokens, output_tokens, cost, run_id))
        return cost

    @property
    def total_cost(self) -> float:
        return round(sum(e.estimated_cost for e in self.entries), 6)

    def summary(self) -> dict:
        by_task: dict[str, dict] = {}
        for e in self.entries:
            t = by_task.setdefault(
                e.task, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            )
            t["calls"] += 1
            t["input_tokens"] += e.input_tokens
            t["output_tokens"] += e.output_tokens
            t["cost"] = round(t["cost"] + e.estimated_cost, 6)
        return {"total_cost": self.total_cost, "calls": len(self.entries), "by_task": by_task}


class ModelRouter:
    """Resolves a corpus task to a concrete model and records measured usage.

    Model and provider are resolved independently: `route(task)` says which model name to
    ask for, `provider_for(task)` says which backend to ask it of. A task can move to a
    different vendor without touching its model name, and vice versa — the same
    override-wins-over-default shape as model routing, one level up.
    """

    def __init__(self, overrides: dict[str, str] | None = None,
                 routing: dict[str, str] | None = None,
                 tiers: dict[str, str] | None = None,
                 provider: str = "openai",
                 provider_overrides: dict[str, str] | None = None) -> None:
        self.routing = dict(routing or DEFAULT_ROUTING)
        self.tiers = dict(tiers or TIERS)
        self.overrides = {k.lower(): v for k, v in (overrides or {}).items()}
        self.provider = provider.lower()
        self.provider_overrides = {k.lower(): v.lower()
                                   for k, v in (provider_overrides or {}).items()}

        self.cost = CostTracker()

    def tier_for(self, task: str) -> str:
        return self.routing.get(task.lower(), DEFAULT_TIER)

    def route(self, task: str) -> str:
        key = task.lower()
        if key in self.overrides:
            return self.overrides[key]
        return self.tiers.get(self.tier_for(key), self.tiers[DEFAULT_TIER])

    def provider_for(self, task: str) -> str:
        return self.provider_overrides.get(task.lower(), self.provider)

    def log_usage(self, task: str, input_tokens: int, output_tokens: int,
                  run_id: str = "", model: str | None = None) -> float:
        return self.cost.log(task, model or self.route(task), input_tokens, output_tokens, run_id)
