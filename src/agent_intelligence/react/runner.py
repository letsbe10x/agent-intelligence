"""Run a YAML-configured ReAct agent end-to-end.

An agent in this framework is:
    - a YAML config with: llm provider, tool_names list, prompt path,
      max_iterations
    - a system prompt file
    - NO Python code

This runner loads the config + prompt + tools, constructs a ReActLoop, and runs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.config import (
    AgentConfig,
    BudgetConfig,
    ObservabilityConfig,
    ProviderConfig,
)
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import ConfigError, RegistryError
from agent_intelligence.observability.budget import BudgetTracker
from agent_intelligence.observability.receipts import Receipt, ReceiptStore
from agent_intelligence.react.loop import ReActLoop, ReActOutcome
from agent_intelligence.registry.registry import registry
from agent_intelligence.tools.base import Tool
from agent_intelligence.tools.registry import tool_registry
from agent_intelligence.tracing.events import TraceEmitter


class ReActAgentConfig(BaseModel):
    """YAML shape for a ReAct agent. Extends the base AgentConfig with tool list."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1"
    prompt_path: str
    provider: ProviderConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    tool_names: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=12, ge=1, le=50)
    max_parse_retries: int = Field(default=2, ge=0, le=10)


def load_react_config(path: str | Path) -> ReActAgentConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    # Resolve env-var refs using the same loader as the base AgentConfig
    from agent_intelligence.core.config import _resolve_env_refs
    raw = _resolve_env_refs(raw)

    # Resolve prompt_path relative to config file
    if "prompt_path" in raw and not Path(raw["prompt_path"]).is_absolute():
        raw["prompt_path"] = str((p.parent / raw["prompt_path"]).resolve())

    try:
        cfg = ReActAgentConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"Invalid ReAct agent config at {p}: {e}") from e

    if not Path(cfg.prompt_path).is_file():
        raise ConfigError(f"Prompt file does not exist: {cfg.prompt_path}")

    return cfg


def _build_provider(cfg: ReActAgentConfig):
    ProviderCls = registry.get_provider(cfg.provider.name)
    return ProviderCls(cfg.provider)


def _resolve_tools(cfg: ReActAgentConfig) -> dict[str, Tool]:
    # Ensure built-ins are imported so they register themselves
    import agent_intelligence.tools.builtins  # noqa: F401

    out: dict[str, Tool] = {}
    for name in cfg.tool_names:
        try:
            out[name] = tool_registry.get(name)
        except KeyError as e:
            raise RegistryError(str(e)) from e
    return out


class ReActAgentResult(BaseModel):
    """Result of running a YAML ReAct agent end-to-end."""

    model_config = ConfigDict(extra="allow")

    output: dict[str, Any]
    receipt_id: str
    receipt_hash: str
    iterations: int
    halted_reason: str
    cost_usd: float
    tokens_in: int
    tokens_out: int
    wallclock_s: float
    steps: list[dict[str, Any]]


async def run_react_agent(
    config_path: str | Path,
    user_input: str | dict[str, Any],
    context: AgentContext | None = None,
    receipt_store: ReceiptStore | None = None,
    emitter: TraceEmitter | None = None,
) -> ReActAgentResult:
    import time as _time

    cfg = load_react_config(config_path)
    provider = _build_provider(cfg)
    tools = _resolve_tools(cfg)
    system_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")

    budget = BudgetTracker(
        max_usd=cfg.budget.max_usd_per_run,
        max_input_tokens=cfg.budget.max_input_tokens_per_run,
        max_output_tokens=cfg.budget.max_output_tokens_per_run,
    )
    provider_with_budget = provider.with_budget(budget)
    ctx = context or AgentContext()
    emitter = emitter or TraceEmitter()

    loop = ReActLoop(
        provider=provider_with_budget,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=cfg.max_iterations,
        max_parse_retries=cfg.max_parse_retries,
        emitter=emitter,
    )

    t0 = _time.perf_counter()
    outcome: ReActOutcome = await loop.run(user_input, ctx)
    wall = _time.perf_counter() - t0

    receipt = Receipt.build(
        agent_name=cfg.name,
        agent_version=cfg.version,
        config_snapshot=cfg.model_dump(exclude={"provider": {"api_key"}}),
        input_payload=user_input,
        output_payload={
            "final_output": outcome.final_output,
            "iterations": outcome.iterations,
            "halted_reason": outcome.halted_reason,
            "steps": [s.__dict__ for s in outcome.steps],
        },
        context=ctx,
        wallclock_s=wall,
        cost_usd=budget.spent_usd,
        tokens_in=budget.spent_input_tokens,
        tokens_out=budget.spent_output_tokens,
        model_calls=budget.call_count,
        status="ok" if outcome.halted_reason == "final_answer" else "partial",
        error_message=None if outcome.halted_reason == "final_answer" else outcome.halted_reason,
    )
    (receipt_store or ReceiptStore(path=None)).put(receipt)

    return ReActAgentResult(
        output=outcome.final_output,
        receipt_id=receipt.receipt_id,
        receipt_hash=receipt.payload_hash,
        iterations=outcome.iterations,
        halted_reason=outcome.halted_reason,
        cost_usd=budget.spent_usd,
        tokens_in=budget.spent_input_tokens,
        tokens_out=budget.spent_output_tokens,
        wallclock_s=wall,
        steps=[s.__dict__ for s in outcome.steps],
    )
