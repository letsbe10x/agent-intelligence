"""Run a YAML-configured ReAct agent end-to-end.

Pluggable everything:
    - LLM provider (anthropic, openai, litellm, mock, ollama, custom)
    - Tool sources (builtin, skillhub, mcp, custom)
    - Runtime overrides for provider name, model name, tool sources

An agent's YAML is a recipe. The runtime is free to override any of it.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.config import (
    BudgetConfig,
    ObservabilityConfig,
    ProviderConfig,
)
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import ConfigError
from agent_intelligence.observability.budget import BudgetTracker
from agent_intelligence.observability.receipts import Receipt, ReceiptStore
from agent_intelligence.react.loop import ReActLoop, ReActOutcome
from agent_intelligence.registry.registry import registry
from agent_intelligence.tools.base import Tool
from agent_intelligence.tools.registry import tool_registry
from agent_intelligence.tools.sources.base import (
    ToolSourceConfig,
    discover_all,
    load_sources,
)
from agent_intelligence.tracing.events import TraceEmitter


class ReActAgentConfig(BaseModel):
    """YAML shape. ``tool_sources`` is the pluggable tool layer; ``tool_names``
    kept for back-compat (treated as a builtin allowlist when set)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1"
    prompt_path: str
    provider: ProviderConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Either:
    tool_names: list[str] = Field(default_factory=list)
    # ...or:
    tool_sources: list[ToolSourceConfig] = Field(default_factory=list)

    max_iterations: int = Field(default=12, ge=1, le=50)
    max_parse_retries: int = Field(default=2, ge=0, le=10)


class RuntimeOverrides(BaseModel):
    """What the caller can override at run time (typically from a UI form)."""

    model_config = ConfigDict(extra="forbid")

    provider_name: str | None = None
    provider_model: str | None = None
    provider_api_key: str | None = None
    provider_base_url: str | None = None
    provider_temperature: float | None = None
    provider_max_output_tokens: int | None = None
    extra_tool_sources: list[ToolSourceConfig] = Field(default_factory=list)
    max_iterations: int | None = None
    max_usd: float | None = None


def load_react_config(path: str | Path) -> ReActAgentConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    from agent_intelligence.core.config import _resolve_env_refs

    raw = _resolve_env_refs(raw)
    if "prompt_path" in raw and not Path(raw["prompt_path"]).is_absolute():
        raw["prompt_path"] = str((p.parent / raw["prompt_path"]).resolve())
    try:
        cfg = ReActAgentConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"Invalid ReAct agent config at {p}: {e}") from e
    if not Path(cfg.prompt_path).is_file():
        raise ConfigError(f"Prompt file does not exist: {cfg.prompt_path}")
    return cfg


def _build_provider(cfg: ReActAgentConfig, overrides: RuntimeOverrides):
    pc_data = cfg.provider.model_dump()
    if overrides.provider_name:
        pc_data["name"] = overrides.provider_name
    if overrides.provider_model:
        pc_data["model"] = overrides.provider_model
    if overrides.provider_api_key:
        pc_data["api_key"] = overrides.provider_api_key
    if overrides.provider_base_url:
        pc_data["base_url"] = overrides.provider_base_url
    if overrides.provider_temperature is not None:
        pc_data["temperature"] = overrides.provider_temperature
    if overrides.provider_max_output_tokens:
        pc_data["max_output_tokens"] = overrides.provider_max_output_tokens

    pc = ProviderConfig.model_validate(pc_data)
    ProviderCls = registry.get_provider(pc.name)
    return ProviderCls(pc)


async def _resolve_tools(cfg: ReActAgentConfig, overrides: RuntimeOverrides) -> dict[str, Tool]:
    # Three resolution modes, in order of precedence:
    # (1) overrides.extra_tool_sources extend whatever cfg specifies.
    # (2) cfg.tool_sources — the modern pluggable surface.
    # (3) cfg.tool_names — back-compat: treat as a builtin allowlist.
    import agent_intelligence.tools.builtins  # noqa: F401 — auto-register

    source_configs: list[ToolSourceConfig] = list(cfg.tool_sources or [])
    if cfg.tool_names and not cfg.tool_sources:
        source_configs.append(
            ToolSourceConfig(
                type="builtin",
                name="builtin",
                config={"allowlist": cfg.tool_names},
            )
        )
    source_configs.extend(overrides.extra_tool_sources)

    if not source_configs:
        # Fall back to builtin-all
        source_configs.append(ToolSourceConfig(type="builtin"))

    sources = load_sources(source_configs)
    return await discover_all(sources)


class ReActAgentResult(BaseModel):
    """End-to-end result with full step trace."""

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
    tool_catalog: list[dict[str, Any]] = Field(default_factory=list)


async def run_react_agent(
    config_path: str | Path,
    user_input: str | dict[str, Any],
    context: AgentContext | None = None,
    receipt_store: ReceiptStore | None = None,
    emitter: TraceEmitter | None = None,
    overrides: RuntimeOverrides | None = None,
) -> ReActAgentResult:
    cfg = load_react_config(config_path)
    overrides = overrides or RuntimeOverrides()

    provider = _build_provider(cfg, overrides)
    tools = await _resolve_tools(cfg, overrides)
    system_prompt = Path(cfg.prompt_path).read_text(encoding="utf-8")

    max_iter = overrides.max_iterations or cfg.max_iterations
    max_usd = overrides.max_usd if overrides.max_usd is not None else cfg.budget.max_usd_per_run

    budget = BudgetTracker(
        max_usd=max_usd,
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
        max_iterations=max_iter,
        max_parse_retries=cfg.max_parse_retries,
        emitter=emitter,
    )

    t0 = _time.perf_counter()
    outcome: ReActOutcome = await loop.run(user_input, ctx)
    wall = _time.perf_counter() - t0

    catalog = [
        {"name": t.name, "description": t.description, "tags": list(getattr(t, "tags", []) or [])}
        for t in tools.values()
    ]

    receipt = Receipt.build(
        agent_name=cfg.name,
        agent_version=cfg.version,
        config_snapshot={**cfg.model_dump(exclude={"provider": {"api_key"}}), "_overrides": overrides.model_dump(exclude={"provider_api_key"})},
        input_payload=user_input,
        output_payload={
            "final_output": outcome.final_output,
            "iterations": outcome.iterations,
            "halted_reason": outcome.halted_reason,
            "steps": [s.__dict__ for s in outcome.steps],
            "tool_catalog": catalog,
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
        tool_catalog=catalog,
    )
