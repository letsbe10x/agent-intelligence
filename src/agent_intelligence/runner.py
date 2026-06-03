"""High-level convenience: load config + construct agent + run, in one call.

Used by the CLI and by control-plane endpoints that don't need to manage
the lifecycle manually.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_intelligence.core.agent import Agent, AgentResult
from agent_intelligence.core.config import AgentConfig, load_config
from agent_intelligence.core.context import AgentContext
from agent_intelligence.observability.receipts import ReceiptStore
from agent_intelligence.registry.registry import registry


def _construct(config: AgentConfig, receipt_store: ReceiptStore | None) -> Agent:
    """Build a configured agent + provider from a config object."""
    AgentCls = registry.get_agent(config.name)
    ProviderCls = registry.get_provider(config.provider.name)
    provider = ProviderCls(config.provider)
    return AgentCls(
        config=config,
        provider=provider,
        receipt_store=receipt_store,
    )


async def run_agent(
    config_path: str | Path,
    input_: dict[str, Any],
    context: AgentContext | None = None,
    receipt_store: ReceiptStore | None = None,
) -> AgentResult:
    """Load config, build agent, run it. The 1-line invocation path."""
    cfg = load_config(config_path)
    agent = _construct(cfg, receipt_store=receipt_store)
    ctx = context or AgentContext()
    return await agent.run(input_, ctx)


def build_agent(
    config_path: str | Path,
    receipt_store: ReceiptStore | None = None,
) -> Agent:
    """Load config + build agent, but don't run yet. Used when callers want to
    call ``agent.run()`` multiple times with different inputs."""
    cfg = load_config(config_path)
    return _construct(cfg, receipt_store=receipt_store)
