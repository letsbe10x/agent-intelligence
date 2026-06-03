"""MCP server: expose agents as tools over the Model Context Protocol.

The implementation here is the *binding layer*. The host application supplies:
    - a list of (agent_name, config_path) pairs to publish
    - an auth callback that resolves an incoming request to an AgentContext
    - a receipt store

We construct an MCP server with one tool per published agent. Each tool's
schema is auto-generated from the agent's InputModel.

This module lazy-imports ``mcp`` so the package can be installed without it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_intelligence.core.context import AgentContext
from agent_intelligence.observability.receipts import ReceiptStore
from agent_intelligence.runner import build_agent


def build_mcp_app(
    publish: list[tuple[str, str | Path]],
    auth_resolver: Callable[[dict[str, Any]], AgentContext],
    receipt_store: ReceiptStore | None = None,
    server_name: str = "agent-intelligence",
):
    """Construct an MCP server exposing the given agents as tools.

    Args:
        publish: List of (agent_name, config_path) pairs. Each becomes an MCP tool
                 named ``ai_<agent_name>``.
        auth_resolver: Callable that maps incoming request metadata to an
                       AgentContext. Host applications typically read a Bearer
                       token here and resolve org_id/user_id.
        receipt_store: Where to persist receipts. None = in-memory only.
        server_name: MCP server identifier.

    Returns:
        An MCP server instance, ready to run with ``server.run_stdio()`` or
        ``server.run_streamable_http(host, port)``.
    """
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install with: pip install 'agent-intelligence[mcp]'"
        ) from e

    app = FastMCP(name=server_name)

    # Build all agents up-front so config errors surface at boot, not at first call.
    agents = {}
    for agent_name, cfg_path in publish:
        agent = build_agent(cfg_path, receipt_store=receipt_store)
        agents[agent_name] = agent

    def _make_handler(agent_name: str):
        async def _handle(input_payload: dict[str, Any], request_metadata: dict[str, Any] | None = None):
            context = auth_resolver(request_metadata or {})
            agent = agents[agent_name]
            result = await agent.run(input_payload, context)
            return {
                "output": result.output.model_dump(),
                "receipt_id": result.receipt.receipt_id,
                "receipt_hash": result.receipt.payload_hash,
                "cost_usd": result.cost_usd,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
            }

        return _handle

    # Register each agent as a tool.
    for agent_name, agent in agents.items():
        tool_name = f"ai_{agent_name}"
        handler = _make_handler(agent_name)
        # The MCP SDK's exact API for dynamic tool registration varies by version.
        # We use FastMCP's decorator-equivalent: app.add_tool(...).
        app.add_tool(
            handler,
            name=tool_name,
            description=(
                f"Run the {agent_name} agent. "
                f"Input schema: {json.dumps(agent.InputModel.model_json_schema())}"
            ),
        )

    return app
