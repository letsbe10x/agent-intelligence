"""MCPToolSource — discover tools from any Model Context Protocol server.

Supports HTTP and stdio MCP servers. Each server's tools become Tool instances
with input schemas derived from the server's tool list.

Config:
    servers: list of MCP server configs, each:
        - name        : friendly label
        - transport   : http | stdio
        - url         : HTTP base URL (when transport=http)
        - command     : process to spawn (when transport=stdio)
        - args        : argv list  (when transport=stdio)
        - env         : optional env vars
        - allowlist   : optional list of tool names to expose
        - prefix      : tool-name prefix (default: mcp.<server.name>.)

If the ``mcp`` python package isn't installed, this source returns an empty list
and logs a warning. Install with: pip install 'agent-intelligence[mcp]'.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from agent_intelligence.core.context import AgentContext
from agent_intelligence.tools.base import Tool
from agent_intelligence.tools.sources.base import ToolSource

logger = logging.getLogger(__name__)


class _MCPToolGeneric(Tool):
    """A Tool that forwards calls to an MCP server."""

    def __init__(
        self,
        *,
        tool_name: str,
        description: str,
        input_schema: dict[str, Any],
        invoker,  # async fn(name, args) -> dict
        server_label: str,
    ) -> None:
        self.name = tool_name
        self.description = description or f"MCP tool from {server_label!r}"
        self._invoker = invoker
        self._server_label = server_label
        self.tags = ["mcp", f"mcp:{server_label}"]

        # Build a dynamic pydantic Input model from the MCP JSON schema.
        # We only honour top-level properties; advanced JSON Schema (refs,
        # oneOf, allOf) round-trips as `Any`. The LLM still sees the raw
        # JSON Schema via to_llm_signature().
        props = input_schema.get("properties") or {}
        required = set(input_schema.get("required") or [])
        fields: dict[str, Any] = {}
        for k, v in props.items():
            field_type: Any
            t = v.get("type")
            if t == "string":
                field_type = str
            elif t == "integer":
                field_type = int
            elif t == "number":
                field_type = float
            elif t == "boolean":
                field_type = bool
            elif t == "array":
                field_type = list
            else:
                field_type = Any
            default = ... if k in required else None
            fields[k] = (field_type, Field(default=default, description=v.get("description", "")))

        # Always allow extras — MCP servers vary
        Cfg = ConfigDict(extra="allow")
        self.InputModel = create_model(  # type: ignore[call-overload]
            f"MCPInput_{tool_name.replace('.', '_')}",
            __config__=Cfg,
            **fields,
        )

        class _MCPOutput(BaseModel):
            model_config = ConfigDict(extra="allow")
            result: dict[str, Any] = Field(default_factory=dict)
            content: list[dict[str, Any]] = Field(default_factory=list)
            error: str | None = None

        self.OutputModel = _MCPOutput

        # Stash raw schema for to_llm_signature
        self._raw_schema = input_schema

    def json_schema(self) -> dict[str, Any]:
        return self._raw_schema

    async def run(self, input_: BaseModel, context: AgentContext) -> BaseModel:  # noqa: ARG002
        args = input_.model_dump(exclude_none=True)
        try:
            result = await self._invoker(self.name, args)
        except Exception as e:  # noqa: BLE001
            return self.OutputModel(error=f"{type(e).__name__}: {e}")
        return self.OutputModel(
            result=result if isinstance(result, dict) else {"value": result},
            content=result.get("content", []) if isinstance(result, dict) else [],
        )


class MCPToolSource(ToolSource):
    type_name = "mcp"

    async def discover(self) -> list[Tool]:
        try:
            from mcp import ClientSession  # noqa: F401
            from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: F401
            from mcp.client.streamable_http import streamablehttp_client  # noqa: F401
        except ImportError:
            logger.warning(
                "mcp package not installed; MCPToolSource returning no tools. "
                "Install with: pip install 'agent-intelligence[mcp]'"
            )
            return []

        servers = self.config.config.get("servers") or []
        out: list[Tool] = []
        for srv in servers:
            try:
                tools = await self._discover_one(srv)
                out.extend(tools)
            except Exception as e:  # noqa: BLE001
                logger.exception("MCP discovery failed for server %s: %s", srv.get("name"), e)
        return out

    async def _discover_one(self, srv: dict[str, Any]) -> list[Tool]:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        label = srv.get("name") or srv.get("url") or srv.get("command") or "mcp"
        prefix = srv.get("prefix") or f"mcp.{label}."
        transport = srv.get("transport") or ("http" if srv.get("url") else "stdio")
        allowlist = set(srv.get("allowlist") or [])

        async def _open_session():
            if transport == "http":
                return streamablehttp_client(srv["url"])
            params = StdioServerParameters(
                command=srv["command"],
                args=srv.get("args") or [],
                env={**os.environ, **(srv.get("env") or {})},
            )
            return stdio_client(params)

        # We open a session, list tools, then close. Persistent sessions could be
        # cached but adds complexity; per-call invocation is fine for v1.
        cm = await _open_session()
        async with cm as conn:
            if transport == "http":
                reader, writer, _ = conn  # streamablehttp returns 3-tuple
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    server_tools = listed.tools
            else:
                reader, writer = conn  # stdio_client returns 2-tuple
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    server_tools = listed.tools

        # Build invokers for each tool
        out: list[Tool] = []
        for t in server_tools:
            if allowlist and t.name not in allowlist:
                continue

            async def _invoker(tool_name: str, args: dict[str, Any], _srv=srv) -> dict[str, Any]:
                cm2 = await _open_session()
                async with cm2 as conn2:
                    if (_srv.get("transport") or "stdio") == "http":
                        reader2, writer2, _ = conn2
                    else:
                        reader2, writer2 = conn2
                    async with ClientSession(reader2, writer2) as session2:
                        await session2.initialize()
                        # Strip our prefix for the actual MCP call
                        bare = tool_name[len(prefix) :] if tool_name.startswith(prefix) else tool_name
                        result = await session2.call_tool(bare, arguments=args)
                        # Normalise — MCP returns a CallToolResult with `content` list
                        return {
                            "content": [
                                getattr(c, "model_dump", lambda: {"text": str(c)})()
                                for c in (result.content or [])
                            ],
                            "isError": bool(getattr(result, "isError", False)),
                        }

            out.append(
                _MCPToolGeneric(
                    tool_name=prefix + t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema if isinstance(t.inputSchema, dict) else json.loads(t.inputSchema or "{}"),
                    invoker=_invoker,
                    server_label=label,
                )
            )
        return out
