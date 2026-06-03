"""MCP server publishing.

Wraps registered agents as MCP tools that any MCP-compatible client (Claude
Code, Cursor, Codex, etc.) can call directly. Authorisation is delegated to
the host application via a callback.
"""

from agent_intelligence.mcp.server import build_mcp_app

__all__ = ["build_mcp_app"]
