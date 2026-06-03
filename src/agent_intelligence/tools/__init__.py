"""Tools — first-class callable units that agents invoke during ReAct loops.

Tools have:
- a unique name
- a pydantic input schema (auto-converted to JSON schema for the LLM)
- an async `run(input, context) -> output` method
- optional metadata (description, tags) for retrieval

Registration is via the `@tool` decorator + a global ToolRegistry. Discovery
also works via entry points (`agent_intelligence.tools`).
"""
from agent_intelligence.tools.base import Tool, ToolResult, tool
from agent_intelligence.tools.registry import ToolRegistry, tool_registry

__all__ = ["Tool", "ToolRegistry", "ToolResult", "tool", "tool_registry"]
