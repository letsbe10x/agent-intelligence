"""BuiltinToolSource — exposes @tool-decorated functions registered in the
global tool_registry. Optional ``allowlist`` config narrows to a subset."""
from __future__ import annotations

from agent_intelligence.tools.base import Tool
from agent_intelligence.tools.registry import tool_registry
from agent_intelligence.tools.sources.base import ToolSource


class BuiltinToolSource(ToolSource):
    """All @tool-registered functions from this package + entry points."""

    type_name = "builtin"

    async def discover(self) -> list[Tool]:
        # Ensure builtins are imported and auto-registered
        import agent_intelligence.tools.builtins  # noqa: F401

        allowlist: list[str] | None = self.config.config.get("allowlist")
        out: list[Tool] = []
        for name in tool_registry.list_names():
            if allowlist is not None and name not in allowlist:
                continue
            out.append(tool_registry.get(name))
        return out
