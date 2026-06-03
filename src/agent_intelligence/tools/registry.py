"""Global tool registry. Singleton-per-process."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_intelligence.tools.base import Tool


class ToolRegistry:
    """Name → Tool instance map. Tools register themselves on import."""

    def __init__(self) -> None:
        self._tools: dict[str, "Tool"] = {}

    def register(self, name: str, instance: "Tool") -> None:
        self._tools[name] = instance

    def get(self, name: str) -> "Tool":
        if name not in self._tools:
            raise KeyError(
                f"Tool {name!r} not registered. Available: {sorted(self._tools.keys()) or '(none)'}"
            )
        return self._tools[name]

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def signatures(self, only: list[str] | None = None) -> list[dict]:
        """Return LLM-ready function declarations for all (or a subset) of tools."""
        names = only or self.list_names()
        return [self._tools[n].to_llm_signature() for n in names if n in self._tools]


tool_registry = ToolRegistry()
"""Global singleton. Import as: from agent_intelligence import tool_registry"""
