"""ToolSource type registry."""
from __future__ import annotations

from importlib.metadata import entry_points

from agent_intelligence.core.errors import RegistryError
from agent_intelligence.tools.sources.base import ToolSource


class _SourceRegistry:
    _GROUP = "agent_intelligence.tool_sources"

    def __init__(self) -> None:
        self._types: dict[str, type[ToolSource]] = {}
        self._discovered = False

    def register(self, type_name: str, cls: type[ToolSource]) -> None:
        cls.type_name = type_name
        self._types[type_name] = cls

    def _discover(self) -> None:
        if self._discovered:
            return
        # Built-ins
        from agent_intelligence.tools.sources.builtin import BuiltinToolSource
        from agent_intelligence.tools.sources.mcp_source import MCPToolSource
        from agent_intelligence.tools.sources.skillhub import SkillHubToolSource

        self.register("builtin", BuiltinToolSource)
        self.register("skillhub", SkillHubToolSource)
        self.register("mcp", MCPToolSource)

        # Entry points
        try:
            for ep in entry_points(group=self._GROUP):
                if ep.name not in self._types:
                    self._types[ep.name] = ep.load()
        except Exception:
            pass
        self._discovered = True

    def get(self, type_name: str) -> type[ToolSource]:
        self._discover()
        if type_name not in self._types:
            raise RegistryError(
                f"ToolSource type {type_name!r} not registered. "
                f"Available: {sorted(self._types.keys())}"
            )
        return self._types[type_name]

    def list_types(self) -> list[str]:
        self._discover()
        return sorted(self._types.keys())


source_registry = _SourceRegistry()
