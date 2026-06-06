"""ToolSource ABC. All discovery happens here. No domain logic."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.tools.base import Tool


class ToolSourceConfig(BaseModel):
    """Source config carried in agent YAML."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="Registered source name (builtin | skillhub | mcp | <custom>)")
    name: str = Field(default="", description="Friendly label; defaults to type.")
    enabled: bool = Field(default=True)
    config: dict[str, Any] = Field(default_factory=dict, description="Source-specific options")


class ToolSource(ABC):
    """Discovers tools. Implementations need only return a list of Tool instances."""

    type_name: str = ""

    def __init__(self, config: ToolSourceConfig) -> None:
        self.config = config

    @abstractmethod
    async def discover(self) -> list[Tool]:
        """Return a list of fully-constructed Tool instances. Cached by caller."""

    @property
    def label(self) -> str:
        return self.config.name or self.type_name


def load_sources(configs: list[ToolSourceConfig]) -> list[ToolSource]:
    """Resolve each ToolSourceConfig to a ToolSource instance via the source registry."""
    from agent_intelligence.tools.sources.registry import source_registry

    out: list[ToolSource] = []
    for c in configs:
        if not c.enabled:
            continue
        SourceCls = source_registry.get(c.type)
        out.append(SourceCls(c))
    return out


async def discover_all(sources: list[ToolSource]) -> dict[str, Tool]:
    """Merge tools from all sources into a single name -> Tool map. Later sources win."""
    out: dict[str, Tool] = {}
    for s in sources:
        tools = await s.discover()
        for t in tools:
            out[t.name] = t
    return out
