"""ToolSource — pluggable tool discovery layer.

A ToolSource produces a list of Tools the agent can call. Three built-in sources:
    - BuiltinToolSource : @tool-decorated functions in this package + entry points
    - SkillHubToolSource: lets-* skills from letsbe10x/skill-hub, exposed as tools
    - MCPToolSource     : any Model Context Protocol server (HTTP or stdio)

Third parties add new sources by subclassing ToolSource and registering via
entry points in group ``agent_intelligence.tool_sources``.
"""
from agent_intelligence.tools.sources.base import ToolSource, ToolSourceConfig
from agent_intelligence.tools.sources.builtin import BuiltinToolSource
from agent_intelligence.tools.sources.skillhub import SkillHubToolSource
from agent_intelligence.tools.sources.mcp_source import MCPToolSource

__all__ = [
    "BuiltinToolSource",
    "MCPToolSource",
    "SkillHubToolSource",
    "ToolSource",
    "ToolSourceConfig",
]
