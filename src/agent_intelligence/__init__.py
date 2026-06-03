"""agent-intelligence — agentic framework where the LLM owns decisions.

The framework provides:
    - ReAct loop primitive (`react.ReActLoop`) — LLM picks tools, decides to stop.
    - Tool registry (`tools.tool_registry`) — built-in + custom tools registered
      via @tool decorator or entry points.
    - Multi-model providers (`providers.*`) — Anthropic, OpenAI, LiteLLM, Mock.
    - Receipts, budgets, OTel-compatible tracing — every step recorded.
    - YAML agents — an agent is a YAML config + prompt + tool list. Zero Python.

Inspired by NVIDIA AgentIQ / NeMo Agent Toolkit. Engineered for the letsbe10x
PVR stack.
"""

from agent_intelligence.core.config import AgentConfig, load_config
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import (
    AgentError,
    BudgetExceededError,
    ConfigError,
    ProviderError,
)
from agent_intelligence.observability.receipts import Receipt, ReceiptStore
from agent_intelligence.providers.base import LLMProvider, LLMRequest, LLMResponse, Message
from agent_intelligence.react.loop import ReActLoop, ReActOutcome, ReActStep
from agent_intelligence.react.runner import (
    ReActAgentConfig,
    ReActAgentResult,
    load_react_config,
    run_react_agent,
)
from agent_intelligence.registry.registry import registry
from agent_intelligence.tools.base import Tool, ToolResult, tool
from agent_intelligence.tools.registry import ToolRegistry, tool_registry
from agent_intelligence.tracing.events import TraceEmitter, TraceEvent

__version__ = "0.2.0"

__all__ = [
    "AgentConfig",
    "AgentContext",
    "AgentError",
    "BudgetExceededError",
    "ConfigError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ProviderError",
    "ReActAgentConfig",
    "ReActAgentResult",
    "ReActLoop",
    "ReActOutcome",
    "ReActStep",
    "Receipt",
    "ReceiptStore",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "TraceEmitter",
    "TraceEvent",
    "load_config",
    "load_react_config",
    "registry",
    "run_react_agent",
    "tool",
    "tool_registry",
]
