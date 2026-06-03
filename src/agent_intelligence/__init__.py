"""agent-intelligence — configurable, multi-model agentic framework with receipt-backed execution.

Public surface:
    from agent_intelligence import Agent, AgentConfig, AgentContext, AgentResult, Receipt
    from agent_intelligence import LLMProvider, LLMRequest, LLMResponse, Message
    from agent_intelligence import load_config, registry, run_agent

The framework's contract:
    - Agents are configured, never hardcoded. Every model, prompt, budget, timeout
      and tool list is read from YAML at runtime.
    - There is no silent fallback. If a config is missing or a provider is
      unreachable, the framework raises a typed error; it does not "best-effort".
    - Every agent invocation produces an immutable Receipt (sha256 of inputs+outputs)
      that downstream systems (PVR, control-plane) can replay and verify.
    - Providers and Agents are discovered via entry points so third parties can
      extend without forking.

Inspired by NVIDIA NeMo Agent Toolkit (Apache-2.0). Engineered independently for
the letsbe10x PVR stack.
"""

from agent_intelligence.core.agent import Agent, AgentResult
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
from agent_intelligence.registry.registry import registry
from agent_intelligence.runner import run_agent

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentError",
    "AgentResult",
    "BudgetExceededError",
    "ConfigError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "ProviderError",
    "Receipt",
    "ReceiptStore",
    "load_config",
    "registry",
    "run_agent",
]
