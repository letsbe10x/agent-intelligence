"""LLMProvider ABC and message types.

Design pattern: **Strategy**.

Every provider implements the same interface. Agents code against the interface,
never against a specific SDK. Multi-model is a runtime choice, not a code change.

Why not use LiteLLM directly: LiteLLM is one valid provider implementation
(``providers/litellm.py``), but native providers exist for cases where you need
provider-specific features (Anthropic prompt caching, OpenAI structured outputs,
etc.). The framework supports both paths through the same ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.config import ProviderConfig
from agent_intelligence.observability.budget import BudgetTracker


class Message(BaseModel):
    """A single message in a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    # Optional tool-call payload. Providers that don't support tools ignore this.
    name: str | None = None
    tool_call_id: str | None = None


class LLMRequest(BaseModel):
    """A normalised request shape that every provider knows how to translate."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    # Per-call overrides of the provider's defaults. None = use config defaults.
    temperature: float | None = None
    max_output_tokens: int | None = None
    # Provider-specific extras. Passed through verbatim to the provider SDK.
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """A normalised response shape."""

    model_config = ConfigDict(extra="forbid")

    content: str
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str
    # Provider-specific extras (e.g. structured output JSON, citations array).
    extra: dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """Provider strategy interface. All LLM access flows through this.

    Lifecycle:
        provider = SomeProvider(config)             # construct from config
        bound = provider.with_budget(tracker)       # bind a budget tracker
        response = await bound.acomplete(request)   # make a call

    The ``with_budget()`` step is mandatory before calling ``acomplete``. It
    returns a new provider view that preflights every call against the budget.
    A provider without a budget tracker raises if called.
    """

    name: str = ""
    """Registered provider name. Must match entry-point key."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._budget: BudgetTracker | None = None

    def with_budget(self, budget: BudgetTracker) -> LLMProvider:
        """Return a new provider view bound to the given budget tracker.

        Implemented in the base; subclasses do not override. The returned view
        shares the same underlying client but enforces budget on every call.
        """
        # Shallow clone via __class__ + __init__ avoidance.
        clone = self.__class__.__new__(self.__class__)
        clone.config = self.config
        clone._budget = budget
        # Subclass-specific state (e.g. client handles) is copied via __dict__.
        for k, v in self.__dict__.items():
            if k != "_budget":
                clone.__dict__[k] = v
        return clone

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Make a chat completion call. Preflight budget; record actual cost.

        Subclasses implement ``_acomplete``. The base wraps it with budget
        enforcement so subclasses cannot skip it.
        """
        if self._budget is None:
            raise RuntimeError(
                f"Provider {self.name!r} called without a bound budget. "
                "Call provider.with_budget(tracker) first."
            )

        # Preflight: estimate input tokens before the call. Cheap upper-bound check.
        est_input_tokens = self._estimate_input_tokens(request)
        self._budget.preflight(
            estimated_input_tokens=est_input_tokens,
            estimated_output_tokens=request.max_output_tokens
            or self.config.max_output_tokens
            or 0,
            estimated_cost_usd=0.0,  # cost unknown until response; budget enforces on accounting
        )

        response = await self._acomplete(request)

        self._budget.account(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )
        return response

    @abstractmethod
    async def _acomplete(self, request: LLMRequest) -> LLMResponse:
        """Subclass implementation. Receives a fully-normalised request."""

    def _estimate_input_tokens(self, request: LLMRequest) -> int:
        """Cheap upper bound. Override for more accurate per-tokenizer estimates.

        Default: chars/4 heuristic. This is conservative for English and a known
        underestimate for code/CJK. Subclasses for production providers should
        use the model's actual tokenizer.
        """
        total_chars = sum(len(m.content) for m in request.messages)
        return total_chars // 4
