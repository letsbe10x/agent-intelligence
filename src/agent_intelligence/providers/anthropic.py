"""Anthropic Messages API provider.

Lazy-imports the ``anthropic`` SDK so the package can be installed without it.
If the SDK is missing at construction time, a precise ImportError is raised.
"""

from __future__ import annotations

from agent_intelligence.core.errors import ProviderError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, LLMResponse

# Cost table is the only "default" we ship — and even it is overridable via the
# ``extra.cost_per_1k_input`` / ``extra.cost_per_1k_output`` config keys. We do
# NOT silently fall back to zero when a model is unknown; we raise.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # model_id: (input $/1k tokens, output $/1k tokens) — Anthropic public pricing.
    "claude-opus-4-7": (15.0 / 1000, 75.0 / 1000),
    "claude-sonnet-4-7": (3.0 / 1000, 15.0 / 1000),
    "claude-haiku-4-7": (0.8 / 1000, 4.0 / 1000),
    "claude-3-7-sonnet-20250219": (3.0 / 1000, 15.0 / 1000),
    "claude-3-5-haiku-20241022": (0.8 / 1000, 4.0 / 1000),
}


class AnthropicProvider(LLMProvider):
    """Native Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        super().__init__(config)
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. "
                "Install with: pip install 'agent-intelligence[anthropic]'"
            ) from e

        from anthropic import AsyncAnthropic  # local import keeps cold-start fast

        client_kwargs: dict[str, object] = {"timeout": config.timeout_s}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = AsyncAnthropic(**client_kwargs)  # type: ignore[arg-type]

    async def _acomplete(self, request: LLMRequest) -> LLMResponse:
        # Split system messages out; Anthropic Messages API takes them at top level.
        system_parts = [m.content for m in request.messages if m.role == "system"]
        non_system = [m for m in request.messages if m.role != "system"]

        try:
            resp = await self._client.messages.create(
                model=self.config.model,
                system="\n\n".join(system_parts) if system_parts else "",
                messages=[
                    {"role": m.role, "content": m.content} for m in non_system
                ],
                max_tokens=(
                    request.max_output_tokens
                    or self.config.max_output_tokens
                    or 1024
                ),
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else (self.config.temperature if self.config.temperature is not None else 0.7)
                ),
                **request.extra,
            )
        except Exception as e:
            raise ProviderError(f"Anthropic call failed: {type(e).__name__}: {e}") from e

        # Anthropic returns content as a list of blocks. We join text blocks.
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        cost = self._compute_cost(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

        return LLMResponse(
            content=text,
            finish_reason=resp.stop_reason,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cost_usd=cost,
            model=resp.model,
            extra={"id": resp.id},
        )

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        # Config can override pricing via extra.cost_per_1k_input/output.
        input_per_1k = self.config.extra.get("cost_per_1k_input")
        output_per_1k = self.config.extra.get("cost_per_1k_output")

        if input_per_1k is None or output_per_1k is None:
            pricing = _DEFAULT_PRICING.get(self.config.model)
            if pricing is None:
                # No silent fallback. The user must declare pricing for unknown models.
                raise ProviderError(
                    f"Anthropic model {self.config.model!r} has no built-in pricing. "
                    "Set provider.extra.cost_per_1k_input and "
                    "provider.extra.cost_per_1k_output in your config."
                )
            input_per_1k, output_per_1k = pricing

        return (input_tokens / 1000.0) * input_per_1k + (output_tokens / 1000.0) * output_per_1k
