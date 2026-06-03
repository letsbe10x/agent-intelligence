"""OpenAI Chat Completions provider."""

from __future__ import annotations

from agent_intelligence.core.errors import ProviderError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, LLMResponse

_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5 / 1000, 10.0 / 1000),
    "gpt-4o-mini": (0.15 / 1000, 0.6 / 1000),
    "gpt-4-turbo": (10.0 / 1000, 30.0 / 1000),
    "o3-mini": (3.0 / 1000, 12.0 / 1000),
}


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions API (or any OpenAI-compatible endpoint)."""

    name = "openai"

    def __init__(self, config: ProviderConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed. "
                "Install with: pip install 'agent-intelligence[openai]'"
            ) from e

        client_kwargs: dict[str, object] = {"timeout": config.timeout_s}
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = AsyncOpenAI(**client_kwargs)  # type: ignore[arg-type]

    async def _acomplete(self, request: LLMRequest) -> LLMResponse:
        try:
            resp = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                max_tokens=request.max_output_tokens or self.config.max_output_tokens,
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else (self.config.temperature if self.config.temperature is not None else 0.7)
                ),
                **request.extra,
            )
        except Exception as e:
            raise ProviderError(f"OpenAI call failed: {type(e).__name__}: {e}") from e

        choice = resp.choices[0]
        usage = resp.usage
        cost = self._compute_cost(
            input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens
        )
        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            model=resp.model,
            extra={"id": resp.id},
        )

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_per_1k = self.config.extra.get("cost_per_1k_input")
        output_per_1k = self.config.extra.get("cost_per_1k_output")
        if input_per_1k is None or output_per_1k is None:
            pricing = _DEFAULT_PRICING.get(self.config.model)
            if pricing is None:
                raise ProviderError(
                    f"OpenAI model {self.config.model!r} has no built-in pricing. "
                    "Set provider.extra.cost_per_1k_input and "
                    "provider.extra.cost_per_1k_output in your config."
                )
            input_per_1k, output_per_1k = pricing
        return (input_tokens / 1000.0) * input_per_1k + (output_tokens / 1000.0) * output_per_1k
