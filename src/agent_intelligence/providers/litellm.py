"""LiteLLM provider — single interface to 100+ models.

Use this when you want max provider coverage without writing per-provider code.
Native providers (anthropic.py, openai.py) are preferable when you need
provider-specific features.
"""

from __future__ import annotations

from agent_intelligence.core.errors import ProviderError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, LLMResponse


class LiteLLMProvider(LLMProvider):
    """LiteLLM unified-interface wrapper."""

    name = "litellm"

    def __init__(self, config: ProviderConfig) -> None:  # type: ignore[name-defined]  # noqa: F821
        super().__init__(config)
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm not installed. "
                "Install with: pip install 'agent-intelligence[litellm]'"
            ) from e

    async def _acomplete(self, request: LLMRequest) -> LLMResponse:
        import litellm
        from litellm import acompletion

        kwargs: dict[str, object] = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "timeout": self.config.timeout_s,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if request.max_output_tokens or self.config.max_output_tokens:
            kwargs["max_tokens"] = request.max_output_tokens or self.config.max_output_tokens
        if request.temperature is not None or self.config.temperature is not None:
            kwargs["temperature"] = (
                request.temperature if request.temperature is not None else self.config.temperature
            )
        kwargs.update(request.extra)

        try:
            resp = await acompletion(**kwargs)
        except Exception as e:
            raise ProviderError(f"LiteLLM call failed: {type(e).__name__}: {e}") from e

        choice = resp.choices[0]
        usage = resp.usage

        # LiteLLM provides cost via completion_cost(). It knows the pricing for
        # most known models. For unknown models it raises — which is what we want.
        try:
            cost = float(litellm.completion_cost(completion_response=resp))
        except Exception:
            override_in = self.config.extra.get("cost_per_1k_input")
            override_out = self.config.extra.get("cost_per_1k_output")
            if override_in is None or override_out is None:
                raise ProviderError(
                    f"LiteLLM cannot price model {self.config.model!r}. "
                    "Set provider.extra.cost_per_1k_input and cost_per_1k_output."
                ) from None
            cost = (usage.prompt_tokens / 1000.0) * override_in + (
                usage.completion_tokens / 1000.0
            ) * override_out

        return LLMResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=cost,
            model=resp.model,
            extra={},
        )
