"""Mock provider for tests and deterministic CI runs.

Returns canned responses keyed by ``model`` or by a user-supplied lookup.
Does not require any external dependency or API key. Use this in tests.
"""

from __future__ import annotations

from agent_intelligence.providers.base import LLMProvider, LLMRequest, LLMResponse


class MockProvider(LLMProvider):
    """Returns canned responses. Useful for unit tests and offline demos.

    Behaviour:
        - If config.extra["scripted_response"] is set, returns that verbatim.
        - Otherwise returns a deterministic echo of the last user message.
        - Token counts are heuristic (chars/4). Cost is always zero.
    """

    name = "mock"

    async def _acomplete(self, request: LLMRequest) -> LLMResponse:
        scripted = self.config.extra.get("scripted_response")
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        content = scripted if scripted is not None else f"[mock] {last_user[:200]}"

        input_chars = sum(len(m.content) for m in request.messages)
        return LLMResponse(
            content=content,
            finish_reason="stop",
            input_tokens=max(1, input_chars // 4),
            output_tokens=max(1, len(content) // 4),
            cost_usd=0.0,
            model=self.config.model or "mock-model",
            extra={"scripted": scripted is not None},
        )
