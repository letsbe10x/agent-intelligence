"""Ollama provider — free local models via the Ollama HTTP API.

Configure once with a model name (e.g. `llama3.1`, `qwen2.5`, `mistral-nemo`)
and Ollama exposes them at http://127.0.0.1:11434. No API key needed.
"""
from __future__ import annotations

from typing import Any

import httpx

from agent_intelligence.core.errors import ProviderError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, LLMResponse


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self._base = config.base_url or "http://127.0.0.1:11434"

    async def _acomplete(self, request: LLMRequest) -> LLMResponse:
        url = f"{self._base.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {
                "temperature": (
                    request.temperature
                    if request.temperature is not None
                    else (self.config.temperature if self.config.temperature is not None else 0.7)
                ),
            },
        }
        if request.max_output_tokens or self.config.max_output_tokens:
            payload["options"]["num_predict"] = request.max_output_tokens or self.config.max_output_tokens
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            raise ProviderError(f"Ollama call failed: {type(e).__name__}: {e}") from e

        msg = data.get("message", {})
        text = msg.get("content", "")
        # Ollama returns prompt_eval_count + eval_count
        return LLMResponse(
            content=text,
            finish_reason=data.get("done_reason"),
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            cost_usd=0.0,  # local
            model=data.get("model") or self.config.model,
            extra={},
        )

    @staticmethod
    async def list_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
        """Probe a local Ollama for installed models. Returns [] if Ollama is not running."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{base_url.rstrip('/')}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in (r.json().get("models") or [])]
        except Exception:
            return []
