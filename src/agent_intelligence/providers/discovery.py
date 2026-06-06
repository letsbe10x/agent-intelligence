"""Provider/model discovery for the UI's runtime-selectable picker.

Each provider exposes:
    - name
    - which env var configures its API key
    - a static catalog of known models (best-effort, supplementable from config)
    - whether the key is currently present in env (for the "ready" indicator)
"""
from __future__ import annotations

import os
from typing import Any


_STATIC_CATALOG: dict[str, dict[str, Any]] = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-7",
        "models": [
            "claude-opus-4-8-thinking-high",
            "claude-opus-4-7-thinking-high",
            "claude-opus-4-7",
            "claude-sonnet-4-7",
            "claude-haiku-4-7",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-haiku-20241022",
        ],
        "supports_streaming": True,
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o3-mini", "gpt-5.5-high", "gpt-5.4-high"],
        "supports_streaming": True,
    },
    "litellm": {
        "env_var": None,
        "default_model": "anthropic/claude-sonnet-4-7",
        "models": [
            "anthropic/claude-sonnet-4-7",
            "openai/gpt-4o",
            "gemini/gemini-2.5-pro",
            "groq/llama-3.3-70b-versatile",
        ],
        "supports_streaming": True,
    },
    "ollama": {
        "env_var": None,
        "default_model": "llama3.1",
        "models": ["llama3.1", "qwen2.5", "mistral-nemo", "deepseek-coder-v2"],
        "supports_streaming": False,
        "is_local": True,
    },
    "mock": {
        "env_var": None,
        "default_model": "mock-model",
        "models": ["mock-model"],
        "supports_streaming": False,
        "is_mock": True,
    },
}


async def discover_providers() -> list[dict[str, Any]]:
    """Return a list of provider descriptors for the UI picker.

    For Ollama, probes the local daemon to enumerate installed models.
    For others, returns the static catalog plus a `ready` flag based on env.
    """
    out: list[dict[str, Any]] = []
    for name, meta in _STATIC_CATALOG.items():
        env_var = meta.get("env_var")
        ready = True
        if env_var:
            ready = bool(os.environ.get(env_var))
        models: list[str] = list(meta.get("models") or [])
        if name == "ollama":
            try:
                from agent_intelligence.providers.ollama import OllamaProvider

                live = await OllamaProvider.list_models()
                if live:
                    models = live
                ready = bool(live)
            except Exception:
                ready = False
        out.append(
            {
                "name": name,
                "default_model": meta.get("default_model"),
                "models": models,
                "env_var": env_var,
                "ready": ready,
                "supports_streaming": meta.get("supports_streaming", False),
                "is_local": meta.get("is_local", False),
                "is_mock": meta.get("is_mock", False),
            }
        )
    return out
