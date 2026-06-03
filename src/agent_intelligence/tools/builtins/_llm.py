"""Shared LLM helper for tools that need a single completion.

Tools that wrap a single LLM call use this helper to:
- pull the provider from the current AgentContext (set by the ReAct loop)
- send one structured-JSON request
- parse the JSON out of the response

The provider is dynamically injected at run time through context.metadata["_provider"].
Tools never construct their own LLM client — that would hardcode the model.
"""
from __future__ import annotations

import json
import re
from typing import Any

from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import ProviderError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, Message


def _extract_json(raw: str) -> dict[str, Any]:
    s = raw.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # find first balanced {...}
    depth = 0
    start = None
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(s[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ProviderError(f"Could not parse JSON from LLM output: {raw[:300]!r}")


async def call_llm_json(context: AgentContext, user_prompt: str) -> dict[str, Any]:
    provider = context.metadata.get("_provider")
    if not isinstance(provider, LLMProvider):
        raise ProviderError(
            "No LLM provider on context. Tool was called outside a ReAct loop, "
            "or the loop forgot to inject context.metadata['_provider']."
        )
    response = await provider.acomplete(
        LLMRequest(
            messages=[
                Message(role="system", content="You produce strictly schema-conforming JSON. No prose outside JSON."),
                Message(role="user", content=user_prompt),
            ],
        )
    )
    return _extract_json(response.content)
