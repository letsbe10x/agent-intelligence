"""Citation Resolver agent tests — uses pytest-httpx-style mocking."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from agent_intelligence.agents.citation_resolver import CitationResolverAgent
from agent_intelligence.core.config import (
    AgentConfig,
    BudgetConfig,
    ObservabilityConfig,
    ProviderConfig,
)
from agent_intelligence.core.context import AgentContext
from agent_intelligence.providers.mock import MockProvider


@pytest.fixture
def prompt_path(tmp_path):
    p = tmp_path / "prompts" / "main.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("placeholder", encoding="utf-8")
    # Also write semantic_check.md for the semantic-check tests.
    sem = p.parent / "semantic_check.md"
    sem.write_text(
        "Claim:\n{claim_text}\n\nSource:\n{source_excerpt}\n",
        encoding="utf-8",
    )
    return p


def _make_agent(prompt_path: Path, params=None, scripted_llm: str | None = None):
    config = AgentConfig(
        name="citation_resolver",
        version="1",
        prompt_path=str(prompt_path),
        provider=ProviderConfig(
            name="mock",
            model="mock-model",
            extra={"scripted_response": scripted_llm or ""},
        ),
        budget=BudgetConfig(),
        observability=ObservabilityConfig(receipts_enabled=True, receipts_path=None),
        params=params or {},
    )
    provider = MockProvider(config.provider)
    return CitationResolverAgent(config=config, provider=provider)


async def test_unknown_scheme_is_flagged(prompt_path):
    agent = _make_agent(prompt_path)
    result = await agent.run(
        {
            "claims": [
                {"text": "X is true", "source_ref": "weirdscheme://something", "confidence_0_1": 0.5}
            ]
        },
        AgentContext(),
    )
    v = result.output.verifications[0]
    assert v.status == "unknown_scheme"
    assert result.output.all_passed is False
    # unknown_scheme blocks: an unverifiable claim is gate-failing, not advisory.
    assert result.output.blocking_failures == 1
    assert "unknown_scheme" in result.output.summary


async def test_signal_id_found(prompt_path):
    agent = _make_agent(prompt_path)
    result = await agent.run(
        {
            "claims": [
                {"text": "Per signal", "source_ref": "signal://sig_42", "confidence_0_1": 0.9}
            ],
            "known_signal_ids": ["sig_42", "sig_43"],
        },
        AgentContext(),
    )
    v = result.output.verifications[0]
    assert v.status == "resolved"
    assert result.output.all_passed is True
    assert result.output.blocking_failures == 0


async def test_signal_id_not_found(prompt_path):
    agent = _make_agent(prompt_path)
    result = await agent.run(
        {
            "claims": [
                {"text": "Per signal", "source_ref": "signal://sig_404", "confidence_0_1": 0.9}
            ],
            "known_signal_ids": ["sig_42"],
        },
        AgentContext(),
    )
    v = result.output.verifications[0]
    assert v.status == "id_not_found"
    assert result.output.blocking_failures == 1
    assert result.output.all_passed is False


async def test_http_url_unreachable_raises_no_exception_but_flagged(prompt_path, monkeypatch):
    """When httpx fails, we record unreachable — we don't bubble the exception."""
    agent = _make_agent(prompt_path)

    class _FailingAsyncClient:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False
        async def head(self, *a, **kw):
            raise httpx.ConnectError("simulated network failure")
        async def get(self, *a, **kw):
            raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr("httpx.AsyncClient", _FailingAsyncClient)
    result = await agent.run(
        {
            "claims": [
                {
                    "text": "Per the docs",
                    "source_ref": "https://example.invalid/docs",
                    "confidence_0_1": 0.5,
                }
            ]
        },
        AgentContext(),
    )
    v = result.output.verifications[0]
    assert v.status == "unreachable"
    assert "simulated network failure" in v.detail
    assert result.output.blocking_failures == 1


async def test_mixed_results_aggregate_correctly(prompt_path, monkeypatch):
    agent = _make_agent(prompt_path)

    class _OkClient:
        def __init__(self, *a, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *exc): return False

        async def head(self, url, *a, **kw):
            req = httpx.Request("HEAD", url)
            return httpx.Response(200, request=req)

        async def get(self, url, *a, **kw):
            req = httpx.Request("GET", url)
            return httpx.Response(200, content=b"<html><body>ok</body></html>", request=req)

    monkeypatch.setattr("httpx.AsyncClient", _OkClient)

    result = await agent.run(
        {
            "claims": [
                {"text": "A", "source_ref": "https://example.com/a", "confidence_0_1": 0.5},
                {"text": "B", "source_ref": "signal://sig_1", "confidence_0_1": 0.5},
                {"text": "C", "source_ref": "signal://sig_missing", "confidence_0_1": 0.5},
                {"text": "D", "source_ref": "garbage://x", "confidence_0_1": 0.5},
            ],
            "known_signal_ids": ["sig_1"],
        },
        AgentContext(),
    )
    summary = result.output.summary
    assert summary.get("resolved", 0) == 2
    assert summary.get("id_not_found", 0) == 1
    assert summary.get("unknown_scheme", 0) == 1
    # Two failures: id_not_found + unknown_scheme are both blocking.
    assert result.output.blocking_failures == 2
    assert result.output.all_passed is False
