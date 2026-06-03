"""Persona Simulator agent tests — uses MockProvider with scripted JSON output."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_intelligence.agents.persona_simulator import PersonaSimulatorAgent
from agent_intelligence.core.config import (
    AgentConfig,
    BudgetConfig,
    ObservabilityConfig,
    ProviderConfig,
)
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import AgentError
from agent_intelligence.providers.mock import MockProvider

SCRIPTED_OK = json.dumps(
    {
        "bet_id": "bet_test_001",
        "reactions": [
            {
                "persona_name": "Sarah, PM at fintech",
                "persona_archetype": "Mid-market product manager at a regulated fintech.",
                "overall_stance": "endorse",
                "confidence_0_1": 0.7,
                "endorsements": ["Solves the export pain on monthly board reports"],
                "objections": ["Audit trail needs to survive Excel re-import"],
                "questions_the_persona_would_ask": ["What's the export format guarantee?"],
            },
            {
                "persona_name": "Raj, Eng lead",
                "persona_archetype": "Engineering lead, infra-conscious.",
                "overall_stance": "resist",
                "confidence_0_1": 0.6,
                "endorsements": [],
                "objections": ["Async pipelines add operational burden"],
                "questions_the_persona_would_ask": ["Who owns the async worker on-call?"],
            },
            {
                "persona_name": "Maya, exec",
                "persona_archetype": "VP-level buyer.",
                "overall_stance": "neutral",
                "confidence_0_1": 0.5,
                "endorsements": ["Could enable better cross-team reporting"],
                "objections": ["Not yet sure on ROI"],
                "questions_the_persona_would_ask": ["What's the projected adoption?"],
            },
        ],
        "aggregate_resistance_pct": 33.3,
        "aggregate_endorsement_pct": 33.3,
        "top_objection_themes": ["operational burden", "audit trail"],
        "claims": [
            {
                "text": "The bet proposes async dashboard exports",
                "source_ref": "bet_test_001",
                "confidence_0_1": 1.0,
            }
        ],
    }
)

YES_MAN_OUTPUT = json.dumps(
    {
        "bet_id": "bet_test_001",
        "reactions": [
            {
                "persona_name": "Endorser 1",
                "persona_archetype": "PM",
                "overall_stance": "endorse",
                "confidence_0_1": 0.9,
                "endorsements": ["Great idea"],
                "objections": [],  # ← violates require_objections
                "questions_the_persona_would_ask": [],
            }
        ],
        "aggregate_resistance_pct": 0,
        "aggregate_endorsement_pct": 100,
        "top_objection_themes": [],
        "claims": [],
    }
)


def _make_agent(scripted: str, prompt_path: Path, params=None):
    config = AgentConfig(
        name="persona_simulator",
        version="1",
        prompt_path=str(prompt_path),
        provider=ProviderConfig(
            name="mock",
            model="mock-model",
            extra={"scripted_response": scripted},
        ),
        budget=BudgetConfig(),
        observability=ObservabilityConfig(receipts_enabled=True, receipts_path=None),
        params=params or {"n_personas": 3, "reaction_depth": "medium", "require_objections": True},
    )
    provider = MockProvider(config.provider)
    return PersonaSimulatorAgent(config=config, provider=provider)


@pytest.fixture
def prompt_path(tmp_path):
    p = tmp_path / "prompts" / "main.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Copy the real prompt so .format() works the same way.
    real = (
        Path(__file__).resolve().parents[2]
        / "src/agent_intelligence/agents/persona_simulator/prompts/main.md"
    )
    p.write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    return p


async def test_persona_simulator_happy_path(prompt_path):
    agent = _make_agent(SCRIPTED_OK, prompt_path)
    result = await agent.run(
        {
            "bet_id": "bet_test_001",
            "hypothesis": "Async dashboard exports lift weekly PM retention by 8% within 30 days",
            "success_metric": "weekly_pm_retention",
            "time_box": "30 days post-ship",
        },
        AgentContext(org_id="acme", bet_id="bet_test_001"),
    )

    out = result.output
    assert out.bet_id == "bet_test_001"
    assert len(out.reactions) == 3
    assert all(r.persona_name for r in out.reactions)
    assert out.aggregate_resistance_pct == pytest.approx(33.3, abs=1)
    assert result.receipt.payload_hash
    assert result.receipt.verify() is True


async def test_persona_simulator_rejects_yes_man(prompt_path):
    agent = _make_agent(YES_MAN_OUTPUT, prompt_path, params={"n_personas": 1, "require_objections": True})
    with pytest.raises(AgentError, match="Yes-Man"):
        await agent.run(
            {
                "bet_id": "bet_test_001",
                "hypothesis": "test hypothesis with enough length",
                "success_metric": "x",
                "time_box": "30 days",
            },
            AgentContext(),
        )


async def test_persona_simulator_repairs_aggregate_percentages(prompt_path):
    # Hand-craft output where the aggregate percentages are wrong.
    bad = json.loads(SCRIPTED_OK)
    bad["aggregate_resistance_pct"] = 99.0  # actually only 33% resist
    bad["aggregate_endorsement_pct"] = 1.0
    agent = _make_agent(json.dumps(bad), prompt_path)
    result = await agent.run(
        {
            "bet_id": "bet_test_001",
            "hypothesis": "test hypothesis with enough length",
            "success_metric": "x",
            "time_box": "30 days",
        },
        AgentContext(),
    )
    assert result.output.aggregate_resistance_pct == pytest.approx(33.3, abs=1)
    assert result.output.aggregate_endorsement_pct == pytest.approx(33.3, abs=1)
