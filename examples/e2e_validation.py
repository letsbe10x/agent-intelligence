"""End-to-end example: persona_simulator → citation_resolver.

Demonstrates how the two agents chain together to produce verified Evidence
for a Bet. Run with:

    # mock mode (no API key required, deterministic)
    python examples/e2e_validation.py --mock

    # real mode (requires ANTHROPIC_API_KEY)
    python examples/e2e_validation.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from agent_intelligence import AgentContext, registry
from agent_intelligence.agents.citation_resolver import CitationResolverAgent
from agent_intelligence.agents.persona_simulator import PersonaSimulatorAgent
from agent_intelligence.core.config import (
    AgentConfig,
    BudgetConfig,
    ObservabilityConfig,
    ProviderConfig,
)
from agent_intelligence.observability.receipts import ReceiptStore
from agent_intelligence.providers.mock import MockProvider


SCRIPTED_PERSONA_OUTPUT = json.dumps(
    {
        "bet_id": "bet_demo_001",
        "reactions": [
            {
                "persona_name": "Sarah, PM at acme-fintech",
                "persona_archetype": "Mid-market product manager, regulated industry.",
                "overall_stance": "endorse",
                "confidence_0_1": 0.7,
                "endorsements": ["Reduces board-report prep time"],
                "objections": ["Audit trail must survive Excel exports per SEC requirements"],
                "questions_the_persona_would_ask": ["What's the export format guarantee?"],
            },
            {
                "persona_name": "Raj, Eng Lead",
                "persona_archetype": "Engineering lead, infra-conscious.",
                "overall_stance": "resist",
                "confidence_0_1": 0.6,
                "endorsements": [],
                "objections": ["Async workers add on-call surface area"],
                "questions_the_persona_would_ask": ["Who owns the async worker on-call?"],
            },
            {
                "persona_name": "Maya, VP Product",
                "persona_archetype": "VP-level buyer, ROI-focused.",
                "overall_stance": "neutral",
                "confidence_0_1": 0.5,
                "endorsements": ["Could improve quarterly reporting"],
                "objections": ["ROI not yet quantified"],
                "questions_the_persona_would_ask": ["What's the projected adoption rate?"],
            },
        ],
        "aggregate_resistance_pct": 33.3,
        "aggregate_endorsement_pct": 33.3,
        "top_objection_themes": ["operational burden", "audit trail compliance", "ROI quantification"],
        "claims": [
            {
                "text": "Async dashboard export is the bet under validation",
                "source_ref": "bet://bet_demo_001",
                "confidence_0_1": 1.0,
            },
            {
                "text": "Per public Anthropic docs, the Messages API supports streaming for long-form outputs",
                "source_ref": "https://docs.anthropic.com/en/api/messages",
                "confidence_0_1": 0.9,
            },
        ],
    }
)


def _make_persona_config(mock: bool, prompt_path: Path) -> AgentConfig:
    if mock:
        provider = ProviderConfig(
            name="mock",
            model="mock-model",
            extra={"scripted_response": SCRIPTED_PERSONA_OUTPUT},
        )
    else:
        provider = ProviderConfig(
            name="anthropic",
            model="claude-sonnet-4-7",
            api_key="${env:ANTHROPIC_API_KEY}",  # resolved earlier by load_config; here we'd use real env
            temperature=0.7,
            max_output_tokens=4096,
        )
    return AgentConfig(
        name="persona_simulator",
        version="1",
        prompt_path=str(prompt_path),
        provider=provider,
        budget=BudgetConfig(max_usd_per_run=2.0),
        observability=ObservabilityConfig(receipts_enabled=True, receipts_path=None),
        params={"n_personas": 3, "reaction_depth": "medium", "require_objections": True},
    )


def _make_citation_config(prompt_dir: Path) -> AgentConfig:
    return AgentConfig(
        name="citation_resolver",
        version="1",
        prompt_path=str(prompt_dir / "main.md"),
        provider=ProviderConfig(name="mock", model="mock-model"),
        budget=BudgetConfig(max_usd_per_run=0.1),
        observability=ObservabilityConfig(receipts_enabled=True, receipts_path=None),
        params={
            "http_timeout_s": 5,
            "http_max_concurrency": 5,
            "require_https_for_external": True,
            "use_llm_for_semantic_check": False,
        },
    )


async def main(mock: bool) -> None:
    # Locate the shipped prompt files (relative to source tree).
    repo_root = Path(__file__).resolve().parents[1]
    persona_prompt = repo_root / "src/agent_intelligence/agents/persona_simulator/prompts/main.md"
    citation_prompt_dir = repo_root / "src/agent_intelligence/agents/citation_resolver/prompts"

    receipts = ReceiptStore(path=Path(tempfile.mkdtemp(prefix="ai-receipts-")))

    # ---------- Step 1: simulate personas ----------
    print("=== Step 1: Persona Simulator ===")
    persona_cfg = _make_persona_config(mock=mock, prompt_path=persona_prompt)
    persona_agent = PersonaSimulatorAgent(
        config=persona_cfg, provider=MockProvider(persona_cfg.provider), receipt_store=receipts
    )
    persona_result = await persona_agent.run(
        {
            "bet_id": "bet_demo_001",
            "hypothesis": (
                "Async dashboard export lifts weekly PM retention by 8% within 30 days "
                "post-ship, by removing the daily-export friction surfaced in 46 signals."
            ),
            "success_metric": "weekly_pm_retention",
            "time_box": "30 days post-ship",
        },
        AgentContext(org_id="acme", bet_id="bet_demo_001"),
    )
    print(f"  → {len(persona_result.output.reactions)} personas")
    print(f"  → resistance: {persona_result.output.aggregate_resistance_pct:.1f}%")
    print(f"  → endorsement: {persona_result.output.aggregate_endorsement_pct:.1f}%")
    print(f"  → receipt: {persona_result.receipt.payload_hash[:16]}...")
    print(f"  → claims produced: {len(persona_result.output.claims)}")

    # ---------- Step 2: resolve citations on the claims ----------
    print("\n=== Step 2: Citation Resolver ===")
    citation_cfg = _make_citation_config(prompt_dir=citation_prompt_dir)
    citation_agent = CitationResolverAgent(
        config=citation_cfg,
        provider=MockProvider(citation_cfg.provider),
        receipt_store=receipts,
    )
    citation_result = await citation_agent.run(
        {
            "claims": [c.model_dump() for c in persona_result.output.claims],
            "known_bet_ids": ["bet_demo_001"],
        },
        AgentContext(org_id="acme", bet_id="bet_demo_001"),
    )
    print(f"  → verifications: {len(citation_result.output.verifications)}")
    print(f"  → summary: {citation_result.output.summary}")
    print(f"  → all_passed: {citation_result.output.all_passed}")
    print(f"  → blocking_failures: {citation_result.output.blocking_failures}")
    print(f"  → receipt: {citation_result.receipt.payload_hash[:16]}...")

    # ---------- Step 3: verify both receipts ----------
    print("\n=== Step 3: Verify receipts ===")
    for receipt_id in receipts.list_ids():
        ok = receipts.verify(receipt_id)
        r = receipts.get(receipt_id)
        print(f"  {receipt_id[:8]}... ({r.agent_name}): verify={ok}, cost=${r.cost_usd:.4f}")

    # ---------- Gate decision ----------
    print("\n=== Gate decision ===")
    if citation_result.output.all_passed:
        print(
            "  ✓ GATE PASSED: all claims have resolvable sources. "
            "Persona simulation can be attached as Evidence."
        )
    else:
        print(
            f"  ✗ GATE BLOCKED: {citation_result.output.blocking_failures} blocking failure(s). "
            "Persona simulation must be re-grounded before attachment."
        )
        for v in citation_result.output.verifications:
            if v.status not in ("resolved", "warning"):
                print(f"    - {v.status}: {v.source_ref}  →  {v.detail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use mock provider (no API key needed)")
    args = parser.parse_args()
    if not args.mock:
        print("Note: real mode not wired in this example. Use --mock.", file=sys.stderr)
        args.mock = True
    asyncio.run(main(mock=args.mock))
