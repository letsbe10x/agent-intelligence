"""Persona tools — generate, evaluate, synthesize.

Each is a single-purpose LLM-backed tool. The AGENT (ReAct loop) decides
WHEN to call them and in what order. None of these tools contains any
agent-decision logic — they take input, call the LLM once, return output.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.providers.base import LLMRequest, Message
from agent_intelligence.tools.base import tool


# ---------------------------------------------------------------------------
# generate_persona — produce one persona reaction
# ---------------------------------------------------------------------------

class GeneratePersonaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bet_hypothesis: str
    bet_metric: str
    bet_time_box: str
    persona_archetype: str = Field(
        description="A 1-2 sentence description of the persona's role + pain + value system"
    )


class GeneratePersonaOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    persona_name: str
    persona_archetype: str
    overall_stance: str
    confidence_0_1: float
    endorsements: list[str]
    objections: list[str]
    questions: list[str]


@tool(
    "persona.generate",
    "Simulate ONE persona's reaction to a product bet. The persona is described by "
    "the `persona_archetype` argument. Returns structured reaction with stance, "
    "endorsements, objections, and diagnostic questions. The persona MUST surface "
    "at least one specific objection to avoid Yes-Man bias.",
    GeneratePersonaInput,
    GeneratePersonaOutput,
    tags=["persona", "validation"],
)
async def persona_generate(
    input_: GeneratePersonaInput, context: AgentContext
) -> GeneratePersonaOutput:
    from agent_intelligence.tools.builtins._llm import call_llm_json

    prompt = (
        f"You ARE the following persona, reacting to a product bet.\n\n"
        f"PERSONA: {input_.persona_archetype}\n\n"
        f"BET:\n"
        f"  hypothesis: {input_.bet_hypothesis}\n"
        f"  success metric: {input_.bet_metric}\n"
        f"  time box: {input_.bet_time_box}\n\n"
        f"Respond as this persona. Be specific and grounded — never generic.\n"
        f"You MUST surface at least one concrete objection even if you endorse.\n\n"
        f"Return JSON only with keys: persona_name, persona_archetype, overall_stance "
        f'("resist"|"neutral"|"endorse"), confidence_0_1, endorsements (array of strings), '
        f"objections (array of strings, min length 1), questions (array of strings)."
    )
    data = await call_llm_json(context, prompt)
    return GeneratePersonaOutput.model_validate(data)


# ---------------------------------------------------------------------------
# evaluate_diversity — score a set of personas for diversity
# ---------------------------------------------------------------------------

class EvaluateDiversityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    personas: list[dict[str, Any]] = Field(
        description="List of persona reactions to evaluate for diversity"
    )


class EvaluateDiversityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diversity_score_0_1: float = Field(
        description="0=all the same, 1=maximally diverse archetypes and stances"
    )
    needs_more_personas: bool
    weakest_persona_index: int | None = Field(
        default=None, description="Index of persona too similar to others; None if all distinct"
    )
    rationale: str


@tool(
    "persona.evaluate_diversity",
    "Evaluate a SET of persona reactions for diversity of stance, archetype, and "
    "objections. Returns a 0-1 score, flags whether more personas are needed, and "
    "identifies the weakest (most-redundant) persona by index. Use after generating "
    "multiple personas to check if the set is rich enough or if one should be retried.",
    EvaluateDiversityInput,
    EvaluateDiversityOutput,
)
async def persona_evaluate_diversity(
    input_: EvaluateDiversityInput, context: AgentContext
) -> EvaluateDiversityOutput:
    from agent_intelligence.tools.builtins._llm import call_llm_json

    prompt = (
        f"Evaluate diversity across these persona reactions:\n\n"
        f"{json.dumps(input_.personas, indent=2)}\n\n"
        f"A diverse set has:\n"
        f"  - mix of stances (resist/neutral/endorse)\n"
        f"  - distinct archetypes (different roles + pain points)\n"
        f"  - non-overlapping objections\n\n"
        f"Return JSON with keys: diversity_score_0_1 (float), needs_more_personas (bool), "
        f"weakest_persona_index (int or null), rationale (1-2 sentences)."
    )
    data = await call_llm_json(context, prompt)
    return EvaluateDiversityOutput.model_validate(data)
