"""Synthesis tools — extract themes, summarise, score."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.tools.base import tool


class ExtractThemesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[str] = Field(description="Raw text items to cluster into themes")
    max_themes: int = Field(default=5, ge=1, le=20)


class ExtractThemesOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    themes: list[str] = Field(description="2-5 word theme labels")


@tool(
    "synthesize.extract_themes",
    "Cluster a list of text items (e.g. persona objections, customer signals) into "
    "2-5 word recurring theme labels. Use to surface what's common across many "
    "individual signals or reactions.",
    ExtractThemesInput,
    ExtractThemesOutput,
)
async def synthesize_extract_themes(
    input_: ExtractThemesInput, context: AgentContext
) -> ExtractThemesOutput:
    from agent_intelligence.tools.builtins._llm import call_llm_json

    prompt = (
        f"Cluster these items into {input_.max_themes} or fewer recurring themes. "
        f"Each theme label is 2-5 words.\n\n"
        f"Items:\n{json.dumps(input_.items, indent=2)}\n\n"
        f'Return JSON: {{"themes": ["label 1", "label 2", ...]}}'
    )
    data = await call_llm_json(context, prompt)
    return ExtractThemesOutput.model_validate(data)


class FinalizeValidationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bet_id: str
    persona_reactions: list[dict[str, Any]]
    diversity_evaluation: dict[str, Any]
    top_themes: list[str]


class FinalizeValidationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bet_id: str
    aggregate_resist_pct: float
    aggregate_neutral_pct: float
    aggregate_endorse_pct: float
    overall_recommendation: str = Field(
        description="proceed_to_spec | needs_more_validation | kill_bet"
    )
    confidence_0_1: float
    rationale: str
    top_themes: list[str]
    persona_reactions: list[dict[str, Any]]


@tool(
    "synthesize.finalize_validation",
    "Assemble the final validation pack output from collected persona reactions + "
    "diversity evaluation + top themes. Returns the structured artifact ready to "
    "attach to a Bet as Evidence at the validation gate.",
    FinalizeValidationInput,
    FinalizeValidationOutput,
)
async def synthesize_finalize_validation(
    input_: FinalizeValidationInput, context: AgentContext
) -> FinalizeValidationOutput:
    from agent_intelligence.tools.builtins._llm import call_llm_json

    prompt = (
        f"Assemble the final validation pack.\n\n"
        f"Bet ID: {input_.bet_id}\n"
        f"Persona reactions: {json.dumps(input_.persona_reactions, indent=2)}\n"
        f"Diversity eval: {json.dumps(input_.diversity_evaluation, indent=2)}\n"
        f"Top themes: {json.dumps(input_.top_themes)}\n\n"
        f"Compute aggregate stance percentages from the persona reactions. "
        f"Make a final recommendation: proceed_to_spec | needs_more_validation | kill_bet. "
        f"Provide a rationale.\n\n"
        f"Return JSON with keys: bet_id, aggregate_resist_pct, aggregate_neutral_pct, "
        f"aggregate_endorse_pct, overall_recommendation, confidence_0_1, rationale, "
        f"top_themes, persona_reactions."
    )
    data = await call_llm_json(context, prompt)
    # Ensure these passthrough fields are preserved even if the LLM omits them
    data.setdefault("bet_id", input_.bet_id)
    data.setdefault("top_themes", input_.top_themes)
    data.setdefault("persona_reactions", input_.persona_reactions)
    return FinalizeValidationOutput.model_validate(data)
