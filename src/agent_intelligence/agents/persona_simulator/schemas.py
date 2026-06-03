"""Pydantic schemas for PersonaSimulatorAgent.

Schemas are the contract — every consumer (UI, MCP, control-plane) sees the
same typed surface. Keep field names explicit; document each one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Input ------------------------------------------------------------------


class PersonaSimulatorInput(BaseModel):
    """What the agent needs from the caller."""

    model_config = ConfigDict(extra="forbid")

    bet_id: str = Field(description="PVR Bet ID this simulation is attached to. Surfaces in receipts.")
    hypothesis: str = Field(
        min_length=10,
        description="The Bet hypothesis. Drives the persona reactions.",
    )
    success_metric: str = Field(description="The metric the Bet claims will move.")
    time_box: str = Field(description="Hypothesis time horizon, e.g. '30 days post-ship'.")
    target_persona_brief: str | None = Field(
        default=None,
        description="Optional caller-supplied persona archetype description. If None, agent generates its own.",
    )


# --- Params -----------------------------------------------------------------


class PersonaSimulatorParams(BaseModel):
    """Agent-specific knobs. Read from YAML ``params:`` block."""

    model_config = ConfigDict(extra="forbid")

    n_personas: int = Field(default=3, ge=1, le=10, description="How many distinct personas to simulate.")
    reaction_depth: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="Controls reaction prose length and how many specific objections/endorsements per persona.",
    )
    require_objections: bool = Field(
        default=True,
        description="If True, every persona MUST surface at least one objection (epistemic hygiene; "
        "prevents Yes-Man simulations that inflate Bet confidence).",
    )


# --- Output -----------------------------------------------------------------


class PersonaReaction(BaseModel):
    """One persona's reaction to the Bet."""

    model_config = ConfigDict(extra="forbid")

    persona_name: str = Field(description="Short label for the persona, e.g. 'Sarah, mid-market PM at fintech'.")
    persona_archetype: str = Field(
        description="1-2 sentence description: role, segment, primary pain, what they value."
    )
    overall_stance: Literal["resist", "neutral", "endorse"] = Field(
        description="Net reaction. Buyers / users PVR can pitch against."
    )
    confidence_0_1: float = Field(
        ge=0, le=1, description="Persona's expressed confidence in their reaction (not the agent's confidence)."
    )
    endorsements: list[str] = Field(
        default_factory=list,
        description="Specific things the persona finds compelling about the Bet. Empty if none.",
    )
    objections: list[str] = Field(
        default_factory=list,
        description="Specific concerns this persona surfaces. At least one required when params.require_objections=True.",
    )
    questions_the_persona_would_ask: list[str] = Field(
        default_factory=list,
        description="Diagnostic questions the persona would ask before adopting. Useful as next-iter validation gates.",
    )


class Claim(BaseModel):
    """An assertion the agent makes that should be traceable to a source.

    Citation pattern adopted from AIQ AI-Q Blueprint. Every claim → source_ref.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The claim itself.")
    source_ref: str = Field(
        description="What the claim is grounded in. For this agent, claims are grounded in the input Bet, "
        "so source_ref is the bet_id. Downstream agents can attach external URLs / signal IDs."
    )
    confidence_0_1: float = Field(ge=0, le=1)


class PersonaSimulatorOutput(BaseModel):
    """Final structured output."""

    model_config = ConfigDict(extra="forbid")

    bet_id: str
    reactions: list[PersonaReaction] = Field(min_length=1)
    aggregate_resistance_pct: float = Field(
        ge=0,
        le=100,
        description="Fraction of personas with overall_stance='resist'. Quick scan number for the UI.",
    )
    aggregate_endorsement_pct: float = Field(ge=0, le=100)
    top_objection_themes: list[str] = Field(
        default_factory=list,
        description="Recurring objection themes across personas. Surfaced to PM as next-validation candidates.",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="All assertions the agent made, with citation chain. Consumed by CitationResolverAgent.",
    )
