"""Verification tools — semantic check (does source actually support the claim?)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.tools.base import tool


class SemanticVerifyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str
    source_excerpt: str = Field(description="Excerpt of the source content to check against")


class SemanticVerifyOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supports: bool
    confidence_0_1: float
    reason: str


@tool(
    "verify.semantic",
    "Check whether a source excerpt actually supports a specific claim. Returns "
    "supports/confidence/reason. Use after fetching a source via http.get to "
    "determine if the citation is meaningful or hallucinated. Be conservative — "
    "supports=true only if the source contains direct evidence for the claim.",
    SemanticVerifyInput,
    SemanticVerifyOutput,
)
async def verify_semantic(
    input_: SemanticVerifyInput, context: AgentContext
) -> SemanticVerifyOutput:
    from agent_intelligence.tools.builtins._llm import call_llm_json

    prompt = (
        f"Verify whether the source supports the claim.\n\n"
        f"CLAIM:\n{input_.claim}\n\n"
        f"SOURCE EXCERPT:\n{input_.source_excerpt[:6000]}\n\n"
        f"Return JSON with keys: supports (bool), confidence_0_1 (float), reason "
        f"(1-2 sentence explanation). Be conservative: supports=true ONLY if the source "
        f"contains direct evidence for the claim. Inferred support is not support."
    )
    data = await call_llm_json(context, prompt)
    return SemanticVerifyOutput.model_validate(data)
