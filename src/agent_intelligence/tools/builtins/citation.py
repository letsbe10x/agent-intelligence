"""Citation aggregation tool — finalise a verification report."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.tools.base import tool


class FinalizeCitationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verifications: list[dict[str, Any]] = Field(
        description="One verification per claim. Each has at least: claim, source_ref, "
        "status (resolved|unreachable|id_not_found|semantic_mismatch|unknown_scheme), detail."
    )


class FinalizeCitationsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verifications: list[dict[str, Any]]
    summary: dict[str, int]
    all_passed: bool
    blocking_failures: int


@tool(
    "citation.finalize",
    "Aggregate per-claim verifications into a final report: counts per status, "
    "all_passed boolean, blocking_failures count. Call once at the end of citation "
    "resolution after all individual claims have been verified.",
    FinalizeCitationsInput,
    FinalizeCitationsOutput,
)
async def citation_finalize(
    input_: FinalizeCitationsInput, context: AgentContext
) -> FinalizeCitationsOutput:
    from agent_intelligence.tools.builtins._llm import call_llm_json

    prompt = (
        f"Aggregate these per-claim verifications. Compute counts per status, "
        f"set all_passed=true only if every claim is 'resolved' or 'warning', and "
        f"count blocking_failures (statuses in: unreachable, id_not_found, "
        f"semantic_mismatch, unknown_scheme).\n\n"
        f"Verifications:\n{json.dumps(input_.verifications, indent=2)}\n\n"
        f"Return JSON with keys: verifications (passthrough), summary (object: status -> count), "
        f"all_passed (bool), blocking_failures (int)."
    )
    data = await call_llm_json(context, prompt)
    data.setdefault("verifications", input_.verifications)
    return FinalizeCitationsOutput.model_validate(data)
