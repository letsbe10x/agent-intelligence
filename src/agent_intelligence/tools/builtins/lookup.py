"""Lookup tools — resolve custom scheme refs (signal://, assumption://, bet://)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.tools.base import tool


class LookupIdInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheme: str = Field(description="One of: signal, assumption, bet")
    identifier: str = Field(description="The ID to look up")


class LookupIdOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    found: bool
    scheme: str
    identifier: str
    detail: str


@tool(
    "lookup.id",
    "Look up a custom-scheme reference (signal://, assumption://, bet://) against "
    "the caller-supplied known-ID set. Returns whether the ID exists. "
    "Use during citation verification when a claim's source_ref uses a non-URL scheme.",
    LookupIdInput,
    LookupIdOutput,
    tags=["lookup", "citation"],
)
async def lookup_id(input_: LookupIdInput, context: AgentContext) -> LookupIdOutput:
    # The known-ID sets are passed through context.metadata at run time.
    known_key = f"known_{input_.scheme}_ids"
    known = set(context.metadata.get(known_key) or [])
    found = input_.identifier in known
    return LookupIdOutput(
        found=found,
        scheme=input_.scheme,
        identifier=input_.identifier,
        detail=(
            f"{input_.scheme} ID {input_.identifier!r} found in known set."
            if found
            else f"{input_.scheme} ID {input_.identifier!r} NOT in known set "
            f"(caller supplied {len(known)} IDs for this scheme)."
        ),
    )
