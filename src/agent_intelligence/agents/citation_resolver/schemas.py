"""Pydantic schemas for CitationResolverAgent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClaimToVerify(BaseModel):
    """A claim the upstream agent produced, awaiting verification."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The claim itself.")
    source_ref: str = Field(
        description="What backs the claim. Accepted patterns: "
        "http(s)://..., signal://<id>, assumption://<id>, bet://<id>."
    )
    confidence_0_1: float = Field(ge=0, le=1, default=0.5)


# --- Input ------------------------------------------------------------------


class CitationResolverInput(BaseModel):
    """The list of claims to verify, plus optional context to help resolution."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimToVerify] = Field(min_length=1)
    # Optional lookup tables provided by the caller. When a claim's source_ref
    # uses a custom scheme (signal://, assumption://, bet://), the resolver
    # checks these maps to verify the ID exists.
    known_signal_ids: list[str] = Field(default_factory=list)
    known_assumption_ids: list[str] = Field(default_factory=list)
    known_bet_ids: list[str] = Field(default_factory=list)


# --- Params -----------------------------------------------------------------


class CitationResolverParams(BaseModel):
    """Agent-specific knobs."""

    model_config = ConfigDict(extra="forbid")

    http_timeout_s: float = Field(default=5.0, gt=0, le=30)
    http_max_concurrency: int = Field(default=10, ge=1, le=50)
    require_https_for_external: bool = Field(
        default=True,
        description="If True, http:// URLs (no TLS) are flagged as warnings.",
    )
    use_llm_for_semantic_check: bool = Field(
        default=False,
        description="If True, after resolving the source, an LLM call verifies the source "
        "actually supports the claim semantically. Adds cost; reduces hallucinated citations.",
    )


# --- Output -----------------------------------------------------------------


VerificationStatus = Literal["resolved", "unreachable", "unknown_scheme", "id_not_found", "semantic_mismatch", "warning"]


class CitationVerification(BaseModel):
    """Per-claim verification result."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str
    source_ref: str
    status: VerificationStatus
    detail: str = Field(description="Human-readable reason for the status.")
    resolved_url: str | None = None
    http_status: int | None = None
    content_sha256: str | None = Field(
        default=None,
        description="If the source is a URL and content was fetched, the sha256 of the body. "
        "Lets downstream callers detect content drift on later re-verification.",
    )
    semantic_confidence_0_1: float | None = Field(
        default=None,
        description="LLM-judged confidence that the source actually supports the claim. "
        "None if semantic check disabled.",
    )


class CitationResolverOutput(BaseModel):
    """Aggregate verification report."""

    model_config = ConfigDict(extra="forbid")

    verifications: list[CitationVerification]
    summary: dict[str, int] = Field(
        description="Counts per status, e.g. {'resolved': 5, 'unreachable': 1, 'unknown_scheme': 0}",
    )
    all_passed: bool = Field(
        description="True iff every verification is 'resolved' or 'warning'. False if any fail."
    )
    blocking_failures: int = Field(
        ge=0,
        description="Count of failures that should block the upstream gate "
        "(unreachable, id_not_found, semantic_mismatch). Excludes warnings.",
    )
