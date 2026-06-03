"""AgentContext — the runtime envelope every agent execution carries.

Threaded through every Agent.run() call. Holds:
    - tenant identity (org_id, user_id) for multi-tenant accounting
    - linked PVR object IDs (bet_id, run_id) for receipt provenance
    - budget caps (per-org $, per-run $)
    - cancellation flag (cooperative)
    - trace context (OTel span)

Immutable. Mutations produce a new context via ``with_overrides()``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Budget:
    """Money + tokens. Either limit can be set, or both, or neither (unlimited)."""

    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def is_unlimited(self) -> bool:
        return all(x is None for x in (self.max_usd, self.max_input_tokens, self.max_output_tokens))


@dataclass(frozen=True)
class AgentContext:
    """Per-invocation envelope. Frozen — use ``with_overrides()`` to derive."""

    # Tenant identity (caller). Multi-tenant systems use this for routing + audit.
    org_id: str | None = None
    user_id: str | None = None

    # PVR object linking. When this agent runs as part of a Bet's validation pack,
    # bet_id is set so receipts can be reverse-indexed.
    bet_id: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Cost / size guardrails. Enforced PREFLIGHT in the provider layer.
    budget: Budget = field(default_factory=Budget)

    # Cooperative cancellation. Long-running agents should poll this between steps.
    # Tests can set this True to verify graceful abort.
    cancelled: bool = False

    # OpenTelemetry trace correlation. Producers set this; emitters consume.
    trace_id: str | None = None
    span_id: str | None = None

    # Free-form correlation tags. Surface in receipts. Useful for filtering audit logs.
    tags: dict[str, str] = field(default_factory=dict)

    # Arbitrary caller-supplied metadata, never used by the framework itself.
    # Carried verbatim into the Receipt. Subject to size limits (32KB serialised).
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> AgentContext:
        """Derive a new context with selected fields overridden. Immutable update."""
        return replace(self, **kwargs)

    def child(self, **kwargs: Any) -> AgentContext:
        """Spawn a child context with a fresh ``run_id`` and the parent linked in tags."""
        new_tags = {**self.tags, "parent_run_id": self.run_id}
        return replace(self, run_id=str(uuid.uuid4()), tags={**new_tags, **kwargs.pop("tags", {})}, **kwargs)
