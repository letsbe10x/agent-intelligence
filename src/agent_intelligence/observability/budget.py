"""BudgetTracker — preflight + per-call accounting of cost/tokens.

Design intent:
    - Budgets are enforced PREFLIGHT. We refuse to make a provider call if the
      call would push us over budget. We do not let the call happen and then
      "discover" we're over.
    - Accounting is per-tracker, not global. One tracker per agent run.
    - Subagents share a tracker by passing the same instance to their providers.

This is a small, deliberately boring object. Keep it that way.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_intelligence.core.errors import BudgetExceededError


@dataclass
class BudgetTracker:
    """Tracks cumulative spend across all provider calls in one agent run."""

    max_usd: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    spent_usd: float = field(default=0.0, init=False)
    spent_input_tokens: int = field(default=0, init=False)
    spent_output_tokens: int = field(default=0, init=False)
    call_count: int = field(default=0, init=False)

    def preflight(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        estimated_cost_usd: float,
    ) -> None:
        """Refuse the call if estimates would exceed budget.

        ``estimated_cost_usd`` is often 0 when called from a provider that
        doesn't know cost until after the response. The token estimates are the
        primary preflight gate in that case.
        """
        if self.max_input_tokens is not None:
            if self.spent_input_tokens + estimated_input_tokens > self.max_input_tokens:
                raise BudgetExceededError(
                    f"Input token budget would be exceeded: "
                    f"spent={self.spent_input_tokens}, "
                    f"+est={estimated_input_tokens}, "
                    f"max={self.max_input_tokens}"
                )
        if self.max_output_tokens is not None:
            if self.spent_output_tokens + estimated_output_tokens > self.max_output_tokens:
                raise BudgetExceededError(
                    f"Output token budget would be exceeded: "
                    f"spent={self.spent_output_tokens}, "
                    f"+est={estimated_output_tokens}, "
                    f"max={self.max_output_tokens}"
                )
        if self.max_usd is not None and estimated_cost_usd > 0:
            if self.spent_usd + estimated_cost_usd > self.max_usd:
                raise BudgetExceededError(
                    f"USD budget would be exceeded: "
                    f"spent=${self.spent_usd:.4f}, "
                    f"+est=${estimated_cost_usd:.4f}, "
                    f"max=${self.max_usd:.4f}"
                )

    def account(
        self,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record actual usage from a completed call.

        Always called after the call returns. If actual cost pushes us over
        the budget after accounting (rare — provider mis-estimated), the NEXT
        preflight will refuse, but this call's response is still surfaced.
        """
        self.spent_input_tokens += input_tokens
        self.spent_output_tokens += output_tokens
        self.spent_usd += cost_usd
        self.call_count += 1
