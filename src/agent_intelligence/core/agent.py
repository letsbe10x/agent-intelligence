"""Agent — the base class every agent extends.

Design pattern: **Template Method**.

The base class owns the lifecycle (validate → preflight → execute → finalise →
receipt). Subclasses implement only the agent-specific bit (``_execute``).
Cross-cutting concerns (budgets, tracing, receipts, cancellation) live in the
template so individual agents cannot forget them.

Why this pattern: it makes it impossible to write an agent that bypasses budget
checks or skips receipt emission. The framework wins by removing the choice.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from agent_intelligence.core.config import AgentConfig
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import AgentError
from agent_intelligence.observability.budget import BudgetTracker
from agent_intelligence.observability.otel import trace_span
from agent_intelligence.observability.receipts import Receipt, ReceiptStore
from agent_intelligence.providers.base import LLMProvider

# Parametric input/output types so subclasses get typed signatures.
InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class AgentResult(BaseModel, Generic[OutputT]):
    """What every agent run returns.

    Schema:
        output      The agent-specific output (typed by subclass).
        receipt     The provenance proof. Always present, even on failure.
        wallclock_s Real seconds the agent took, including provider RTTs.
        cost_usd    Best-effort cost estimate from the provider.
        tokens_in   Total input tokens across all provider calls.
        tokens_out  Total output tokens.
        model_calls How many provider round-trips the run made.
    """

    output: OutputT
    receipt: Receipt
    wallclock_s: float
    cost_usd: float
    tokens_in: int
    tokens_out: int
    model_calls: int


class Agent(ABC, Generic[InputT, OutputT]):
    """Base class for all agents in the framework.

    Subclasses MUST:
        - Set ``name``, ``InputModel``, ``OutputModel``, ``ParamsModel`` (class attrs).
        - Implement ``_execute(input, params, context, provider)`` returning the OutputT.

    Subclasses MUST NOT:
        - Call providers directly without using the injected ``provider`` arg.
        - Bypass the budget tracker.
        - Mutate ``context``. It is frozen.
    """

    # --- Class-level declarations (subclasses override) ---------------------

    name: str = ""
    """Registered agent name. Must match the entry-point key."""

    InputModel: type[BaseModel]
    """Pydantic model for the agent's input. Validated before ``_execute``."""

    OutputModel: type[BaseModel]
    """Pydantic model for the agent's output. Validated after ``_execute``."""

    ParamsModel: type[BaseModel]
    """Pydantic model for the agent's params (the ``params`` dict in YAML)."""

    # --- Construction -------------------------------------------------------

    def __init__(
        self,
        config: AgentConfig,
        provider: LLMProvider,
        receipt_store: ReceiptStore | None = None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.receipt_store = receipt_store or ReceiptStore(path=None)  # in-memory

        # Validate and freeze the agent-specific params at construction time.
        # Misconfigured agents fail loudly here, not on first run.
        try:
            self.params = self.ParamsModel.model_validate(self.config.params)
        except Exception as e:
            raise AgentError(
                f"Agent {self.name!r}: invalid params under `params:` in config: {e}"
            ) from e

        # Load the prompt template from disk.
        with open(self.config.prompt_path, encoding="utf-8") as f:
            self.prompt_template = f.read()

    # --- Public entry point (the template) ----------------------------------

    async def run(self, input_: InputT | dict[str, Any], context: AgentContext) -> AgentResult[OutputT]:
        """Run the agent. This is the only entry point external callers should use.

        Steps:
            1. Validate input against InputModel.
            2. Preflight budget check (no provider call if budget would be exceeded).
            3. Start OTel span.
            4. Call subclass ``_execute``.
            5. Validate output against OutputModel.
            6. Compute final cost.
            7. Build receipt.
            8. Persist receipt.
            9. Return AgentResult.

        Any exception in steps 1-5 still produces a receipt with status="error" so
        the audit log always has a row.
        """
        if not isinstance(input_, self.InputModel):
            try:
                input_ = self.InputModel.model_validate(input_)
            except Exception as e:
                raise AgentError(f"Invalid input to agent {self.name!r}: {e}") from e

        if context.cancelled:
            raise AgentError(f"Agent {self.name!r} cancelled before start")

        budget = BudgetTracker(
            max_usd=context.budget.max_usd or self.config.budget.max_usd_per_run,
            max_input_tokens=context.budget.max_input_tokens
            or self.config.budget.max_input_tokens_per_run,
            max_output_tokens=context.budget.max_output_tokens
            or self.config.budget.max_output_tokens_per_run,
        )

        start = time.perf_counter()
        status = "ok"
        error_message: str | None = None
        output: OutputT | None = None

        with trace_span(
            name=f"agent.{self.name}.run",
            attributes={
                "agent.name": self.name,
                "agent.version": self.config.version,
                "agent.org_id": context.org_id or "",
                "agent.bet_id": context.bet_id or "",
                "agent.run_id": context.run_id,
                "agent.provider": self.config.provider.name,
                "agent.model": self.config.provider.model,
            },
        ):
            try:
                # Inject budget tracker into the provider for this run.
                provider_with_budget = self.provider.with_budget(budget)
                output = await self._execute(input_, self.params, context, provider_with_budget)

                # Validate the subclass actually returned an OutputModel instance.
                if not isinstance(output, self.OutputModel):
                    try:
                        output = self.OutputModel.model_validate(output)
                    except Exception as e:
                        raise AgentError(
                            f"Agent {self.name!r} returned malformed output: {e}"
                        ) from e
            except Exception as e:
                status = "error"
                error_message = f"{type(e).__name__}: {e}"
                # Re-raise after the receipt is built and persisted in `finally`.
                raise
            finally:
                wallclock = time.perf_counter() - start
                receipt = Receipt.build(
                    agent_name=self.name,
                    agent_version=self.config.version,
                    config_snapshot=self.config.model_dump(exclude={"provider": {"api_key"}}),
                    input_payload=input_.model_dump() if isinstance(input_, BaseModel) else input_,
                    output_payload=(output.model_dump() if isinstance(output, BaseModel) else None),
                    context=context,
                    wallclock_s=wallclock,
                    cost_usd=budget.spent_usd,
                    tokens_in=budget.spent_input_tokens,
                    tokens_out=budget.spent_output_tokens,
                    model_calls=budget.call_count,
                    status=status,
                    error_message=error_message,
                )
                self.receipt_store.put(receipt)

        # Output is non-None here because if _execute raised, we would have
        # re-raised in the except block above.
        assert output is not None
        return AgentResult(
            output=output,
            receipt=receipt,
            wallclock_s=wallclock,
            cost_usd=budget.spent_usd,
            tokens_in=budget.spent_input_tokens,
            tokens_out=budget.spent_output_tokens,
            model_calls=budget.call_count,
        )

    def run_sync(self, input_: InputT | dict[str, Any], context: AgentContext) -> AgentResult[OutputT]:
        """Sync convenience wrapper for callers outside an event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(input_, context))
        else:
            raise AgentError(
                "run_sync called from inside a running event loop. "
                "Use `await agent.run(...)` instead."
            )

    # --- Subclass contract --------------------------------------------------

    @abstractmethod
    async def _execute(
        self,
        input_: InputT,
        params: BaseModel,
        context: AgentContext,
        provider: LLMProvider,
    ) -> OutputT:
        """Subclass implementation. Called once per ``run()``.

        ``provider`` is a budget-bound view of the injected provider — it will
        raise BudgetExceededError if a call would exceed the budget.

        Subclasses may make multiple provider calls. They should poll
        ``context.cancelled`` between calls to support cooperative cancellation.
        """
