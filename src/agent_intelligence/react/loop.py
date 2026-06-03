"""The ReAct loop. The agent's brain is the LLM; this is just plumbing."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import AgentError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, Message
from agent_intelligence.react.parser import (
    ReActAction,
    ReActFinish,
    ReActParseError,
    parse_react,
)
from agent_intelligence.tools.base import Tool
from agent_intelligence.tracing.events import TraceEmitter, TraceEvent


@dataclass
class ReActStep:
    """One iteration of the loop. Emitted as a trace event and stored in the receipt."""
    iteration: int
    kind: str  # "think_act" | "observation" | "final" | "parse_retry"
    thought: str | None = None
    tool: str | None = None
    tool_input: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    raw_llm_output: str | None = None
    error: str | None = None
    wallclock_s: float = 0.0


@dataclass
class ReActOutcome:
    """What the loop produces."""
    final_output: dict[str, Any]
    steps: list[ReActStep] = field(default_factory=list)
    iterations: int = 0
    halted_reason: str = "final_answer"  # final_answer | max_iterations | parse_failures


def _build_tool_catalog(tools: dict[str, Tool]) -> str:
    """Render the tool list as text for the LLM."""
    lines = []
    for name, t in tools.items():
        schema = json.dumps(t.json_schema(), separators=(",", ":"))
        lines.append(f"- {name}: {t.description}\n  Input schema: {schema}")
    return "\n".join(lines)


def _build_system_prompt(user_system: str, tools: dict[str, Tool]) -> str:
    catalog = _build_tool_catalog(tools)
    return f"""{user_system}

You are an agent that solves tasks by reasoning step-by-step and calling tools.

# Available tools
{catalog}

# Output format (STRICT)
On every step, emit EXACTLY ONE of these two structures.

To call a tool:
    Thought: <your reasoning about what to do next and why>
    Action: <one tool name from the list above>
    Action Input: <a JSON object that matches that tool's input schema>

When you have enough information to answer the user, emit:
    Thought: <your reasoning about the final result>
    Final Answer: <a JSON object with the structured final output>

Rules:
- Emit ONE step at a time. Wait for the Observation after each tool call.
- Action Input MUST be valid JSON, matching the tool's schema exactly.
- Final Answer MUST be a JSON object (not prose).
- Do not invent tools that are not in the list.
- Do not skip the Thought line.
- When you have enough evidence, stop and emit Final Answer. Do not call extra tools.
"""


class ReActLoop:
    """LLM-driven loop. The LLM picks tools and decides when to stop."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tools: dict[str, Tool],
        system_prompt: str,
        max_iterations: int = 12,
        max_parse_retries: int = 2,
        emitter: TraceEmitter | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_parse_retries = max_parse_retries
        self.emitter = emitter or TraceEmitter()

    async def run(self, user_input: str | dict[str, Any], context: AgentContext) -> ReActOutcome:
        """Run the loop until the LLM emits Final Answer, we hit max_iterations,
        or parsing fails too many times. Returns ReActOutcome."""
        outcome = ReActOutcome(final_output={}, halted_reason="")

        # Inject the provider into context so tools can use it via the helper.
        # Tools never construct their own provider.
        ctx = context.with_overrides(
            metadata={**context.metadata, "_provider": self.provider}
        )

        scratchpad: list[Message] = []
        scratchpad.append(
            Message(
                role="user",
                content=user_input if isinstance(user_input, str) else json.dumps(user_input, indent=2),
            )
        )

        parse_failures = 0
        await self.emitter.emit(
            TraceEvent(
                kind="loop.start",
                run_id=ctx.run_id,
                data={
                    "max_iterations": self.max_iterations,
                    "tool_names": list(self.tools.keys()),
                },
            )
        )

        for iteration in range(1, self.max_iterations + 1):
            outcome.iterations = iteration

            t0 = time.perf_counter()
            response = await self.provider.acomplete(
                LLMRequest(
                    messages=[Message(role="system", content=self.system_prompt), *scratchpad]
                )
            )
            llm_wall = time.perf_counter() - t0

            await self.emitter.emit(
                TraceEvent(
                    kind="llm.response",
                    run_id=ctx.run_id,
                    data={
                        "iteration": iteration,
                        "wallclock_s": llm_wall,
                        "preview": response.content[:300],
                        "tokens_in": response.input_tokens,
                        "tokens_out": response.output_tokens,
                    },
                )
            )

            # Try to parse. The LLM either gives us a tool call or a final answer.
            try:
                parsed = parse_react(response.content)
            except ReActParseError as e:
                parse_failures += 1
                step = ReActStep(
                    iteration=iteration,
                    kind="parse_retry",
                    raw_llm_output=response.content,
                    error=str(e),
                    wallclock_s=llm_wall,
                )
                outcome.steps.append(step)
                await self.emitter.emit(
                    TraceEvent(kind="parse.retry", run_id=ctx.run_id, data=step.__dict__)
                )
                # Push a corrective message into the scratchpad and let the LLM try again.
                scratchpad.append(
                    Message(role="assistant", content=response.content)
                )
                scratchpad.append(
                    Message(
                        role="user",
                        content=(
                            f"Your previous response could not be parsed: {e}\n"
                            f"Re-emit your step using the exact required format. "
                            f"Either a tool call (Thought / Action / Action Input) or "
                            f"a Final Answer."
                        ),
                    )
                )
                if parse_failures > self.max_parse_retries:
                    outcome.halted_reason = "parse_failures"
                    break
                continue

            # Routing on what the LLM emitted. NOT an agent decision — just inspecting LLM output.
            match parsed:
                case ReActFinish(thought=thought, output=output):
                    step = ReActStep(
                        iteration=iteration,
                        kind="final",
                        thought=thought,
                        raw_llm_output=response.content,
                        wallclock_s=llm_wall,
                    )
                    outcome.steps.append(step)
                    outcome.final_output = output
                    outcome.halted_reason = "final_answer"
                    await self.emitter.emit(
                        TraceEvent(kind="final.answer", run_id=ctx.run_id, data={"output": output})
                    )
                    return outcome

                case ReActAction(thought=thought, tool=tool_name, tool_input=tool_input):
                    step = ReActStep(
                        iteration=iteration,
                        kind="think_act",
                        thought=thought,
                        tool=tool_name,
                        tool_input=tool_input,
                        raw_llm_output=response.content,
                        wallclock_s=llm_wall,
                    )
                    outcome.steps.append(step)
                    await self.emitter.emit(
                        TraceEvent(
                            kind="tool.start",
                            run_id=ctx.run_id,
                            data={"tool": tool_name, "input": tool_input, "thought": thought},
                        )
                    )

                    scratchpad.append(Message(role="assistant", content=response.content))

                    # Dispatch. If the tool name isn't registered, feed that fact back to the LLM
                    # so it picks a real tool — do not raise.
                    tool = self.tools.get(tool_name)
                    if tool is None:
                        observation = {
                            "error": f"Unknown tool {tool_name!r}. Pick one of: {list(self.tools.keys())}"
                        }
                    else:
                        try:
                            validated_input = tool.InputModel.model_validate(tool_input)
                            t1 = time.perf_counter()
                            tool_output = await tool.run(validated_input, ctx)
                            tool_wall = time.perf_counter() - t1
                            observation = {
                                "ok": True,
                                "output": tool_output.model_dump() if hasattr(tool_output, "model_dump") else tool_output,
                                "wallclock_s": tool_wall,
                            }
                        except Exception as e:  # noqa: BLE001 — observation, not crash
                            observation = {"ok": False, "error": f"{type(e).__name__}: {e}"}

                    obs_step = ReActStep(
                        iteration=iteration,
                        kind="observation",
                        tool=tool_name,
                        observation=observation,
                    )
                    outcome.steps.append(obs_step)
                    await self.emitter.emit(
                        TraceEvent(
                            kind="tool.end",
                            run_id=ctx.run_id,
                            data={"tool": tool_name, "observation": observation},
                        )
                    )

                    scratchpad.append(
                        Message(
                            role="user",
                            content=f"Observation: {json.dumps(observation, indent=2)}",
                        )
                    )
                    continue

                case _:
                    raise AgentError(f"Unreachable: unknown parse result {parsed!r}")

        # Loop ran out of iterations without a final answer.
        if not outcome.halted_reason:
            outcome.halted_reason = "max_iterations"
        await self.emitter.emit(
            TraceEvent(
                kind="loop.halt",
                run_id=ctx.run_id,
                data={"reason": outcome.halted_reason, "iterations": outcome.iterations},
            )
        )
        return outcome
