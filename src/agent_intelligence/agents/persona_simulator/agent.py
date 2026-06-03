"""PersonaSimulatorAgent implementation."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from agent_intelligence.agents.persona_simulator.schemas import (
    PersonaSimulatorInput,
    PersonaSimulatorOutput,
    PersonaSimulatorParams,
)
from agent_intelligence.core.agent import Agent
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import AgentError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, Message


class PersonaSimulatorAgent(Agent[PersonaSimulatorInput, PersonaSimulatorOutput]):
    """Generates N ICP personas and simulates each reacting to a Bet hypothesis.

    Design notes:
        - One LLM call. We coax the model into producing all personas in a
          single structured JSON response. This keeps cost bounded and
          deterministic for receipt purposes.
        - Output is strictly schema-validated. A malformed JSON response or a
          schema violation raises AgentError — we do not "best effort".
        - All assertions get added to the `claims` array with `source_ref`
          pointing back to the Bet — feeds directly into CitationResolverAgent.
    """

    name = "persona_simulator"
    InputModel = PersonaSimulatorInput
    OutputModel = PersonaSimulatorOutput
    ParamsModel = PersonaSimulatorParams

    async def _execute(
        self,
        input_: PersonaSimulatorInput,
        params: BaseModel,
        context: AgentContext,
        provider: LLMProvider,
    ) -> PersonaSimulatorOutput:
        assert isinstance(params, PersonaSimulatorParams)

        target_persona_section = (
            f"\n- **Target persona brief:** {input_.target_persona_brief}\n"
            if input_.target_persona_brief
            else ""
        )

        prompt = self.prompt_template.format(
            n_personas=params.n_personas,
            reaction_depth=params.reaction_depth,
            hypothesis=input_.hypothesis,
            success_metric=input_.success_metric,
            time_box=input_.time_box,
            bet_id=input_.bet_id,
            target_persona_section=target_persona_section,
        )

        if context.cancelled:
            raise AgentError("persona_simulator cancelled before model call")

        response = await provider.acomplete(
            LLMRequest(
                messages=[
                    Message(role="system", content="You produce strictly schema-conforming JSON. No prose outside JSON."),
                    Message(role="user", content=prompt),
                ],
            )
        )

        return self._parse_and_validate(response.content, input_, params)

    # --- Output parsing -----------------------------------------------------

    def _parse_and_validate(
        self,
        raw: str,
        input_: PersonaSimulatorInput,
        params: PersonaSimulatorParams,
    ) -> PersonaSimulatorOutput:
        """Parse the model's JSON output, validate schema, enforce doctrine constraints."""
        payload = self._extract_json(raw)

        try:
            output = PersonaSimulatorOutput.model_validate(payload)
        except Exception as e:
            raise AgentError(
                f"persona_simulator: model returned JSON that fails schema validation: {e}. "
                f"Raw output (first 500 chars): {raw[:500]!r}"
            ) from e

        # Doctrine enforcement: if require_objections=True, every persona must surface
        # at least one objection. The model occasionally drifts on this; we catch it.
        if params.require_objections:
            bad = [r.persona_name for r in output.reactions if not r.objections]
            if bad:
                raise AgentError(
                    f"persona_simulator: require_objections=True but {bad} surfaced no objections. "
                    "This is a Yes-Man simulation; reject and retry."
                )

        # Sanity: aggregate percentages must match the counts.
        n = len(output.reactions)
        resist_n = sum(1 for r in output.reactions if r.overall_stance == "resist")
        endorse_n = sum(1 for r in output.reactions if r.overall_stance == "endorse")
        expected_resist_pct = (resist_n / n) * 100 if n > 0 else 0.0
        expected_endorse_pct = (endorse_n / n) * 100 if n > 0 else 0.0
        # Tolerance for rounding in model output.
        if abs(output.aggregate_resistance_pct - expected_resist_pct) > 5:
            # Repair rather than reject — the structured data is the source of truth.
            output = output.model_copy(update={"aggregate_resistance_pct": expected_resist_pct})
        if abs(output.aggregate_endorsement_pct - expected_endorse_pct) > 5:
            output = output.model_copy(update={"aggregate_endorsement_pct": expected_endorse_pct})

        # Enforce bet_id consistency. If the model echoed a different one, fix it.
        if output.bet_id != input_.bet_id:
            output = output.model_copy(update={"bet_id": input_.bet_id})

        return output

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Find a JSON object inside the model's response.

        Models sometimes wrap JSON in ```json fences despite being asked not to,
        or add a sentence before/after. This is the robust extractor.
        """
        s = raw.strip()
        # Strip code fences if present.
        fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", s, re.DOTALL)
        if fence:
            s = fence.group(1)

        # Fast path: the whole thing is valid JSON.
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        # Slow path: find the first { ... } block that parses.
        depth = 0
        start = None
        for i, c in enumerate(s):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue

        raise AgentError(
            f"persona_simulator: could not extract JSON from model output. "
            f"First 500 chars: {raw[:500]!r}"
        )
