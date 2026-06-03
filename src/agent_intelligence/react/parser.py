"""ReAct output parser.

The LLM emits text in this exact format:

    Thought: <reasoning>
    Action: <tool_name>
    Action Input: <JSON>

…OR…

    Thought: <reasoning>
    Final Answer: <JSON>

The parser converts the LLM's text into either ReActAction or ReActFinish. This is
the ONLY routing in the framework. It is NOT an agent decision — it's just reading
the LLM's structured output.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ReActAction:
    """The LLM wants to call a tool."""
    thought: str
    tool: str
    tool_input: dict[str, Any]


@dataclass
class ReActFinish:
    """The LLM is done; here is the final answer."""
    thought: str
    output: dict[str, Any]


class ReActParseError(Exception):
    pass


_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action:|Final Answer:)|\Z)", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(r"Action:\s*([^\n]+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input:\s*(.*?)(?=\n(?:Observation:|Thought:|Action:|Final Answer:)|\Z)", re.DOTALL | re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer:\s*(.*?)$", re.DOTALL | re.IGNORECASE)


def _strip_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"```(?:json)?\s*\n(.*?)\n```", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def _try_json_object(s: str) -> dict[str, Any]:
    """Find the first balanced {...} block that parses. Raises ReActParseError otherwise."""
    s = _strip_fence(s)
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
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
                try:
                    v = json.loads(s[start : i + 1])
                    if isinstance(v, dict):
                        return v
                except json.JSONDecodeError:
                    start = None
    raise ReActParseError(f"Expected a JSON object; got: {s[:200]!r}")


def parse_react(text: str) -> ReActAction | ReActFinish:
    """Parse the LLM's ReAct-formatted output.

    Routing logic — NOT agent decisions, just text inspection:
      - 'Final Answer:' present → ReActFinish
      - 'Action:' + 'Action Input:' present → ReActAction
      - Otherwise → ReActParseError (retry-able by the loop)
    """
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    final_m = _FINAL_RE.search(text)
    if final_m:
        return ReActFinish(thought=thought, output=_try_json_object(final_m.group(1)))

    action_m = _ACTION_RE.search(text)
    input_m = _INPUT_RE.search(text)
    if action_m and input_m:
        tool = action_m.group(1).strip().strip("\"'`")
        tool_input = _try_json_object(input_m.group(1))
        return ReActAction(thought=thought, tool=tool, tool_input=tool_input)

    raise ReActParseError(
        f"Could not parse ReAct step. Need either 'Final Answer: {{json}}' or "
        f"'Action: <tool>\\nAction Input: {{json}}'. Got: {text[:300]!r}"
    )
