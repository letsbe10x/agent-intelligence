"""ReAct loop primitive — think → act → observe, driven entirely by the LLM.

Pattern (mirrors NVIDIA AgentIQ react_agent):
    1. Send LLM: system prompt + tool catalog + conversation so far
    2. Parse LLM output: did it emit Final Answer, or Action+Action Input?
    3. If Final Answer → END
    4. If Action → dispatch tool, observe, append to scratchpad, loop
    5. If parse fails N times → return whatever we have

The agent never decides what to do via Python conditionals. The agent decides
by emitting structured text that the parser routes. The framework code only
asks: "did the LLM say it's done, or did it ask to call a tool?"
"""
from agent_intelligence.react.loop import ReActLoop, ReActOutcome
from agent_intelligence.react.parser import ReActAction, ReActFinish, ReActParseError, parse_react

__all__ = [
    "ReActAction",
    "ReActFinish",
    "ReActLoop",
    "ReActOutcome",
    "ReActParseError",
    "parse_react",
]
