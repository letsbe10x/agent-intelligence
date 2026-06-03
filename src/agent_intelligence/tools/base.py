"""Tool ABC + decorator. Tools are the actions an agent can take in a ReAct loop."""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolResult(BaseModel):
    """What a tool returns to the ReAct loop."""

    model_config = ConfigDict(extra="forbid")

    output: dict[str, Any] = Field(description="Tool output as JSON-serialisable dict")
    error: str | None = Field(default=None, description="Set if tool failed; output may be partial")
    cost_usd: float = Field(default=0.0, ge=0)
    wallclock_s: float = Field(default=0.0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Tool(ABC, Generic[InputT, OutputT]):
    """Abstract tool. Subclass + override `run`, or use @tool decorator."""

    name: str = ""
    description: str = ""
    InputModel: type[BaseModel]
    OutputModel: type[BaseModel]
    tags: list[str] = []

    @abstractmethod
    async def run(self, input_: InputT, context: AgentContext) -> OutputT:
        """Execute the tool. Returns a typed OutputT instance."""

    def json_schema(self) -> dict[str, Any]:
        """Expose the input schema to the LLM as JSON schema."""
        return self.InputModel.model_json_schema()

    def to_llm_signature(self) -> dict[str, Any]:
        """Return the OpenAI/Anthropic-style function declaration for tool-calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema(),
        }


def tool(
    name: str,
    description: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    tags: list[str] | None = None,
) -> Callable[[Callable], type[Tool]]:
    """Decorator: wrap an async function into a Tool subclass and auto-register.

    Usage:
        class GetUrlInput(BaseModel):
            url: str
        class GetUrlOutput(BaseModel):
            status: int
            body_preview: str

        @tool("http.get", "Fetch a URL", GetUrlInput, GetUrlOutput)
        async def http_get(input_: GetUrlInput, context: AgentContext) -> GetUrlOutput:
            ...
    """

    def _wrap(fn: Callable) -> type[Tool]:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@tool {name!r}: function must be async")

        class _Decorated(Tool[input_model, output_model]):  # type: ignore[valid-type]
            pass

        _Decorated.name = name
        _Decorated.description = description
        _Decorated.InputModel = input_model
        _Decorated.OutputModel = output_model
        _Decorated.tags = tags or []

        # Bind the user's async function as the `run` method, and clear the
        # ABC's abstractmethod set so the subclass can be instantiated.
        async def _run_proxy(self, input_, context):  # noqa: ANN001
            return await fn(input_, context)

        _Decorated.run = _run_proxy  # type: ignore[assignment]
        _Decorated.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]
        _Decorated.__name__ = fn.__name__
        _Decorated.__qualname__ = fn.__qualname__

        # Auto-register
        from agent_intelligence.tools.registry import tool_registry as _reg
        _reg.register(name, _Decorated())
        return _Decorated

    return _wrap
