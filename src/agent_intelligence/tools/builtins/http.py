"""HTTP tools: GET, HEAD, hash-fetch."""
from __future__ import annotations

import hashlib
import time

import httpx
from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.tools.base import tool


class HttpGetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(description="Absolute URL to fetch")
    timeout_s: float = Field(default=5.0, gt=0, le=30)


class HttpGetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    status_code: int
    body_preview: str = Field(description="First 2 KB of response body")
    content_sha256: str
    wallclock_s: float


@tool(
    "http.get",
    "Fetch a URL and return status + body preview + content hash. "
    "Use for citation verification, web research, or fetching external resources.",
    HttpGetInput,
    HttpGetOutput,
    tags=["network", "evidence"],
)
async def http_get(input_: HttpGetInput, context: AgentContext) -> HttpGetOutput:  # noqa: ARG001
    start = time.perf_counter()
    async with httpx.AsyncClient(
        timeout=input_.timeout_s,
        follow_redirects=True,
        headers={"User-Agent": "agent-intelligence/tool.http.get"},
    ) as client:
        resp = await client.get(input_.url)
    elapsed = time.perf_counter() - start
    body_bytes = resp.content[:2048]
    return HttpGetOutput(
        url=str(resp.url),
        status_code=resp.status_code,
        body_preview=body_bytes.decode("utf-8", errors="ignore"),
        content_sha256=hashlib.sha256(resp.content).hexdigest(),
        wallclock_s=elapsed,
    )
