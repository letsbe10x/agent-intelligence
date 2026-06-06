"""SkillHubToolSource — wraps letsbe10x/skill-hub skills as agent tools.

Each ``lets-*/SKILL.md`` is converted into a Tool:
  - tool name    : skill.<skill-name>  (e.g. skill.lets-research-competitive-scan)
  - description  : the skill's frontmatter ``description``
  - input schema : { goal: str, context: str? }  (LLM-friendly generic shape)
  - run()        : either shells out to ``lets <skill>`` if available, or
                   uses an LLM to act according to the skill's SKILL.md content

The latter (LLM as skill runner) is the default and keeps the source pure-Python
+ self-contained for the demo. Production deployments can shell out.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_intelligence.core.context import AgentContext
from agent_intelligence.providers.base import LLMProvider, LLMRequest, Message
from agent_intelligence.tools.base import Tool
from agent_intelligence.tools.sources.base import ToolSource


class SkillToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(description="What you want this skill to accomplish.")
    context: str = Field(default="", description="Optional grounding context (prior results, source data).")


class SkillToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: str
    result: str = Field(description="The skill's summarised output as the LLM produced it.")
    confidence_0_1: float = Field(ge=0, le=1, default=0.5)


def _read_skill_md(path: Path) -> dict:
    """Parse SKILL.md frontmatter + body."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        return {"_frontmatter": {}, "_body": text}
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        fm = {}
    return {"_frontmatter": fm, "_body": m.group(2)}


class _SkillTool(Tool):
    """Generic LLM-driven wrapper around a skill. Shared shape across all skills."""

    InputModel = SkillToolInput
    OutputModel = SkillToolOutput

    def __init__(self, skill_name: str, description: str, skill_md_body: str) -> None:
        self.name = f"skill.{skill_name}"
        self.description = description or f"Invoke the {skill_name!r} skill from skill-hub."
        self._skill_name = skill_name
        self._body = skill_md_body
        self.tags = ["skill", "skillhub"]

    async def run(self, input_: SkillToolInput, context: AgentContext) -> SkillToolOutput:
        provider = context.metadata.get("_provider")
        if not isinstance(provider, LLMProvider):
            return SkillToolOutput(
                skill=self._skill_name,
                result=(
                    f"[stub] skill {self._skill_name!r} would run with goal: "
                    f"{input_.goal!r}. No LLM provider attached to context."
                ),
                confidence_0_1=0.0,
            )

        # Use the skill's own SKILL.md content as the system prompt; let the LLM
        # produce the result the skill is supposed to produce.
        system = (
            f"You ARE the {self._skill_name!r} skill. Your full instructions live "
            f"in the markdown below. When called, execute the skill's job on the "
            f"caller's goal and return a structured summary of what you did + the "
            f"result.\n\n"
            f"## SKILL.md\n\n{self._body[:6000]}\n\n"
            f"# Output\n"
            f'Return JSON only: {{"skill": "{self._skill_name}", '
            f'"result": "<your result as plain text>", "confidence_0_1": <0..1>}}.'
        )
        user = f"GOAL: {input_.goal}\nCONTEXT: {input_.context or '(none)'}"
        response = await provider.acomplete(
            LLMRequest(
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=user),
                ]
            )
        )
        from agent_intelligence.tools.builtins._llm import _extract_json
        try:
            data = _extract_json(response.content)
        except Exception:
            data = {"skill": self._skill_name, "result": response.content[:1200], "confidence_0_1": 0.3}
        data.setdefault("skill", self._skill_name)
        return SkillToolOutput.model_validate(data)


class SkillHubToolSource(ToolSource):
    """Discovers skill-hub skills and exposes each as a tool.

    Config:
      path:      filesystem path to skill-hub (default: ~/lets/skill-hub)
      allowlist: optional list of skill names to include (lets-research-*, etc.)
      exclude:   optional list of skill names to skip
    """

    type_name = "skillhub"

    async def discover(self) -> list[Tool]:
        cfg = self.config.config
        root = Path(cfg.get("path") or str(Path.home() / "lets" / "skill-hub"))
        if not root.is_dir():
            return []
        allowlist = set(cfg.get("allowlist") or [])
        exclude = set(cfg.get("exclude") or [])

        out: list[Tool] = []
        skill_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("lets-"))
        # Read in parallel
        loop = asyncio.get_event_loop()
        results = await asyncio.gather(
            *(loop.run_in_executor(None, _read_skill_md, d / "SKILL.md") for d in skill_dirs if (d / "SKILL.md").is_file())
        )
        for sd, parsed in zip(skill_dirs, results, strict=False):
            name = sd.name  # e.g. "lets-research-competitive-scan"
            if allowlist and name not in allowlist:
                continue
            if name in exclude:
                continue
            fm = parsed.get("_frontmatter") or {}
            description = (fm.get("description") or fm.get("summary") or "")[:600]
            out.append(_SkillTool(skill_name=name, description=description, skill_md_body=parsed.get("_body") or ""))
        return out
