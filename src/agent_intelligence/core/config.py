"""Config — YAML-driven agent configuration.

Design intent:
    1. **No hardcoded defaults inside agents.** Every parameter that affects
       behaviour (model, temperature, prompt path, retry count, budget, tools)
       lives in YAML. Code reads from the config; it does not "fall back to a
       sensible default".

    2. **Schema is pydantic.** Malformed config is rejected at load time with a
       precise error pointing at the offending field. Not at run time.

    3. **Composition via ``include``.** An org-level config can include a
       baseline; per-bet runs can include the org-level. Order is resolved
       deterministically (last-write-wins on conflict, with the override path
       recorded).

    4. **Secrets via env-var indirection.** Config never contains a literal
       secret. ``api_key: ${env:ANTHROPIC_API_KEY}`` is resolved at load.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_intelligence.core.errors import ConfigError

# --- Provider sub-config ----------------------------------------------------


class ProviderConfig(BaseModel):
    """How to talk to one specific LLM provider."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Registered provider name, e.g. 'anthropic', 'openai', 'litellm'.")
    model: str = Field(description="Model identifier, e.g. 'claude-sonnet-4.7'.")
    api_key: str | None = Field(default=None, description="Resolved at load time from env.")
    base_url: str | None = Field(default=None, description="Override default endpoint (proxy, local).")
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_output_tokens: int | None = Field(default=None, gt=0)
    # Extra provider-specific kwargs (e.g. {"top_p": 0.9}). Passed through verbatim.
    extra: dict[str, Any] = Field(default_factory=dict)


# --- Budget sub-config ------------------------------------------------------


class BudgetConfig(BaseModel):
    """Cost guardrails. Enforced preflight by the provider layer."""

    model_config = ConfigDict(extra="forbid")

    max_usd_per_run: float | None = Field(default=None, gt=0)
    max_input_tokens_per_run: int | None = Field(default=None, gt=0)
    max_output_tokens_per_run: int | None = Field(default=None, gt=0)
    on_exceed: Literal["raise", "truncate"] = Field(default="raise")


# --- Observability sub-config -----------------------------------------------


class ObservabilityConfig(BaseModel):
    """OTel + Receipts settings."""

    model_config = ConfigDict(extra="forbid")

    otel_enabled: bool = Field(default=True)
    otel_service_name: str = Field(default="agent-intelligence")
    receipts_enabled: bool = Field(default=True)
    receipts_path: str | None = Field(
        default=None,
        description="Local dir for receipt files. None = no on-disk persistence (in-memory only).",
    )


# --- Agent config (top-level) -----------------------------------------------


class AgentConfig(BaseModel):
    """Top-level agent configuration.

    One YAML file produces one AgentConfig. The same shape is used for all agents;
    each agent declares its own ``params`` schema for agent-specific knobs.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Registered agent name (entry point key).")
    version: str = Field(default="1", description="Config schema version for the agent.")

    provider: ProviderConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Path to the prompt template file (relative to the config file).
    # Externalised so prompt edits don't require code changes.
    prompt_path: str = Field(description="Filesystem path to the prompt template file.")

    # Agent-specific parameters. The agent's pydantic Params class validates this.
    # Kept as a free dict here; the agent calls Params.model_validate(params) on it.
    params: dict[str, Any] = Field(default_factory=dict)

    # Optional list of tools the agent may call (declarative). Tools resolved by
    # the agent at run time from the registry.
    tools: list[str] = Field(default_factory=list)

    @field_validator("prompt_path")
    @classmethod
    def _strip_prompt_path(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt_path must not be empty")
        return v


# --- Loader -----------------------------------------------------------------


_ENV_REF = re.compile(r"\$\{env:([A-Z][A-Z0-9_]*)(?::-(.*?))?\}")


def _resolve_env_refs(value: Any) -> Any:
    """Recursively resolve ``${env:VAR}`` and ``${env:VAR:-default}`` references."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            v = os.environ.get(var)
            if v is None:
                if default is not None:
                    return default
                raise ConfigError(
                    f"Config references env var ${{{var}}} but it is not set "
                    "and no default was provided. Set the env var, or change the "
                    "config to use ${env:VAR:-default}."
                )
            return v

        return _ENV_REF.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_refs(v) for v in value]
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge. Lists are replaced, not concatenated; dicts are merged recursively."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_with_includes(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load YAML, recursively applying ``include`` directives. Detect cycles."""
    seen = seen or set()
    real = path.resolve()
    if real in seen:
        raise ConfigError(f"Config include cycle detected at {real}")
    seen.add(real)

    if not real.is_file():
        raise ConfigError(f"Config file not found: {real}")

    try:
        raw = yaml.safe_load(real.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {real}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping in {real}, got {type(raw).__name__}")

    includes = raw.pop("include", []) or []
    if not isinstance(includes, list):
        raise ConfigError(f"`include` must be a list of paths in {real}")

    merged: dict[str, Any] = {}
    for inc in includes:
        inc_path = (real.parent / inc).resolve()
        merged = _merge(merged, _load_with_includes(inc_path, seen))
    merged = _merge(merged, raw)
    return merged


def load_config(path: str | Path) -> AgentConfig:
    """Load an AgentConfig from a YAML path.

    The path can include ``include:`` directives pointing at other YAML files
    (composition). Env-var references like ``${env:VAR}`` are resolved at load
    time. Pydantic validates the final shape and raises ``ConfigError`` on any
    schema violation, with field paths in the message.

    The prompt path is resolved relative to the config file's directory.
    """
    path = Path(path)
    raw = _load_with_includes(path)
    raw = _resolve_env_refs(raw)

    try:
        cfg = AgentConfig.model_validate(raw)
    except Exception as e:
        raise ConfigError(f"Config validation failed for {path}: {e}") from e

    # Resolve prompt_path relative to the config file directory.
    if not Path(cfg.prompt_path).is_absolute():
        cfg = cfg.model_copy(update={"prompt_path": str((path.parent / cfg.prompt_path).resolve())})

    if not Path(cfg.prompt_path).is_file():
        raise ConfigError(
            f"Prompt file referenced by config does not exist: {cfg.prompt_path}"
        )

    return cfg
