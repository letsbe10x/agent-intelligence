"""Typed errors. The framework never silently swallows; every failure has a type."""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all framework errors."""


class ConfigError(AgentError):
    """A YAML config is missing, malformed, or fails schema validation."""


class ProviderError(AgentError):
    """An LLM provider is unreachable, rate-limited, or returned a malformed response."""


class BudgetExceededError(AgentError):
    """The agent would exceed its configured token / dollar budget.

    Raised BEFORE the provider call is made (preflight) so we never spend money
    we cannot account for.
    """


class CitationError(AgentError):
    """A claim's ``source_ref`` cannot be resolved (404, deleted, hash mismatch).

    Raised by CitationResolverAgent when verification fails.
    """


class ReceiptError(AgentError):
    """A receipt fails its hash recompute. Tampering or corruption suspected."""


class RegistryError(AgentError):
    """An agent or provider was requested by name but is not installed."""
