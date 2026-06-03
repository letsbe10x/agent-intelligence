"""Global plugin registry.

Two namespaces:
    - agents     : maps name → Agent subclass
    - providers  : maps name → LLMProvider subclass

Discovery:
    1. At first use, scan entry points in groups
       ``agent_intelligence.agents`` and ``agent_intelligence.providers``.
    2. Lazy-load each entry point only when its name is requested.
    3. Manual registration is also supported (for tests + dynamic plugins).
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from agent_intelligence.core.errors import RegistryError

if TYPE_CHECKING:
    from agent_intelligence.core.agent import Agent
    from agent_intelligence.providers.base import LLMProvider


class _Registry:
    """Namespaced registry. Used as a singleton at module level."""

    _AGENT_GROUP = "agent_intelligence.agents"
    _PROVIDER_GROUP = "agent_intelligence.providers"

    def __init__(self) -> None:
        self._agents: dict[str, type[Agent]] = {}
        self._providers: dict[str, type[LLMProvider]] = {}
        self._discovered = False

    # --- Manual registration (tests, dynamic plugins) ----------------------

    def register_agent(self, name: str, cls: type[Agent]) -> None:
        self._agents[name] = cls

    def register_provider(self, name: str, cls: type[LLMProvider]) -> None:
        self._providers[name] = cls

    # --- Discovery ----------------------------------------------------------

    def _discover(self) -> None:
        """Scan entry points once. Lazy — won't import classes here."""
        if self._discovered:
            return

        try:
            for ep in entry_points(group=self._AGENT_GROUP):
                if ep.name not in self._agents:
                    self._agents[ep.name] = _LazyClass(ep)  # type: ignore[assignment]
            for ep in entry_points(group=self._PROVIDER_GROUP):
                if ep.name not in self._providers:
                    self._providers[ep.name] = _LazyClass(ep)  # type: ignore[assignment]
        except Exception:
            # Entry-point discovery is best-effort. Manually-registered plugins
            # still work even if entry-point scanning fails (e.g., broken meta).
            pass
        self._discovered = True

    # --- Lookup -------------------------------------------------------------

    def get_agent(self, name: str) -> type[Agent]:
        self._discover()
        if name not in self._agents:
            raise RegistryError(
                f"Agent {name!r} is not registered. "
                f"Available: {sorted(self._agents.keys()) or '(none)'}. "
                "Install the package that provides it, or register manually with "
                "registry.register_agent(name, cls)."
            )
        cls = self._agents[name]
        if isinstance(cls, _LazyClass):
            real = cls.load()
            self._agents[name] = real
            return real
        return cls

    def get_provider(self, name: str) -> type[LLMProvider]:
        self._discover()
        if name not in self._providers:
            raise RegistryError(
                f"Provider {name!r} is not registered. "
                f"Available: {sorted(self._providers.keys()) or '(none)'}. "
                "Install the package that provides it, or register manually with "
                "registry.register_provider(name, cls)."
            )
        cls = self._providers[name]
        if isinstance(cls, _LazyClass):
            real = cls.load()
            self._providers[name] = real
            return real
        return cls

    def list_agents(self) -> list[str]:
        self._discover()
        return sorted(self._agents.keys())

    def list_providers(self) -> list[str]:
        self._discover()
        return sorted(self._providers.keys())


class _LazyClass:
    """A deferred reference to a class loaded from an entry point."""

    def __init__(self, ep) -> None:
        self._ep = ep

    def load(self) -> type:
        return self._ep.load()


registry = _Registry()
"""Singleton registry instance. Import as ``from agent_intelligence import registry``."""
