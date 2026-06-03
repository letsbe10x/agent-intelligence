"""Entry-point-based plugin discovery.

Agents and providers are discovered via Python entry points. This means:
    - Third parties publish their own package with an entry point and it
      auto-registers when installed.
    - Users do not have to import specific agent classes; they reference them
      by name in YAML.
    - Tests can register lightweight in-process implementations without
      packaging.
"""

from agent_intelligence.registry.registry import registry

__all__ = ["registry"]
