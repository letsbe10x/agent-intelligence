"""Citation Resolver agent.

Verifies that every claim in an Evidence artifact has a valid, resolvable
``source_ref`` (URL, signal ID, AssumptionRecord ID). Pattern inspired by
NVIDIA AI-Q's citation-backed research approach.

The resolver does three things:
    1. URL refs    : HTTP HEAD check + (optional) content-hash compare
    2. Signal refs : Lookup against a signal store (caller-supplied)
    3. Other refs  : Lookup against a generic resolver (caller-supplied)

Output is a structured verification report: which claims hold, which broke,
how the broken ones broke. Downstream PVR uses this to either accept the
Evidence at a gate (all green) or block the gate with explicit failure reasons.
"""

from agent_intelligence.agents.citation_resolver.agent import CitationResolverAgent

__all__ = ["CitationResolverAgent"]
