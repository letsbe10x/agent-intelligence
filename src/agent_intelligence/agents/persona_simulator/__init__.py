"""Persona Simulator agent.

Generates N ICP personas and simulates each one reacting to a product Bet.
Output is a list of persona-specific reactions with structured signals
(resistance / agreement / specific objections / specific endorsements) plus
a citation chain back to the input Bet.
"""

from agent_intelligence.agents.persona_simulator.agent import PersonaSimulatorAgent

__all__ = ["PersonaSimulatorAgent"]
