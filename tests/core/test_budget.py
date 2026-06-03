"""Budgets must enforce preflight, not after-the-fact."""

from __future__ import annotations

import pytest

from agent_intelligence.core.errors import BudgetExceededError
from agent_intelligence.observability.budget import BudgetTracker


def test_unlimited_budget_allows_anything():
    b = BudgetTracker()
    b.preflight(estimated_input_tokens=10**9, estimated_output_tokens=10**9, estimated_cost_usd=10**9)
    b.account(input_tokens=1000, output_tokens=2000, cost_usd=100.0)
    assert b.spent_usd == 100.0


def test_input_token_budget_preflight_blocks_call():
    b = BudgetTracker(max_input_tokens=100)
    b.preflight(estimated_input_tokens=50, estimated_output_tokens=0, estimated_cost_usd=0)
    b.account(input_tokens=50, output_tokens=0, cost_usd=0)
    with pytest.raises(BudgetExceededError):
        b.preflight(estimated_input_tokens=60, estimated_output_tokens=0, estimated_cost_usd=0)


def test_output_token_budget_preflight_blocks_call():
    b = BudgetTracker(max_output_tokens=100)
    with pytest.raises(BudgetExceededError):
        b.preflight(estimated_input_tokens=0, estimated_output_tokens=200, estimated_cost_usd=0)


def test_usd_budget_preflight_blocks_call():
    b = BudgetTracker(max_usd=1.0)
    b.account(input_tokens=0, output_tokens=0, cost_usd=0.99)
    with pytest.raises(BudgetExceededError):
        b.preflight(estimated_input_tokens=0, estimated_output_tokens=0, estimated_cost_usd=0.5)


def test_accounting_records_per_call():
    b = BudgetTracker()
    b.account(input_tokens=10, output_tokens=20, cost_usd=0.01)
    b.account(input_tokens=5, output_tokens=10, cost_usd=0.005)
    assert b.spent_input_tokens == 15
    assert b.spent_output_tokens == 30
    assert b.spent_usd == pytest.approx(0.015)
    assert b.call_count == 2
