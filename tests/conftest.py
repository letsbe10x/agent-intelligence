"""Test fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_intelligence.core.context import AgentContext


@pytest.fixture
def tmp_receipts(tmp_path: Path) -> Path:
    d = tmp_path / "receipts"
    d.mkdir()
    return d


@pytest.fixture
def context() -> AgentContext:
    return AgentContext(org_id="acme", bet_id="bet_001")
