"""Config loader: env-var resolution, includes, schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_intelligence.core.config import load_config
from agent_intelligence.core.errors import ConfigError


def _write_prompt(d: Path) -> Path:
    p = d / "prompt.md"
    p.write_text("hello {x}", encoding="utf-8")
    return p


def test_load_minimal_valid_config(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-fake")
    _write_prompt(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: mock_agent
prompt_path: prompt.md
provider:
  name: mock
  model: mock-model
  api_key: ${env:FAKE_KEY}
params: {}
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.name == "mock_agent"
    assert cfg.provider.api_key == "sk-fake"
    assert Path(cfg.prompt_path).is_file()


def test_missing_env_var_raises_config_error(tmp_path):
    _write_prompt(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: mock_agent
prompt_path: prompt.md
provider:
  name: mock
  model: mock-model
  api_key: ${env:NEVER_SET_VAR}
params: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="NEVER_SET_VAR"):
        load_config(cfg_path)


def test_env_var_default_is_used(tmp_path):
    _write_prompt(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: mock_agent
prompt_path: prompt.md
provider:
  name: mock
  model: mock-model
  api_key: ${env:NEVER_SET_VAR:-fallback-default}
params: {}
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    assert cfg.provider.api_key == "fallback-default"


def test_unknown_field_at_top_level_is_rejected(tmp_path):
    _write_prompt(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: mock_agent
prompt_path: prompt.md
provider:
  name: mock
  model: mock-model
typo_field: oops
params: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_missing_prompt_file_raises(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
name: mock_agent
prompt_path: nonexistent.md
provider:
  name: mock
  model: mock-model
params: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Prompt file"):
        load_config(cfg_path)


def test_include_composition(tmp_path):
    _write_prompt(tmp_path)
    base = tmp_path / "base.yaml"
    base.write_text(
        """
name: mock_agent
prompt_path: prompt.md
provider:
  name: mock
  model: base-model
  timeout_s: 30
params: {}
""",
        encoding="utf-8",
    )
    over = tmp_path / "override.yaml"
    over.write_text(
        """
include:
  - base.yaml
provider:
  model: overridden-model
  temperature: 0.1
""",
        encoding="utf-8",
    )
    cfg = load_config(over)
    assert cfg.provider.model == "overridden-model"
    assert cfg.provider.timeout_s == 30   # from base
    assert cfg.provider.temperature == 0.1


def test_include_cycle_detected(tmp_path):
    _write_prompt(tmp_path)
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(
        """
include: [b.yaml]
name: x
prompt_path: prompt.md
provider:
  name: mock
  model: m
params: {}
""",
        encoding="utf-8",
    )
    b.write_text(
        """
include: [a.yaml]
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="cycle"):
        load_config(a)
