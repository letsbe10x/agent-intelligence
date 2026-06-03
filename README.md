# agent-intelligence

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> **Configurable, multi-model, receipt-backed agentic framework for the letsbe10x PVR stack.** Inspired by [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/AgentIQ); engineered independently for governed product-decision workflows.

---

## Why this exists

Most agent frameworks treat configuration as an afterthought: prompts are hardcoded, models are pinned, budgets are honour-system, and provenance is *"there's a log somewhere."*

`agent-intelligence` flips those defaults:

| Conventional agent code | `agent-intelligence` |
|---|---|
| `model = "claude-sonnet-4.7"` baked in | Every model, prompt, budget, and tool list comes from YAML |
| Best-effort fallback if provider down | No silent fallback — typed errors with precise context |
| Trust the caller to track cost | Budget preflighted on **every** provider call — refused before the spend, not after |
| Audit log is a `print()` | Every run produces an immutable, sha256-hashed Receipt that `verify()`s in one call |
| `if openai: ... elif anthropic: ...` | Strategy-pattern providers behind a single `LLMProvider` ABC; multi-model is a runtime config change |
| Add a new agent → fork the framework | Entry-point discovery — third parties publish their own agents and they auto-register |

This is the agentic substrate underneath the [Product Velocity Runtime](https://github.com/letsbe10x) (PVR). Every PVR validation pack, every spec-groom run, every impact analysis ultimately invokes an `Agent` from this framework — and every one of them produces a receipt that downstream PVR can replay and verify.

## What's in v0.1

- **Core framework** — `Agent` ABC, `AgentConfig`, `AgentContext`, `AgentResult`
- **Multi-model providers** — Anthropic, OpenAI, LiteLLM (100+ models), Mock (tests)
- **Budgets** — preflight USD + token caps per run
- **Receipts** — sha256-hashed, file-persistable, `verify()`able
- **OpenTelemetry** — opt-in, host-controlled tracer provider
- **MCP server** — publish agents as Model Context Protocol tools (Cursor / Claude Code / Codex consume them directly)
- **CLI** — `ai-cli list-agents | describe | run | verify-receipt`
- **Two production agents:**
  - **`persona_simulator`** — generates N ICP personas + simulates each reacting to a Bet hypothesis; enforces "no Yes-Man" doctrine
  - **`citation_resolver`** — verifies every claim's `source_ref` actually resolves (HTTP, signal://, assumption://, bet://) with optional LLM-driven semantic check

## Install

```bash
pip install agent-intelligence            # core only
pip install 'agent-intelligence[anthropic,openai]'  # native providers
pip install 'agent-intelligence[mcp]'     # MCP server
pip install 'agent-intelligence[all]'     # everything
```

For development:

```bash
git clone https://github.com/letsbe10x/agent-intelligence
cd agent-intelligence
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,all]'
pytest -v
```

## Quickstart — run an agent in 6 lines

```python
import asyncio
from agent_intelligence import run_agent, AgentContext

result = asyncio.run(run_agent(
    config_path="configs/persona_simulator.production.yaml",
    input_={
        "bet_id": "bet_001",
        "hypothesis": "Async dashboard export lifts weekly PM retention by 8% in 30 days",
        "success_metric": "weekly_pm_retention",
        "time_box": "30 days post-ship",
    },
    context=AgentContext(org_id="acme", bet_id="bet_001"),
))

for r in result.output.reactions:
    print(f"{r.persona_name} ({r.overall_stance}): {r.objections}")

# Every run produces a verifiable receipt.
print("receipt:", result.receipt.payload_hash)
assert result.receipt.verify()
```

Or from the shell:

```bash
ai-cli run configs/persona_simulator.production.yaml input.json --org-id acme --bet-id bet_001
```

## Configuration is the product

Every aspect of an agent's behaviour lives in YAML:

```yaml
# configs/persona_simulator.production.yaml
name: persona_simulator
version: "1"
prompt_path: prompts/main.md            # externalised — edit without code changes

provider:
  name: anthropic                       # switch to 'openai' or 'litellm' freely
  model: claude-sonnet-4-7
  api_key: ${env:ANTHROPIC_API_KEY}     # no literal secrets in config
  temperature: 0.7
  max_output_tokens: 4096

budget:
  max_usd_per_run: 0.50                 # refused PREFLIGHT if a call would exceed
  max_input_tokens_per_run: 8000
  max_output_tokens_per_run: 4000
  on_exceed: raise

observability:
  otel_enabled: true
  receipts_enabled: true
  receipts_path: ./.ai-receipts

params:                                 # agent-specific, schema-validated
  n_personas: 5
  reaction_depth: long
  require_objections: true              # doctrine: no Yes-Man simulations
```

**Compose configs** with `include:`:

```yaml
# configs/persona_simulator.bet_high_value.yaml
include:
  - persona_simulator.production.yaml
budget:
  max_usd_per_run: 5.00                  # deeper validation for high-value bets
params:
  n_personas: 10
  reaction_depth: long
```

## Architecture (one diagram)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Caller (CLI, control-plane FastAPI route, MCP client, your code)            │
└────────────────────────────────────────┬─────────────────────────────────────┘
                                         │  AgentContext (org_id, bet_id, budget)
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Agent (template method)                                                     │
│  ───────────────────────                                                     │
│  1. validate input (pydantic)                                                │
│  2. budget preflight                                                         │
│  3. open OTel span                                                           │
│  4. ── call subclass._execute(input, params, ctx, provider) ──┐              │
│  5. validate output (pydantic)                                │              │
│  6. emit Receipt (sha256)                                     │              │
│  7. return AgentResult                                        │              │
└───────────────────────────────────────────────────────────────┼──────────────┘
                                                                │
                          ┌─────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LLMProvider (strategy pattern)                                              │
│   ┌─────────────────┬─────────────────┬─────────────────┬──────────────────┐ │
│   │ AnthropicProvider│  OpenAIProvider │ LiteLLMProvider │  MockProvider    │ │
│   └─────────────────┴─────────────────┴─────────────────┴──────────────────┘ │
│  Each enforces BudgetTracker preflight + accounting on every call.           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## The two shipped agents

### `persona_simulator`

**Why:** every PM tool today auto-generates "what users want" summaries. None of them surface what users would object to. This agent enforces the opposite default: every persona must surface at least one specific objection. No Yes-Man simulations.

**Input:** Bet hypothesis + metric + time box.
**Output:** N personas with structured stance (resist / neutral / endorse), endorsements, objections, diagnostic questions, top objection themes, and a claims array (every assertion linked to the input Bet for citation chain).
**Doctrine enforcement:** if `require_objections=true`, an output where any persona has zero objections is rejected at the agent layer — the framework retries or fails loud.

### `citation_resolver`

**Why:** PVR's moat is receipt-backed Evidence. Receipts are worthless if the claims they back point at sources that 404 or contradict the claim. This agent verifies every claim's `source_ref` actually resolves.

**Input:** a list of claims with `source_ref`s plus optional known-ID maps for `signal://`, `assumption://`, `bet://` schemes.
**Output:** per-claim verification (resolved / unreachable / unknown_scheme / id_not_found / semantic_mismatch / warning) + aggregate counts + a boolean `all_passed` + a count of blocking failures.
**Modes:** HTTP HEAD for URLs (with optional GET + LLM-driven semantic check), local ID lookup for custom schemes.

## MCP — publish agents to any IDE

The same agents work as MCP tools. Any Cursor, Claude Code, Codex, or other MCP-compatible client can call them with auth scoped to their tenant:

```python
from agent_intelligence.mcp import build_mcp_app
from agent_intelligence import AgentContext

def resolve_auth(meta: dict) -> AgentContext:
    token = meta.get("authorization", "")
    org_id = your_auth_resolver(token)        # your tenant logic here
    return AgentContext(org_id=org_id)

app = build_mcp_app(
    publish=[
        ("persona_simulator", "configs/persona_simulator.production.yaml"),
        ("citation_resolver", "configs/citation_resolver.production.yaml"),
    ],
    auth_resolver=resolve_auth,
)

app.run_streamable_http(host="0.0.0.0", port=8765)
```

Now a Cursor user runs `@ai_persona_simulator` from their chat and gets a receipt-backed validation pack inside their IDE, consuming their org's budget — without ever visiting your website.

## Extensibility — write your own agent in ~50 lines

```python
from pydantic import BaseModel
from agent_intelligence import Agent, AgentContext, LLMProvider, LLMRequest, Message


class Input(BaseModel):
    question: str


class Output(BaseModel):
    answer: str
    confidence: float


class Params(BaseModel):
    style: str = "concise"


class FAQAgent(Agent[Input, Output]):
    name = "faq"
    InputModel = Input
    OutputModel = Output
    ParamsModel = Params

    async def _execute(self, input_, params, context, provider):
        response = await provider.acomplete(
            LLMRequest(messages=[
                Message(role="system", content=f"Answer in {params.style} style."),
                Message(role="user", content=input_.question),
            ])
        )
        return Output(answer=response.content, confidence=0.7)


# Register either via entry point in pyproject.toml:
#   [project.entry-points."agent_intelligence.agents"]
#   faq = "my_pkg.faq:FAQAgent"
#
# ... or manually for tests:
from agent_intelligence import registry
registry.register_agent("faq", FAQAgent)
```

A YAML config + this class is all you need to ship a new agent with full receipts, budgets, observability, and MCP publishing — all inherited from the framework.

## Design pattern map

The framework uses well-known patterns deliberately. Each is named in the source where it appears.

| Pattern | Where | Why |
|---|---|---|
| **Template Method** | `Agent.run()` | Lifecycle (budget → trace → execute → receipt) is owned by the base; subclasses cannot skip steps |
| **Strategy** | `LLMProvider` + concrete providers | Multi-model is runtime config, not code |
| **Plugin Registry** | `registry/registry.py` via entry points | Third-party agents/providers auto-discover |
| **Builder** | `runner.build_agent()` | Decouple config-load from invocation |
| **Repository** | `ReceiptStore` | Receipt persistence backend swappable (file → S3/Postgres) |
| **Decorator** *(implicit)* | `provider.with_budget(tracker)` | Wraps any provider with budget enforcement |
| **Observer** *(implicit)* | OTel span emission | Host process subscribes; framework just emits |

## Inspirations vs novel work

We learned from [NVIDIA NeMo Agent Toolkit](https://github.com/NVIDIA/AgentIQ) (Apache-2.0) and the [AI-Q Blueprint](https://github.com/NVIDIA-AI-Blueprints/aiq). The specific patterns we adopted:

- **Framework-agnostic provider abstraction** (their LangChain/CrewAI wrapping → our `LLMProvider` ABC)
- **OpenTelemetry-compatible observability** (their built-in profiling → our opt-in spans)
- **MCP bidirectional support** (their FastMCP runtime → our `build_mcp_app()`)
- **Citation-backed evidence chain** (AI-Q Blueprint's `source_ref` pattern → our `Claim.source_ref` + `CitationResolverAgent`)
- **Per-call budget / token-cap enforcement** (their tool budgets → our `BudgetTracker`)
- **Shallow / Deep validation tiers** (AI-Q's shallow/deep researcher → our agent-level `params.reaction_depth`)

What's our own:

- **Tamper-evident receipts** — sha256-canonical-JSON of (config, input, output, context, execution). AIQ profiles runs; we *prove* them.
- **No-fallback philosophy** — every error is typed. If a model is unknown, we raise rather than silently price-zero.
- **Doctrine enforcement at the agent boundary** — e.g. `persona_simulator` rejects Yes-Man outputs in the framework, not by hope-and-prompt.
- **Per-config env-var resolution with explicit `${env:VAR:-default}` syntax** — secrets stay out of the YAML.

## Status

- ✅ Core framework complete
- ✅ 4 providers (Anthropic, OpenAI, LiteLLM, Mock)
- ✅ 2 production agents with full test coverage
- ✅ Receipts with verify endpoint
- ✅ MCP server scaffolding
- ✅ CLI (`ai-cli`)
- 🟡 control-plane integration (in flight)
- 🟡 More built-in agents (Clarifier, Hypothesis Tester, Outcome Decomposer)

## License

Apache-2.0. See [LICENSE](LICENSE).

Built for and by [letsbe10x](https://github.com/letsbe10x).
