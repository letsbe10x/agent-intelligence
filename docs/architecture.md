# Architecture

## Layered overview

```
caller
  └─► Agent.run()             [template method — lifecycle owned by base]
        ├─► validate input    [pydantic InputModel]
        ├─► budget preflight  [BudgetTracker — refused before provider call]
        ├─► OTel span         [opt-in, host-provider-managed]
        ├─► _execute()        [subclass — agent-specific logic]
        ├─► validate output   [pydantic OutputModel]
        ├─► build Receipt     [sha256(canonical JSON of inputs+outputs+context)]
        ├─► persist Receipt   [ReceiptStore — file or in-memory or custom]
        └─► return AgentResult
```

## Module responsibilities

| Package | Owns | Does not own |
|---|---|---|
| `core.agent` | Lifecycle template, AgentResult shape | Agent-specific behaviour |
| `core.config` | YAML loading, env-var resolution, include composition, schema validation | Per-agent params shape (each agent defines its own) |
| `core.context` | Per-invocation envelope; tenant/run/budget/trace | State mutation (frozen) |
| `core.errors` | Typed error hierarchy | Recovery logic (caller's choice) |
| `providers.base` | LLMProvider ABC, normalised request/response shapes | Per-provider SDK details |
| `providers.{anthropic,openai,litellm,mock}` | Per-provider translation + pricing | Anything else |
| `observability.budget` | Preflight + accounting | Persistence (caller picks where to log) |
| `observability.otel` | Span emission | Tracer provider setup (host) |
| `observability.receipts` | Hash, verify, persist | Distribution (S3 sync → subclass) |
| `registry.registry` | Entry-point discovery, name → class lookup | Construction (callers/runner) |
| `runner` | High-level "load → build → run" convenience | Anything customisable |
| `mcp.server` | Wrap agents as MCP tools, dispatch + auth callout | MCP protocol details (delegated to FastMCP) |
| `cli.main` | Shell-friendly commands | Programmatic API (use the library) |

## Key invariants

1. **Every agent run produces exactly one Receipt.** Even on error. The Receipt
   captures status="error" and the error message so audit logs are never empty.
2. **Receipts are tamper-evident.** sha256 over canonical JSON; verify() returns
   false on any mutation.
3. **Budget is preflight, not post-hoc.** A call that would exceed budget never
   happens. Subclasses cannot bypass this — the base class wraps the provider.
4. **Providers must be budget-bound before use.** Calling `provider.acomplete`
   without `provider.with_budget(tracker)` raises immediately.
5. **Config validation is at load time.** Schema errors surface immediately, with
   the offending field path in the message. Never at first run.
6. **Secrets are never in YAML.** Use `${env:VAR}` or `${env:VAR:-default}`. The
   loader resolves at load time and the resolved value is what flows downstream.
7. **No silent fallbacks.** If a model is unknown to a provider's pricing table,
   we raise. If a config is missing a required field, we raise. If an entry-point
   class fails to import, we raise. The framework never "best-effort"s.

## Design patterns (named in source)

| Pattern | Module | Lines |
|---|---|---|
| Template Method | `core/agent.py:Agent.run` | The fixed lifecycle owned by base |
| Strategy | `providers/base.py:LLMProvider` | Multi-model via runtime swap |
| Plugin Registry | `registry/registry.py:_Registry` | Entry-point auto-discovery |
| Builder | `runner.py:build_agent` | Decouple config-load from invocation |
| Repository | `observability/receipts.py:ReceiptStore` | Pluggable storage backend |
| Decorator | `providers/base.py:LLMProvider.with_budget` | Wraps any provider in budget enforcement |
| Observer (implicit) | `observability/otel.py:trace_span` | Host process subscribes; framework emits |

## Extension points

### Add a new provider

1. Subclass `LLMProvider`. Implement `_acomplete(request) -> LLMResponse`.
2. Handle pricing — read from `config.extra.cost_per_1k_input` /
   `cost_per_1k_output` first, fall back to a built-in table if you have one,
   raise `ProviderError` if neither is available. Do not silently price-zero.
3. Publish via entry point in your package's `pyproject.toml`:
   ```toml
   [project.entry-points."agent_intelligence.providers"]
   my_provider = "my_pkg.provider:MyProvider"
   ```

### Add a new agent

1. Subclass `Agent[InputT, OutputT]`. Set class attrs `name`, `InputModel`,
   `OutputModel`, `ParamsModel`.
2. Implement `async _execute(input_, params, context, provider) -> OutputT`.
3. Externalise prompt(s) to `prompts/*.md` in the agent's package dir. Reference
   from YAML.
4. Publish via entry point:
   ```toml
   [project.entry-points."agent_intelligence.agents"]
   my_agent = "my_pkg.agent:MyAgent"
   ```

### Add a new receipt backend

1. Subclass `ReceiptStore` (or write a duck-typed class).
2. Override `put`, `get`, `verify`, `list_ids`.
3. Pass an instance to the agent constructor or to `build_mcp_app(receipt_store=...)`.

## Threading / concurrency

- The framework is **async-first**. All provider calls are async. `run_sync()`
  is a convenience for non-loop callers.
- The CitationResolver agent runs HTTP checks concurrently with a semaphore.
  Other agents run sequentially per-call but may be invoked in parallel from
  the host application.
- Receipts are written atomically to the store. The default file backend uses
  one JSON file per receipt — no shared file, no locking needed.
- Budget trackers are not thread-safe. One tracker per run; do not share across
  concurrent runs.
