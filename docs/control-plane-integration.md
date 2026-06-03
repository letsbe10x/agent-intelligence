# Integrating with letsbe10x control-plane

This document is the bridge between `agent-intelligence` and the
`letsbe10x/control-plane` API. It shows how the PVR control plane exposes
agent-intelligence agents over HTTP for the React UI to consume.

## Architecture

```
React UI (control-plane/ui)
        │  POST /v1/agents/{name}/run     (JSON input)
        ▼
FastAPI route (control-plane/api/agents.py)
        │  build_agent(config_path)
        │  agent.run(input, context)
        ▼
agent-intelligence (this repo)
        │  -> AnthropicProvider / OpenAIProvider / ...
        ▼
LLM API
```

## The two endpoints control-plane exposes

```python
from fastapi import APIRouter, Depends, HTTPException
from agent_intelligence import AgentContext, build_agent
from agent_intelligence.observability.receipts import ReceiptStore

router = APIRouter(prefix="/v1/agents", tags=["agents"])

# Receipts go to R2 / Postgres in prod; local dir for dev.
_receipts = ReceiptStore(path="/var/lib/control-plane/receipts")


@router.post("/{agent_name}/run")
async def run_agent_endpoint(
    agent_name: str,
    input_payload: dict,
    org=Depends(authenticated_org),  # your auth dependency
    bet_id: str | None = None,
):
    # Map agent_name → config path. In prod this is a per-org table lookup.
    config_path = f"/etc/control-plane/agent-configs/{agent_name}.yaml"
    try:
        agent = build_agent(config_path, receipt_store=_receipts)
    except FileNotFoundError:
        raise HTTPException(404, f"Agent {agent_name!r} not configured for this org")

    context = AgentContext(
        org_id=org.id,
        user_id=org.current_user_id,
        bet_id=bet_id,
        tags={"source": "control-plane-ui"},
    )

    try:
        result = await agent.run(input_payload, context)
    except Exception as e:
        # Receipts are still emitted on failure; surface a structured error.
        raise HTTPException(422, str(e))

    return {
        "output": result.output.model_dump(),
        "receipt_id": result.receipt.receipt_id,
        "receipt_hash": result.receipt.payload_hash,
        "cost_usd": result.cost_usd,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "wallclock_s": result.wallclock_s,
    }


@router.get("/receipts/{receipt_id}/verify")
async def verify_receipt(receipt_id: str):
    if _receipts.verify(receipt_id):
        receipt = _receipts.get(receipt_id)
        return {
            "verified": True,
            "receipt_id": receipt.receipt_id,
            "payload_hash": receipt.payload_hash,
            "agent": receipt.agent_name,
            "bet_id": receipt.bet_id,
        }
    raise HTTPException(422, f"Receipt {receipt_id} failed verification")
```

## The two React panels the UI gains

```tsx
// control-plane/ui/src/components/PersonaSimulation.tsx
import { useState } from "react";
import { useAgentRun } from "@/lib/useAgentRun";

export function PersonaSimulation({ betId, hypothesis, metric, timeBox }: Props) {
  const { run, result, loading, error } = useAgentRun("persona_simulator");

  return (
    <Panel title="Persona Simulation">
      <Button onClick={() => run({
        bet_id: betId,
        hypothesis,
        success_metric: metric,
        time_box: timeBox,
      })} disabled={loading}>
        Run validation pack
      </Button>

      {result?.output.reactions.map((r, i) => (
        <PersonaCard
          key={i}
          name={r.persona_name}
          archetype={r.persona_archetype}
          stance={r.overall_stance}
          confidence={r.confidence_0_1}
          endorsements={r.endorsements}
          objections={r.objections}
          questions={r.questions_the_persona_would_ask}
        />
      ))}

      {result && (
        <ReceiptBadge
          hash={result.receipt_hash}
          cost={result.cost_usd}
          onVerify={() => verifyReceipt(result.receipt_id)}
        />
      )}
    </Panel>
  );
}
```

```tsx
// control-plane/ui/src/components/CitationCheck.tsx
export function CitationCheck({ claims }: { claims: Claim[] }) {
  const { run, result, loading } = useAgentRun("citation_resolver");

  return (
    <Panel title="Citation verification">
      <Button onClick={() => run({ claims })} disabled={loading}>
        Verify all citations
      </Button>

      {result?.output.verifications.map((v, i) => (
        <CitationRow
          key={i}
          claim={v.claim_text}
          source={v.source_ref}
          status={v.status}
          detail={v.detail}
        />
      ))}

      {result && (
        <GateBadge
          allPassed={result.output.all_passed}
          blocking={result.output.blocking_failures}
        />
      )}
    </Panel>
  );
}
```

## Shared hook

```ts
// control-plane/ui/src/lib/useAgentRun.ts
import { useState } from "react";

export function useAgentRun<TOutput = any>(agentName: string) {
  const [result, setResult] = useState<AgentRunResponse<TOutput> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(input: Record<string, any>, betId?: string) {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(
        `/api/v1/agents/${agentName}/run${betId ? `?bet_id=${betId}` : ""}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(input),
          credentials: "include",
        }
      );
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return { run, result, loading, error };
}
```

## Migration path

1. Install `agent-intelligence` in `control-plane`:
   ```bash
   cd ~/lets/control-plane
   pip install -e ../agent-intelligence[anthropic]
   ```
2. Drop `agents.py` router into `control-plane/api/`.
3. Mount in `main.py`: `app.include_router(agents_router)`.
4. Place YAML configs under `/etc/control-plane/agent-configs/`.
5. Add the React components and hook to the existing UI tree.
6. Add a new route in the React app's router for the validation surface.

## Why this is the right boundary

- `agent-intelligence` knows nothing about control-plane's tenant model. It
  accepts an `org_id` and writes it into receipts; that's the entire surface.
- `control-plane` knows nothing about how agents work internally. It loads YAML,
  calls `agent.run()`, returns the typed result.
- The Receipt is the contract between them. control-plane can verify any
  receipt it stored against any future agent-intelligence version (the hash
  scheme is stable).
