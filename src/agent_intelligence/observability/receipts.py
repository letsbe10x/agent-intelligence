"""Receipts — tamper-evident proofs of agent execution.

A Receipt is a sha256 hash over a canonical JSON serialisation of:
    agent name + version, config snapshot (secrets redacted),
    input payload, output payload, context (org/bet/run IDs),
    timing, cost, status.

Two operations are supported:
    1. ``build()``  — compute the hash on write.
    2. ``verify()`` — recompute and compare on read.

A Receipt that fails ``verify()`` indicates tampering or corruption. The
framework raises ReceiptError in that case.

Receipts can be persisted to a local directory (one JSON per receipt) or kept
in-memory. Production deployments will swap in S3/R2/Postgres backends via
subclassing ReceiptStore.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import ReceiptError


def _canonical_json(payload: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace. The receipt hash input."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


@dataclass(frozen=True)
class Receipt:
    """An immutable receipt. Every agent run produces exactly one."""

    receipt_id: str          # uuid (matches AgentContext.run_id)
    agent_name: str
    agent_version: str
    org_id: str | None
    bet_id: str | None
    status: str              # "ok" | "error"
    error_message: str | None
    wallclock_s: float
    cost_usd: float
    tokens_in: int
    tokens_out: int
    model_calls: int
    created_at: float        # unix timestamp
    payload_hash: str        # sha256 of canonical(config + input + output + context)
    payload: dict[str, Any]  # the canonical payload itself; preserved for replay

    @classmethod
    def build(
        cls,
        *,
        agent_name: str,
        agent_version: str,
        config_snapshot: dict[str, Any],
        input_payload: Any,
        output_payload: Any,
        context: AgentContext,
        wallclock_s: float,
        cost_usd: float,
        tokens_in: int,
        tokens_out: int,
        model_calls: int,
        status: str,
        error_message: str | None,
    ) -> Receipt:
        """Construct and hash a receipt."""
        payload = {
            "agent": {"name": agent_name, "version": agent_version},
            "config": config_snapshot,
            "input": input_payload,
            "output": output_payload,
            "context": {
                "org_id": context.org_id,
                "bet_id": context.bet_id,
                "run_id": context.run_id,
                "user_id": context.user_id,
                "tags": context.tags,
                "metadata": context.metadata,
            },
            "execution": {
                "status": status,
                "error_message": error_message,
                "wallclock_s": round(wallclock_s, 4),
                "cost_usd": round(cost_usd, 6),
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "model_calls": model_calls,
            },
        }
        payload_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

        return cls(
            receipt_id=context.run_id,
            agent_name=agent_name,
            agent_version=agent_version,
            org_id=context.org_id,
            bet_id=context.bet_id,
            status=status,
            error_message=error_message,
            wallclock_s=wallclock_s,
            cost_usd=cost_usd,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model_calls=model_calls,
            created_at=time.time(),
            payload_hash=payload_hash,
            payload=payload,
        )

    def verify(self) -> bool:
        """Recompute the hash and compare. True iff tamper-free."""
        recomputed = hashlib.sha256(_canonical_json(self.payload).encode("utf-8")).hexdigest()
        return recomputed == self.payload_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "org_id": self.org_id,
            "bet_id": self.bet_id,
            "status": self.status,
            "error_message": self.error_message,
            "wallclock_s": self.wallclock_s,
            "cost_usd": self.cost_usd,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model_calls": self.model_calls,
            "created_at": self.created_at,
            "payload_hash": self.payload_hash,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Receipt:
        return cls(**d)


class ReceiptStore:
    """Persists receipts. Default backend: per-file JSON in a directory, or in-memory."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, Receipt] = {}

    def put(self, receipt: Receipt) -> None:
        self._memory[receipt.receipt_id] = receipt
        if self.path:
            target = self.path / f"{receipt.receipt_id}.json"
            target.write_text(
                json.dumps(receipt.to_dict(), indent=2, default=str), encoding="utf-8"
            )

    def get(self, receipt_id: str) -> Receipt:
        if receipt_id in self._memory:
            return self._memory[receipt_id]
        if self.path:
            target = self.path / f"{receipt_id}.json"
            if target.is_file():
                return Receipt.from_dict(json.loads(target.read_text(encoding="utf-8")))
        raise ReceiptError(f"Receipt not found: {receipt_id}")

    def verify(self, receipt_id: str) -> bool:
        receipt = self.get(receipt_id)
        if not receipt.verify():
            raise ReceiptError(
                f"Receipt {receipt_id} failed hash verification. "
                "The receipt has been tampered with or corrupted."
            )
        return True

    def list_ids(self) -> list[str]:
        ids = set(self._memory.keys())
        if self.path:
            ids.update(p.stem for p in self.path.glob("*.json"))
        return sorted(ids)
