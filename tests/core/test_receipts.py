"""Receipts are the moat. These tests are non-negotiable."""

from __future__ import annotations

import pytest

from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import ReceiptError
from agent_intelligence.observability.receipts import Receipt, ReceiptStore


def _make_receipt() -> Receipt:
    return Receipt.build(
        agent_name="test",
        agent_version="1",
        config_snapshot={"name": "test"},
        input_payload={"a": 1},
        output_payload={"b": 2},
        context=AgentContext(org_id="acme", bet_id="b1"),
        wallclock_s=1.0,
        cost_usd=0.01,
        tokens_in=10,
        tokens_out=20,
        model_calls=1,
        status="ok",
        error_message=None,
    )


def test_receipt_hash_is_deterministic():
    r1 = _make_receipt()
    r2 = _make_receipt()
    # Different run_ids → different hashes (expected).
    assert r1.payload_hash != r2.payload_hash
    # But the same payload always hashes the same.
    same_payload = r1.payload
    import hashlib
    import json as _json

    canonical = _json.dumps(same_payload, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == r1.payload_hash


def test_receipt_verify_passes_for_untampered():
    r = _make_receipt()
    assert r.verify() is True


def test_receipt_verify_fails_when_payload_tampered():
    r = _make_receipt()
    # Mutate the payload in-place by replacing via __dict__ (bypassing frozen).
    object.__setattr__(r, "payload", {**r.payload, "tampered": True})
    assert r.verify() is False


def test_receipt_store_put_get_roundtrip(tmp_receipts):
    store = ReceiptStore(path=tmp_receipts)
    r = _make_receipt()
    store.put(r)
    fetched = store.get(r.receipt_id)
    assert fetched.payload_hash == r.payload_hash
    assert fetched.verify() is True


def test_receipt_store_persists_to_disk(tmp_receipts):
    store = ReceiptStore(path=tmp_receipts)
    r = _make_receipt()
    store.put(r)
    files = list(tmp_receipts.glob("*.json"))
    assert len(files) == 1
    assert files[0].stem == r.receipt_id


def test_receipt_store_verify_raises_on_tampered_file(tmp_receipts):
    import json

    store = ReceiptStore(path=tmp_receipts)
    r = _make_receipt()
    store.put(r)
    # Tamper the on-disk file.
    path = tmp_receipts / f"{r.receipt_id}.json"
    data = json.loads(path.read_text())
    data["payload"]["input"]["a"] = 999
    path.write_text(json.dumps(data))
    fresh_store = ReceiptStore(path=tmp_receipts)
    with pytest.raises(ReceiptError):
        fresh_store.verify(r.receipt_id)
