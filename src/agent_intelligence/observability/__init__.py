"""Observability primitives: budget tracking, OTel tracing, receipts.

Each primitive is decoupled — agents can be built with any subset of them
enabled. Receipts are the strongest invariant (every run produces one); OTel
is opt-in via config.
"""
