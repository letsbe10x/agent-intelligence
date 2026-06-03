"""OpenTelemetry helpers — opt-in tracing.

If the OTel SDK is configured by the host application, our spans plug in
automatically. If not, ``trace_span`` becomes a no-op context manager.

We never construct a TracerProvider ourselves. The host process is responsible
for OTel setup (exporter, sampler, resource attributes). This keeps the
framework embedding-friendly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from opentelemetry import trace

_tracer = trace.get_tracer("agent_intelligence")


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Yield an OTel span. No-op if no provider is configured.

    Attributes are stringified; OTel rejects non-primitive types. Lists are
    joined with ','. Anything else is repr'd.
    """
    safe_attrs: dict[str, str | int | float | bool] = {}
    for k, v in (attributes or {}).items():
        if isinstance(v, (str, int, float, bool)):
            safe_attrs[k] = v
        elif isinstance(v, (list, tuple)):
            safe_attrs[k] = ",".join(str(x) for x in v)
        else:
            safe_attrs[k] = repr(v)

    with _tracer.start_as_current_span(name, attributes=safe_attrs) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
