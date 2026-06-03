"""CitationResolverAgent implementation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

import httpx
from pydantic import BaseModel

from agent_intelligence.agents.citation_resolver.schemas import (
    CitationResolverInput,
    CitationResolverOutput,
    CitationResolverParams,
    CitationVerification,
    ClaimToVerify,
)
from agent_intelligence.core.agent import Agent
from agent_intelligence.core.context import AgentContext
from agent_intelligence.core.errors import AgentError
from agent_intelligence.providers.base import LLMProvider, LLMRequest, Message

# URL scheme detection.
_URL_SCHEMES = ("http://", "https://")
_CUSTOM_SCHEMES = ("signal://", "assumption://", "bet://")

# Statuses that count as "blocking" for upstream gates.
# unknown_scheme is blocking: a claim whose source_ref cannot be parsed is
# unverifiable, which is functionally the same as unreachable for gate purposes.
_BLOCKING_STATUSES = {"unreachable", "id_not_found", "semantic_mismatch", "unknown_scheme"}


class CitationResolverAgent(Agent[CitationResolverInput, CitationResolverOutput]):
    """Verifies every claim's ``source_ref`` actually resolves.

    Three modes per claim, picked from the source_ref scheme:

        http(s)://...   →  HTTP HEAD + (optional) GET for semantic check
        signal://<id>   →  Lookup in input.known_signal_ids
        assumption://   →  Lookup in input.known_assumption_ids
        bet://          →  Lookup in input.known_bet_ids

    Unknown schemes are flagged as ``unknown_scheme`` and counted as failures.

    The agent runs HTTP checks concurrently up to ``params.http_max_concurrency``.
    If semantic checking is enabled, an LLM call follows each successful fetch.
    """

    name = "citation_resolver"
    InputModel = CitationResolverInput
    OutputModel = CitationResolverOutput
    ParamsModel = CitationResolverParams

    async def _execute(
        self,
        input_: CitationResolverInput,
        params: BaseModel,
        context: AgentContext,
        provider: LLMProvider,
    ) -> CitationResolverOutput:
        assert isinstance(params, CitationResolverParams)

        semaphore = asyncio.Semaphore(params.http_max_concurrency)

        async with httpx.AsyncClient(
            timeout=params.http_timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "agent-intelligence/citation-resolver"},
        ) as http:
            tasks = [
                self._verify_claim(claim, params, input_, http, semaphore, context, provider)
                for claim in input_.claims
            ]
            verifications = await asyncio.gather(*tasks, return_exceptions=False)

        summary: dict[str, int] = {}
        for v in verifications:
            summary[v.status] = summary.get(v.status, 0) + 1

        blocking = sum(1 for v in verifications if v.status in _BLOCKING_STATUSES)
        all_passed = blocking == 0

        return CitationResolverOutput(
            verifications=verifications,
            summary=summary,
            all_passed=all_passed,
            blocking_failures=blocking,
        )

    # --- Per-claim verification ---------------------------------------------

    async def _verify_claim(
        self,
        claim: ClaimToVerify,
        params: CitationResolverParams,
        input_: CitationResolverInput,
        http: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        context: AgentContext,
        provider: LLMProvider,
    ) -> CitationVerification:
        ref = claim.source_ref.strip()

        # URL ref
        if ref.startswith(_URL_SCHEMES):
            async with semaphore:
                return await self._verify_url(claim, ref, params, http, context, provider)

        # Custom scheme — local lookup
        for scheme, known in (
            ("signal://", input_.known_signal_ids),
            ("assumption://", input_.known_assumption_ids),
            ("bet://", input_.known_bet_ids),
        ):
            if ref.startswith(scheme):
                ident = ref[len(scheme) :]
                if ident in known:
                    return CitationVerification(
                        claim_text=claim.text,
                        source_ref=ref,
                        status="resolved",
                        detail=f"{scheme[:-3]} ID {ident!r} found in caller-supplied known set.",
                    )
                return CitationVerification(
                    claim_text=claim.text,
                    source_ref=ref,
                    status="id_not_found",
                    detail=f"{scheme[:-3]} ID {ident!r} not in the known set "
                    f"(caller provided {len(known)} known IDs).",
                )

        return CitationVerification(
            claim_text=claim.text,
            source_ref=ref,
            status="unknown_scheme",
            detail=f"source_ref does not match any known scheme. "
            f"Expected one of: http(s)://, {', '.join(_CUSTOM_SCHEMES)}",
        )

    # --- URL verification ---------------------------------------------------

    async def _verify_url(
        self,
        claim: ClaimToVerify,
        url: str,
        params: CitationResolverParams,
        http: httpx.AsyncClient,
        context: AgentContext,
        provider: LLMProvider,
    ) -> CitationVerification:
        # HTTPS warning
        is_warning = False
        warning_detail: str | None = None
        if params.require_https_for_external and url.startswith("http://"):
            is_warning = True
            warning_detail = "Source uses plain HTTP. Consider HTTPS for tamper-resistance."

        # HEAD first (cheap). If HEAD fails or returns 405, fall back to GET.
        try:
            resp = await http.head(url)
            if resp.status_code in (405, 501):
                # HEAD not supported — try GET.
                resp = await http.get(url)
            elif resp.status_code >= 400:
                # HEAD said error; some servers lie on HEAD but accept GET.
                resp = await http.get(url)
        except httpx.RequestError as e:
            return CitationVerification(
                claim_text=claim.text,
                source_ref=url,
                status="unreachable",
                detail=f"HTTP request failed: {type(e).__name__}: {e}",
            )

        if resp.status_code >= 400:
            return CitationVerification(
                claim_text=claim.text,
                source_ref=url,
                status="unreachable",
                detail=f"HTTP {resp.status_code} {resp.reason_phrase}",
                resolved_url=str(resp.url),
                http_status=resp.status_code,
            )

        # If we already got the body (from a GET), hash it.
        content_sha: str | None = None
        body: str | None = None
        if resp.request.method == "GET":
            body_bytes = resp.content
            content_sha = hashlib.sha256(body_bytes).hexdigest()
            # Only keep small bodies for semantic check.
            if len(body_bytes) < 200_000:
                body = body_bytes.decode("utf-8", errors="ignore")

        # Semantic check (optional, costs an LLM call)
        semantic_conf: float | None = None
        if params.use_llm_for_semantic_check and body:
            semantic_conf, semantic_supported, semantic_reason = await self._semantic_check(
                claim_text=claim.text,
                source_excerpt=self._make_excerpt(body),
                provider=provider,
            )
            if not semantic_supported:
                return CitationVerification(
                    claim_text=claim.text,
                    source_ref=url,
                    status="semantic_mismatch",
                    detail=f"Source resolved but does not support claim: {semantic_reason}",
                    resolved_url=str(resp.url),
                    http_status=resp.status_code,
                    content_sha256=content_sha,
                    semantic_confidence_0_1=semantic_conf,
                )

        status = "warning" if is_warning else "resolved"
        detail = warning_detail if is_warning else "Source reachable."
        return CitationVerification(
            claim_text=claim.text,
            source_ref=url,
            status=status,
            detail=detail,
            resolved_url=str(resp.url),
            http_status=resp.status_code,
            content_sha256=content_sha,
            semantic_confidence_0_1=semantic_conf,
        )

    # --- LLM semantic check helpers ----------------------------------------

    async def _semantic_check(
        self,
        claim_text: str,
        source_excerpt: str,
        provider: LLMProvider,
    ) -> tuple[float, bool, str]:
        """Run the LLM check. Returns (confidence, supports, reason)."""
        # We loaded prompt_template at agent construction; for the semantic
        # check we use a separate template file.
        prompt_path = self.config.prompt_path
        # Convention: semantic_check.md lives in the same prompts/ dir.
        from pathlib import Path

        sem_path = Path(prompt_path).parent / "semantic_check.md"
        if not sem_path.is_file():
            raise AgentError(
                f"Semantic check enabled but prompt file missing: {sem_path}. "
                "Disable params.use_llm_for_semantic_check or add the file."
            )
        template = sem_path.read_text(encoding="utf-8")
        prompt = template.format(claim_text=claim_text, source_excerpt=source_excerpt)

        response = await provider.acomplete(
            LLMRequest(
                messages=[
                    Message(role="system", content="You produce strict JSON. No prose outside JSON."),
                    Message(role="user", content=prompt),
                ],
            )
        )

        payload = self._extract_json(response.content)
        try:
            supports = bool(payload["supports"])
            confidence = float(payload["confidence_0_1"])
            reason = str(payload.get("reason", ""))
        except (KeyError, TypeError, ValueError) as e:
            raise AgentError(
                f"citation_resolver: semantic check returned malformed JSON: {e}. "
                f"Raw (first 200 chars): {response.content[:200]!r}"
            ) from e
        return confidence, supports, reason

    @staticmethod
    def _make_excerpt(body: str, max_chars: int = 6000) -> str:
        """Pull a sensible excerpt of an HTML/text body for the semantic check.

        Strips HTML tags crudely; keeps the first max_chars of the result.
        Good enough for the semantic supports/not-supports decision.
        """
        # Crude tag strip. For HTML pages this loses structure; that's OK because
        # the LLM just needs the textual signal.
        text = re.sub(r"<script[^>]*>.*?</script>", " ", body, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Same robust extractor used by persona_simulator."""
        s = raw.strip()
        fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", s, re.DOTALL)
        if fence:
            s = fence.group(1)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        depth = 0
        start = None
        for i, c in enumerate(s):
            if c == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
        raise AgentError(
            f"citation_resolver: could not extract JSON from LLM output. "
            f"First 200 chars: {raw[:200]!r}"
        )
