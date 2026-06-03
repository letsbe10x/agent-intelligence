You are verifying whether a fetched source actually supports a specific claim.

# Claim
{claim_text}

# Source content (excerpt)
{source_excerpt}

# Your job

Decide whether the source content supports the claim. Output JSON only:

```json
{{
  "supports": true | false,
  "confidence_0_1": 0.0,
  "reason": "1-2 sentence explanation"
}}
```

Constraints:
- "supports=true" only if the source contains evidence for the claim. Inferred support is not support.
- "supports=false" if the source is silent on the claim, or contradicts it.
- Be conservative. False positives are worse than false negatives here.

Return ONLY the JSON. No prose preamble.
