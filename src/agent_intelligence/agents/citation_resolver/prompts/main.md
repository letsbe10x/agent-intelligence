# Citation Resolver — unused prompt placeholder

This agent does not use a main LLM prompt by default. The HTTP / lookup
verification logic is deterministic. An optional semantic check uses
``semantic_check.md`` in this same directory.

This file exists because the framework's AgentConfig requires `prompt_path`
to point to a real file. It is intentionally empty of agent logic; do not edit
unless you are adding a main-prompt LLM phase to citation resolution.
