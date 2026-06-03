"""citation_resolver — YAML-defined ReAct agent.

The agent has NO Python code. It is defined entirely by:
  - citation_resolver.yaml
  - prompt.md

The LLM decides which tool to call for each claim:
  http.get   for http(s):// refs
  lookup.id  for signal:// / assumption:// / bet:// refs
  verify.semantic  for content checks
  citation.finalize  to assemble the final report
"""
