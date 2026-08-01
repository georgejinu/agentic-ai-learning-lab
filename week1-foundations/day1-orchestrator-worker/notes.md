# Day 1 — Orchestrator/Worker Pattern

## Goal
Understand and implement the orchestrator/worker agent pattern: a central agent decomposes a task and dispatches sub-tasks to worker agents, then synthesizes their results.

## Write-up prompt answer

Implemented in `orchestrator_demo.py`, run with:

```
python orchestrator_demo.py "the pros and cons of remote work"
```

The orchestrator calls the LLM with structured output (a `Plan` of `Section`s)
to decide, per input topic, how many report sections are needed and what
each should cover — that decomposition isn't fixed at graph-build time,
which is what separates this from [Routing](../../week3-langgraph/02-routing/).
`assign_workers` fans those sections out via LangGraph's `Send` API so each
worker runs in parallel; a `synthesizer` joins the `completed_sections` list
(accumulated with an `operator.add` reducer) into the final report.

## Notes

- Orchestrator-workers vs. routing: routing picks one of N pre-defined paths;
  orchestrator-workers decides the shape of the work itself per input.
- Orchestrator-workers vs. parallelization/sectioning (see
  `week3-langgraph/03-parallelization/`): sectioning's branches are fixed at
  build time (e.g. always "sentiment" + "keywords"); here the branches
  themselves are planned at runtime.
- Uses Vertex AI Gemini (`gemini-2.5-flash`, `global` endpoint) via
  `common/llm.py` — see repo root `.env.example` for setup.
