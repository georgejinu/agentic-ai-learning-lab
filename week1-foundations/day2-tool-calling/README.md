# Day 2 — Tool Calling

## Goal

See the agent loop -- LLM call -> decide if a tool is needed -> execute tool
-> feed result back -> repeat until the model has enough to answer -- from
the inside, using a framework other than LangGraph.

## Implementation

`tool_calling_demo.py` deliberately switches frameworks from Day 1's
LangGraph to **Google ADK** (Agent Development Kit), to see the same agent
loop expressed differently. It defines one plain Python function tool,
`check_prior_auth_requirement` (a mock PA-eligibility lookup), wraps it in an
ADK `Agent`, and runs it via `InMemoryRunner`.

Run with:

```
python tool_calling_demo.py "Does plan PPO-100 need prior auth for J1745?"
```

Each step of the loop is printed as it happens: the model's tool call
request, the tool's return value, and the model's final synthesized answer.

ADK's Gemini integration goes through the same `google-genai` client as
`common/llm.py` uses for LangChain, so it reads the same `.env` (Vertex AI
project/location) -- no separate credentials needed.

## Notes

- Tool calling mechanics: the LLM doesn't execute anything itself -- it
  outputs structured JSON naming a function and its arguments (a
  `function_call` part). The framework (ADK here, LangGraph's `ToolNode`
  elsewhere) matches that to the real Python function, runs it, and feeds
  the return value back to the model as a `function_response` part.
- The docstring and type hints on `check_prior_auth_requirement` *are* the
  tool schema the model sees -- that's what it uses to decide when to call
  it and how to fill in arguments.
- LangGraph (Day 1) makes the graph shape explicit and code-defined; ADK's
  `Agent` hides the loop inside the framework -- you get an `instruction`,
  a model, and a list of tools, and the tool-call/respond/repeat loop runs
  implicitly until the model stops requesting tools.
- Multiple CLI args become separate turns in one session (`_ensure_session`
  creates one `session_id` shared across all of them), so later turns can
  reference earlier ones -- e.g. "What about drug J3490 on the *same plan*?"
  only resolves correctly if session history is actually retained.
- A single compound question (e.g. asking about two plans at once) can make
  the model request multiple tool calls in *one* turn -- worth distinguishing
  from the loop genuinely repeating across turns. It only goes parallel like
  that when the calls don't depend on each other; if the second call's
  arguments depended on the first call's result, the model would be forced
  into a real sequential loop instead.

## Optional: tracing to LangSmith

ADK ships its own OpenTelemetry instrumentation (`google.adk.telemetry` --
LLM calls, tool calls, using the standard `gen_ai.*` semantic convention) but
has no LangSmith-specific integration. `_configure_langsmith_tracing()` in
`tool_calling_demo.py` doesn't touch ADK at all -- it just points the
process's global OTel `TracerProvider` at LangSmith's OTLP endpoint before
any ADK code runs, so the spans ADK already emits end up there.

To enable: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and
`LANGSMITH_PROJECT` in `.env` (see `.env.example`). Requires
`opentelemetry-exporter-otlp-proto-http`, pinned to match the OTel API/SDK
version `google-adk` itself depends on (installing the exporter unpinned
pulls a newer `opentelemetry-api`/`sdk` that conflicts with ADK's declared
bounds).
