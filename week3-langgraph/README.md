# Week 3 — LangGraph

Four of Anthropic's five "building effective agents" workflow patterns,
implemented as small runnable LangGraph graphs. The fifth,
orchestrator-workers, lives in
[`week1-foundations/day1-orchestrator-worker/`](../week1-foundations/day1-orchestrator-worker/)
since that folder was already scaffolded for it.

All demos call Vertex AI Gemini (`gemini-2.5-flash`) through
[`common/llm.py`](../common/llm.py) at the repo root. Setup:

```
python -m venv .venv          # from repo root, once
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env          # fill in GOOGLE_CLOUD_PROJECT
```

## Patterns

| # | Pattern | Folder | Graph shape |
|---|---|---|---|
| 1 | Prompt chaining | [`01-prompt-chaining/`](01-prompt-chaining/) | `A -> B`, fixed order, no conditional edges |
| 2 | Routing | [`02-routing/`](02-routing/) | `classifier -> conditional_edge -> (path A \| B \| C)` |
| 3a | Parallelization (sectioning) | [`03-parallelization/sectioning_demo.py`](03-parallelization/sectioning_demo.py) | fan-out to independent branches -> join |
| 3b | Parallelization (voting) | [`03-parallelization/voting_demo.py`](03-parallelization/voting_demo.py) | fan-out to N identical-purpose branches -> tally |
| 4 | Orchestrator-workers | [`../week1-foundations/day1-orchestrator-worker/`](../week1-foundations/day1-orchestrator-worker/) | orchestrator plans -> `Send` fan-out (dynamic) -> synthesize |
| 5 | Evaluator-optimizer | [`05-evaluator-optimizer/`](05-evaluator-optimizer/) | `generate -> evaluate -> conditional_edge -> (retry \| END)` |

Each folder's script is runnable directly, e.g.:

```
python 02-routing/routing_demo.py "My internet keeps dropping every 10 minutes"
```

## What actually distinguishes them

- **Chaining vs. routing**: chaining always runs every step in order;
  routing picks exactly one path per input.
- **Routing vs. orchestrator-workers**: routing's conditional edge is a
  lookup over a small fixed set of categories; orchestrator-workers has the
  model decide the number/shape of sub-tasks per input — closer to agentic
  judgment, but the overall shape (plan -> delegate -> synthesize) is still
  fixed, so Anthropic still classifies it as a workflow.
- **Sectioning vs. voting**: sectioning runs *different* jobs on the same
  input in parallel and merges complementary results; voting runs the
  *same* job multiple times in parallel and aggregates by agreement.
- **Evaluator-optimizer vs. everything else**: the only pattern with a
  cycle in the graph — it's the workflow-pattern name for what the
  Reflection/Reflexion literature calls the same graph shape.
