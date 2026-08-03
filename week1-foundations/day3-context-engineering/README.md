# Day 3 — Context Engineering

## Goal

Make the three memory types and the four context-engineering levers
concrete and runnable, rather than just conceptual -- each memory type
lives in a structurally distinct place in the code, and each lever is its
own visible graph node.

## Implementation

`context_engineering_demo.py` rotates back to LangGraph (Day 1's framework,
not Day 2's ADK) since the ticket explicitly frames short-term memory in
terms of "LangGraph GraphState." It's a 4-node graph over the same
PA-eligibility domain as Days 1-2:

```
START -> retrieve_context -> compress_history -> build_prompt_and_generate -> write_memory -> END
           (select)             (compress)              (isolate)              (write)
```

Run:

```
python context_engineering_demo.py "Does PPO-100 need prior auth for J1745?"
```

Multi-turn (same session, later turns can omit details the earlier turn
established):

```
python context_engineering_demo.py \
    "Does PPO-100 need prior auth for J1745?" \
    "What about J3490 on the same plan?" \
    "And HMO-200 for J1745?"
```

## The three memory types -- each in a different place in the code

- **Short-term / working state**: `GraphState` fields that get recomputed
  fresh every turn and never accumulate -- `selected_context`,
  `compressed_history`, `answer`.
- **Session memory**: the `history` field. It *does* accumulate across
  turns, but only because it's typed `Annotated[list[str], operator.add]`
  and the graph is compiled with `checkpointer=InMemorySaver()` -- LangGraph
  persists `GraphState` between `.invoke()` calls that share a `thread_id`.
  Same concept as Day 2's ADK `session_id`, different mechanism.
- **Long-term memory**: `long_term_memory.json`, written/read directly by
  `retrieve_context`/`write_memory` -- it's never part of `GraphState` at
  all. It survives a fresh *process run*, not just a fresh thread, which is
  the real distinction from session memory. Stands in for pa-slm-poc's
  Qdrant-backed RAG: real Gemini embeddings (`gemini-embedding-001`) and
  cosine similarity, just persisted to a JSON file instead of a vector DB.
  Delete the file to reset to the seed KB.

## The four levers -- each its own node

- **select** (`retrieve_context`): embeds the question, cosine-ranks it
  against the long-term KB, keeps only the top 2 matches. Verified: after
  running the single-question example once, a later run's `write_memory`
  fact ("Previously asked and answered: ...") shows up as a top-2 hit for a
  similar question -- proof the KB genuinely persists and gets searched, not
  just appended.
- **compress** (`compress_history`): passes short history through as-is,
  but once there's more than one prior turn, summarizes it with an LLM call
  instead of sending the raw transcript. Verified: turn 3 of a 3-turn run
  printed `"2 prior turn(s) -- summarizing"` and produced a real 1-sentence
  summary before answering.
- **isolate** (`build_prompt_and_generate`): the actual prompt sent to the
  model is built from exactly 3 fields (`selected_context`,
  `compressed_history`, `question`) -- never the full `GraphState`, never
  the whole long-term KB. Printed prompt length confirms this stays small
  regardless of how big the KB or history grows.
- **write** (`write_memory`): persists a new fact -- this turn's Q&A --
  back to the long-term KB so a *future* run can retrieve it. Explicitly
  skips writing a duplicate if the exact fact is already there.

## Notes

- This intentionally uses three separate storage mechanisms for the three
  memory types (state field vs. checkpointer-persisted field vs. on-disk
  file) rather than three different-looking-but-functionally-similar dict
  keys, so the architectural distinction is forced, not just labeled.
- The embedding model (`gemini-embedding-001`) and similarity function
  (plain-Python cosine, no numpy) were chosen to avoid adding a new heavy
  dependency for a 5-document demo KB -- would not scale past a few hundred
  documents without a real vector index.
