"""Day 3: Memory types & context engineering levers, via LangGraph.

Three memory types, each living in an architecturally distinct place in
this code -- not just described, but structurally forced apart:

  - Short-term / working state: GraphState fields that get recomputed fresh
    every turn (selected_context, compressed_history, answer). Nothing
    accumulates here.
  - Session memory: the `history` field, which DOES accumulate across turns
    -- but only because LangGraph's checkpointer persists GraphState between
    `.invoke()` calls that share a thread_id, combined with an operator.add
    reducer. Same idea as Day 2's ADK session_id, different mechanism.
  - Long-term memory: a small embedded knowledge base persisted to a JSON
    file on disk, living entirely OUTSIDE GraphState. It survives even a
    fresh process run, not just a fresh thread_id. Stands in for
    pa-slm-poc's Qdrant-backed RAG without needing a real vector DB.

Four context engineering levers, each its own graph node so the concept
maps 1:1 onto runnable code:

  select   -> retrieve_context           pick relevant long-term memory for
                                          this question via cosine similarity
  compress -> compress_history           condense old turns instead of
                                          sending the full transcript
  isolate  -> build_prompt_and_generate  the LLM call sees only 3 fields --
                                          never the full state, never the
                                          whole KB
  write    -> write_memory               persist a new fact back to
                                          long-term memory for future runs

Run (single turn):
    python context_engineering_demo.py "Does PPO-100 need prior auth for J1745?"

Run (multi-turn, same session -- later turns can omit details the earlier
turn already established, same idea as Day 2's session test):
    python context_engineering_demo.py \
        "Does PPO-100 need prior auth for J1745?" \
        "What about J3490 on the same plan?"
"""

import json
import math
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google import genai
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from common.llm import get_llm

llm = get_llm()

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
_genai_client = genai.Client()
EMBED_MODEL = "gemini-embedding-001"

# ---------------------------------------------------------------------------
# Long-term memory: persisted to disk, read/written directly by nodes --
# never passed through GraphState. Deleting this file resets it to the seed
# KB; run the script twice without deleting it to see write_memory's effect
# (a fact from turn 1 becomes retrievable in a later, separate run).
# ---------------------------------------------------------------------------
KB_PATH = Path(__file__).parent / "long_term_memory.json"

SEED_KB = [
    "Plan PPO-100 requires prior authorization for J1745 (infliximab): step therapy, methotrexate trial required first.",
    "Plan PPO-100 does not require prior authorization for J3490 (unclassified biologic).",
    "Plan HMO-200 requires prior authorization for J1745: specialist referral plus diagnosis code confirmation.",
    "Plan HMO-200 requires prior authorization for J9035 (bevacizumab): oncology diagnosis code required.",
    "Plan PPO-100 requires a step therapy exception form if the patient already failed methotrexate elsewhere.",
]


def _embed(text: str) -> list[float]:
    result = _genai_client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def _load_long_term_memory() -> list[dict]:
    if KB_PATH.exists():
        return json.loads(KB_PATH.read_text())
    print(f"[long-term memory] no cache at {KB_PATH.name} -- embedding seed KB (one-time cost)")
    kb = [{"text": text, "embedding": _embed(text)} for text in SEED_KB]
    KB_PATH.write_text(json.dumps(kb))
    return kb


def _save_long_term_memory(kb: list[dict]) -> None:
    KB_PATH.write_text(json.dumps(kb))


# ---------------------------------------------------------------------------
# Short-term/working state (recomputed each turn) + session memory
# (`history`, accumulated across turns by the checkpointer).
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    history: Annotated[list[str], operator.add]
    selected_context: list[str]
    compressed_history: str
    answer: str


def retrieve_context(state: GraphState) -> dict:
    """select: pull only the relevant slice of long-term memory in."""
    kb = _load_long_term_memory()
    q_vec = _embed(state["question"])
    scored = sorted(kb, key=lambda e: _cosine(q_vec, e["embedding"]), reverse=True)
    top = [e["text"] for e in scored[:2]]
    print(f"[select] chose {len(top)}/{len(kb)} long-term memory entries for this question:")
    for t in top:
        print(f"  - {t}")
    return {"selected_context": top}


def compress_history(state: GraphState) -> dict:
    """compress: summarize old turns instead of sending the raw transcript."""
    history = state["history"]
    print(f"[session memory] raw `history` accumulated so far ({len(history)} entries):")
    if history:
        for h in history:
            print(f"    {h!r}")
    else:
        print("    (empty -- this is the first turn in the session)")
    if len(history) <= 1:
        print("[compress] history short enough to pass through as-is")
        return {"compressed_history": "\n".join(history)}
    print(f"[compress] {len(history)} prior turn(s) -- summarizing instead of sending raw transcript")
    summary = llm.invoke(
        "Summarize this conversation history in 1-2 sentences, keeping any "
        "plan names or drug codes mentioned:\n" + "\n".join(history)
    ).content
    print(f"  -> {summary}")
    return {"compressed_history": summary}


def build_prompt_and_generate(state: GraphState) -> dict:
    """isolate: the model sees only these 3 fields, never full state/KB."""
    prompt = (
        "You are a pharmacy benefits assistant. Use ONLY the facts below; "
        "don't guess.\n\n"
        f"Known facts:\n{chr(10).join(state['selected_context'])}\n\n"
        f"Conversation so far: {state['compressed_history'] or '(none)'}\n\n"
        f"Question: {state['question']}"
    )
    print(f"[isolate] prompt built from 3 fields only, not the full state or full KB ({len(prompt)} chars)")
    answer = llm.invoke(prompt).content
    return {"answer": answer, "history": [f"Q: {state['question']}\nA: {answer}"]}


def write_memory(state: GraphState) -> dict:
    """write: persist a new fact back to long-term memory."""
    kb = _load_long_term_memory()
    note = f"Previously asked and answered: {state['question']} -> {state['answer']}"
    if not any(e["text"] == note for e in kb):
        kb.append({"text": note, "embedding": _embed(note)})
        _save_long_term_memory(kb)
        print(f"[write] persisted this Q&A to {KB_PATH.name} -- retrievable in future runs")
    else:
        print("[write] already in long-term memory, skipping")
    return {}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("compress_history", compress_history)
    graph.add_node("build_prompt_and_generate", build_prompt_and_generate)
    graph.add_node("write_memory", write_memory)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "compress_history")
    graph.add_edge("compress_history", "build_prompt_and_generate")
    graph.add_edge("build_prompt_and_generate", "write_memory")
    graph.add_edge("write_memory", END)
    return graph.compile(checkpointer=InMemorySaver())


def main():
    questions = sys.argv[1:] or ["Does PPO-100 need prior auth for J1745?"]
    app = build_graph()
    thread_id = "demo-thread"

    for i, question in enumerate(questions, start=1):
        print(f"\n--- turn {i} {'-' * 40}")
        print(f"[user] {question}")
        result = app.invoke(
            {"question": question, "history": []},
            config={"configurable": {"thread_id": thread_id}},
        )
        print(f"\n[answer] {result['answer']}")


if __name__ == "__main__":
    main()
