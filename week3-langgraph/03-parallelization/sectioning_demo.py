"""Pattern 3a: Parallelization -- Sectioning.

Split a task into independent subtasks that don't depend on each other,
run them in parallel, then combine the results in a join node. Unlike
routing, every branch runs on every input; unlike orchestrator-workers,
the branches are fixed at graph-build time, not decided per-input.

Graph shape: START fans out to N nodes -> join node combines -> END

Here: sentiment analysis and keyword extraction on the same text run
independently and simultaneously, then a combine step merges them into
one structured summary.

Run:
    python sectioning_demo.py "The new checkout flow is so much faster, but the confirmation email never arrived."
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph

from common.llm import get_llm

llm = get_llm()


class State(TypedDict):
    text: str
    sentiment: str
    keywords: str
    summary: str


def analyze_sentiment(state: State) -> dict:
    resp = llm.invoke(
        f"In one short phrase, what is the sentiment of this text?\n{state['text']}"
    )
    print("[analyze_sentiment] done")
    return {"sentiment": resp.content}


def extract_keywords(state: State) -> dict:
    resp = llm.invoke(
        f"List 3-5 comma-separated keywords from this text:\n{state['text']}"
    )
    print("[extract_keywords] done")
    return {"keywords": resp.content}


def combine(state: State) -> dict:
    """Join node: both parallel branches have finished by the time this runs."""
    summary = f"sentiment: {state['sentiment']} | keywords: {state['keywords']}"
    return {"summary": summary}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("extract_keywords", extract_keywords)
    graph.add_node("combine", combine)

    # Fan out: both branches start directly from START, so LangGraph runs
    # them in the same superstep (concurrently).
    graph.add_edge(START, "analyze_sentiment")
    graph.add_edge(START, "extract_keywords")
    # Fan in: combine only runs once both branches have written their state.
    graph.add_edge("analyze_sentiment", "combine")
    graph.add_edge("extract_keywords", "combine")
    graph.add_edge("combine", END)
    return graph.compile()


def main():
    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "The new checkout flow is so much faster, but the confirmation "
        "email never arrived."
    )
    app = build_graph()
    result = app.invoke({"text": text})
    print(f"\n{result['summary']}")


if __name__ == "__main__":
    main()
