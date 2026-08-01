"""Pattern 1: Prompt Chaining.

A task broken into fixed sequential steps, each an LLM call, where the
output of one call is a clear prerequisite for the next. The order is
fixed at graph-build time -- no conditional edges, no runtime branching.

Graph shape: node_A -> node_B -> node_C

Here: draft an announcement paragraph for a product feature, then tighten
it against a fixed style rule (<= 40 words, active voice, one sentence
call-to-action). Each step needs the previous step's output as input.

Run:
    python chaining_demo.py "a dark mode toggle in settings"
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph

from common.llm import get_llm

llm = get_llm()


class State(TypedDict):
    feature: str
    draft: str
    final: str


def generate_draft(state: State) -> dict:
    """Step A: write a first-pass announcement paragraph."""
    resp = llm.invoke(
        f"Write a short announcement paragraph for this new product feature: "
        f"{state['feature']}. Don't worry about length yet, just cover what it "
        "is and why it matters."
    )
    print("[generate_draft] done")
    return {"draft": resp.content}


def tighten_draft(state: State) -> dict:
    """Step B: rewrite the draft against a fixed style rule, using step A's output."""
    resp = llm.invoke(
        "Rewrite this announcement to fit these rules exactly:\n"
        "- 40 words or fewer\n"
        "- active voice\n"
        "- end with one sentence that is a clear call-to-action\n\n"
        f"Draft:\n{state['draft']}"
    )
    print("[tighten_draft] done")
    return {"final": resp.content}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("generate_draft", generate_draft)
    graph.add_node("tighten_draft", tighten_draft)

    graph.add_edge(START, "generate_draft")
    graph.add_edge("generate_draft", "tighten_draft")
    graph.add_edge("tighten_draft", END)
    return graph.compile()


def main():
    feature = sys.argv[1] if len(sys.argv) > 1 else "a dark mode toggle in settings"
    app = build_graph()
    result = app.invoke({"feature": feature})

    print("\n--- draft ---")
    print(result["draft"])
    print("\n--- final (tightened) ---")
    print(result["final"])


if __name__ == "__main__":
    main()
