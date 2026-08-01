"""Pattern 2: Routing.

Classify the input first, then send it down exactly one of several
specialized paths. The classifier's output drives a conditional edge, but
the edge logic itself is a plain switch/lookup on a fixed set of
categories -- not the model exercising open-ended judgment about what to
do next. That's what keeps this a workflow, not an agent.

Graph shape: classifier -> conditional_edge -> (billing | technical | general) -> END

Run:
    python routing_demo.py "My internet keeps dropping every 10 minutes"
"""

import sys
from pathlib import Path
from typing import Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.llm import get_llm

llm = get_llm()

Category = Literal["billing", "technical", "general"]


class Classification(BaseModel):
    category: Category = Field(description="Which queue this message belongs in")


class State(TypedDict):
    message: str
    category: str
    response: str


def classify(state: State) -> dict:
    classifier = llm.with_structured_output(Classification)
    result = classifier.invoke(
        f"Classify this customer message into billing, technical, or general:\n"
        f"{state['message']}"
    )
    print(f"[classify] -> {result.category}")
    return {"category": result.category}


def route(state: State) -> str:
    """Plain lookup on the classifier's output -- a switch, not a judgment call."""
    return state["category"]


def handle_billing(state: State) -> dict:
    resp = llm.invoke(
        "You are a billing support specialist. Reply helpfully and briefly to:\n"
        f"{state['message']}"
    )
    return {"response": resp.content}


def handle_technical(state: State) -> dict:
    resp = llm.invoke(
        "You are a technical support specialist. Ask a clarifying diagnostic "
        f"question or give a first troubleshooting step for:\n{state['message']}"
    )
    return {"response": resp.content}


def handle_general(state: State) -> dict:
    resp = llm.invoke(
        "You are a general customer support agent. Reply helpfully and briefly to:\n"
        f"{state['message']}"
    )
    return {"response": resp.content}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("classify", classify)
    graph.add_node("billing", handle_billing)
    graph.add_node("technical", handle_technical)
    graph.add_node("general", handle_general)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route,
        {"billing": "billing", "technical": "technical", "general": "general"},
    )
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("general", END)
    return graph.compile()


def main():
    message = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "My internet keeps dropping every 10 minutes"
    )
    app = build_graph()
    result = app.invoke({"message": message})

    print(f"\ncategory: {result['category']}")
    print(f"response: {result['response']}")


if __name__ == "__main__":
    main()
