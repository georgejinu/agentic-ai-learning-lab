"""Day 1: Orchestrator/worker pattern demo.

A central LLM (the orchestrator) looks at a topic and decides, at runtime,
which report sections are needed -- the number and names of sections are
NOT known ahead of time. It then dispatches one worker per section in
parallel (LangGraph's Send API), and a synthesizer joins their output into
a final report.

This is what distinguishes orchestrator-workers from routing: routing picks
one of a few pre-defined paths, this dynamically decomposes the task itself.

Run:
    python orchestrator_demo.py "the pros and cons of remote work"
"""

import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common.llm import get_llm

llm = get_llm()


class Section(BaseModel):
    name: str = Field(description="Short section heading")
    description: str = Field(description="What this section should cover")


class Plan(BaseModel):
    sections: list[Section] = Field(description="The sections needed to cover the topic well")


class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list[str], operator.add]
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list[str], operator.add]


def orchestrator(state: State) -> dict:
    """Decide how many sections this topic needs, and what each covers."""
    planner = llm.with_structured_output(Plan)
    plan = planner.invoke(
        f"Plan a short report on: {state['topic']}\n"
        "Break it into 2-4 sections. Each section should be independent enough "
        "to be written on its own."
    )
    print(f"[orchestrator] planned {len(plan.sections)} sections: "
          f"{[s.name for s in plan.sections]}")
    return {"sections": plan.sections}


def assign_workers(state: State) -> list[Send]:
    """Fan out: one worker per planned section, dispatched in parallel."""
    return [Send("worker", {"section": s}) for s in state["sections"]]


def worker(state: WorkerState) -> dict:
    """Write one section of the report."""
    section = state["section"]
    result = llm.invoke(
        f"Write a short paragraph (3-5 sentences) for a report section.\n"
        f"Heading: {section.name}\nCovers: {section.description}\n"
        "Return only the paragraph body, no heading."
    )
    print(f"[worker] finished section: {section.name}")
    return {"completed_sections": [f"## {section.name}\n\n{result.content}"]}


def synthesizer(state: State) -> dict:
    """Join all completed sections into the final report."""
    report = "\n\n".join(state["completed_sections"])
    return {"final_report": report}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("worker", worker)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", assign_workers, ["worker"])
    graph.add_edge("worker", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "the pros and cons of remote work"
    app = build_graph()
    result = app.invoke({"topic": topic, "completed_sections": []})

    print("\n" + "=" * 60)
    print(f"FINAL REPORT: {topic}")
    print("=" * 60)
    print(result["final_report"])


if __name__ == "__main__":
    main()
