"""Pattern 5: Evaluator-Optimizer.

One LLM generates, a second LLM evaluates the result against explicit
criteria and gives feedback, and the loop repeats until the evaluator is
satisfied or a max-iteration cap is hit. Graph-shape twin of the
Reflection/Reflexion pattern -- same idea, different name.

Graph shape: generator -> evaluator -> conditional_edge ->
             (loop back to generator | END)

Here: generate a product tagline that must (a) be <= 10 words,
(b) mention "speed", (c) end with a call-to-action. Capped at 3 attempts.

Run:
    python evaluator_optimizer_demo.py "a project management app"
"""

import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.llm import get_llm

llm = get_llm()

MAX_ITERATIONS = 3

CRITERIA = (
    "- 6 words or fewer\n"
    "- must contain the exact word 'speed' (not a synonym like 'fast' or 'quick')\n"
    "- must contain the exact word 'now'\n"
    "- must contain a number\n"
    "- must not contain the words 'app' or 'download'"
)


class Evaluation(BaseModel):
    passed: bool
    feedback: str = Field(description="What to fix, if it did not pass")


class State(TypedDict):
    product: str
    tagline: str
    feedback: str
    iteration: int
    passed: bool


def generate(state: State) -> dict:
    feedback_note = (
        f"\n\nYour previous attempt failed review. Feedback: {state['feedback']}"
        if state.get("feedback")
        else ""
    )
    resp = llm.invoke(
        f"Write one marketing tagline for: {state['product']}\n"
        f"It must follow these rules:\n{CRITERIA}{feedback_note}\n"
        "Return only the tagline, nothing else."
    )
    iteration = state.get("iteration", 0) + 1
    print(f"[generate] attempt {iteration}: {resp.content!r}")
    return {"tagline": resp.content, "iteration": iteration}


def evaluate(state: State) -> dict:
    evaluator = llm.with_structured_output(Evaluation)
    result = evaluator.invoke(
        f"Does this tagline satisfy every rule below? Be strict.\n\n"
        f"Rules:\n{CRITERIA}\n\nTagline: {state['tagline']}"
    )
    print(f"[evaluate] passed={result.passed} feedback={result.feedback!r}")
    return {"passed": result.passed, "feedback": result.feedback}


def should_continue(state: State) -> str:
    if state["passed"] or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges(
        "evaluate", should_continue, {"retry": "generate", "end": END}
    )
    return graph.compile()


def main():
    product = sys.argv[1] if len(sys.argv) > 1 else "a project management app"
    app = build_graph()
    result = app.invoke({"product": product, "iteration": 0, "feedback": ""})

    print(f"\nfinal tagline (after {result['iteration']} attempt(s), "
          f"passed={result['passed']}):")
    print(result["tagline"])


if __name__ == "__main__":
    main()
 