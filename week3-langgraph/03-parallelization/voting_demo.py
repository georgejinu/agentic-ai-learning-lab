"""Pattern 3b: Parallelization -- Voting.

Run the same task multiple times in parallel (different prompt framings,
or higher temperature for diversity) and aggregate the answers by vote,
for higher confidence than a single call. Contrast with sectioning: here
every branch does the SAME job on the SAME input, not different jobs.

Graph shape: START fans out to N identical-purpose nodes -> aggregator
tallies votes -> END

Here: three independent LLM calls each judge whether a message looks like
spam; flag it only if at least 2 of 3 agree.

Run:
    python voting_demo.py "Congratulations! You've won a free prize, click here now!!!"
"""

import operator
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.llm import get_llm

Verdict = Literal["spam", "not_spam"]


class Judgment(BaseModel):
    verdict: Verdict
    reason: str = Field(description="One short sentence")


class State(TypedDict):
    message: str
    votes: Annotated[list[str], operator.add]
    final_verdict: str


# A slightly different temperature per voter gives genuinely independent
# judgments instead of three identical calls.
def _voter(prompt_framing: str, temperature: float):
    def vote(state: State) -> dict:
        llm = get_llm(temperature=temperature).with_structured_output(Judgment)
        result = llm.invoke(f"{prompt_framing}\n\nMessage:\n{state['message']}")
        print(f"[vote temp={temperature}] -> {result.verdict} ({result.reason})")
        return {"votes": [result.verdict]}
    return vote


vote_1 = _voter("Is this message spam? Judge strictly on urgency/prize language.", 0.0)
vote_2 = _voter("Is this message spam? Judge based on overall intent and tone.", 0.7)
vote_3 = _voter("Is this message spam? Would a typical email provider flag it?", 0.7)


def aggregate(state: State) -> dict:
    spam_votes = state["votes"].count("spam")
    final = "spam" if spam_votes >= 2 else "not_spam"
    print(f"[aggregate] votes={state['votes']} -> {final}")
    return {"final_verdict": final}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("vote_1", vote_1)
    graph.add_node("vote_2", vote_2)
    graph.add_node("vote_3", vote_3)
    graph.add_node("aggregate", aggregate)

    graph.add_edge(START, "vote_1")
    graph.add_edge(START, "vote_2")
    graph.add_edge(START, "vote_3")
    graph.add_edge("vote_1", "aggregate")
    graph.add_edge("vote_2", "aggregate")
    graph.add_edge("vote_3", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile()


def main():
    message = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Congratulations! You've won a free prize, click here now!!!"
    )
    app = build_graph()
    result = app.invoke({"message": message, "votes": []})
    print(f"\nfinal verdict: {result['final_verdict']}")


if __name__ == "__main__":
    main()
