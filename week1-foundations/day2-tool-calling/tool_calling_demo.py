"""Day 2: Tool calling & the agent loop, built with Google ADK.

Day 1 used LangGraph; this one deliberately switches frameworks to see the
same agent loop -- LLM call -> decide if a tool is needed -> execute tool ->
feed result back -> repeat until the model has enough to answer -- expressed
through Google's Agent Development Kit (ADK) instead.

ADK's Gemini wrapper goes through the same google-genai client as
common/llm.py, so it honors the same Vertex AI env vars (GOOGLE_CLOUD_PROJECT
etc.) from .env.

Run (single turn):
    python tool_calling_demo.py "Does plan PPO-100 need prior auth for J1745?"

Run (multiple turns in one session -- each arg is a separate turn, later
turns can reference earlier ones since they share session history):
    python tool_calling_demo.py \
        "Does plan PPO-100 need prior auth for J1745?" \
        "What about drug J3490 on the same plan?" \
        "And HMO-200 for J1745?"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

if sys.platform == "win32":
    # aiohttp's connector cleanup races with ProactorEventLoop teardown at
    # interpreter exit, producing a harmless but noisy "Exception ignored in
    # __del__" traceback. SelectorEventLoop doesn't have that race.
    # set_event_loop_policy is deprecated as of Python 3.14 (removal in 3.16)
    # in favor of passing loop_factory to asyncio.run/Runner -- but that only
    # helps when we control the asyncio.run() call site. ADK's Runner.run()
    # calls asyncio.run() internally on its own thread, so the global policy
    # is the only lever available here.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")


def _configure_langsmith_tracing():
    """Export ADK's own built-in OpenTelemetry spans to LangSmith via OTLP.

    ADK already instruments itself with OTel (google.adk.telemetry -- LLM
    calls, tool calls, using the standard gen_ai.* semantic conventions) and
    calls trace.get_tracer() internally. It has no LangSmith-specific hook;
    what makes this "the ADK way" rather than a manual @traceable wrapper is
    that we're not touching ADK's code at all -- we're just pointing the
    process's global OTel TracerProvider at LangSmith's OTLP endpoint before
    any ADK code runs, so the spans ADK already emits end up there.

    Returns the TracerProvider (so callers can force_flush() before exit --
    the exporter batches spans on a background thread, which a short-lived
    script can outrun) or None if tracing wasn't enabled/configured.
    """
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return None
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("[tracing] LANGSMITH_TRACING=true but LANGSMITH_API_KEY is unset -- skipping.")
        return None

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    base = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/")
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    exporter = OTLPSpanExporter(
        endpoint=f"{base}/otel/v1/traces",
        headers={"x-api-key": api_key, "Langsmith-Project": project},
    )
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    print(f"[tracing] exporting to LangSmith project {project!r}")
    return provider


_TRACER_PROVIDER = _configure_langsmith_tracing()

# Mock prior-authorization rules table -- stands in for a real eligibility
# lookup (e.g. a plan's PA criteria database) so the demo has no external
# dependencies.
PA_RULES = {
    ("PPO-100", "J1745"): {"requires_pa": True, "criteria": "step therapy: methotrexate trial required first"},
    ("PPO-100", "J3490"): {"requires_pa": False, "criteria": None},
    ("HMO-200", "J1745"): {"requires_pa": True, "criteria": "specialist referral + diagnosis code confirmation"},
}


def check_prior_auth_requirement(plan_id: str, drug_code: str) -> dict:
    """Look up whether a plan requires prior authorization for a drug.

    Args:
        plan_id: The insurance plan identifier, e.g. "PPO-100".
        drug_code: The drug's billing code (HCPCS/J-code), e.g. "J1745".

    Returns:
        A dict with `requires_pa` (bool) and `criteria` (str or None).
    """
    result = PA_RULES.get((plan_id, drug_code))
    print(f"[tool] check_prior_auth_requirement(plan_id={plan_id!r}, drug_code={drug_code!r}) -> {result}")
    if result is None:
        return {"requires_pa": None, "criteria": "no rule on file for this plan/drug combination"}
    return result


root_agent = Agent(
    name="pa_eligibility_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a pharmacy benefits assistant. When asked whether a drug "
        "needs prior authorization for a plan, always call "
        "check_prior_auth_requirement rather than guessing -- never answer "
        "from your own knowledge. Summarize the tool result in one sentence."
    ),
    tools=[check_prior_auth_requirement],
)


def print_event(event) -> None:
    """Surface each step of the agent loop as it happens."""
    for call in event.get_function_calls():
        print(f"[loop] model requested tool call: {call.name}({call.args})")
    for response in event.get_function_responses():
        print(f"[loop] tool result fed back to model: {response.response}")
    if event.is_final_response() and event.content and event.content.parts:
        text = "".join(p.text or "" for p in event.content.parts)
        print(f"[loop] final answer: {text}")


async def _ensure_session(runner: InMemoryRunner, user_id: str, session_id: str) -> None:
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )


DEFAULT_QUESTIONS = [
    # Turn 1: single fact, single tool call, single loop round-trip.
    "Does plan PPO-100 need prior auth for J1745?",
    # Turn 2: compound question -- both facts are independent (neither
    # depends on the other's result), so the model can request both tool
    # calls in one response instead of looping sequentially. Still a
    # separate turn/trace from turn 1 above, just not a separate trace from
    # itself -- one turn, one trace, two function_call parts in it.
    "Does PPO-100 need prior auth for J1745, and does HMO-200 also need it for J1745?",
]


def main():
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    runner = InMemoryRunner(agent=root_agent)
    user_id, session_id = "demo-user", "demo-session"
    asyncio.run(_ensure_session(runner, user_id, session_id))

    for i, question in enumerate(questions, start=1):
        print(f"--- turn {i} {'-' * 40}")
        print(f"[user] {question}\n")
        message = types.Content(role="user", parts=[types.Part(text=question)])
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=message):
            print_event(event)
        print()

    if _TRACER_PROVIDER is not None:
        # BatchSpanProcessor exports on a background thread; a short-lived
        # script can exit before it flushes on its own, silently dropping
        # spans. Force it before the process ends.
        _TRACER_PROVIDER.force_flush()


if __name__ == "__main__":
    main()
