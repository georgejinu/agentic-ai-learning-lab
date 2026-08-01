# Agentic AI Learning Lab

A hands-on learning path for building agentic AI systems — from LLM fundamentals to multi-agent architectures.

## Roadmap

### Week 1 — Foundations
- [x] Day 1: Orchestrator/worker pattern (`week1-foundations/day1-orchestrator-worker/`)
- [ ] Day 2: Tool calling (`week1-foundations/day2-tool-calling/`)
- [ ] Day 3: Context engineering (`week1-foundations/day3-context-engineering/`)
- [ ] Day 4: ReAct & reflection (`week1-foundations/day4-react-reflection/`)
- [ ] Day 5: Guardrails & evals (`week1-foundations/day5-guardrails-evals/`)

### Week 2 — Vertex AI & Gemini
- [ ] Building agentic workflows on Vertex AI / Gemini (`week2-vertex-gemini/`)

### Week 3 — LangGraph
- [ ] Graph-based agent orchestration with LangGraph (`week3-langgraph/`)

### Week 4 — Production
- [ ] Deployment, observability, and operating agents in production (`week4-production/`)

### Week 5 — Applied Build
- [ ] Capstone project applying skills from prior weeks (`week5-applied-build/`)

### Week 6 — Leadership
- [ ] Leading agentic AI adoption and strategy (`week6-leadership/`)

## Structure

```
agentic-ai-learning-lab/
├── README.md
├── week1-foundations/
│   ├── day1-orchestrator-worker/
│   ├── day2-tool-calling/
│   ├── day3-context-engineering/
│   ├── day4-react-reflection/
│   └── day5-guardrails-evals/
├── week2-vertex-gemini/
├── week3-langgraph/
├── week4-production/
├── week5-applied-build/
└── week6-leadership/
```

Each day/week folder holds notes, write-ups, and runnable code for that topic.

Runnable demos share one LLM client, [`common/llm.py`](common/llm.py) (Vertex
AI Gemini). Setup:

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # .venv/bin/pip on macOS/Linux
cp .env.example .env                            # fill in GOOGLE_CLOUD_PROJECT
```

## Status

🚧 In progress — Day 1 (orchestrator-workers) and the LangGraph workflow
patterns in `week3-langgraph/` are implemented; the rest of the roadmap is
still ahead.
