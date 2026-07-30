# Agentic AI Learning Lab

A hands-on learning path for building agentic AI systems — from LLM fundamentals to multi-agent architectures.

## Roadmap

### 1. Foundations
- [ ] LLM basics: tokens, context windows, sampling params
- [ ] Prompting techniques: zero/few-shot, chain-of-thought, structured output
- [ ] Working with an LLM API (Anthropic/OpenAI SDK basics)

### 2. Tool Use & Function Calling
- [ ] Defining tools/functions for an LLM to call
- [ ] Handling tool-call loops (request → execute → return result)
- [ ] Error handling and retries in tool execution

### 3. Retrieval-Augmented Generation (RAG)
- [ ] Embeddings and vector stores
- [ ] Chunking strategies
- [ ] Building a basic RAG pipeline
- [ ] Evaluating retrieval quality

### 4. Memory & State
- [ ] Short-term (conversation) memory
- [ ] Long-term memory (persisted facts/summaries)
- [ ] Session/state management across turns

### 5. Agent Frameworks
- [ ] Build a minimal agent loop from scratch (no framework)
- [ ] Explore an existing framework (e.g. LangGraph, Claude Agent SDK)
- [ ] Compare tradeoffs: control vs. convenience

### 6. Multi-Agent Systems
- [ ] Agent-to-agent communication patterns
- [ ] Orchestrator/worker architectures
- [ ] Parallel vs. sequential task decomposition

### 7. Evaluation & Safety
- [ ] Building eval sets for agent behavior
- [ ] Guardrails and input/output validation
- [ ] Handling failure modes (hallucination, infinite loops, runaway costs)

### 8. Deployment
- [ ] Packaging an agent as a service
- [ ] Observability: logging, tracing agent decisions
- [ ] Cost and latency optimization

## Structure

Each module will live in its own folder with runnable examples and notes as the lab progresses.

## Status

🚧 Just getting started.
