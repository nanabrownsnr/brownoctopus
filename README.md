# 🐙 Brown Octopus

**Brown Octopus is a dynamic capability-context manager for AI agents.**

AI agents can have access to hundreds or thousands of tools, but passing every tool definition to the LLM on every turn increases context size, latency, and tool-selection complexity.

Brown Octopus sits **before the agent** and dynamically determines which subset of the agent's available tools should be exposed for the current conversation.

```text
Tool Universe
     │
     ▼
User Request
     │
     ▼
Brown Octopus
     │
     ├── Understand operational intents
     ├── Retrieve relevant capabilities
     ├── Preserve recently useful capabilities
     └── Expire / evict stale capabilities
     │
     ▼
Small Active Tool Context
     │
     ▼
Agent / LLM
     │
     ▼
Normal Tool Execution
```

Brown Octopus does **not** decide which tool the agent must call. It manages the capability surface from which the agent can choose.

## How It Works

### 1. Tool Discovery

Brown Octopus connects to configured MCP servers and discovers their tools dynamically.

Tool metadata is normalized, namespaced, deduplicated, and stored in an in-memory registry.

### 2. Intent Analysis

User requests are decomposed into operational intents.

For example:

```text
"Get the sprint work items and create a Word document summarizing them."
```

becomes approximately:

```text
Get the sprint work items
Create a Word document
Summarize them
```

This prevents compound requests from being treated as a single semantic search problem.

### 3. Capability Retrieval

Tool descriptions are embedded using:

```text
Qwen/Qwen3-Embedding-0.6B
```

Each intent independently retrieves its most relevant tools.

```text
Intent 1 → Top K
Intent 2 → Top K
Intent 3 → Top K
              │
              ▼
        Merge + Dedupe
```

The current default is **top 4 tools per intent** with no global retrieval cap.

The goal is high capability recall: every capability required to complete the request should survive into the agent's context.

### 4. Active Tool Context

Brown Octopus maintains capability context across conversation turns.

Previously relevant tools remain available so follow-up requests do not have to rediscover the entire task context.

```text
Turn 1: "Check my leave balance"
        → Leave tools

Turn 2: "Email it to Tom"
        → Leave + Email tools

Turn 3: "Actually send it to Sarah"
        → Leave + Email tools
```

The LLM remains responsible for understanding conversational references such as `"it"`. Brown Octopus is responsible for ensuring the required capabilities remain available.

### 5. Context Control

The active context is bounded using two mechanisms:

```text
TTL
Tools not retrieved for more than 8 turns expire.

MAX ACTIVE TOOLS
The context is capped at 30 tools.

If the cap is exceeded, the least recently retrieved
capabilities are evicted first.
```

This creates a rolling capability context that evolves with the conversation without growing indefinitely.

## Architecture

```text
MCP Servers
    │
    ▼
Tool Discovery
    │
    ▼
Tool Registry + Embedding Index
    │
    │
User Query
    ▼
Intent Analyzer
    │
    ▼
Per-Intent Retrieval
    │
    ▼
Merge + Dedupe
    │
    ▼
Active Tool Memory
    │
    ├── Refresh
    ├── TTL
    └── Capacity Eviction
    │
    ▼
Selected Tool Definitions
    │
    ▼
Agent
```

Brown Octopus is designed to be **agent-harness agnostic**. LangGraph, custom agents, or other orchestration systems can consume the selected tool definitions without Brown Octopus controlling the agent's execution loop.

## Current Status

The current prototype includes:

- Dynamic MCP tool discovery
- Namespaced tool registry
- Transformer-based operational intent analysis
- Qwen semantic tool retrieval
- Independent retrieval for compound intents
- Batched intent embedding
- Stateful active-tool memory
- 8-turn capability TTL
- 30-tool active-context ceiling
- Least-recently-retrieved eviction
- Unit, pipeline, end-to-end, and multi-turn conversation tests

The next major phase is **evaluation at scale**: measuring capability recall, context reduction, and latency across a larger realistic tool universe and request dataset.

## Core Principle

> **Brown Octopus does not need to know exactly which tool the agent will use. It needs to make sure the agent does not lose access to the tools it may need.**
