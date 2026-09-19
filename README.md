# 🐙 Brown Octopus

**Dynamic capability-context management for AI agents**

Brown Octopus is a stateful, harness-agnostic capability-context manager for tool-using AI agents.

Instead of exposing an agent to every available tool on every turn, Brown Octopus dynamically identifies the operational intents in a request, retrieves a compact neighborhood of relevant capabilities for each intent, and maintains useful capabilities across a multi-turn conversation.

The goal is simple:

> Give the agent the capabilities it is likely to need without forcing the LLM to reason over the entire tool universe.

Brown Octopus does **not** execute tools and does **not** replace the agent's tool-selection logic.

It determines which tool definitions should be available to the agent. The agent remains responsible for deciding which tools to call.

---

# Why Brown Octopus?

Modern agents may have access to tens, hundreds, or eventually thousands of tools.

Passing every tool definition to the LLM on every turn creates several problems:

- larger context windows
- increased token usage
- increased inference latency
- greater semantic competition between tools
- harder tool selection
- poor scalability as capability libraries grow

A common solution is retrieval:

```text
user request
    ↓
semantic search
    ↓
top K tools
    ↓
LLM
```

But fixed `K` introduces another problem.

Different tasks have different capability widths.

A web-search request may require only one or two relevant capabilities, while an email, project-management, repository, or document-authoring task may benefit from a larger family of related operations.

Brown Octopus therefore treats tool exposure as **dynamic capability-context management**, rather than fixed-size tool retrieval.

---

# Core Idea

Brown Octopus exposes a **capability neighborhood** around each operational intent.

For example:

```text
"Find the latest Nvidia news and email a summary to Tom"
```

is decomposed into:

```text
Intent 1
"Find the latest Nvidia news"

Intent 2
"email a summary to Tom"
```

Each intent receives its own semantic ranking and adaptive capability boundary.

An observed retrieval may look like:

```text
Find the latest Nvidia news
        ↓
github search
web search

email a summary to Tom
        ↓
compose email
send email
reply email
find email address
search mail
...
```

The results are then merged and deduplicated before being exposed to the agent.

This allows different intents to consume different amounts of the capability budget.

---

# Architecture

```text
                     Tool Universe
                          │
                          │
                    Persisted Index
                          │
                          ▼
User Message ──► Operational Intent Analysis
                          │
                          ▼
                 Per-Intent Semantic Ranking
                          │
                          ▼
                    Bounded Max Gap
                          │
                ┌─────────┴─────────┐
                │                   │
           Intent A             Intent B
        neighborhood         neighborhood
                │                   │
                └─────────┬─────────┘
                          ▼
                    Merge + Dedupe
                          │
                          ▼
                 Active Capability Memory
                    TTL + Context Cap
                          │
                          ▼
                   Tool Definitions
                          │
                          ▼
                     Agent / LLM
                          │
                          ▼
                    Tool Execution
```

Brown Octopus sits **before** the agent's normal tool-calling loop.

The normal execution path remains:

```text
LLM → tool call → tool execution → LLM
```

---

# Adaptive Capability Selection

Brown Octopus currently uses **Bounded Max Gap**.

For each operational intent:

1. Rank the tool universe semantically.
2. Inspect the top 16 candidates plus rank 17 for boundary measurement.
3. Measure adjacent similarity-score gaps.
4. Find the strongest gap inside the bounded candidate neighborhood.
5. If the strongest gap is at least 2%, expose everything above that boundary.
6. Otherwise expose the full 16-tool candidate neighborhood.

Current defaults:

```text
MAX_TOOLS = 16
MIN_GAP_PERCENT = 2.0
```

Conceptually:

```text
semantic ranking
      ↓
inspect bounded neighborhood
      ↓
find strongest adjacent gap
      ↓
meaningful boundary?
   ┌───────┴───────┐
  yes              no
   │                │
cut there      return full cap
```

The hard bound is important.

An unrestricted maximum-gap search can find large score cliffs deep in the irrelevant tail of the ranking. Brown Octopus only searches for a boundary within the bounded candidate neighborhood.

Therefore:

```text
maximum exposed tools per intent = 16
```

regardless of the size of the underlying tool universe.

---

# Why Capability Neighborhoods?

The most obvious tool is not always sufficient to complete a task autonomously.

Consider:

```text
Send an email to Tom
```

The primary capability may be:

```text
send_email
```

but the agent may first need:

```text
find_email_address
```

Similarly:

```text
Create a Word report
```

may require:

```text
create_word_document
add_heading
add_paragraph
add_table
```

Brown Octopus therefore prioritizes **capability recall** over selecting the mathematically smallest possible tool set.

The objective is not:

> Find the single tool that best matches the user's wording.

It is:

> Expose a compact capability neighborhood containing the primary and plausible supporting capabilities required for autonomous task completion.

---

# Stateful Capability Memory

Capability requirements persist across conversational turns.

Example:

```text
Turn 1:
"Find the latest Nvidia news."

Turn 2:
"Email it to Tom."
```

The second turn contains a conversational reference to the previous result.

Brown Octopus therefore maintains active capability state.

Current defaults:

```text
Capability TTL:       8 turns
Global active cap:    30 tools
```

Retrieved capabilities refresh their TTL.

When the global capability context exceeds the cap, the oldest capabilities are evicted.

This creates a rolling capability context rather than rebuilding the entire tool environment independently on every turn.

---

# Operational Intent Analysis

Brown Octopus decomposes compound requests into operational actions before retrieval.

For example:

```text
"Create a Word document summarizing the sprint"
```

becomes approximately:

```text
Create a Word document
summarizing the sprint
```

and:

```text
"Search the web for the latest Nvidia news and send what you find to Tom"
```

becomes:

```text
Search the web for the latest Nvidia news
send what you find to Tom
```

Intent analysis currently uses:

```text
spaCy
en_core_web_trf
```

The analyzer attempts to isolate operational actions while avoiding treating every subordinate verb as an independent tool requirement.

---

# Semantic Retrieval

Tool retrieval currently uses:

```text
Qwen/Qwen3-Embedding-0.6B
```

Tool embeddings are built from:

```text
tool name + tool description
```

The current index stores:

```text
data/indexes/default/
├── tools.json
├── embeddings.pt
└── metadata.json
```

A persisted index allows normal startup without rediscovering MCP servers or recomputing tool embeddings.

---

# MCP Discovery

Brown Octopus can construct its capability universe from MCP servers.

The discovery pipeline:

```text
MCP URLs
   ↓
connect
   ↓
list_tools()
   ↓
normalize metadata
   ↓
deduplicate
   ↓
semantic index
```

The current development universe contains **122 tools** collected from multiple MCP servers covering capabilities including:

- email
- web search
- GitHub
- project management
- SharePoint
- Word/document generation
- databases
- HR workflows
- Railway/deployment operations
- file storage

The architecture itself is not MCP-dependent. MCP currently provides a convenient standardized source of tool metadata.

---

# Public API

```python
from octopus import Octopus


octopus = Octopus(
    index_path="data/indexes/default",
    catalog_path="data/mcps.json",
)

await octopus.initialize()
```

Retrieve capabilities without modifying conversational state:

```python
tools = octopus.retrieve(
    "Find the latest Nvidia news"
)
```

Process a conversational turn and update capability memory:

```python
result = octopus.process(
    "Find the latest Nvidia news"
)
```

Reset conversational capability state:

```python
octopus.reset()
```

Explicitly rebuild the capability universe and persisted index:

```python
await octopus.update()
```

Normal initialization loads the persisted index and does not require MCP rediscovery or re-embedding.

---

# Example Retrieval Behavior

Using the current 122-tool development universe:

| Request                            | Tools exposed |
| ---------------------------------- | ------------: |
| Send an email to Tom               |            11 |
| Create a Word report               |             4 |
| Get my leave applications          |             4 |
| Submit a leave application         |             4 |
| Find a GitHub repository           |            16 |
| Check deployment status on Railway |            10 |
| Get current sprint work items      |            12 |
| Find Nvidia news + email summary   |            13 |

For the compound Nvidia request, the two operational intents independently produced approximately:

```text
news retrieval → 2 capabilities
email          → 11 capabilities
```

before merge/deduplication.

These are development observations, not benchmark results.

---

# Testing

The current implementation passes:

```text
89 tests
```

covering:

- operational intent analysis
- semantic retrieval
- bounded Max Gap selection
- per-intent retrieval
- cross-intent merge/deduplication
- active capability memory
- TTL expiration
- global capability caps
- persisted indexes
- MCP discovery
- pipeline behavior
- multi-turn conversations
- public Octopus API
- end-to-end flows

Run the suite with:

```bash
uv run pytest -v
```

---

# Project Structure

```text
octopus/
├── data/
│   ├── mcps.json
│   ├── experiments/
│   └── indexes/
│       └── default/
│           ├── tools.json
│           ├── embeddings.pt
│           └── metadata.json
│
├── scripts/
│   ├── build_index.py
│   ├── experiment_adaptive_cutoff.py
│   └── experiment_v2_smoke_test.py
│
├── src/octopus/
│   ├── octopus.py
│   ├── analyzer.py
│   ├── retriever.py
│   ├── selection.py
│   ├── active_tools.py
│   ├── pipeline.py
│   ├── index_store.py
│   ├── tool_registry.py
│   ├── mcp_discovery.py
│   ├── sources.py
│   └── bootstrap.py
│
└── tests/
```

---

# Research Status

Brown Octopus is currently a research project.

The implementation architecture has been frozen sufficiently to begin systematic evaluation.

The next phase evaluates whether dynamic capability-context management can reduce exposed tool context while preserving or improving the capabilities available to an agent.

Planned comparisons include:

```text
ALL TOOLS
vs
external dynamic tool-retrieval/context approaches
vs
BROWN OCTOPUS
```

Evaluation dimensions include:

- required capability recall
- supporting capability recall
- tool-context reduction
- tool-schema token reduction
- retrieval latency
- downstream task success
- multi-tool task completion
- multi-turn capability continuity
- scalability as the tool universe grows

Standardized tool-retrieval and tool-use benchmarks will also be investigated.

---

# Research Question

The central question behind Brown Octopus is:

> Can an AI agent operate with a small, dynamically maintained capability context while retaining the task performance of an agent given access to the entire tool universe?

A secondary hypothesis is that exposing **capability neighborhoods**, rather than only the highest-ranked tools, may improve autonomous task completion by preserving supporting operations required for intermediate steps.

---

# Status

```text
MCP discovery                 ✓
Persisted capability index    ✓
Operational intent analysis   ✓
Semantic capability ranking   ✓
Bounded adaptive selection    ✓
Per-intent retrieval          ✓
Merge / deduplication         ✓
Stateful capability memory    ✓
TTL / context rotation        ✓
Public API                    ✓
Regression suite              89 passing
Systematic evaluation         next
```

---

# License

License to be determined.

---

# Author

**Nana Brown**

Brown Octopus is an independent research project exploring scalable capability management for tool-using AI agents.
