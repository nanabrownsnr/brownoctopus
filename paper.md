# Brown Octopus — Research & Development Notes

> Living research document. This file records the problem, hypotheses, experiments, architectural changes, findings, and open questions behind Brown Octopus. It is intended to serve as source material for a future research paper.

---

# 1. Research Problem

Modern AI agents can be connected to increasingly large collections of tools through MCP servers, APIs, plugins, and internal capabilities.

A straightforward agent architecture exposes all available tool definitions to the language model:

```text
User Request
     ↓
LLM + Entire Tool Universe
     ↓
Tool Selection
     ↓
Tool Execution
```

This becomes increasingly undesirable as the tool universe grows.

A large capability surface can increase:

- context consumption,
- inference cost,
- latency,
- tool-selection complexity,
- semantic competition between similar tools,
- and the amount of irrelevant information presented to the model.

This led to the initial research question:

> **Can an agent dynamically expose only the tools relevant to the current request without removing capabilities that the agent may need?**

The problem initially appeared to be a **tool recommendation problem**.

Our work so far suggests it is more accurately a **dynamic capability-context management problem**.

---

# 2. Initial Hypothesis

The original Brown Octopus concept was a deterministic preprocessing layer between the user and the LLM.

```text
Large Tool Universe
        │
        ▼
User → Tool Recommender → Small Tool Set → LLM
```

Instead of providing hundreds of tools to the model, a lightweight retrieval system would identify a small subset before inference.

The original design emphasized:

- deterministic retrieval,
- semantic similarity,
- low latency,
- independence from the main LLM,
- and high recall.

The recommender would not execute tools.

Its only responsibility would be determining which tools should be made available to the agent.

---

# 3. Early Retrieval Approach

The first retrieval experiments treated tool selection similarly to conventional semantic search.

Each tool was represented using metadata such as:

```text
tool name
tool description
input schema
MCP/server information
```

A query representation was generated and compared against tool embeddings.

Early experiments also explored separately representing:

```text
Action
Target
Context
```

and combining their similarity scores.

Conceptually:

```text
Q = wA × Action + wT × Target + wC × Context
```

This approach produced useful rankings for simple requests but revealed a more fundamental problem when requests contained multiple operational intents.

---

# 4. Discovery: Tool Retrieval Is Not Always a Single Search Problem

Consider:

```text
"Get the sprint work items and create a Word document summarizing them."
```

This request requires multiple capabilities:

```text
Sprint / project management
Word document creation
Summarization
```

A single global semantic ranking can allow one semantic neighborhood to dominate the available top-K positions.

For example, project-management tools may occupy most of the ranking while document capabilities disappear below the retrieval boundary.

This changed the retrieval problem from:

```text
query
 ↓
rank all tools
 ↓
global top K
```

to:

```text
query
 ↓
identify operational intents
 ↓
retrieve independently for each intent
 ↓
merge
 ↓
deduplicate
```

This became one of the central architectural changes in Brown Octopus.

---

# 5. Operational Intent Analysis

Brown Octopus currently uses spaCy's transformer English pipeline:

```text
en_core_web_trf
```

to identify operational actions within a request.

The analyzer attempts to distinguish between:

- primary requested actions,
- coordinated actions,
- nested operational actions,
- conditions,
- descriptive clauses,
- and supporting linguistic structure.

For example:

```text
"Search the web for the latest Nvidia news and send what you find to Tom."
```

is decomposed approximately into:

```text
Search the web for the latest Nvidia news
Send what you find to Tom
```

Importantly:

```text
find
```

is not treated as another independent operational request.

Similarly:

```text
"Create a Word document summarizing the sprint."
```

can expose both:

```text
Create a Word document
Summarizing the sprint
```

The guiding rule that emerged was approximately:

> An operational action owns its local linguistic subtree except where another identified operational action begins.

This allows Brown Octopus to construct local natural-language intent spans without relying on crude sentence or conjunction splitting.

---

# 6. Local Intent Retrieval

Each operational intent is embedded independently.

The current embedding model is:

```text
Qwen/Qwen3-Embedding-0.6B
```

Tool embeddings are constructed primarily from:

```text
tool name + tool description
```

The current retrieval process is:

```text
Raw Query
    ↓
Intent Analyzer
    ↓
Intent 1 ──→ Top 4 tools
Intent 2 ──→ Top 4 tools
Intent 3 ──→ Top 4 tools
    ↓
Merge
    ↓
Deduplicate
```

There is currently **no global retrieval cap after the per-intent retrieval stage**.

This is deliberate.

A global cap could recreate the original problem by removing the tools associated with one of the identified intents.

---

# 7. Shift in Evaluation Philosophy

Early experimentation naturally focused on ranking quality:

```text
Was the correct tool ranked #1?
```

This turned out not to match the actual role Brown Octopus plays.

The downstream LLM remains responsible for selecting and invoking tools.

Brown Octopus instead controls which capabilities are visible to that LLM.

Therefore the more important question is:

> **Did every capability required by the request survive into the tool context?**

This suggests that **capability recall** is more important than perfect ranking precision.

Conceptually:

```text
Capability Recall =

required capabilities successfully exposed
───────────────────────────────────────────
total required capabilities
```

If a task requires:

```text
Web Search
Email
Word
```

and Brown Octopus exposes all three plus several unnecessary tools, the retrieval may still be successful.

If it exposes only Web Search and Email, the system has failed because the agent has lost access to a required capability.

This led to the working principle:

> **Brown Octopus does not need to know exactly which tool the agent will use. It needs to make sure the agent does not lose access to the tools it may need.**

---

# 8. Batched Intent Embedding

Compound requests originally caused multiple sequential embedding calls:

```text
Intent 1 → encode()
Intent 2 → encode()
Intent 3 → encode()
```

The retriever was changed to batch all intent spans:

```text
[Intent 1, Intent 2, Intent 3]
              ↓
         one encode()
              ↓
      multiple embeddings
```

Each resulting embedding still receives its own independent retrieval neighborhood.

CPU experiments did not demonstrate a universal latency improvement from batching.

However, production deployment is intended for GPU execution, where batching represents a more appropriate inference architecture.

Therefore batching was retained.

---

# 9. Dynamic MCP Tool Discovery

The original prototype contained static or synthetic tool definitions.

This was replaced with live MCP discovery.

Brown Octopus now:

1. reads configured MCP URLs,
2. connects to each MCP,
3. retrieves server metadata,
4. calls `list_tools()`,
5. constructs namespaced tool identities,
6. deduplicates the resulting universe,
7. stores the tools in an in-memory registry,
8. builds the retrieval index.

Conceptually:

```text
MCP URLs
   ↓
MCP Discovery
   ↓
Server Metadata + list_tools()
   ↓
Namespaced Tools
   ↓
Deduplication
   ↓
Tool Registry
   ↓
Embedding Index
```

The current experimental universe contains approximately 122 discovered tools across services including email, project management, document creation, source control, file storage, databases, web search, and other capabilities.

Partial MCP failure is tolerated so one unavailable MCP does not prevent Brown Octopus from initializing the rest of the capability universe.

---

# 10. Discovery: Tool Context Must Be Stateful

The initial architecture retrieved tools independently on every turn.

Conversation exposed another problem.

Consider:

```text
Turn 1:
"Check my remaining leave balance."

Turn 2:
"Email it to Tom."
```

The second message requires an email capability, but the conversation still depends on the capability introduced during the first turn.

Brown Octopus therefore evolved from a stateless retriever into a stateful capability-context manager.

```text
Current Retrieval
       +
Previously Active Capabilities
       ↓
Active Tool Context
       ↓
Agent
```

Example:

```text
Turn 1
Leave capability

Turn 2
Leave + Email

Turn 3
Leave + Email

Turn 4
Leave + Email + Web Search
```

Brown Octopus does not attempt to resolve what pronouns such as `"it"` refer to.

That remains the responsibility of the language model and its conversational context.

The responsibility boundary is:

```text
LLM
→ understand conversational meaning

Brown Octopus
→ maintain capability availability
```

---

# 11. Capability Expiration

Persistent capability memory introduces another problem:

```text
Turn 1  → tools accumulate
Turn 2  → more tools
Turn 3  → more tools
...
Turn N  → potentially the entire universe
```

This would eventually recreate the original tool-overload problem.

Brown Octopus therefore introduced a capability TTL.

Current value:

```text
ACTIVE_TOOL_TTL = 8 turns
```

Each active tool stores its most recent retrieval turn.

Retrieving the tool again refreshes that timestamp.

A capability expires when:

```text
current_turn - last_retrieved_turn > TTL
```

An interesting experimental observation was that semantic retrieval can occasionally retrieve an unnecessary capability and therefore refresh its lifetime.

For the current high-recall design, this behavior has been accepted rather than introducing confidence thresholds or more complicated refresh policies.

The assumption is that some unnecessary active tools are acceptable provided the active context remains bounded.

---

# 12. Hard Context Ceiling

TTL alone does not guarantee a bounded context.

A conversation may continually introduce new capabilities faster than older capabilities expire.

Brown Octopus therefore now also enforces:

```text
MAX_ACTIVE_TOOLS = 30
```

After retrieval timestamps are refreshed and TTL expiration occurs:

```text
if active tools > 30:
    remove least recently retrieved tools
```

The resulting policy is effectively an LRU-style capability cache.

```text
                    ACTIVE CONTEXT
                          │
             ┌────────────┴────────────┐
             │                         │
            TTL                     CAPACITY
          8 turns                    30 tools
             │                         │
     stale capability?       context still too large?
             │                         │
           eject                  eject oldest
```

The two mechanisms solve different problems:

**TTL**

> Has this capability stopped being relevant?

**Capacity ceiling**

> Even if many capabilities are recent, is the context becoming too large?

---

# 13. Multi-Turn Context Experiments

The active-context mechanism has now been tested at multiple levels.

### Unit behavior

Tests verify:

- capabilities persist,
- newly retrieved capabilities are added,
- duplicate capabilities are not created,
- TTL boundaries behave correctly,
- retrieval refreshes timestamps,
- stale capabilities expire,
- context cannot exceed 30 tools,
- multiple old capabilities are evicted when a batch pushes the context beyond the ceiling.

### Pipeline behavior

A pipeline test demonstrated:

```text
30 active capabilities
        +
4 newly retrieved capabilities
        ↓
34 candidates
        ↓
4 oldest capabilities evicted
        ↓
30 active tool definitions
```

This demonstrated that eviction affects the actual definitions exposed to the agent rather than only internal bookkeeping.

### Conversation behavior

A multi-turn experiment demonstrated:

```text
Turn 1 → 10 active
Turn 2 → 20 active
Turn 3 → 30 active
Turn 4 → +4 new
             ↓
            34
             ↓
      remove 4 oldest
             ↓
            30
```

This demonstrated that Brown Octopus behaves as a rolling capability context over successive turns.

---

# 14. Current Architecture

The current system can be summarized as:

```text
                         MCP SERVERS
                              │
                              ▼
                       Tool Discovery
                              │
                              ▼
                    Registry + Embeddings
                              │
                              │
USER MESSAGE ─────────────────┘
      │
      ▼
Operational Intent Analysis
      │
      ▼
Local Intent Spans
      │
      ▼
Batched Qwen Embedding
      │
      ▼
Independent Top-K Retrieval
      │
      ▼
Merge + Deduplicate
      │
      ▼
Active Capability Context
      │
      ├── Add
      ├── Refresh
      ├── TTL Expiration
      └── Capacity Eviction
      │
      ▼
Selected Tool Definitions
      │
      ▼
Agent / LLM
      │
      ▼
Normal LLM → Tool → LLM Loop
```

Brown Octopus intentionally does not execute tools.

It is also intended to remain agent-harness agnostic.

LangGraph, custom orchestration systems, or other agent runtimes should be able to consume the resulting capability context.

---

# 15. Current Performance Findings

CPU profiling has revealed two separate performance concerns.

## Runtime Retrieval

Warm retrieval currently ranges approximately from hundreds of milliseconds for simple requests to around two seconds for larger compound requests on the development CPU environment.

This is not considered representative of the intended production environment because GPU deployment is planned.

## Initialization

Startup profiling identified a much larger CPU bottleneck.

Approximate measurements:

```text
Analyzer initialization       ~5 s
Embedding model load         ~12 s
MCP discovery                ~25 s
Tool indexing               ~829 s
```

Tool embedding therefore dominated initialization time in the CPU development environment.

A likely future optimization is persistent embedding caching:

```text
Tool unchanged
→ load cached embedding

Tool added
→ embed new tool

Tool description changed
→ regenerate embedding

Tool removed
→ remove cached embedding
```

This has not yet been implemented because production GPU measurements should be collected first.

---

# 16. What Has Been Established So Far

The experiments currently support several architectural conclusions.

### 1. Tool selection benefits from operational decomposition

Compound requests should not necessarily compete inside one global semantic ranking.

### 2. High recall is more important than perfect ranking

Brown Octopus controls capability availability rather than making the final tool-selection decision.

### 3. Tool context should be conversationally stateful

Capabilities useful on previous turns can remain necessary even when they are not explicitly mentioned again.

### 4. Stateful capability context must be bounded

TTL and capacity-based eviction prevent the active capability surface from eventually returning to the full tool universe.

### 5. Capability management can remain separate from agent execution

Brown Octopus can operate before the LLM without replacing the normal LLM → tool → LLM execution loop.

---

# 17. Current Research Hypothesis

The project can now be expressed more precisely.

> **A stateful capability-context manager can substantially reduce the number of tool definitions exposed to an AI agent while preserving high recall of the capabilities required to complete user requests across multi-turn conversations.**

Brown Octopus is the experimental system being developed to test this hypothesis.

The central trade-off is:

```text
             SMALLER TOOL CONTEXT
                     ↑
                     │
                     │
       How far can we reduce it?
                     │
                     ↓
          CAPABILITY PRESERVATION
```

Reducing the context is only useful if the agent retains the capabilities necessary to perform its task.

---

# 18. What We Have Not Yet Proven

The current prototype demonstrates architectural feasibility, but several important questions remain unanswered.

## Retrieval Quality at Scale

We have not yet measured capability recall across a sufficiently large and diverse evaluation dataset.

## Scale

The current real tool universe is approximately 122 tools.

Behavior with hundreds or thousands of tools remains to be measured.

## Production GPU Latency

Current latency measurements are CPU measurements.

Production GPU retrieval and indexing performance remains unknown.

## Agent-Level Impact

We have not yet performed the critical comparison:

```text
Agent + Entire Tool Universe
            VS
Agent + Brown Octopus Context
```

The effect on:

- task completion,
- tool-selection accuracy,
- token usage,
- inference latency,
- and overall cost

remains to be experimentally measured.

## Dynamic Universe Updates

The current lifecycle primarily indexes tools during initialization.

Incremental handling of added, removed, or modified MCP tools remains future work.

---

# 19. Next Research Phase

The next major phase should focus on **measurement rather than additional architecture**.

A realistic evaluation dataset should be constructed containing categories such as:

```text
single-intent requests
compound requests
nested operational requests
multi-turn conversations
ambiguous requests
requests requiring no tool
similar/competing tools
multiple providers exposing similar capabilities
```

Each test case should identify the capabilities required to complete the task.

Primary metrics should include:

### Capability Recall

```text
required capabilities exposed
─────────────────────────────
total required capabilities
```

### Context Reduction

```text
1 - (
    exposed tools
    ─────────────
    available tools
)
```

### Retrieval Latency

Time added by Brown Octopus before agent inference.

### Active Context Size

Tool count across multi-turn conversations.

Later experiments should measure downstream agent outcomes including:

- correct tool selection,
- task completion,
- input tokens,
- latency,
- and inference cost.

---

# 20. Planned Experimental Comparison

A future controlled experiment should compare at least:

```text
A. Full Tool Universe
   User → LLM + all tools

B. Stateless Retrieval
   User → semantic top-K → LLM

C. Brown Octopus
   User
     → operational decomposition
     → per-intent retrieval
     → stateful bounded context
     → LLM
```

This comparison should help isolate whether the additional architecture provides measurable benefits beyond ordinary semantic tool retrieval.

---

# 21. Longer-Term Questions

Several questions remain intentionally open.

- What is the optimal per-intent K?
- Is 30 the appropriate active-context ceiling?
- Is an 8-turn TTL appropriate across different task types?
- Should TTL be adaptive?
- Should active context operate at tool level or capability level?
- Does sparse+dense hybrid retrieval improve capability recall?
- How should duplicate capabilities from different providers be handled?
- At what tool-universe size does Brown Octopus begin producing significant benefits?
- How does tool-context reduction affect LLM tool-selection accuracy?
- Can the same intent representation eventually route models as well as tools?
- Can capability routing generalize beyond MCP tools?

These should remain experimental questions rather than being prematurely encoded into the architecture.

---

# 22. Current Principle

The project currently rests on one central idea:

> **The goal is not to predict exactly what the agent will do. The goal is to dynamically maintain the capability surface from which the agent can successfully decide what to do.**

Brown Octopus therefore acts as a layer between a large capability universe and an autonomous agent:

```text
Everything the agent COULD use
              ↓
        Brown Octopus
              ↓
What the agent MAY NEED now
              ↓
             LLM
              ↓
What the agent CHOOSES to use
```

That distinction is the core of the current research direction.

---

## Research Status

**Phase:** Architecture validated / evaluation design beginning

**Current real tool universe:** ~122 discovered tools

**Current retrieval:** Qwen3-Embedding-0.6B, top 4 per operational intent

**Context memory:** 8-turn TTL

**Maximum active context:** 30 tools

**Next milestone:** Build a capability-recall evaluation dataset and benchmark Brown Octopus against baseline tool-context strategies.
