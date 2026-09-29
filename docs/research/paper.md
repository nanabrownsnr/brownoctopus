# Brown Octopus: Stateful Dynamic Capability-Context Management for Tool-Using AI Agents

**Nana Brown**

_Working research manuscript — September 2026_

---

## Abstract

Large language model agents increasingly interact with external tools, APIs, and services through structured function definitions. As the number of available tools grows, exposing the complete tool universe to the language model on every turn becomes increasingly inefficient and may make tool selection more difficult. Existing retrieval-based approaches can reduce this context by selecting tools relevant to a user request, but fixed-size retrieval introduces a different limitation: different operational intents may require capability neighborhoods of substantially different sizes.

This paper introduces **Brown Octopus**, a stateful, harness-agnostic capability-context manager for tool-using AI agents. Brown Octopus decomposes user requests into operational intents, performs semantic capability ranking independently for each intent, estimates a bounded semantic neighborhood using a maximum adjacent-score-gap policy, merges the resulting capabilities, and maintains useful capabilities across conversational turns using time-to-live and global context constraints.

Rather than attempting to identify only the single best tool for a request, Brown Octopus is designed to preserve both primary and plausible supporting capabilities that an autonomous agent may require during task completion. In the current implementation, each intent is allowed a maximum neighborhood of 16 tools, while a score-distribution boundary can reduce that neighborhood when a stronger semantic separation is observed.

A 122-tool development universe demonstrates that the method produces variable capability neighborhoods across tasks: compact neighborhoods for web search and document authoring, and larger neighborhoods for email, repository management, deployment management, and project-management requests. These observations motivate a systematic evaluation against full-tool exposure and external dynamic tool-retrieval approaches. The principal research question is whether dynamic, stateful capability-context management can substantially reduce the tool context presented to an agent while preserving the capabilities required for successful autonomous task completion.

---

# 1. Introduction

Tool-using large language models are increasingly expected to interact with external systems rather than generate text alone.

An enterprise agent may have access to capabilities for:

- email
- calendars
- project management
- databases
- document generation
- file systems
- source control
- web search
- infrastructure
- HR systems
- internal applications

A straightforward architecture exposes all available tool definitions to the LLM:

```text
Tool Universe
     ↓
LLM
     ↓
Tool Selection
     ↓
Execution
```

This architecture is simple when the number of tools is small.

As the capability universe grows, however, every additional tool introduces schema and descriptive context that the language model must process even when that capability is irrelevant to the current task.

The resulting problem is not merely tool retrieval.

It is **capability-context management**.

The system must determine:

1. which capabilities are relevant to the current request,
2. how many capabilities should be exposed,
3. which supporting capabilities may become necessary during execution,
4. which capabilities should remain available across conversational turns, and
5. how the context should change as the conversation evolves.

Brown Octopus addresses these questions through a stateful retrieval layer positioned before the agent's ordinary tool-selection and execution loop.

---

# 2. Problem Definition

Let the complete capability universe be:

$$
T = \{t_1, t_2, \ldots, t_N\}
$$

where each \(t_i\) is a tool definition containing at minimum a name and description and potentially an input schema and execution metadata.

A conventional tool-enabled agent may receive the complete set \(T\) on every inference step.

Brown Octopus instead attempts to construct a much smaller active capability context:

$$
A_t \subseteq T
$$

for conversational turn \(t\).

The objective is not simply:

$$
\min |A_t|
$$

because an arbitrarily small context may omit capabilities required by the agent.

Instead, the objective can be expressed conceptually as:

$$
\min |A_t|
$$

subject to:

$$
R(A_t, q_t) \approx R(T, q_t)
$$

where \(R\) represents the availability of capabilities required to successfully address query \(q_t\).

For multi-turn interaction, the active context also depends on previous capability requirements:

$$
A_t = f(q_t, A_{t-1}, T)
$$

This distinguishes Brown Octopus from a purely stateless query-to-tool retrieval function.

---

# 3. Capability Recall Rather Than Exact Tool Prediction

A central design decision is that Brown Octopus does not attempt to determine exactly which tool the downstream agent will call.

That responsibility remains with the agent.

Instead, Brown Octopus attempts to preserve a sufficiently useful **capability neighborhood**.

Consider the request:

> Send an email to Tom.

The obvious primary capability is an email-sending function.

However, successful autonomous completion may first require discovering Tom's email address.

A minimal lexical or semantic mapping may expose:

```text
send_email
```

while a broader capability neighborhood may contain:

```text
compose_email
send_email
find_email_address
search_mail
```

Similarly:

> Create a Word report.

may require a workflow involving:

```text
create_word_document
add_heading
add_paragraph
add_table
```

The retrieval problem therefore differs from identifying one canonical tool label.

Brown Octopus prioritizes **capability recall**: ensuring that the agent retains access to both obvious and plausible supporting operations.

---

# 4. System Architecture

Brown Octopus is positioned between the capability universe and the downstream agent.

```text
Capability Universe
        ↓
Persisted Semantic Index
        ↓
User Request
        ↓
Operational Intent Analysis
        ↓
Per-Intent Semantic Ranking
        ↓
Bounded Capability Selection
        ↓
Merge and Deduplication
        ↓
Active Capability Memory
        ↓
Agent Tool Context
        ↓
LLM
        ↓
Normal Tool Execution
```

Brown Octopus does not execute the selected tools.

It therefore remains independent of the downstream orchestration framework.

The agent may use any compatible execution architecture after receiving the selected capability definitions.

---

# 5. Operational Intent Decomposition

Natural-language requests frequently contain multiple operational actions.

For example:

> Find the latest Nvidia news and email a summary to Tom.

This request contains at least two capability requirements:

$$
I_1 = \text{find the latest Nvidia news}
$$

$$
I_2 = \text{email a summary to Tom}
$$

Retrieving tools against the complete sentence risks allowing one semantic component to dominate the embedding representation.

Brown Octopus therefore decomposes the request into operational intents:

$$
Q \rightarrow \{I_1, I_2, \ldots, I_m\}
$$

and performs capability retrieval independently for each intent.

The current implementation uses the `en_core_web_trf` spaCy pipeline and dependency-based operational-action rules.

The analyzer identifies root operational verbs and selected coordinated or nested operational actions while attempting to exclude verbs that primarily express content or conditions.

For example:

```text
Search the web for the latest Nvidia news
and send what you find to Tom
```

is represented approximately as:

```text
Search the web for the latest Nvidia news
send what you find to Tom
```

rather than treating every verb in the sentence as an independent capability requirement.

---

# 6. Semantic Capability Ranking

Each operational intent is ranked against the complete tool universe.

The current implementation uses:

```text
Qwen/Qwen3-Embedding-0.6B
```

Tool representations are constructed from:

```text
tool name + tool description
```

For intent \(I_j\), semantic retrieval produces:

$$
S_j =
[(t_1,s_1),(t_2,s_2),\ldots,(t_N,s_N)]
$$

where:

$$
s_1 \ge s_2 \ge \ldots \ge s_N
$$

The remaining problem is determining where the relevant capability neighborhood ends.

---

# 7. Why Fixed-K Retrieval Was Insufficient

An early Brown Octopus prototype used a fixed number of tools per operational intent.

This provided a simple baseline but revealed a structural problem.

Capability neighborhoods have different widths.

A web-search intent may exhibit:

```text
web_search
----------------
unrelated tools
```

while an email intent may exhibit:

```text
send_email
compose_email
reply_email
forward_email
find_email_address
search_mail
----------------
unrelated capabilities
```

A fixed \(K\) therefore creates an unavoidable tradeoff.

Small \(K\):

- reduces context,
- but may remove useful supporting capabilities.

Large \(K\):

- improves recall,
- but unnecessarily expands context for narrow intents.

This motivated an adaptive boundary-selection policy.

The fixed-K implementation is no longer part of the production retrieval interface but remains conceptually useful as part of the system's development history.

---

# 8. Bounded Maximum-Gap Selection

Brown Octopus currently estimates the capability boundary using a bounded maximum adjacent-score gap.

Let:

$$
s_1 \ge s_2 \ge \ldots \ge s_N
$$

be the semantic ranking for one intent.

Define adjacent normalized gaps:

$$
g_i =
\frac{s_i - s_{i+1}}{s_1}
$$

The system searches for:

$$
i^* =
\arg\max_i g_i
$$

within a bounded candidate neighborhood.

Current parameters are:

$$
K_{\max}=16
$$

and:

$$
g_{\min}=0.02
$$

The system inspects ranks 1 through 17 so that the boundary after rank 16 can be measured, while never exposing more than 16 tools for one intent.

The selection rule is:

$$
K =
\begin{cases}
i^*, & g_{i^*} \ge g_{\min} \\
K_{\max}, & \text{otherwise}
\end{cases}
$$

This produces:

$$
C_j = \{t_1,\ldots,t_K\}
$$

for intent \(I_j\).

---

# 9. Why the Search Is Bounded

Early experiments considered searching for large score cliffs over the complete ranking.

This produced pathological boundaries deep in weak-ranking tails.

Observed candidate cutoffs included ranks such as:

```text
61
98
121
```

These boundaries were mathematical artifacts rather than useful capability neighborhoods.

Brown Octopus therefore separates two concepts:

**Context budget**

$$
K_{\max}
$$

and:

**Semantic boundary**

$$
i^*
$$

The maximum-gap policy may reduce the number of capabilities below the budget, but it cannot expand the context beyond that budget.

With the current configuration:

$$
|C_j| \le 16
$$

for every operational intent.

---

# 10. Multi-Intent Merge

After adaptive retrieval is performed independently for every operational intent, the resulting capability sets are combined:

$$
C =
\bigcup_{j=1}^{m} C_j
$$

Duplicate tools are removed while preserving the first selected occurrence.

This allows capability allocation to vary by intent.

For example, a development observation for:

> Find the latest Nvidia news and email a summary to Tom.

produced approximately:

```text
News intent:
2 capabilities

Email intent:
11 capabilities

Merged context:
13 capabilities
```

The system therefore does not impose one global retrieval size on a compound request.

---

# 11. Stateful Capability Memory

Tool requirements do not disappear at sentence boundaries.

Consider:

```text
Turn 1:
Find the latest Nvidia news.

Turn 2:
Email it to Tom.
```

The second request relies on conversational state.

Brown Octopus maintains an active capability set:

$$
A_t
$$

Capabilities retrieved on the current turn are merged with capabilities retained from previous turns.

Each active capability records the turn on which it was last retrieved.

The current implementation uses:

```text
TTL = 8 turns
```

A capability remains active through its TTL boundary and expires afterward unless retrieved again.

Retrieval refreshes its last-retrieved turn.

---

# 12. Global Capability Budget

Persistent capability memory introduces another scaling problem: without eviction, active context can grow indefinitely.

Brown Octopus therefore maintains a global active-context cap:

```text
30 tools
```

When the active capability context exceeds the cap, the oldest capabilities are evicted.

Thus the conversational capability state behaves as a bounded rolling context rather than an ever-growing history.

---

# 13. Capability Discovery and Indexing

The current implementation constructs its development capability universe from Model Context Protocol servers.

The discovery process is:

```text
MCP catalog
    ↓
connect to server
    ↓
list tools
    ↓
normalize metadata
    ↓
deduplicate
    ↓
embed
    ↓
persist index
```

The current development universe contains:

```text
122 tools
```

across capability domains including email, web search, GitHub, project management, Word/document generation, databases, HR operations, SharePoint, infrastructure, and file storage.

MCP is currently the capability source, but Brown Octopus is designed as a harness-agnostic context manager rather than an MCP execution framework.

---

# 14. Persisted Capability Index

Normal application startup should not require live discovery of every capability source.

Brown Octopus therefore persists:

```text
tools.json
embeddings.pt
metadata.json
```

The index records the normalized capability universe and precomputed semantic embeddings.

Normal initialization loads the persisted index.

Capability rediscovery and index rebuilding occur only through an explicit update operation.

This separates:

```text
capability-universe maintenance
```

from:

```text
runtime capability retrieval
```

---

# 15. Development Observations

The current 122-tool development universe produces variable capability-neighborhood sizes.

| Query                                | Selected capabilities |
| ------------------------------------ | --------------------: |
| Send an email to Tom                 |                    11 |
| Create a Word report                 |                     4 |
| Get my leave applications            |                     4 |
| Submit a leave application           |                     4 |
| Find a GitHub repository             |                    16 |
| Check deployment status on Railway   |                    10 |
| Get current sprint work items        |                    12 |
| Find Nvidia news and email a summary |                    13 |

These values are not presented as benchmark results.

They demonstrate that the adaptive policy behaves differently from fixed-size retrieval and that different semantic neighborhoods produce different capability allocations.

The email case is particularly illustrative.

The selected neighborhood includes the expected send/compose operations but also includes an address-discovery capability at approximately rank 10.

This provides a concrete example of a supporting capability that may be useful during autonomous task execution even though it is not the highest-scoring semantic match.

---

# 16. Implementation Validation

The current implementation contains tests covering:

- operational intent decomposition
- semantic ranking
- bounded maximum-gap selection
- candidate-boundary behavior
- per-intent retrieval
- cross-intent deduplication
- active capability state
- TTL behavior
- global capability eviction
- MCP discovery
- persisted indexes
- pipeline behavior
- public API behavior
- multi-turn conversations
- end-to-end capability flow

At the current implementation freeze:

```text
89 tests pass
```

with one dependency deprecation warning unrelated to Brown Octopus behavior.

---

# 17. Research Hypotheses

The primary hypothesis is:

> A stateful capability-context manager can substantially reduce the tool definitions exposed to an AI agent while preserving high recall of the capabilities required to complete user requests.

A second hypothesis concerns supporting capabilities:

> Retrieving a semantic capability neighborhood rather than only the highest-ranked tools improves the availability of intermediate capabilities required for autonomous multi-step completion.

A third hypothesis concerns state:

> Maintaining capability context across conversational turns can reduce repeated retrieval requirements while preserving capabilities relevant to follow-up requests.

These hypotheses require systematic evaluation and are not treated as established results in the present manuscript.

---

# 18. Evaluation Plan

The evaluation will separate retrieval quality from downstream agent behavior.

## 18.1 Experimental Conditions

The principal comparison is intended to include:

### ALL

Expose the complete capability universe to the agent.

This provides a maximum-context reference condition.

### External Dynamic Retrieval Baseline

Compare Brown Octopus against a strong external tool-retrieval or dynamic capability-selection approach.

Candidate approaches will be selected based on reproducibility and compatibility with the evaluation environment.

### BROWN OCTOPUS

Operational intent decomposition followed by per-intent semantic ranking, bounded maximum-gap selection, merge/deduplication, and stateful capability memory.

Historical fixed-K retrieval may be retained as an ablation or development baseline but is not intended to serve as the primary external comparison.

---

## 18.2 Retrieval-Level Evaluation

Retrieval evaluation should measure:

### Required Capability Recall

Did the selected context contain the capabilities required for task completion?

### Supporting Capability Recall

Did the context contain useful intermediate capabilities that may be required during autonomous execution?

### Context Reduction

$$
Reduction =
1 -
\frac{|C|}{|T|}
$$

### Tool-Schema Token Reduction

Because tools vary substantially in schema size, raw tool count is insufficient.

The token cost of the exposed tool definitions should therefore also be measured.

### Retrieval Latency

The cost of capability management itself must be included.

---

## 18.3 Downstream Agent Evaluation

Retrieval quality alone does not establish agent usefulness.

The same downstream LLM and execution harness should therefore be evaluated under each capability-context condition.

Metrics should include:

- task completion
- correct tool usage
- unnecessary tool calls
- failed tool-selection attempts
- total tool-schema tokens
- end-to-end latency
- total model tokens
- multi-step task completion

This evaluation tests whether reduced capability context preserves or improves actual agent behavior.

---

## 18.4 Multi-Turn Evaluation

Brown Octopus is explicitly stateful.

Evaluation should therefore include conversations where capability requirements persist or evolve across turns.

Example:

```text
Turn 1:
Find the latest Nvidia news.

Turn 2:
Summarize the most important development.

Turn 3:
Email it to Tom.
```

The experiment should measure:

- capability continuity,
- unnecessary capability retention,
- TTL behavior,
- context growth,
- context rotation,
- task success.

---

## 18.5 Scaling Evaluation

The capability universe should be expanded beyond the current 122 tools.

Evaluation should measure behavior as:

$$
N \rightarrow 10^2, 10^3, 10^4, \ldots
$$

where feasible.

Important measurements include:

- retrieval latency,
- index memory,
- context reduction,
- capability recall,
- agent task success.

Synthetic distractor capabilities may be useful for controlled scaling experiments, while public benchmarks provide externally defined retrieval corpora.

---

# 19. Related Work

Brown Octopus intersects several areas of tool-using language-model research.

## Tool Retrieval

Tool retrieval treats capability selection as an information-retrieval problem over tool descriptions.

The ToolRet benchmark introduced a heterogeneous benchmark containing thousands of retrieval tasks over tens of thousands of tools and demonstrated that strong general-purpose retrieval models do not necessarily transfer cleanly to tool retrieval.

This motivates evaluating Brown Octopus on standardized retrieval data rather than relying only on its development MCP universe.

## ToolBench and StableToolBench

ToolBench provides large-scale tool-use data containing both single-tool and multi-tool tasks.

StableToolBench addresses instability in large-scale tool evaluation through simulated and cached API behavior.

These benchmarks are relevant to evaluating downstream task completion after capability-context selection.

## Dynamic Tool Retrieval

Recent systems increasingly consider dynamic rather than one-shot tool availability.

Dynamic Tool Dependency Retrieval (DTDR), for example, conditions retrieval on both the original query and an evolving tool-calling plan, explicitly addressing multi-step dependencies between tools.

Brown Octopus approaches the problem from a complementary capability-context perspective: it constructs semantic capability neighborhoods per operational intent and maintains useful capability context across conversational turns.

A rigorous comparison must distinguish these architectural differences experimentally rather than assuming superiority from design alone.

## Dynamic Tool Selection During Reasoning

Other recent work explores changing tool availability during agent reasoning rather than treating the available tool set as fixed.

This reinforces the broader observation that capability selection is becoming part of the agent architecture itself rather than merely a preprocessing optimization.

---

# 20. Distinction from Tool Execution

Brown Octopus is not:

- an agent planner,
- a tool executor,
- an MCP client exposed as a tool,
- an LLM-based router,
- or a replacement for the agent's function-calling mechanism.

Its responsibility ends after constructing the capability context.

This separation is intentional.

It allows Brown Octopus to be evaluated and integrated independently of the downstream agent harness.

---

# 21. Limitations

The current work has several limitations.

First, the primary development universe contains only 122 tools.

This is sufficient for implementation development but not sufficient to establish large-scale retrieval performance.

Second, the current operational-intent analyzer relies on English dependency parsing and handcrafted structural rules.

Third, the bounded maximum-gap parameters are currently based on development experiments rather than a large held-out benchmark.

Fourth, the current semantic representation primarily embeds tool names and descriptions rather than full input schemas.

Fifth, capability-neighborhood quality has not yet been validated through controlled downstream agent experiments.

Sixth, stateful capability retention introduces its own hyperparameters, including TTL and global context cap, which require ablation.

These limitations define the next evaluation phase.

---

# 22. Reproducibility

The implementation maintains explicit separation between:

```text
capability discovery
index construction
runtime initialization
intent analysis
semantic ranking
capability selection
state management
agent execution
```

The current implementation freeze uses:

```text
Embedding model:
Qwen/Qwen3-Embedding-0.6B

Intent analyzer:
spaCy en_core_web_trf

Maximum capability neighborhood:
16

Minimum normalized adjacent gap:
2%

Capability TTL:
8 turns

Global active capability cap:
30 tools

Development capability universe:
122 tools
```

The complete test suite currently reports:

```text
89 passing tests
```

---

# 23. Research Question

The central research question is:

> Can an AI agent operate with a small, dynamically selected and statefully maintained capability context while retaining task performance comparable to an agent given access to the complete tool universe?

The capability-neighborhood hypothesis adds a second question:

> Does preserving semantically related supporting capabilities improve autonomous multi-step task completion compared with narrower tool retrieval?

And the stateful architecture introduces a third:

> Does maintaining capability context across turns provide measurable benefits over independent per-turn tool retrieval?

---

# 24. Conclusion

Brown Octopus reframes large-scale tool exposure as a capability-context management problem.

Rather than presenting every available tool to the language model or retrieving a fixed number of tools for every request, the system decomposes requests into operational intents, identifies bounded semantic capability neighborhoods, merges those neighborhoods, and maintains useful capabilities across conversational turns.

The architecture is designed around a distinction between **choosing a tool** and **maintaining access to capabilities**.

The downstream agent chooses tools.

Brown Octopus determines the capability context within which that decision is made.

The implementation is now sufficiently stable to move from architecture development to systematic evaluation. The next stage will determine whether the observed reductions in capability context preserve retrieval completeness and downstream task performance across standardized tool-retrieval benchmarks, multi-tool agent tasks, and increasing capability-universe sizes.
