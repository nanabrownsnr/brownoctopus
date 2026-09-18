# 🐙 Octopus

**High-recall capability retrieval for AI agents**

Octopus is a lightweight tool-retrieval layer designed to sit between a user's natural-language request and an AI agent's LLM.

Its job is not to decide exactly which tool the agent must execute.

Its job is to reduce a potentially large tool catalog into a small set of relevant capabilities while preserving the tools needed to satisfy **every meaningful intent in the user's request**.

```text
User Request
     │
     ▼
┌─────────────────┐
│     Octopus     │
│                 │
│ Intent Analysis │
│       ↓         │
│ Tool Retrieval  │
│       ↓         │
│ Coverage Merge  │
└────────┬────────┘
         │
         ▼
Small Candidate Tool Set
         │
         ▼
        LLM
         │
         ▼
Final Tool Selection / Execution
```

The core philosophy is:

> **Octopus optimizes for capability recall and coverage, not perfect final tool ranking.**

The downstream LLM remains responsible for choosing which of the exposed tools should actually be called.

---

# 1. The Problem

Modern agents may have access to dozens, hundreds, or eventually thousands of tools.

Passing every available tool to the LLM creates several problems:

- larger prompts
- higher token usage
- increased latency
- more difficult tool selection
- increased confusion between similar tools
- poorer scalability as the capability catalog grows

A straightforward solution is semantic retrieval:

```text
User query
    ↓
Embedding
    ↓
Compare against tool embeddings
    ↓
Top-K tools
    ↓
LLM
```

This works reasonably well for simple requests.

It becomes problematic for **compound requests**.

Consider:

```text
"Check my calendar and reply to Tom."
```

There are two independent capabilities required:

```text
check calendar
reply to Tom
```

A traditional global Top-K search can allow one semantic neighborhood to consume the entire result set.

For example:

```text
check_calendar_availability
search_calendar
list_calendar_events
get_calendar_event
create_calendar_event
```

All five tools are relevant to part of the query, but the email/reply capability has disappeared.

This is a **coverage problem**.

The most semantically similar tools globally are not necessarily the set of tools required to complete the entire request.

---

# 2. Design Goal

Given:

```text
Natural-language request
+
Large tool catalog
```

Octopus should return:

```text
A small candidate set
containing at least one appropriate capability
for each meaningful user intent.
```

For V1, the primary objective is therefore:

> **Capability coverage over perfect ranking precision.**

If a user asks:

```text
"Find the quarterly report."
```

and Octopus returns:

```text
search_web
search_files
search_calendar
```

the ranking is not perfect.

However, the required capability:

```text
search_files
```

survived retrieval.

The downstream LLM can still determine that searching existing files is the appropriate operation.

For V1, this is considered successful retrieval.

---

# 3. Current Architecture

The current pipeline is:

```text
User Request
      │
      ▼
┌──────────────────────┐
│ Intent Analyzer      │
│                      │
│ spaCy Transformer    │
└──────────┬───────────┘
           │
           ▼
   Structured Intents
           │
           ▼
┌──────────────────────┐
│ Intent Retriever     │
│                      │
│ BGE Embeddings       │
│ Weighted Scoring     │
└──────────┬───────────┘
           │
           ▼
 Per-Intent Top-K Tools
           │
           ▼
┌──────────────────────┐
│ Merge + Deduplicate  │
└──────────┬───────────┘
           │
           ▼
 Candidate Tool Set
           │
           ▼
          LLM
```

The major difference from conventional retrieval is:

```text
NOT:

whole query → one retrieval → global Top-K
```

Instead:

```text
whole query
    ↓
identify requested actions
    ↓
retrieve independently for each action
    ↓
merge candidates
```

This provides a much stronger coverage guarantee for compound requests.

---

# 4. Project Structure

Current repository structure:

```text
octopus/
├── pyproject.toml
├── README.md
├── src/
│   └── octopus/
│       ├── __init__.py
│       ├── analyzer.py
│       ├── retriever.py
│       └── tools.py
└── tests/
    ├── test_intent_pipeline.py
    └── test_retrieval.py
```

The major responsibilities are:

```text
analyzer.py
    Natural language → structured intents

retriever.py
    Structured intents → candidate tools

tools.py
    Tool/capability catalog

tests/
    Behavioral and retrieval coverage tests
```

---

# 5. Intent Analyzer

Octopus first performs lightweight linguistic analysis of the user's request.

The current analyzer uses:

```text
spaCy
+
en_core_web_trf
```

The transformer-based spaCy model was selected after testing the smaller `en_core_web_sm` model.

## Why the transformer parser?

Consider:

```text
"Check my calendar and reply to Tom."
```

The small model incorrectly interpreted `reply` as a noun associated with `calendar`.

The transformer model produced the more useful dependency structure:

```text
Check       VERB   ROOT
calendar    NOUN   dobj → Check
and         CCONJ  cc   → Check
reply       VERB   conj → Check
to          ADP    prep → reply
Tom         PROPN  pobj → to
```

That gives Octopus two requested actions:

```text
check
reply
```

rather than incorrectly treating the sentence as one operation.

---

# 6. Intent Representation

Each detected intent is represented approximately as:

```python
{
    "action": str,
    "target": str | None,
    "context": str,
}
```

For example:

```text
"Check my calendar and reply to Tom."
```

becomes conceptually:

```python
[
    {
        "action": "check",
        "target": "my calendar",
        "context": "Check my calendar and reply to Tom.",
    },
    {
        "action": "reply",
        "target": "Tom",
        "context": "Check my calendar and reply to Tom.",
    },
]
```

The distinction between `action` and `target` later becomes important during retrieval.

---

# 7. Analyzer Philosophy: Hints, Not Truth

An important design decision was to avoid making the analyzer responsible for perfectly understanding the request.

Early experimentation showed how quickly clause splitting and dependency rules can become complicated.

Natural language contains:

- conjunctions
- subordinate clauses
- conditions
- pronouns
- indirect objects
- relative clauses
- sequential instructions
- conversational fragments

Attempting to perfectly decompose every sentence would effectively turn Octopus into another language model.

That is not its purpose.

The analyzer therefore follows this principle:

> **Analyzer output is retrieval guidance, not semantic ground truth.**

The analyzer should provide useful signals while remaining high-recall and relatively lightweight.

---

# 8. Action Detection

The current analyzer treats the sentence ROOT as the primary requested action.

For example:

```text
"Check my calendar."
```

produces:

```text
action = check
```

Coordinated verbs connected through `conj` are also treated as requested actions.

Example:

```text
"Check my calendar and reply to Tom."
```

produces:

```text
check
reply
```

The analyzer also supports certain sequential constructions where another verb is attached weakly to the root but clearly owns its own object.

At the same time, not every verb in a sentence should become an intent.

Consider:

```text
"Check whether I submitted Christmas leave
and email my manager if I haven't."
```

Words such as:

```text
submitted
haven't
```

may occur as verbs grammatically, but they do not necessarily represent independent actions the user is requesting from the agent.

The analyzer therefore tries to distinguish:

```text
requested operations
```

from:

```text
verbs describing conditions or content
```

---

# 9. Target Extraction

Once an action has been identified, Octopus extracts the object or target associated with it.

Examples:

```text
check my calendar
      └─────────┘
         target

reply to Tom
         └─┘
        target

schedule a meeting with Sarah tomorrow
         └───────────────────────────┘
                     target
```

Target extraction intentionally favors recall over perfect cleanliness.

It is preferable to retrieve:

```text
"a meeting with Sarah tomorrow"
```

rather than accidentally reduce the target to something too narrow and lose useful semantic information.

---

# 10. Why We Did Not Build a Complex Clause Splitter

We experimented conceptually with splitting compound requests into independent clauses before retrieval.

That approach quickly became brittle.

For example:

```text
"Check whether I submitted Christmas leave
and email my manager if I haven't."
```

contains both conjunction and conditional structure.

A rule-based clause splitter can easily confuse:

```text
things the user wants done
```

with:

```text
things describing the conditions under which they should be done.
```

The current design therefore relies on dependency-based action hints instead of attempting perfect clause reconstruction.

This keeps the analyzer focused on its actual purpose:

> generating useful retrieval signals.

---

# 11. Tool Catalog

The current development catalog contains roughly 50 tools across intentionally overlapping domains.

Current categories include:

```text
Email
Calendar
Leave / HR
Tasks / Project Management
Files / Documents
Contacts / People
Chat / Messaging
Knowledge / Search
Reporting / Data
```

The catalog deliberately contains confusable tools.

Examples:

```text
send_email
send_chat_message

reply_email
reply_chat_message

search_emails
search_calendar
search_files
search_contacts
search_chat_messages
search_knowledge_base
search_web

create_calendar_event
create_leave_application
create_work_item
create_document
create_contact
```

This is intentional.

A retrieval system that only works when every tool is semantically unique is not representative of a production agent platform.

---

# 12. Tool Naming Convention

One of the strongest findings from experimentation has been that **tool naming materially affects embedding retrieval**.

The recommended naming structure is:

```text
<verb>_<primary object>[_qualifier]
```

Examples:

```text
send_email
reply_email
search_files
search_web
search_calendar
schedule_meeting
get_leave_balance
submit_leave_application
query_database
```

In simplified form:

```text
VERB + NOUN
```

or:

```text
ACTION + OBJECT
```

This mirrors the representation produced by the analyzer:

```text
action + target
```

For example:

```text
User intent:

find + quarterly report
```

is structurally closer to:

```text
search + files
```

than an inconsistently named capability.

---

# 13. Evidence That Naming Matters

Originally the web search tool was named:

```text
web_search
```

For:

```text
"find the quarterly report"
```

its retrieval score was approximately:

```text
web_search       1.6038
search_files     1.5308
```

We renamed it:

```text
web_search
    ↓
search_web
```

without changing its fundamental capability.

Its score dropped to approximately:

```text
search_web       1.5626
search_files     1.5308
```

The ranking did not completely reverse, but the gap became significantly smaller.

This demonstrated that the identifier itself is contributing semantic information to the embedding.

Therefore:

> Tool names should be treated as retrieval metadata, not arbitrary implementation identifiers.

---

# 14. Tool Descriptions Matter Too

Tool descriptions also materially affect retrieval.

Originally:

```python
{
    "name": "search_files",
    "description": "Search files and documents by name, keyword, or content.",
}
```

For:

```text
"find the quarterly report"
```

the tool scored approximately:

```text
search_files = 1.4882
```

We enriched its description:

```python
{
    "name": "search_files",
    "description": (
        "Find, locate, or search for existing files, documents, reports, "
        "presentations, spreadsheets, and stored content by name, "
        "keyword, or content."
    ),
}
```

Its score increased to approximately:

```text
search_files = 1.5308
```

and it moved from roughly third to second in the candidate ranking.

This gives us another important principle:

> **Descriptions should describe the semantic capability of the tool, not merely its implementation.**

Useful descriptions should include:

- the operation
- the object/domain
- important synonyms
- the type of content handled
- meaningful distinctions from neighboring tools

However, descriptions should not be artificially stuffed with keywords solely to manipulate retrieval.

They should remain accurate descriptions of what the tool actually does.

---

# 15. Embedding Model

Current semantic retrieval uses:

```text
BAAI/bge-small-en-v1.5
```

through Sentence Transformers.

Tool representations are constructed from:

```python
f"{tool['name']}: {tool['description']}"
```

For example:

```text
search_files:
Find, locate, or search for existing files,
documents, reports, presentations, spreadsheets...
```

These tool representations are embedded once when the retrieval service initializes.

Conceptually:

```python
tool_texts = [
    f"{tool['name']}: {tool['description']}"
    for tool in TOOLS
]

tool_embeddings = model.encode(
    tool_texts,
    convert_to_tensor=True,
)
```

Precomputing these embeddings is important because the tool catalog does not need to be re-embedded for every request.

---

# 16. Initial Retrieval: Cosine Similarity

The first implementation used standard cosine similarity.

Conceptually:

```python
query_embedding = model.encode(query)

scores = cosine_similarity(
    query_embedding,
    tool_embeddings,
)
```

Cosine similarity measures the angle between two vectors rather than their absolute magnitude.

Conceptually:

```text
                  A · B
cos(A, B) = ─────────────────
             ||A|| × ||B||
```

This is useful for conventional semantic search because it asks:

> How similar is the direction of these two semantic vectors?

The initial architecture therefore looked like:

```text
Full User Request
      ↓
BGE Embedding
      ↓
Cosine Similarity
      ↓
Global Top-K
```

---

# 17. Failure of Flat Global Top-K

For:

```text
"Check my calendar and reply to Tom."
```

a flat retrieval with `top_k=5` produced calendar-heavy results such as:

```text
check_calendar_availability
search_calendar
list_calendar_events
get_calendar_event
create_calendar_event
```

No reply capability survived.

Increasing K eventually recovered email tools.

At approximately K=8, results included:

```text
check_calendar_availability
search_calendar
list_calendar_events
get_calendar_event
create_calendar_event
update_calendar_event
reply_email
reply_chat_message
```

This proved something important:

> The embedding model could identify both semantic regions, but global Top-K allowed the stronger region to consume the candidate budget.

Increasing K solves the problem only by brute force.

With a much larger catalog, continually increasing K defeats the purpose of retrieval.

---

# 18. Per-Intent Retrieval

The solution was to retrieve independently for each detected intent.

Instead of:

```text
"check my calendar and reply to Tom"
                  ↓
             one retrieval
```

Octopus performs:

```text
check + calendar
      ↓
Top-K

reply + Tom
      ↓
Top-K

      ↓
merge + deduplicate
```

With `per_intent_k=3`, the same request produced approximately:

```text
check_calendar_availability
search_calendar
search_web

reply_email
reply_chat_message
forward_email
```

The exact ranking is imperfect, but both capability families survive.

This is the central architectural insight behind Octopus V1:

> **Perform retrieval per intent, then merge for coverage.**

---

# 19. Why Action and Target Are Embedded Separately

A simple intent representation could be:

```text
"check my calendar"
```

embedded as one string.

However, the action and target play different semantic roles.

Consider:

```text
find the quarterly report
```

There are two signals:

```text
ACTION
find

TARGET
quarterly report
```

The action tells us the type of operation.

The target tells us the domain/object on which the operation should occur.

We therefore experimented with embedding them independently:

```python
action_embedding = model.encode(intent["action"])

target_embedding = model.encode(
    intent["target"] or ""
)
```

This allows Octopus to control how strongly each signal influences retrieval.

---

# 20. Why We Weight the Action

Without additional weighting, nouns can dominate semantic similarity.

For example:

```text
find quarterly report
```

contains the highly meaningful noun:

```text
report
```

A semantic model may therefore favor:

```text
generate_report
```

even though the user did not ask to generate anything.

The difference between:

```text
find report
```

and:

```text
generate report
```

is primarily encoded by the **verb**.

That led to the hypothesis:

> The action should receive slightly more influence than the target during capability retrieval.

---

# 21. Why the Current Action Weight Is 1.5

We experimentally tested stronger action weighting.

A weight of:

```text
2.0
```

helped cases such as:

```text
find quarterly report
```

but introduced another failure.

Consider:

```text
"Use the send email tool to send Tom a message..."
```

The parser identifies:

```text
action = use
target = the send email tool
```

Here, `use` is grammatically the root verb but semantically weak.

Giving it a 2× weight caused the generic action to overpower the highly informative target:

```text
the send email tool
```

As a result, `send_email` could be pushed down or lost.

Reducing the action weight to:

```text
1.5
```

produced a better balance across the tested cases.

The current weighting is therefore:

```text
1.5 × action signal
+
1.0 × target signal
```

Importantly:

> **1.5 is not considered a mathematically optimal or final constant.**

It is simply the best value observed in the current test suite.

Future evaluation over a much larger dataset may produce a different value or a dynamic weighting strategy.

---

# 22. Why Dot Product Instead of Cosine Similarity

The initial retriever used cosine similarity.

We later experimented with weighted action and target representations.

The intent representation became conceptually:

```text
Q = 1.5 × Action + 1.0 × Target
```

We then score tools using a dot product:

```text
score = Q · Tool
```

Expanding this gives:

```text
score =
1.5 × (Action · Tool)
+
1.0 × (Target · Tool)
```

This is useful because the weighted components directly contribute to the final score.

A crucial observation is that simply multiplying an entire query embedding by a scalar would not meaningfully solve the ranking problem under cosine similarity.

If:

```text
Q' = 2Q
```

then cosine normalization removes that magnitude difference:

```text
cos(2Q, T) = cos(Q, T)
```

So simply saying:

```text
"make the query vector 2× stronger"
```

does not change cosine ranking.

Instead, we need to change the **composition of the query representation**:

```text
1.5 × Action
+
1.0 × Target
```

and then compare that weighted representation against the tools.

Dot-product scoring makes this weighting relationship straightforward.

---

# 23. Current Retrieval Formula

Conceptually, Octopus currently performs:

```text
A = embedding(action)
T = embedding(target)

Q = 1.5A + T

score(tool) = Q · embedding(tool)
```

Then:

```text
Top K per intent
```

are retained.

Current default experimentation uses:

```text
per_intent_k = 3
```

Finally:

```text
results from all intents
        ↓
merge
        ↓
deduplicate
        ↓
candidate tool set
```

---

# 24. Why We Do Not Require Rank #1 in V1

An important finding during testing was that demanding perfect ranking encourages overfitting.

For example:

```text
"Find the quarterly report."
```

currently produces approximately:

```text
search_web        1.5626
search_files      1.5308
search_calendar   1.4902
```

Ideally:

```text
search_files
```

would rank first.

However, Octopus is not the final decision-maker.

The LLM receives the shortlist and can determine which tool is appropriate.

Therefore, V1 tests primarily answer:

```text
Did the appropriate capability survive retrieval?
```

rather than:

```text
Did the appropriate tool always rank #1?
```

This distinction prevents us from over-engineering the retrieval layer into something that duplicates the downstream LLM's responsibility.

---

# 25. Current V1 Success Criterion

For every meaningful requested intent:

```text
At least one appropriate capability
must survive into the candidate set.
```

For example:

```text
"Check my leave balance,
schedule a meeting with Sarah tomorrow,
and email my manager the details."
```

should expose capabilities from all three neighborhoods:

```text
Leave
Meeting / Calendar
Email
```

The exact order is secondary.

This is **coverage-oriented retrieval**.

---

# 26. Three-Intent Stress Testing

Compound requests were used deliberately to test whether the architecture preserves multiple capability families.

Example:

```text
"Check my leave balance,
schedule a meeting with Sarah tomorrow,
and email my manager the details."
```

The analyzer identifies approximately:

```text
check → my leave balance

schedule → a meeting with Sarah tomorrow

email → the details
```

Retrieval then runs independently for each.

Observed candidates included:

```text
get_leave_balance
approve_leave_application
check_calendar_availability

schedule_meeting
list_calendar_events
search_calendar

get_email
reply_email
search_emails
```

The email ranking is imperfect, but all three required capability neighborhoods survive.

---

# 27. Another Compound Example

Consider:

```text
"Find the quarterly report,
send it to Tom,
and if he replies schedule a meeting with him."
```

The analyzer identifies the requested actions approximately as:

```text
find
send
schedule
```

while avoiding treating:

```text
replies
```

as a separate requested operation.

Current retrieval produces candidates similar to:

```text
find:
    search_web
    search_files
    search_calendar

send:
    send_chat_message
    send_email
    forward_email

schedule:
    schedule_meeting
    search_calendar
    create_calendar_event
```

Again, ranking is not perfect.

Coverage is successful.

---

# 28. Merge and Deduplication

Per-intent retrieval can produce duplicate tools.

For example:

```text
search_calendar
```

might appear for multiple intents.

Octopus therefore merges candidate indices while preserving their first appearance and removes duplicates.

Conceptually:

```python
seen = set()
results = []

for tool in candidates:
    if tool not in seen:
        seen.add(tool)
        results.append(tool)
```

This keeps the final candidate set smaller without sacrificing discovered capability families.

---

# 29. Why Tool Embeddings Are Precomputed

Embedding every tool on every request would be wasteful.

The tool catalog changes far less frequently than user requests.

Therefore:

```text
Application startup
       ↓
Load BGE
       ↓
Embed entire tool catalog once
       ↓
Cache embeddings
```

Then each request only requires embeddings for:

```text
action
target
```

followed by inexpensive vector comparisons.

This is important for eventual API deployment.

---

# 30. Runtime and Latency

Current pytest runs may take roughly tens of seconds because each test process initializes large ML dependencies.

This is not representative of production request latency.

A production service would behave approximately like:

```text
SERVER STARTUP

Load spaCy transformer
        ↓
Load BGE model
        ↓
Embed tool catalog
        ↓
Keep process alive
```

Then:

```text
REQUEST

Parse sentence
      ↓
Embed action/target
      ↓
Vector scoring
      ↓
Merge candidates
      ↓
Return
```

The expensive model initialization happens once rather than once per API request.

Future benchmarking should measure **warm request latency**, not pytest startup time.

---

# 31. Test-Driven Development Strategy

Octopus is being developed using behavioral tests rather than attempting to design the entire retrieval architecture upfront.

The development loop is:

```text
Write realistic request
       ↓
Define expected capability coverage
       ↓
Run retrieval
       ↓
Observe failure
       ↓
Identify general failure mode
       ↓
Make smallest general change
       ↓
Run entire regression suite
```

This is important because retrieval systems are easy to accidentally optimize for individual phrases.

The goal is not:

```text
make this sentence pass
```

but:

```text
discover a general retrieval principle
from this failure.
```

---

# 32. Key Findings So Far

## 32.1 Global Top-K Is Not Enough

Flat semantic retrieval can work for single intents but can lose entire capability families for compound requests.

```text
compound query
→ one embedding
→ global Top-K
```

creates a winner-takes-most effect.

---

## 32.2 Retrieve Per Intent

Breaking retrieval into action-level branches dramatically improves capability coverage.

```text
intent 1 → Top-K
intent 2 → Top-K
intent 3 → Top-K
        ↓
       merge
```

This is currently the most important architectural improvement in Octopus.

---

## 32.3 Analyzer Output Should Be Advisory

Perfect linguistic decomposition is unnecessary and potentially counterproductive.

The analyzer should produce useful hints while allowing downstream retrieval and the LLM to handle ambiguity.

---

## 32.4 Verbs Carry Important Operational Meaning

The difference between:

```text
find report
```

and:

```text
generate report
```

is largely carried by the verb.

Slightly increasing action influence improved retrieval coverage.

---

## 32.5 Too Much Action Weighting Is Harmful

Not every grammatical root is semantically useful.

For example:

```text
"use the send email tool..."
```

has:

```text
use
```

as a generic/meta action.

A 2.0 action weight caused this weak verb to overpower useful target information.

A weight of 1.5 performed better across the current tests.

---

## 32.6 Tool Names Are Retrieval Metadata

Changing:

```text
web_search
```

to:

```text
search_web
```

measurably changed semantic scores.

Tool identifiers therefore influence retrieval and should follow a consistent semantic convention.

Recommended:

```text
VERB + OBJECT
```

Examples:

```text
search_web
search_files
send_email
reply_email
schedule_meeting
get_leave_balance
```

---

## 32.7 Descriptions Improve Capability Recognition

Expanding `search_files` to explicitly mention:

```text
files
documents
reports
presentations
spreadsheets
stored content
```

increased its score for:

```text
"find the quarterly report"
```

and moved it higher in the ranking.

Descriptions are therefore a meaningful part of the retrieval representation.

---

## 32.8 High Score Does Not Necessarily Mean Correct

Raw embedding scores should not yet be interpreted as calibrated confidence.

For example, a semantically broad tool can receive a high score while still being less appropriate than another candidate.

Therefore:

```text
score > X
```

should not yet be interpreted as:

```text
X% confidence
```

Thresholds should be calibrated empirically against a larger evaluation dataset.

---

## 32.9 Coverage Matters More Than Ordering for V1

The current system is better at ensuring:

```text
the right tool survives
```

than ensuring:

```text
the right tool is always #1.
```

That is acceptable for the current architecture because Octopus feeds another reasoning system.

The division of responsibility is:

```text
Octopus
    ↓
Which capabilities might reasonably be needed?

LLM
    ↓
Which capability should actually be used?

Tool
    ↓
Execute the operation.
```

---

# 33. Current Limitations

Octopus V1 still has several known limitations.

### Ranking precision

Correct tools may appear second or third rather than first.

Example:

```text
find quarterly report

search_web
search_files
search_calendar
```

### Generic root verbs

Words such as:

```text
use
want
need
help
try
```

may receive too much influence if treated like operational actions.

A future version may distinguish:

```text
operational verbs
```

from:

```text
meta/request verbs.
```

### Pronoun resolution

Requests such as:

```text
"Find the report and send it to Tom."
```

contain:

```text
it
```

whose meaning depends on earlier context.

The downstream LLM can currently compensate because multiple candidate tools are retained, but Octopus does not yet perform full coreference resolution.

### Candidate budget

With:

```text
per_intent_k = 3
```

a three-action request can theoretically expose approximately nine tools before deduplication.

Future versions may need a global candidate budget while still guaranteeing per-intent coverage.

### Conversational follow-ups

Consider:

```text
User: Can you book my leave?

Assistant: What dates?

User: Monday to Thursday.
```

The final message:

```text
Monday to Thursday
```

contains no explicit operational verb.

A stateless Octopus invocation may therefore fail to recover the existing leave capability.

Future versions should support active conversational capability state.

---

# 34. Planned Conversational State

A likely future strategy is:

```text
New message
    ↓
Octopus analysis
    │
    ├── strong new intent
    │       ↓
    │   update/augment active capabilities
    │
    └── weak/no new intent
            ↓
        preserve previous active capabilities
```

For example:

```text
Turn 1:
"Book my leave."

Active capabilities:
create_leave_application
submit_leave_application
...

Turn 2:
"Monday to Thursday."

No strong new action detected.

Therefore:
retain leave capability context.
```

This prevents conversational argument completion from accidentally destroying useful tool context.

---

# 35. Potential Future Improvements

The current architecture deliberately prioritizes simplicity.

Potential future work includes:

### Dynamic action weighting

Instead of:

```text
all actions = 1.5
```

classify actions as:

```text
operational:
send
search
create
schedule
approve

generic/meta:
use
want
need
help
try
```

and weight them differently.

### Explicit tool-name matching

A request such as:

```text
"use the send email tool"
```

could receive deterministic exact-name or alias matching before semantic retrieval.

### Rich capability metadata

Future tool representations could include:

```text
name
description
aliases
domain
action
object
examples
input schema
permissions
ownership
availability
```

instead of relying entirely on:

```text
name + description
```

### Hybrid retrieval

Semantic embeddings could eventually be combined with lexical retrieval such as BM25.

Conceptually:

```text
Dense retrieval
       +
Lexical retrieval
       ↓
Fusion
       ↓
Per-intent candidate set
```

### Global coverage optimization

Rather than simply:

```text
3 tools × N intents
```

Octopus could optimize:

```text
minimum number of tools
that covers maximum intent space
```

### Score calibration

A larger labeled evaluation dataset could help determine whether:

```text
top score
score margin
score distribution
```

can reliably indicate retrieval confidence.

---

# 36. Current Technology Stack

```text
Python 3.13
uv
spaCy
en_core_web_trf
Sentence Transformers
BAAI/bge-small-en-v1.5
PyTorch
NumPy
pytest
```

---

# 37. Current V1 Retrieval Algorithm

At a high level:

```python
def retrieve_tools(query, per_intent_k=3):

    intents = analyze_intents(query)

    candidates = []

    for intent in intents:

        action = embed(intent.action)
        target = embed(intent.target)

        query_vector = (
            1.5 * action
            + 1.0 * target
        )

        scores = dot_product(
            query_vector,
            tool_embeddings,
        )

        candidates.extend(
            top_k(scores, per_intent_k)
        )

    return deduplicate(candidates)
```

This is intentionally simple.

The current objective is to validate the architecture before introducing more sophisticated retrieval machinery.

---

# 38. Example End-to-End Flow

Input:

```text
"Check my leave balance,
schedule a meeting with Sarah tomorrow,
and email my manager the details."
```

### Step 1 — Analyzer

```text
Intent 1
action: check
target: my leave balance

Intent 2
action: schedule
target: a meeting with Sarah tomorrow

Intent 3
action: email
target: the details
```

### Step 2 — Independent retrieval

```text
Intent 1
    ↓
leave-related candidates

Intent 2
    ↓
calendar/meeting candidates

Intent 3
    ↓
email candidates
```

### Step 3 — Merge

```text
get_leave_balance
approve_leave_application
check_calendar_availability

schedule_meeting
list_calendar_events
search_calendar

get_email
reply_email
search_emails
```

### Step 4 — LLM

The LLM receives a much smaller capability set than the complete catalog while retaining tools relevant to all three user intents.

The LLM performs final reasoning and tool selection.

---

# 39. Design Principles

The current work has produced several principles that should guide further development.

### 1. Optimize for coverage first

Missing the correct capability is worse than exposing one or two additional candidates.

### 2. Separate discovery from selection

Octopus discovers likely capabilities.

The LLM selects the final capability.

### 3. Compound requests require independent retrieval paths

One global semantic ranking is insufficient.

### 4. Treat tool metadata as part of the retrieval model

Names and descriptions directly influence retrieval quality.

### 5. Prefer action-first naming

Use:

```text
verb_object
```

where practical.

### 6. Give verbs slightly more influence, but not absolute control

Current experimental weighting:

```text
action = 1.5
target = 1.0
```

### 7. Do not mistake embedding score for confidence

Scores are ranking signals until properly calibrated.

### 8. Keep the analyzer lightweight

Octopus should not become another general-purpose language model.

### 9. Test behavior, not individual phrases

Every failure should lead to a general architectural improvement rather than a hardcoded exception.

### 10. Let the downstream LLM reason

Octopus should reduce the search space, not duplicate the final reasoning layer.

---

# 40. V1 Definition

Octopus V1 can be summarized as:

> **A lightweight, intent-aware capability retrieval layer that decomposes compound user requests into action/target signals, retrieves tools independently for each intent using weighted semantic embeddings, and merges those results into a small high-recall candidate set for a downstream LLM.**

The current emphasis is:

```text
High recall
+
Intent coverage
+
Small tool context
+
Simple architecture
```

rather than:

```text
Perfect ranking
+
Perfect parsing
+
Final tool execution
```

That boundary is intentional.

---

# 41. Why "Octopus"?

An octopus can extend multiple arms toward different objects simultaneously.

That mirrors the retrieval problem this project is trying to solve.

A compound request may require reaching into several capability domains at once:

```text
                  ┌── Email
                  │
                  ├── Calendar
User Request ──► 🐙 ── Files
                  │
                  ├── HR
                  │
                  └── Project Management
```

Rather than forcing the entire request through one winner-takes-all ranking, Octopus can extend a retrieval branch toward each intent and bring the relevant capabilities back together.

---

# Status

**Current stage:** V1 retrieval architecture / experimentation.

Working:

- transformer-based intent analysis
- compound action detection
- action/target extraction
- BGE tool embeddings
- precomputed catalog embeddings
- weighted action/target retrieval
- dot-product scoring
- per-intent Top-K retrieval
- merge and deduplication
- multi-intent capability coverage
- standardized action-first tool naming
- retrieval-oriented capability descriptions
- pytest behavioral regression suite

Still under investigation:

- ranking precision
- candidate budget optimization
- generic/meta verb handling
- conversational state
- follow-up utterances
- score calibration
- hybrid lexical + dense retrieval
- production latency benchmarking
- large-scale tool catalogs

---

## Guiding Principle

> **Octopus does not need to know exactly which tool the agent will use. It needs to make sure the agent does not lose access to the tool it needs.**
