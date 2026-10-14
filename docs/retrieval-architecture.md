# Brown Octopus retrieval architecture

Brown Octopus is best described in plain English as an intent-aware dense
retrieval and context-management layer for agent capabilities.

It is not a new embedding model, an LLM router, or an agent executor. Its job
is to decide which capability definitions should be placed in the host
agent's context for the current request and session.

## The current pipeline

1. **Intent analysis** decomposes the request into operational signals such as
   actions, objects, and constraints. This gives retrieval more structure than
   embedding the raw sentence alone.
2. **Dense retrieval** compares the request representation with indexed
   capability descriptions.
3. **Bounded selection** keeps a useful minimum and expands only when the
   score distribution supports it. The production `brown_octopus_v3` policy
   currently uses Min-4 with a bounded maximum-gap cutoff.
4. **Context management** carries selected capabilities across a session while
   applying `active_cap` and capability TTL rules.
5. **Index lifecycle management** discovers, embeds, persists, and incrementally
   publishes capability snapshots. The host still owns execution, credentials,
   authorization, and conversation history.

The result is a small, request-scoped capability context rather than the full
tool universe. That is Brown Octopus's primary contribution: controlling the
capability surface presented to an agent over time.

## What “dynamic K” means here

There are two different K values:

- **Selection K** is the number of capabilities returned for a request. The
  current bounded-gap policy already makes this variable within configured
  limits.
- **Candidate K** is the number of nearest neighbours retrieved before the
  final selection policy runs. Candidate-K optimization is a future experiment
  and must not be confused with the current selector.

The retrieval baseline measures both the returned count and the quality at
K=1, 4, 8, and 16. This lets us test whether a future dynamic policy reaches
the required capabilities with fewer returned tools, rather than assuming that
“more tools” means better retrieval.

## What is distinctive—and what is not

Intent-aware dense retrieval is a known design pattern. Brown Octopus should
not claim novelty merely for combining those two components. Its practical
distinction is the surrounding capability-context contract: operational intent,
bounded selection, session TTL/cap controls, source and snapshot lifecycle,
incremental updates, and backend-aware persistence in one host-controlled
library.

Whether the selector itself is materially better remains an empirical question.
The reviewed retrieval harness is the mechanism for answering it. A dynamic-K
policy earns its place only if it improves required-tool recall or nDCG while
reducing selected-tool count and payload characters on the same reviewed task
set.

