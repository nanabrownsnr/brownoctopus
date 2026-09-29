# Brown Octopus Current Implementation Report

This report describes the implementation currently in the repository. It does
not propose or apply behavior changes.

## 1. Component interfaces and classes

The stable contracts are defined in `src/brown_octopus/contracts.py`:

- `IntentAnalyzer.analyze(query) -> list[dict]`
- `CandidateRetriever.retrieve(query, intent) -> list[dict]`
- `SelectionStrategy.select(query, intent, candidates) -> list[dict]`
- `CapabilityContextManager.update(retrieved_tools, turn) -> RetrievalResult`
- `CapabilityContextManager.reset() -> None`

Current implementations:

- `DeterministicIntentAnalyzer` in `analyzer.py`: adapter over the deterministic
  spaCy analyzer using `en_core_web_trf`.
- `QwenCandidateRetriever` in `retriever.py`: dense Qwen retrieval used by the
  frozen V3 pipeline.
- `BoundedMaxGapSelectionStrategy` in `strategies.py`: frozen V3 Min-4 plus
  Bounded Max Gap selection per intent.
- `JevSelectionStrategy` in `strategies.py`: retained as an internal
  experimental implementation and not selected by default.
- `ActiveCapabilityContext` in `context_manager.py`: merge, TTL expiry, and
  active-context capping.
- `CapabilityPipeline` in `pipeline.py`: coordinates the four components and
  performs merge/deduplication before context management.
- `RetrievalResult` in `contracts.py`: stable result object containing tool IDs,
  definitions, scores where available, and metadata.

`Octopus` remains the public coordinator. It does not expose a public
`mode=`, `approach=`, or `strategy=` selector. `retrieve_result()` exposes the
stable result object while the existing `retrieve()` and `process()` methods
remain available for compatibility.

Capability identity is stored separately from the readable exposed name. MCP
discovery adds `capability_id`, composed from the normalized MCP/server name,
a stable hash of the MCP URL, and the normalized tool name. Retrieval results,
deduplication, active-state keys, and `RetrievalResult.tool_ids` use this stable
ID. Older persisted tools without `capability_id` continue to use their legacy
`name` as a compatibility fallback.

## 2. `/v1/capabilities` schema

Endpoint:

```http
POST /v1/capabilities
Content-Type: application/json
```

Request body:

```json
{
  "session_id": "conversation-123",
  "query": "Find the latest Nvidia news"
}
```

`query` is required by the handler. `session_id` is optional and defaults to
`"default"`.

The current handler accepts an untyped JSON object. It does not explicitly
validate that `query` is a string before passing it to the pipeline.

Response body:

```json
{
  "retrieved_tools": [
    {
      "name": "web_search",
      "description": "Search the public web for current information and news.",
      "input_schema": {}
    }
  ],
  "intents": [
    {
      "action": "search",
      "target": "the latest Nvidia news",
      "text": "Search the web for the latest Nvidia news",
      "context": "Find the latest Nvidia news"
    }
  ],
  "per_intent": [
    {
      "intent": {},
      "candidate_count": 122,
      "selected_count": 1
    }
  ],
  "strategy": "brown_octopus_v3",
  "active_state": {
    "web_search": 1
  },
  "active_tools": [
    {
      "name": "web_search",
      "description": "Search the public web for current information and news.",
      "input_schema": {}
    }
  ],
  "tools": [
    {
      "name": "web_search",
      "description": "Search the public web for current information and news.",
      "input_schema": {}
    }
  ],
  "tool_ids": ["web_search-<server-hash>:web_search"]
}
```

The actual tool definitions and intent fields depend on the persisted index and
the analyzer output. `retrieved_tools` is the current-turn V3 selection;
`active_tools`, `tools`, and `tool_ids` represent the final active context.
The response uses `capability_id` as the capability ID for
newly discovered tools and falls back to legacy `name` for older persisted
definitions.

Additional endpoint:

```http
POST /v1/sessions/{session_id}/reset
```

returns:

```json
{"status": "reset"}
```

## 3. Jev selection behavior (internal, non-default)

The normal `Octopus` constructor builds frozen V3 with
`BoundedMaxGapSelectionStrategy`. It does not select Jev. The following
describes the retained Jev implementation for future internal experiments.

For each detected operational intent:

1. The internal Jev strategy receives the candidate list supplied by its caller.
   The normal V3 pipeline does not invoke this strategy.
2. Each candidate is serialized for Jev as:

   ```text
   <tool name>: <tool description>
   ```

   Input schemas and other MCP metadata are not sent to Jev.
3. `JevSelectionStrategy` calls:

   ```python
   reranker.relevance_rerank(
       intent_text,
       candidate_documents,
       threshold=0.2,
       return_documents=False,
   )
   ```

4. Jev's default mode is `listwise`, so the candidates for one intent are
   evaluated together in one relevance-ranking context. The Jev library may
   internally split requests if its own limits require it. The configured mode
   can be changed to `pointwise`, but the current default is listwise.
5. Jev returns result entries containing a candidate `document_index` and a
   relevance `score`. The library applies the supplied threshold and returns
   only entries passing it. Octopus maps those indices back to the original
   candidate definitions and stores the score as `jev_score`.
6. The final number is genuinely variable: it is the number of candidates Jev
   returns, not a fixed K. It can be zero through the candidate limit, subject
   to Jev behavior.

Current defaults:

```text
candidate K: not configured by the production V3 adapter
Jev mode: listwise
Jev relevance threshold: 0.2
```

For a multi-intent request, the analyzer produces multiple intents. Qwen and
Jev run independently for each intent. Selected tools are then merged in intent
order and deduplicated by `capability_id` before active-state management.

## 4. Failure and fallback behavior

The default production path does not depend on Jev. There is deliberately no
silent substitution of a different selector when the internal Jev strategy is
used.

| Condition | Current behavior |
|---|---|
| Jev package unavailable | `StrategyUnavailableError` with an install-extra message when the internal strategy is first used. |
| Jev API key missing or invalid | The underlying Jev configuration/API error propagates; there is no automatic fallback. |
| Jev timeout or transient API error | `jev-reranker` performs its own configured retry behavior; if it still fails, the exception propagates. Octopus does not catch it or substitute another selector. |
| Qwen/retriever failure | The exception propagates to the caller. There is no fallback retriever. |
| Analyzer failure | Exceptions propagate. A narrow parser safeguard handles a document without a spaCy `ROOT` by treating the entire original request as one opaque intent. |
| Empty query | The analyzer returns no intents; no tools are newly retrieved. |
| Empty Jev selection | No new tools are added. Existing active tools remain subject to TTL and active-cap processing. |
| Malformed capability definition | There is no dedicated schema validator. Missing `name` fields can cause failures during ranking, merge, or context lookup. |
| Missing persisted index | `Octopus.initialize()` raises a clear index-not-found error. |

## 5. Session and active-state behavior

`Octopus` maps each opaque `session_id` to a `SessionContext` in the
`InMemorySessionStore`, containing:

- an independent `ActiveCapabilityContext`;
- an independent conversation turn counter;
- a session-local lock;
- the shared analyzer, Qwen candidate retriever, and V3 selector objects.

Each selected tool refreshes its last-retrieved turn. A tool remains active while:

```text
current_turn - last_retrieved_turn <= 8
```

The active capability limit is `30` tools per session. If the limit is exceeded,
the least recently retrieved tools are evicted. The context limit is per session,
not global across all conversations.

Session cleanup is currently explicit only:

- `reset_session(session_id)` clears the session but keeps it available;
- `delete_session(session_id)` removes the session;
- process restart removes all sessions.

There is no idle-session timeout, persistent session store, or automatic session
garbage collection. The in-memory store can therefore grow if callers create
unbounded unique session IDs.

The store lock protects session creation, reset, and deletion. Each session then
uses its own lock while incrementing its turn and processing a request. Different
sessions can process concurrently; requests for the same session are serialized.

## 6. Startup behavior

`brown_octopus.server` creates an `Octopus` instance and initializes it during FastAPI
startup. Initialization loads:

1. spaCy `en_core_web_trf` once through the analyzer module global;
2. Qwen/Qwen3-Embedding-0.6B once through the retriever module global;
3. persisted `tools.json`, `embeddings.pt`, and `metadata.json`;
4. the runtime tool registry;
5. the replaceable production pipeline.

The Qwen index is loaded from disk; it is not rebuilt per request. Qwen query
encoding, intent analysis, and V3 selection occur per request. Startup time is
not currently benchmarked in the repository. It is dominated by model loading,
and first startup can also include model download if the model is not cached.

## 7. Environment variables and secrets

| Variable | Default | Purpose |
|---|---:|---|
| `OCTOPUS_INDEX_PATH` | `data/indexes/default` | Persisted capability index location. |
| `OCTOPUS_CATALOG_PATH` | `data/mcps.json` | MCP catalog used by update/discovery. |
| `OCTOPUS_CAPABILITY_TTL` | `8` | Active capability lifetime in turns. |
| `OCTOPUS_ACTIVE_CAP` | `30` | Maximum active capabilities per session. |
| `OCTOPUS_LOG_LEVEL` | `INFO` | Structured logging level. |
| `TYPESAFE_API_KEY` | none | Secret used only by internal Jev experiments. |

The key must not be committed to source control or baked into an image.

## 8. Docker deployment

The repository contains `Dockerfile` and `.dockerignore`.

Build:

```bash
docker build -t brown-octopus .
```

Run:

```bash
docker run --rm \
  -p 8000:8000 \
  -e TYPESAFE_API_KEY="$TYPESAFE_API_KEY" \
  brown-octopus
```

The service listens on port `8000` and binds to `0.0.0.0`. The image installs the
`service` and `jev` optional dependency groups and copies the default persisted
index and MCP catalog into the image. Installing Jev does not activate it.

## 9. End-to-end example

Request:

```json
{
  "session_id": "demo-1",
  "query": "Find the latest Nvidia news and email a summary to Tom"
}
```

Deterministic analyzer output, schematically:

```json
[
  {"action": "find", "text": "Find the latest Nvidia news", "target": "the latest Nvidia news"},
  {"action": "email", "text": "email a summary to Tom", "target": "a summary to Tom"}
]
```

Qwen candidate retrieval produces a ranked list per intent for the V3 selector:

```json
{
  "find": [
    {"name": "web_search", "rank": 1, "score": 0.91},
    {"name": "news_search", "rank": 2, "score": 0.88}
  ],
  "email": [
    {"name": "send_email", "rank": 1, "score": 0.90},
    {"name": "find_email_address", "rank": 2, "score": 0.84}
  ]
}
```

The lists above are illustrative; the actual persisted universe and model
scores determine the real candidates.

The internal Jev strategy, when used in an experiment, receives candidate
document lists separately for each intent:

```text
Query: Find the latest Nvidia news
Documents:
  0. web_search: Search the public web for current information and news.
  1. news_search: Search news sources.

Query: email a summary to Tom
Documents:
  0. send_email: Send an email message.
  1. find_email_address: Find a contact's email address.
```

Assume Jev returns `web_search`, `send_email`, and `find_email_address` above
the `0.2` threshold. Octopus merges and deduplicates them, updates active
state, and returns:

```json
{
  "strategy": "brown_octopus_v3",
  "tool_ids": ["web_search", "send_email", "find_email_address"],
  "active_state": {
    "web_search": 1,
    "send_email": 1,
    "find_email_address": 1
  },
  "tools": [
    {"name": "web_search", "description": "...", "input_schema": {}},
    {"name": "send_email", "description": "...", "input_schema": {}},
    {"name": "find_email_address", "description": "...", "input_schema": {}}
  ]
}
```

The consuming agent harness receives these definitions and remains responsible
for deciding whether and how to execute them.

## 10. Structured logging

`JsonFormatter` currently emits:

```json
{
  "timestamp": "2026-09-23T12:00:00Z",
  "level": "INFO",
  "logger": "octopus.pipeline",
  "message": "capability_context_updated",
  "data": {
    "turn": 1,
    "strategy": "brown_octopus_v3",
    "retrieved_count": 3,
    "exposed_count": 3
  }
}
```

Current fields are timestamp, level, logger, message, turn, strategy,
retrieved-count, and exposed-count. The current logs do not include session ID,
request ID, query text, latency, Jev request ID, Jev scores, token counts,
selection failures, or error stack metadata. Those fields are therefore not yet
sufficient for complete production evaluation observability.
