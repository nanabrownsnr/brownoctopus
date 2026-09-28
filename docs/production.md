# Production integration

This guide describes how to use Brown Octopus from an application. Brown
Octopus manages capability discovery, indexing, retrieval, and per-session
capability context. It does not execute tools or manage conversation history.

## Responsibilities

Brown Octopus owns:

- capability discovery through a `CapabilitySource`;
- the persisted capability index;
- intent analysis, retrieval, and V3 selection;
- active capability context and TTL handling;
- session capability state through a `SessionStore`.

The host application owns:

- source credentials and permissions;
- when discovery and index updates happen;
- the LLM and agent harness;
- tool binding and execution;
- authentication, authorization, and user identity;
- conversation history and session lifecycle policy.

Brown Octopus returns capability definitions. It never calls an MCP server or
executes a selected capability on behalf of the host.

## Installation and model setup

```bash
uv add brown-octopus
uv run brown-octopus setup-models
uv run brown-octopus doctor
```

The same commands work with an activated environment without the `uv run`
prefix. Model setup prepares `en_core_web_trf` and
`Qwen/Qwen3-Embedding-0.6B`. `initialize()` never downloads models. If a
model is missing, initialization raises an actionable error directing the host
to `brown-octopus setup-models`.

## Runtime pipeline

```text
user request
    -> deterministic operational-intent analysis
    -> capability-oriented retrieval text
    -> Qwen dense retrieval
    -> Min-4 + Bounded Max Gap selection per intent
    -> merge and deduplicate by capability_id
    -> session active capability context
    -> RetrievalResult
```

The frozen runtime settings are:

```text
embedding model:       Qwen/Qwen3-Embedding-0.6B
tool representation:   name + description
selection:             Min-4 + Bounded Max Gap
minimum per intent:    4 capabilities
maximum per intent:    16 capabilities
capability TTL:        8 turns
active capability cap: 30 capabilities
```

Jev remains available internally as an experimental implementation but is not
the default strategy.

## Capability index lifecycle

These operations have deliberately different responsibilities:

```text
brown-octopus setup-models
    prepare local model assets

await octopus.update()
    discover capabilities and build/publish an index

await octopus.initialize()
    load the existing models and index for runtime use
```

### Build or update an index

Create `setup_index.py`:

```python
import asyncio

from brown_octopus import Octopus
from brown_octopus.sources import LocalMcpCatalogSource


async def main():
    source = LocalMcpCatalogSource("data/mcps.json")
    octopus = Octopus(
        capability_source=source,
        index_path="data/indexes/default",
    )

    report = await octopus.update()
    print(f"Indexed: {report.tool_count}")
    print(f"Added: {len(report.added)}")
    print(f"Changed: {len(report.changed)}")
    print(f"Removed: {len(report.removed)}")
    print(f"Failed sources: {report.failed_sources}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it after model setup:

```bash
uv run python setup_index.py
```

`update()` discovers the source, prepares the new embeddings and metadata
outside the live snapshot, validates the result, and atomically publishes it.
Retrievals already in progress keep using their valid snapshot; subsequent
retrievals use the new one.

The current implementation rebuilds embeddings for the complete resulting
universe. It is not an incremental append-only operation.

### Add or change an MCP

The local catalog is a complete configured source snapshot. To add an MCP, add
it to the existing catalog and run the update again:

```json
{
  "items": [
    {"url": "https://example.com/outlook/mcp"},
    {"url": "https://example.com/word/mcp"},
    {"url": "https://example.com/github/mcp"}
  ]
}
```

Brown Octopus compares capabilities by stable `capability_id`:

```text
new capability                         -> added
existing ID with changed metadata      -> replaced
unchanged capability                   -> retained
missing from authoritative snapshot    -> removed
temporarily failed source              -> preserved
```

Passing a second catalog containing only a new MCP does not append it to the
previous catalog automatically. That file becomes the configured source
snapshot. Include all MCPs that should remain available, or implement a source
that defines the desired merge behavior.

Brown Octopus does not run a background refresh scheduler. The host decides
when updates happen.

### Normal runtime initialization

Once an index exists, application startup only loads local runtime state:

```python
import asyncio

from brown_octopus import Octopus


async def main():
    octopus = Octopus(index_path="data/indexes/default")
    await octopus.initialize()

    result = octopus.retrieve_result(
        "Reply to Tom's email",
        session_id="conversation-123",
    )

    print(f"Turn: {result.turn}")
    for capability in result.tools:
        print(capability["name"])


if __name__ == "__main__":
    asyncio.run(main())
```

`initialize()` does not discover capabilities, rebuild the index, or download
models. Calling it on every application start is expected.

## Capability sources

`LocalMcpCatalogSource` is the built-in source used by default. If no custom
source is supplied, `Octopus` creates one using its `catalog_path`:

```python
octopus = Octopus(
    catalog_path="data/mcps.json",
    index_path="data/indexes/default",
)
```

The catalog format is:

```json
{
  "items": [
    {"url": "https://example.com/outlook/mcp"},
    {"url": "https://example.com/word/mcp"}
  ]
}
```

The local source reads each URL, discovers its tools, and returns normalized
capability definitions. It does not store credentials in the catalog. MCP
authentication and execution remain the responsibility of the host or MCP
client.

### Custom sources

An application can inject a source from an internal API, database, registry,
marketplace, or any other system:

```python
from brown_octopus import Octopus

octopus = Octopus(
    capability_source=my_source,
    index_path="data/indexes/default",
)
```

The public contract is:

```python
from brown_octopus import CapabilityDiscoveryResult


class CapabilitySource:
    async def discover(self) -> CapabilityDiscoveryResult:
        ...
```

Example source backed by an internal registry:

```python
from brown_octopus import CapabilityDiscoveryResult


class InternalRegistrySource:
    async def discover(self) -> CapabilityDiscoveryResult:
        tools = [
            {
                "capability_id": "internal:crm:search_customers",
                "source_id": "internal-crm",
                "name": "search_customers",
                "description": "Search the customer database.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ]
        return CapabilityDiscoveryResult(
            tools=tools,
            successful_sources=["internal-crm"],
            failed_sources={},
            authoritative=True,
        )
```

Use the source when constructing `Octopus` and update the index:

```python
import asyncio

from brown_octopus import Octopus


async def main():
    octopus = Octopus(
        capability_source=InternalRegistrySource(),
        index_path="data/indexes/default",
    )
    report = await octopus.update()
    print(f"Indexed {report.tool_count} capabilities")


if __name__ == "__main__":
    asyncio.run(main())
```

Each normalized capability should provide:

```text
capability_id  stable globally unique capability identity
source_id      stable discovery provenance
name           readable capability name
description    text used for retrieval
input_schema   schema supplied to the host agent
```

MCP-specific fields such as `mcp_url`, `mcp_name`, and `tool_name` may be
included when applicable. They are not required for non-MCP capabilities.
`source_id` identifies discovery provenance; `mcp_url` is execution metadata
when MCP is involved.

### Discovery failures and removals

Sources should distinguish temporary failure from authoritative absence:

```python
return CapabilityDiscoveryResult(
    tools=tools_from_available_sources,
    successful_sources=["crm", "github"],
    failed_sources={"outlook": "connection timed out"},
    authoritative=True,
)
```

Brown Octopus preserves previously indexed capabilities whose `source_id` is
listed in `failed_sources`. It does not interpret a temporary outage as a
deletion.

If the complete discovery result is incomplete, return `authoritative=False`.
Absent capabilities are then preserved because the result cannot safely be
used as a complete removal snapshot. Mark a result authoritative only when
the source has determined the complete current state.

## Sessions and active capability context

Pass an opaque conversation identifier to retrieval:

```python
result = octopus.retrieve_result(
    "Send Sarah an email",
    session_id="conversation-123",
)
```

One engine can serve many conversations:

```text
                         Octopus
                   shared models and index
                              |
              +---------------+---------------+
              |               |               |
          session A       session B       session C
           turn/state      turn/state      turn/state
```

The analyzer, Qwen model, capability universe, embeddings, retriever, and
selector are shared. Each session has its own turn counter, active capability
IDs, TTL timestamps, active-cap eviction, and synchronization.

`session_id` identifies a conversation, not a user. It may be a LangGraph
`thread_id`, web chat ID, n8n session ID, or host-defined conversation key.

Session lifecycle operations are host-controlled:

```python
snapshot = octopus.get_session("conversation-123")
octopus.reset_session("conversation-123")
octopus.delete_session("conversation-123")
```

`reset_session()` clears the turn and active state while keeping the session
usable. `delete_session()` removes its state. Unknown sessions are created on
first retrieval by the default in-memory store.

The final context puts current-turn capabilities before retained capabilities.
TTL is 8 turns and the active capability cap is 30.

## SessionStore

The default store is zero-configuration in-process storage:

```python
from brown_octopus import Octopus

octopus = Octopus()
```

For persistence across processes or restarts, inject a host-owned store:

```python
octopus = Octopus(session_store=my_session_store)
```

The persisted public state is:

```python
from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    active_state: dict[str, int] = field(default_factory=dict)
```

`active_state` maps stable capability IDs to the turn on which they were last
retrieved. It contains no locks, runtime contexts, pipelines, models, tool
definitions, credentials, or conversation history.

The store contract is:

```python
from typing import Callable, Protocol, TypeVar

from brown_octopus import SessionState

T = TypeVar("T")


class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState: ...

    def mutate(
        self,
        session_id: str,
        operation: Callable[[SessionState], T],
    ) -> T: ...

    def reset(self, session_id: str) -> SessionState: ...

    def delete(self, session_id: str) -> None: ...

    def snapshot(self, session_id: str) -> dict | None: ...
```

The important boundary is `mutate()`. Brown Octopus performs the turn, TTL,
and active-cap changes inside that operation. A custom store must make the
callback atomic for a given session.

For an in-process store, use a per-session lock. For an external store, use its
transaction or distributed synchronization mechanism. A Redis implementation
could use optimistic locking or a transaction; a relational implementation
could use a transaction and row lock. This pattern is unsafe across processes:

```python
state = store.get(session_id)
state.turn += 1
store.save(session_id, state)
```

Two workers can read the same turn and overwrite one another. Implement the
complete read-modify-write operation behind `mutate()` instead.

## Retrieval results and agent integration

```python
result = octopus.retrieve_result(
    "Reply to Tom's email",
    session_id="conversation-123",
)
```

Stable result fields are:

```text
result.retrieved_tools  current-turn V3 selection only
result.tools            final active context for the session
result.tool_ids         capability IDs in result.tools
result.session_id       supplied session ID
result.turn             session turn after this request
```

The host normally exposes `result.tools` to the agent:

```text
user message
    -> octopus.retrieve_result(...)
    -> result.tools
    -> host binds tools to its LLM/agent
    -> host executes tool calls
```

LangGraph-specific code remains outside Brown Octopus. Brown Octopus has no
dependency on a particular agent harness.

## HTTP service

The optional FastAPI adapter exposes retrieval without executing tools. Install
the service dependencies and run the provided ASGI application with Uvicorn:

```bash
uv add "brown-octopus[service]"
uv run uvicorn brown_octopus.server:app --host 0.0.0.0 --port 8000
```

Endpoints include:

```text
GET  /health
GET  /ready
POST /v1/capabilities
POST /v1/sessions/{session_id}/reset
```

Request example:

```json
{
  "session_id": "conversation-123",
  "query": "Reply to Tom's email"
}
```

Authentication, authorization, credentials, and execution remain outside the
service boundary.

## Configuration and diagnostics

Important environment variables include:

```text
OCTOPUS_INDEX_PATH=data/indexes/default
OCTOPUS_CAPABILITY_TTL=8
OCTOPUS_ACTIVE_CAP=30
BROWN_OCTOPUS_MODEL_DIR=...       optional managed model directory
```

Useful commands:

```bash
brown-octopus --help
brown-octopus doctor
brown-octopus inspect
brown-octopus inspect --json
```

`doctor` checks package, model, and index readiness. `inspect` reports the
active index and runtime configuration without performing capability discovery.
Neither command exposes credentials.

## Logging and deployment

Brown Octopus uses Python logging and is quiet by default as a library. Hosts
can configure the `brown_octopus` logger. Context-update events include fields
such as `session_id`, `turn`, `strategy`, `retrieved_count`, `exposed_count`,
and timing data. Debug logs may include capability IDs, but complete schemas
and credentials should not be logged.

Before deployment, verify that the process has models prepared, a compatible
index, a configured capability source, and a session store appropriate for its
worker topology. The default in-memory store is suitable for one process. For
multiple workers sharing conversations, provide a store whose `mutate()` is
atomic across workers.

## Release validation from a clone

```bash
uv sync --dev
uv run brown-octopus setup-models
uv run pytest
uv build
```

Install the built wheel into a separate clean environment before distributing
it. The source checkout must not be required at runtime.
