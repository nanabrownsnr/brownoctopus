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

## Public API at a glance

The normal application-facing API is intentionally small:

```python
from brown_octopus import Octopus

octopus = Octopus()
await octopus.initialize()

result = octopus.retrieve_result(
    user_message,
    session_id="conversation-123",
)

tools = result.tools
```

The public lifecycle is split deliberately:

```text
brown-octopus setup-models
    prepare local model assets

await index.create()
    perform the initial capability discovery and build the persisted index

await index.update()
    synchronize the configured sources after the index exists

await octopus.initialize()
    load existing local models and index for runtime retrieval

octopus.retrieve_result(...)
    retrieve the current active capability context
```

`Octopus` is the runtime facade. `CapabilitySource`, `SessionStore`, and the
lower-level analyzer/retriever/selector protocols are extension points for
applications that need custom infrastructure. Most applications should use
`OctopusIndex.create()` during setup, `OctopusIndex.update()` for later source
synchronization, and `Octopus.initialize()` at application startup.

For applications that want a simpler setup facade, `OctopusIndex` wraps the
same source and snapshot lifecycle:

```python
from brown_octopus import OctopusIndex

index = OctopusIndex.from_mcp_server(
    "https://example.com/outlook/mcp",
    index_path="data/indexes/default",
)
await index.create()

octopus = index.runtime()
await octopus.initialize()
```

The facade also supports normalized JSON and HTTP API sources:

```python
file_index = OctopusIndex.from_file(
    "data/capabilities.json",
    items_path="capabilities",
)

api_index = OctopusIndex.from_api(
    "https://registry.example.com/capabilities",
    items_path="data.items",
    fields={
        "capability_id": "id",
        "source_id": "server_id",
        "name": "name",
        "description": "description",
        "input_schema": "input_schema",
    },
    headers={"Authorization": "Bearer <host-provided-token>"},
)
```

`from_api()` expects one normalized capability record per item. It preserves
execution metadata when supplied:

```json
{
  "capability_id": "outlook:send_email",
  "source_id": "outlook-prod",
  "name": "outlook_send_email",
  "description": "Send an email",
  "input_schema": {"type": "object"},
  "mcp_url": "https://example.com/outlook/mcp",
  "tool_name": "send_email"
}
```

The API source expects JSON records and field mappings; credentials belong to
the host application.

For registries that return MCP server records rather than normalized
capabilities, use the dedicated registry source. It fetches all pages,
discovers tools from each server URL, and records partial source failures:

```python
registry_index = OctopusIndex.from_mcp_registry(
    "https://registry.example.com/mcp-servers",
    items_path="items",
    headers={"Authorization": "Bearer <host-provided-token>"},
    page_size=50,
    index_path="data/indexes/default",
)

report = await registry_index.create()
```

The registry source supports both page/limit and cursor pagination. Cursor
responses are followed through fields such as `next_cursor` or
`pagination.next_cursor`; repeated cursors are rejected to prevent an
infinite refresh loop.

If a registry token can expire, the host can provide a one-time refresh
callback:

```python
async def refresh_headers():
    token = await get_new_registry_token()
    return {"Authorization": f"Bearer {token}"}


registry_index = OctopusIndex.from_mcp_registry(
    registry_url,
    headers={"Authorization": "Bearer <short-lived-token>"},
    refresh_headers=refresh_headers,
)
```

`mcp_url` and `tool_name` are execution metadata. Brown Octopus returns them
to the host but does not execute the MCP tool. The host owns credentials,
authorization, and execution. Header values and tokens are not logged by the
source.

The MCP convenience source supports one HTTP MCP server. stdio MCP
configuration is not currently part of this facade.

Index lifecycle methods are:

```python
await index.create()                         # initial/full synchronization
await index.add(other_source)                # add a configured source
await index.update()                          # synchronize all sources
await index.remove_source("outlook-prod")    # remove one source
await index.remove_capability("crm_search")  # suppress one capability
index.reset()                                 # remove the persisted index
```

`OctopusIndex.from_sources([...])` is available for advanced applications that
already have custom `CapabilitySource` objects. `OctopusIndex` does not replace
the lower-level `CapabilitySource` contract; it provides convenience adapters
for common source shapes.

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

At runtime, Brown Octopus detects the best accelerator visible to the local
PyTorch installation. It prefers CUDA, then Apple MPS or Intel XPU when
available, and otherwise uses CPU. Qwen receives this device explicitly. spaCy
also requests GPU execution when its compatible GPU runtime is available and
otherwise remains on CPU. No GPU is required.

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

await index.create()
    initial discovery and index creation through the setup facade

await index.update()
    later synchronization of the configured sources

await octopus.initialize()
    load the existing models and index for runtime use
```

For normal applications, use `OctopusIndex.create()` and
`OctopusIndex.update()`. The direct `Octopus.update()` method remains an
advanced lower-level path for applications that intentionally construct the
runtime with a custom `CapabilitySource`.

### Build or update an index

Create `setup_index.py`:

```python
import asyncio

from brown_octopus import OctopusIndex
from brown_octopus.sources import LocalMcpCatalogSource


async def main():
    source = LocalMcpCatalogSource("data/mcps.json")
    index = OctopusIndex.from_sources(
        [source],
        index_path="data/indexes/default",
    )

    report = await index.create()
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

`create()` discovers the source, prepares the new embeddings and metadata
outside the live snapshot, validates the result, and atomically publishes it.
Retrievals already in progress keep using their valid snapshot; subsequent
retrievals use the new one.

After the initial index exists, use `await index.update()` to synchronize the
same configured sources. Use `await octopus.update()` only when working with
the lower-level runtime API directly.

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

Use the source with `OctopusIndex` and create the initial index:

```python
import asyncio

from brown_octopus import OctopusIndex


async def main():
    index = OctopusIndex.from_sources(
        [InternalRegistrySource()],
        index_path="data/indexes/default",
    )
    report = await index.create()
    print(f"Indexed {report.tool_count} capabilities")


if __name__ == "__main__":
    asyncio.run(main())
```

### Source backed by an HTTP API

The source can call an API owned by the host application. The API client and
its credentials belong to the host; Brown Octopus only receives the normalized
discovery result.

```python
import httpx

from brown_octopus import CapabilityDiscoveryResult


class RegistryApiSource:
    def __init__(self, base_url: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token

    async def discover(self) -> CapabilityDiscoveryResult:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{self.base_url}/capabilities",
                headers=headers,
            )
            response.raise_for_status()
            records = response.json()

        tools = [
            {
                "capability_id": record["id"],
                "source_id": record.get("source_id", "internal-registry"),
                "name": record["name"],
                "description": record.get("description", ""),
                "input_schema": record.get("input_schema", {}),
            }
            for record in records
        ]
        return CapabilityDiscoveryResult(
            tools=tools,
            successful_sources=["internal-registry"],
            failed_sources={},
            authoritative=True,
        )
```

Use it like any other source through `OctopusIndex`:

```python
index = OctopusIndex.from_sources(
    [RegistryApiSource(
        base_url="https://registry.example.com",
        api_token=registry_token,
    )],
    index_path="data/indexes/default",
)
await index.create()
```

The token is application configuration. Do not put it in capability metadata,
the index, or session state.

### Source backed by a database

A database source follows the same pattern. The database schema and connection
pool are host concerns:

```python
from brown_octopus import CapabilityDiscoveryResult


class CapabilityDatabaseSource:
    def __init__(self, db):
        self.db = db

    async def discover(self) -> CapabilityDiscoveryResult:
        rows = await self.db.fetch_all(
            """
            SELECT capability_id, source_id, name, description, input_schema
            FROM capabilities
            WHERE enabled = TRUE
            """
        )

        tools = [
            {
                "capability_id": row["capability_id"],
                "source_id": row["source_id"],
                "name": row["name"],
                "description": row["description"] or "",
                "input_schema": row["input_schema"] or {},
            }
            for row in rows
        ]
        return CapabilityDiscoveryResult(
            tools=tools,
            successful_sources=["capability-database"],
            failed_sources={},
            authoritative=True,
        )
```

The host creates `db` using its chosen database library, connection pool, and
credentials. Brown Octopus does not require or select an ORM. If the query
fails, either raise the error so `update()` preserves the existing snapshot,
or return a result with the affected source in `failed_sources`.

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

### SQLite example

For a small single-host deployment, a custom store can keep only the
serializable session fields in SQLite. `BEGIN IMMEDIATE` makes the mutation a
serialized write transaction:

```python
import json
import sqlite3
from copy import deepcopy

from brown_octopus import SessionState


class SQLiteSessionStore:
    def __init__(self, path: str = "data/sessions.sqlite3"):
        self.path = path
        with sqlite3.connect(self.path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS octopus_sessions (
                    session_id TEXT PRIMARY KEY,
                    turn INTEGER NOT NULL,
                    active_state TEXT NOT NULL
                )
                """
            )

    def _read(self, db, session_id: str) -> SessionState:
        row = db.execute(
            "SELECT turn, active_state FROM octopus_sessions "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return SessionState(session_id=session_id)
        return SessionState(
            session_id=session_id,
            turn=row[0],
            active_state=json.loads(row[1]),
        )

    def _write(self, db, state: SessionState) -> None:
        db.execute(
            """
            INSERT INTO octopus_sessions(session_id, turn, active_state)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                turn = excluded.turn,
                active_state = excluded.active_state
            """,
            (
                state.session_id,
                state.turn,
                json.dumps(state.active_state),
            ),
        )

    def get(self, session_id: str) -> SessionState:
        with sqlite3.connect(self.path) as db:
            return deepcopy(self._read(db, session_id))

    def mutate(self, session_id, operation):
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            state = self._read(db, session_id)
            result = operation(state)
            self._write(db, state)
            return result

    def reset(self, session_id: str) -> SessionState:
        def clear(state):
            state.turn = 0
            state.active_state.clear()
            return deepcopy(state)

        return self.mutate(session_id, clear)

    def delete(self, session_id: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                "DELETE FROM octopus_sessions WHERE session_id = ?",
                (session_id,),
            )

    def snapshot(self, session_id: str) -> dict | None:
        state = self.get(session_id)
        return {
            "session_id": state.session_id,
            "turn": state.turn,
            "active_state": dict(state.active_state),
            "active_tool_ids": list(state.active_state),
        }
```

For a multi-process deployment, use the transaction and locking primitives
recommended by the selected database. Test the implementation under the
actual worker topology before relying on it in production.

### Redis example

Redis is not a Brown Octopus dependency. A host that already uses Redis can
implement the same contract with its Redis client. The key requirement is an
atomic compare-and-set around the complete callback:

```python
import json
from dataclasses import asdict

from redis import Redis, WatchError

from brown_octopus import SessionState


class RedisSessionStore:
    def __init__(self, client: Redis, prefix: str = "brown-octopus:session:"):
        self.client = client
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    @staticmethod
    def _decode(session_id: str, raw) -> SessionState:
        if raw is None:
            return SessionState(session_id=session_id)
        value = json.loads(raw)
        return SessionState(**value)

    def get(self, session_id: str) -> SessionState:
        return self._decode(session_id, self.client.get(self._key(session_id)))

    def mutate(self, session_id, operation):
        key = self._key(session_id)
        while True:
            try:
                with self.client.pipeline() as pipe:
                    pipe.watch(key)
                    state = self._decode(session_id, pipe.get(key))
                    result = operation(state)
                    pipe.multi()
                    pipe.set(key, json.dumps(asdict(state)))
                    pipe.execute()
                    return result
            except WatchError:
                # Another worker changed this session; retry from fresh state.
                continue

    def reset(self, session_id: str) -> SessionState:
        def clear(state):
            state.turn = 0
            state.active_state.clear()
            return state

        return self.mutate(session_id, clear)

    def delete(self, session_id: str) -> None:
        self.client.delete(self._key(session_id))

    def snapshot(self, session_id: str) -> dict | None:
        state = self.get(session_id)
        return {
            "session_id": state.session_id,
            "turn": state.turn,
            "active_state": dict(state.active_state),
            "active_tool_ids": list(state.active_state),
        }
```

The host installs and configures its Redis client, then injects the store:

```python
from redis import Redis
from brown_octopus import Octopus

store = RedisSessionStore(Redis.from_url(redis_url))
octopus = Octopus(session_store=store)
```

The callback may run more than once if another worker wins the optimistic-lock
race. The callback must therefore be deterministic and must not perform an
external side effect such as sending an email. Brown Octopus's callback only
updates capability state; tool execution remains outside the store.

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

### MCP URL prefilter

Hosts can restrict one retrieval request to a set of MCP endpoints:

```python
result = octopus.retrieve_result(
    "Reply to Tom's email",
    session_id="conversation-123",
    allowed_mcp_urls=[
        "https://example.com/outlook/mcp",
        "https://example.com/gmail/mcp",
    ],
)
```

The processing order is:

```text
indexed capabilities
    -> intent analysis
    -> allowed_mcp_urls filter
    -> Qwen ranking
    -> V3 selection
    -> active-context update
```

The allowlist is request-scoped and never mutates the shared index. If it is
omitted (`None`), all indexed capabilities participate as before. An empty
list means that no MCP capabilities are allowed. Matching is exact after
trimming a trailing slash, so these are treated as equivalent:

```text
https://example.com/outlook/mcp
https://example.com/outlook/mcp/
```

Capabilities without an `mcp_url` do not match an active MCP URL allowlist.
The allowlist is also applied to the active session context. A capability from
an MCP that is not allowed on the current request cannot remain exposed merely
because it was retrieved on an earlier turn.

The HTTP adapter accepts the same field in its JSON request:

```json
{
  "session_id": "conversation-123",
  "query": "Reply to Tom's email",
  "allowed_mcp_urls": [
    "https://example.com/outlook/mcp"
  ]
}
```

### Example response

The Python result is a `RetrievalResult`. A representative response looks like:

```python
RetrievalResult(
    tool_ids=["outlook-123:send_email"],
    tools=[
        {
            "rank": 1,
            "score": 0.91,
            "capability_id": "outlook-123:send_email",
            "source_id": "outlook-123",
            "name": "outlook_send_email",
            "tool_name": "send_email",
            "mcp_name": "Outlook Mail",
            "mcp_url": "https://example.com/outlook/mcp",
            "description": "Send an email.",
            "input_schema": {"type": "object"},
        }
    ],
    retrieved_tools=[...],
    session_id="conversation-123",
    turn=1,
)
```

`retrieved_tools` contains the current-turn selection. `tools` contains the
final active context, including retained capabilities where applicable. Both
contain the complete capability definition and metadata supplied by the source,
including `mcp_url` when the capability is MCP-backed.

### Method return values

```python
report = await index.create()
# initial CapabilityUpdateReport

report = await index.update()
# later synchronization: added, changed, removed, failed_sources, tool_count

loaded_tools = await octopus.initialize()
# list[dict] containing the tools loaded from the existing index

active_tools = octopus.retrieve("Reply to Tom's email")
# list[dict] containing the final active context

result = octopus.retrieve_result("Reply to Tom's email")
# RetrievalResult with retrieved_tools, tools, tool_ids, session_id, and turn

details = octopus.process("Reply to Tom's email")
# dict containing the pipeline result, session metadata, and timing data

session = octopus.get_session("conversation-123")
# dict containing session_id, turn, active_state, and active_tool_ids
```

`reset_session()` and `delete_session()` return `None`. `update()` and
`initialize()` are asynchronous because they perform model/index I/O;
retrieval and session inspection are synchronous.

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

## Registry integration checks

For a registry-backed deployment, validate the complete path before enabling
it in production:

```text
registry API
    -> pagination
    -> MCP server discovery
    -> capability index
    -> runtime retrieval
    -> host-side MCP execution
```

The registry source handles pagination and preserves prior capabilities when a
source is temporarily unavailable. It can refresh registry request headers
once after an authorization failure when the host supplies `refresh_headers`.
It does not refresh tokens used by the host's later MCP execution request.
That token lifecycle belongs to the host execution layer.

Run the external validation described in
[`registry_integration_test_plan.md`](registry_integration_test_plan.md).
Use a read-only MCP allowlist first. Run write-capable operations only against
an explicitly approved disposable service.

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

If retrieval succeeds but MCP execution returns HTTP 401 or 403, Brown
Octopus has already completed its responsibility. Refresh the host execution
token and retry the MCP call; do not treat that response as a retrieval or
index failure.

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

Known boundaries:

- Brown Octopus does not execute MCP tools or own their access tokens.
- Registry discovery can preserve a previous snapshot after a partial source
  failure, but the host must decide when to retry the source.
- External `SessionStore` implementations must provide atomic `mutate()`
  behavior across workers.
- The default in-memory session store is process-local.
- The V3 selector is recall-oriented and may expose adjacent capabilities.
