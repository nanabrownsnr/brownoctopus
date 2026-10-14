# Brown Octopus production guide

This document is the detailed integration reference for Brown Octopus 0.5. It
covers index lifecycle, capability sources, runtime retrieval, sessions,
embedding providers, local and database-backed index storage, diagnostics, and
the host/application boundary.

For the shortest complete setup, use the [README](../README.md). The README
is intentionally a first-run guide; this document explains the alternatives.

## 1. What Brown Octopus owns

Brown Octopus manages capability context:

- capability discovery adapters;
- capability indexing and snapshot publication;
- deterministic operational-intent analysis;
- candidate retrieval and bounded selection;
- active capability context;
- capability-context session state.

The host application owns:

- the LLM and agent framework;
- conversation history;
- capability binding and execution;
- credentials, authentication, and authorization;
- user identity;
- update timing and session lifecycle policy.

Brown Octopus returns capability definitions. It does not execute MCP or
application tools.

## 2. Installation and model setup

```bash
pip install brown-octopus
brown-octopus setup-models
brown-octopus doctor
```

With uv:

```bash
uv add brown-octopus
uv run brown-octopus setup-models
uv run brown-octopus doctor
```

The default setup prepares `en_core_web_trf` and
`Qwen/Qwen3-Embedding-0.6B`. Select one asset when needed:

```bash
brown-octopus setup-models --spacy
brown-octopus setup-models --embedding
```

The selectors are mutually exclusive. With no selector, both assets are
prepared. Setup is idempotent.

Normal runtime initialization is offline-by-default:

```python
from brown_octopus import Octopus

octopus = Octopus(index_path="data/indexes/default")
await octopus.initialize()
```

`initialize()` never downloads models, discovers capabilities, or updates an
index. If a model is missing, the error directs the operator to
`brown-octopus setup-models`.

Managed spaCy assets are kept outside the consumer virtual environment. Set
`BROWN_OCTOPUS_MODEL_DIR` for a container volume or shared model directory:

```powershell
$env:BROWN_OCTOPUS_MODEL_DIR = "D:\brown-octopus-models"
brown-octopus setup-models
```

```bash
export BROWN_OCTOPUS_MODEL_DIR=/var/lib/brown-octopus/models
brown-octopus setup-models
```

## 3. The lifecycle

```text
setup-models
    prepare runtime model assets

OctopusIndex.create()
    create and publish a snapshot only if one is missing

OctopusIndex.create(replace=True)
    deliberately rebuild and publish a new snapshot

OctopusIndex.update()
    host-requested synchronization of configured sources

Octopus.runtime().initialize()
    load models and the current snapshot for retrieval
```

### Create the initial index

```python
import asyncio

from brown_octopus import OctopusIndex
from brown_octopus.sources import LocalMcpCatalogSource


async def main() -> None:
    index = OctopusIndex.from_sources(
        [LocalMcpCatalogSource("data/mcps.json")],
        index_path="data/indexes/default",
    )
    report = await index.create()
    print(f"Indexed {report.tool_count} capabilities")
    print(report.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
```

### Start the runtime

```python
import asyncio

from brown_octopus import Octopus


async def main() -> None:
    octopus = Octopus(index_path="data/indexes/default")
    loaded = await octopus.initialize()
    print(f"Loaded {len(loaded)} capabilities")

    result = octopus.retrieve_result(
        "Reply to Tom's email",
        session_id="conversation-123",
    )
    print([tool["name"] for tool in result.tools])


if __name__ == "__main__":
    asyncio.run(main())
```

### Start before an index exists

`OctopusIndex` may be created without sources and without a persisted index:

```python
index = OctopusIndex(index_path="data/indexes/new-index")
octopus = index.runtime()
await octopus.initialize()

result = octopus.retrieve_result("Reply to Tom's email")
assert result.tools == []
assert result.retrieved_tools == []
assert result.tool_ids == []
```

This is a valid staged-provisioning state. The host can establish the empty
index first, then add a source later:

```python
from brown_octopus import McpServerSource

await index.create()
await index.add(McpServerSource("https://example.com/outlook/mcp"))
```

`add()` is incremental. It requires an existing index, discovers only the
supplied source, embeds only new or changed capabilities, and preserves the
rest of the snapshot. It does not synchronize or remove capabilities from
other sources. Several sources can be added in one operation:

```python
await index.add(
    McpServerSource("https://example.com/outlook/mcp", source_id="outlook"),
    McpServerSource("https://example.com/word/mcp", source_id="word"),
)
```

### Indexing reports

`create()`, `update()`, `add()`, and `remove_source()` return a
`CapabilityUpdateReport`. These operations are synchronous from the library
caller’s perspective: when the method returns, `completed` is `True` and the
snapshot has either been published or an exception has been raised. The report
does not represent a background job.

The report exposes both typed fields and a JSON-safe representation:

```python
report = await index.add(McpServerSource("https://example.com/maps/mcp"))

print(report.operation)          # "add"
print(report.completed)          # True
print(report.added)              # capability IDs newly indexed
print(report.changed)            # capability IDs replaced
print(report.removed)            # capability IDs removed
print(report.failed_sources)     # source -> discovery error
print(report.tool_count)         # resulting snapshot size
print(report.authoritative)      # complete source view for update(), when true
print(report.embedded_count)     # capabilities actually sent to the embedder
print(report.duration_ms)        # complete operation duration in milliseconds
print(report.embedding_duration_ms)

http_response_body = report.to_dict()
```

`to_dict()` includes `counts` for added, changed, removed, and failed sources,
while preserving the established response shape by default. Hosts that need
the new safe timing and embedding counters can use
`report.to_dict(include_metrics=True)`. Prompts and authentication headers are
not included.

For a local indexing baseline that does not load models or contact MCP servers,
run:

```text
uv run python scripts/benchmark_indexing.py --sizes 10 100 1000 --json
```

This synthetic benchmark measures lifecycle and incremental-work behavior; it
does not represent real embedding throughput. Use a configured local provider
separately to measure CPU/GPU inference.

To measure a cached local embedding model without permitting an implicit model
download:

```text
uv run python scripts/benchmark_indexing.py --real-embedding --device cpu --sizes 10 100
```

Use `--device cuda` to test a CUDA host. The command reports model, device,
batch size, dimensionality, throughput, Python peak memory, and CUDA memory
when available. It fails clearly if the requested model is not already cached.
It contains no MCP headers, credentials, or full tool definitions. A host API
should return this object instead of a generic “success” message so callers
can distinguish a successful no-op from a mutation or partial discovery.

### Background indexing jobs

`await index.add(...)`, `await index.update()`, and `await index.create()` are
still the direct lifecycle APIs. For a web service or agent host that cannot
hold an HTTP request open during discovery and embedding, use the optional
per-index job queue:

```python
index = OctopusIndex(
    index_path="data/indexes/default",
    max_pending_jobs=1,
    busy_policy="coalesce",
)

job = index.submit_add(source)
print(job.snapshot())
await job.wait()
report = job.result()
```

The queue has one active worker. `coalesce` merges pending `submit_add()` calls
into one job; `reject` raises `IndexBusyError` when pending capacity is full;
`wait` accepts work into the queue. An active job is never replaced by a newer
request. `submit_update()` and `submit_create(replace=True)` are also
available. `await index.close()` drains the active operation and cancels jobs
that have not started.

The queue is process-local. Multi-worker FastAPI deployments must route all
jobs for one index to one owner process or add a host-level distributed queue
and lock. Brown Octopus does not claim to coordinate separate worker
processes.

For a new index, establish the snapshot first:

```python
await index.create()
await index.add(McpServerSource("https://example.com/outlook/mcp"))
```

A missing path is not an implicit discovery request. A malformed existing
snapshot remains an error because it may indicate corruption or an incomplete
deployment.

For application startup, it is also safe to call `create()` as an
ensure-created operation. If the configured store already has an active
snapshot, `create()` leaves it untouched. Use `create(replace=True)` only when
the host explicitly wants to rediscover sources and replace the active
snapshot.

## 4. `OctopusIndex`

### Constructor

```text
OctopusIndex(
    sources=None,
    *,
    index_path="data/indexes/default",
    session_store=None,
    config=None,
    embedding_provider=None,
    index_store=None,
)
```

`sources` is optional. Supply a sequence of `CapabilitySource` objects when
the index should discover capabilities. Omit it when opening an existing
index, starting empty, or adding sources later.

### Factory methods

```text
from_sources(sources, **kwargs)       custom CapabilitySource objects
from_mcp_server(url, **kwargs)        one HTTP MCP server
from_mcp_registry(url, **kwargs)      registry records and MCP discovery
from_file(path, **kwargs)             normalized capabilities from JSON
from_api(url, **kwargs)               normalized capabilities from HTTP JSON
```

### Lifecycle methods

```text
await index.create()                  create snapshot only if missing
await index.create(replace=True)      deliberately replace snapshot
await index.update()                  synchronize configured sources
await index.add(source)               incrementally add to an existing index
await index.add(source_a, source_b)  batch incremental additions
await index.remove_source(source_id) incrementally remove a source by ID or URL
index.runtime()                       return the Octopus runtime
index.octopus                         compatibility alias for runtime()
index.reset()                         delete persisted index state
```

For a normal ``McpServerSource``, ``source_id`` defaults to the MCP URL, so the
following removes all capabilities belonging to that MCP without rediscovering
the remaining sources:

```python
report = await index.remove_source("https://example.com/word/mcp")
print(report.removed)
```

Removal also works when the index was reopened with no source objects. Registry
records that use explicit server IDs should be removed using that persisted
source ID.

### Add and remove sources

```python
from brown_octopus import McpServerSource, OctopusIndex

index = OctopusIndex(index_path="data/indexes/default")
await index.add(
    McpServerSource(
        "https://example.com/word/mcp",
        source_id="word",
    )
)
await index.remove_source("word")
```

`remove_source()` changes only the persisted snapshot and the current runtime
state. It does not call MCP discovery or require source objects to be
recreated. Credentials and source configuration are not stored in the index
snapshot. Use `update()` when you want to reconcile all configured sources.

## 5. Capability source contract

The public protocol is:

```python
class CapabilitySource(Protocol):
    async def discover(self) -> CapabilityDiscoveryResult:
        ...
```

The result separates discovery health from returned tools:

```python
CapabilityDiscoveryResult(
    tools=[...],
    successful_sources=["source-id"],
    failed_sources={"offline-source": "temporary error"},
    authoritative=True,
)
```

Normalized capabilities should contain `capability_id`, `source_id`, `name`,
`description`, and `input_schema`. MCP records may also contain `mcp_url`,
`tool_name`, and `mcp_name`.

`source_id` is discovery provenance. `mcp_url` is an execution endpoint. The
generic identity remains useful for API, database, marketplace, and stdio
sources that do not have an HTTP URL.

## 6. Built-in sources

### Local MCP catalog

The catalog is an object containing an `items` list. Each item needs `url`; an
`id` is recommended:

```json
{
  "items": [
    {"id": "crm", "name": "CRM", "url": "https://example.com/crm/mcp"}
  ]
}
```

```python
from brown_octopus import OctopusIndex
from brown_octopus.sources import LocalMcpCatalogSource

index = OctopusIndex.from_sources(
    [LocalMcpCatalogSource("data/mcps.json")],
    index_path="data/indexes/default",
)
await index.create()
```

### One HTTP MCP server

```python
index = OctopusIndex.from_mcp_server(
    "https://example.com/outlook/mcp",
    source_id="outlook",
    mcp_headers={"Authorization": "Bearer <short-lived-token>"},
    index_path="data/indexes/outlook",
)
await index.create()
```

``mcp_headers`` is optional and is used only while the host discovers tools
from that MCP. Header values are not included in capability records, persisted
snapshots, or logs. The host remains responsible for obtaining and refreshing
short-lived credentials.

### Normalized JSON

`JsonCapabilitySource` reads one normalized capability per item:

```json
{
  "capabilities": [
    {
      "capability_id": "crm_search",
      "source_id": "crm",
      "name": "search_customers",
      "description": "Search customer records",
      "input_schema": {"type": "object"}
    }
  ]
}
```

```python
index = OctopusIndex.from_file(
    "data/capabilities.json",
    items_path="capabilities",
    index_path="data/indexes/file-index",
)
await index.create()
```

### Normalized capability API

```python
index = OctopusIndex.from_api(
    "https://registry.example.com/capabilities",
    items_path="data.items",
    fields={
        "capability_id": "id",
        "source_id": "server.id",
        "name": "tool_name",
        "description": "summary",
        "input_schema": "schema",
    },
    headers={"Authorization": "Bearer <host-token>"},
    index_path="data/indexes/api-index",
)
await index.create()
```

The adapter preserves MCP metadata when it is present. Do not put credentials
in capability records or logs.

### MCP registry API

```python
index = OctopusIndex.from_mcp_registry(
    "https://registry.example.com/mcp-servers",
    items_path="items",
    headers={"Authorization": "Bearer <host-token>"},
    mcp_headers={"Authorization": "Bearer <mcp-token>"},
    page_size=50,
    cursor_param="cursor",
    index_path="data/indexes/registry-index",
)
await index.create()
```

The adapter supports page/limit and cursor pagination. A host can refresh an
expired registry token once:

```python
async def refresh_headers() -> dict[str, str]:
    token = await get_new_registry_token()
    return {"Authorization": f"Bearer {token}"}


index = OctopusIndex.from_mcp_registry(
    registry_url,
    headers={"Authorization": "Bearer <short-lived-token>"},
    refresh_headers=refresh_headers,
)
```

``headers`` authenticate requests to the registry API. ``mcp_headers`` are
sent to each discovered MCP during tool discovery. They may be the same
mapping when the registry and MCP servers accept the same token, or different
mappings when they use separate credentials. MCP headers are not persisted or
returned as capability metadata.

If one MCP server fails, prior capabilities are preserved when the result is
non-authoritative. An authoritative absence is treated as removal.

### Custom source

```python
from brown_octopus import CapabilityDiscoveryResult, OctopusIndex


class InternalCapabilitySource:
    source_id = "internal-registry"

    async def discover(self) -> CapabilityDiscoveryResult:
        records = await fetch_records_from_the_host_application()
        return CapabilityDiscoveryResult(
            tools=records,
            successful_sources=[self.source_id],
            failed_sources={},
            authoritative=True,
        )


index = OctopusIndex.from_sources(
    [InternalCapabilitySource()],
    index_path="data/indexes/internal",
)
await index.create()
```

The source owns its network client and credentials. Brown Octopus only
requires the normalized result.

## 7. Runtime API

```text
await initialize() -> list[dict]
retrieve(query, session_id="default", allowed_mcp_urls=None) -> list[dict]
retrieve_result(query, session_id="default", allowed_mcp_urls=None)
process(query, session_id="default", allowed_mcp_urls=None) -> dict
await update() -> CapabilityUpdateReport
get_session(session_id="default") -> dict | None
reset_session(session_id) -> None
delete_session(session_id) -> None
reset() -> None
reset_index() -> None
```

The result contract is:

```python
result = octopus.retrieve_result(
    "Reply to Tom's email",
    session_id="conversation-123",
)

result.retrieved_tools  # current-turn capabilities
result.tools            # final active context
result.tool_ids         # IDs for result.tools
result.session_id       # conversation identity
result.turn             # session turn
result.metadata         # timing and pipeline metadata
```

## 8. Sessions and session stores

The default `InMemorySessionStore` is process-local:

```python
from brown_octopus import InMemorySessionStore, Octopus

octopus = Octopus(session_store=InMemorySessionStore())
```

Durable state contains only `session_id`, `turn`, and `active_state`. It does
not contain locks, models, clients, credentials, or conversation history.

The public custom-store contract is:

```python
class SessionStore(Protocol):
    def get(self, session_id: str) -> SessionState: ...
    def mutate(self, session_id: str, operation: Callable[[SessionState], T]) -> T: ...
    def reset(self, session_id: str) -> SessionState: ...
    def delete(self, session_id: str) -> None: ...
    def snapshot(self, session_id: str) -> dict | None: ...
```

`mutate()` must be atomic for a session. Redis should use `WATCH`/`MULTI` or a
distributed lock. A database should use a transaction and row lock. A plain
`get()` → modify → `save()` sequence is unsafe across workers.

Brown Octopus does not require Redis. A host may inject its own implementation:

```python
from redis import Redis

from brown_octopus import Octopus

redis_client = Redis.from_url("redis://localhost:6379/0")
session_store = MyAtomicRedisSessionStore(redis_client)
octopus = Octopus(
    index_path="data/indexes/default",
    session_store=session_store,
)
await octopus.initialize()
```

`MyAtomicRedisSessionStore` is host code and must persist only serializable
session state.

## 9. Index stores

Index storage and session storage are separate. An index store owns snapshots
and embeddings; a session store owns active conversation state.

### Local files

```python
from brown_octopus import LocalIndexStore, OctopusIndex

index = OctopusIndex.from_file(
    "data/capabilities.json",
    index_store=LocalIndexStore("data/indexes/default"),
)
await index.create()
```

Updates prepare a complete snapshot outside the active snapshot and atomically
swap the current pointer. A failed update leaves the previous snapshot usable.

### PostgreSQL

```bash
pip install "brown-octopus[postgres]"
```

```python
import os

from brown_octopus import OctopusIndex, PostgresIndexStore

store = PostgresIndexStore(os.environ["DATABASE_URL"], index_name="capabilities")
index = OctopusIndex.from_mcp_registry(
    os.environ["REGISTRY_URL"],
    index_store=store,
)
await index.create()
octopus = index.runtime()
await octopus.initialize()
```

If the server exposes the `vector` extension, native cosine search is used.
Plain PostgreSQL remains supported and falls back to local retrieval.

### MongoDB

```bash
pip install "brown-octopus[mongo]"
```

```python
import os

from brown_octopus import MongoIndexStore, OctopusIndex

store = MongoIndexStore(
    os.environ["MONGODB_URI"],
    database="brown_octopus",
    collection="index_snapshots",
    index_name="capabilities",
)
index = OctopusIndex.from_mcp_registry(
    os.environ["REGISTRY_URL"],
    index_store=store,
)
await index.create()
```

Without `vector_search_index`, MongoDB provides snapshot storage and local
retrieval fallback. Atlas Vector Search can be enabled with a compatible Atlas
index name:

```python
store = MongoIndexStore(
    os.environ["MONGODB_URI"],
    vector_search_index="brown-octopus-capabilities",
)
```

The Atlas index must use the `embedding` path and the selected provider's
dimension.

## 10. Embedding providers

The default local provider uses Qwen:

```python
from brown_octopus import Octopus

octopus = Octopus(index_path="data/indexes/default")
await octopus.initialize()
```

The provider uses the best device visible to the installed PyTorch build:
CUDA, Apple MPS, Intel XPU, or CPU. CPU-only PyTorch cannot use a GPU.

Indexing uses a conservative default batch size of 8 and truncates model input
to 512 tokens. This keeps one unusually long capability description from
consuming the whole GPU. `add()` embeds only new or changed capabilities, and
`update()` reuses unchanged embedding rows when the existing snapshot was
built with the same provider. Use `create(replace=True)` when a deliberate
full rebuild is required.

For a small GPU, reduce the batch size or force CPU mode:

```text
OCTOPUS_EMBEDDING_DEVICE=cuda:1
OCTOPUS_EMBEDDING_BATCH_SIZE=2
OCTOPUS_EMBEDDING_MAX_SEQ_LENGTH=384
OCTOPUS_EMBEDDING_FALLBACK_CPU=1
```

`OCTOPUS_EMBEDDING_DEVICE` may be `cpu`, `cuda`, `cuda:N`, `mps`, or `xpu`.
If several GPUs are visible, prefer `CUDA_VISIBLE_DEVICES=N` to isolate one
physical GPU; inside the container that GPU is then addressed as `cuda:0`.
With fallback enabled, a CUDA out-of-memory failure moves the local embedding
model to CPU and retries with batch size 1. This preserves the previous
snapshot if indexing still fails, but can be slower than selecting a smaller
batch size up front.

Use another local model:

```python
from brown_octopus import LocalEmbeddingProvider, OctopusIndex

provider = LocalEmbeddingProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
index = OctopusIndex.from_file(
    "data/capabilities.json",
    index_path="data/indexes/minilm",
    embedding_provider=provider,
)
await index.create()
```

Or a local directory:

```python
provider = LocalEmbeddingProvider(model_path="/models/my-sentence-transformer")
```

Use an HTTP provider:

```python
from brown_octopus import HttpEmbeddingProvider, OctopusIndex

provider = HttpEmbeddingProvider(
    url="http://localhost:11434/api/embed",
    model="qwen3-embedding:0.6b",
)
index = OctopusIndex.from_file(
    "data/capabilities.json",
    index_path="data/indexes/http",
    embedding_provider=provider,
)
await index.create()
```

The HTTP provider accepts Ollama-style `embeddings` or OpenAI-style `data`
responses. Use the same provider/model when loading the index; dimension
mismatches are rejected.

## 11. CLI and configuration

```bash
brown-octopus --help
brown-octopus --version
brown-octopus version
brown-octopus setup-models --help
brown-octopus doctor
brown-octopus inspect
brown-octopus inspect --json
```

Help/version do not load ML models. `doctor` checks package, models, and the
configured index. `inspect` reports configuration and index metadata.

```text
BROWN_OCTOPUS_MODEL_DIR       managed model directory
OCTOPUS_INDEX_PATH             default runtime index path
OCTOPUS_CATALOG_PATH          default MCP catalog path
OCTOPUS_CAPABILITY_TTL        active capability TTL; default 8
OCTOPUS_ACTIVE_CAP             active capability cap; default 30
OCTOPUS_EMBEDDING_DEVICE       embedding device; auto-detected by default
OCTOPUS_EMBEDDING_BATCH_SIZE   local embedding batch size; default 8
OCTOPUS_EMBEDDING_MAX_SEQ_LENGTH local input limit; default 512 tokens
OCTOPUS_EMBEDDING_FALLBACK_CPU retry CUDA OOM on CPU; default 1
OCTOPUS_LOG_LEVEL              Python logging level
```

### Runtime context limits

The default context limits remain:

```text
TTL:        8 turns
Active cap: 30 capabilities per session
```

An application can override them directly on the runtime or index facade:

```python
from brown_octopus import Octopus, OctopusIndex

index = OctopusIndex(
    index_path="data/indexes/default",
)
await index.create()
octopus = index.runtime(capability_ttl=3, active_cap=20)
await octopus.initialize()
```

The equivalent environment configuration is:

```text
OCTOPUS_CAPABILITY_TTL=3
OCTOPUS_ACTIVE_CAP=20
```

`capability_ttl` controls how long a capability remains active after its last
retrieval. `active_cap` limits the number of capabilities retained in a
session. These settings change active-session context management only; they do
not change embedding, ranking, Min-4, or Bounded Max Gap behavior. Existing
applications that do not configure either value continue to use TTL 8 and
active cap 30.

## 12. Errors and operational behavior

```text
ModelInitializationError   required model is unavailable
IndexLoadError             snapshot cannot be read
IncompatibleIndexError     snapshot/provider is incompatible
CapabilityUpdateError      discovery or snapshot preparation failed
```

A missing index path is a valid empty staged-provisioning state. A malformed
existing directory is an error. Runtime diagnostics use the `brown_octopus`
logger and do not log tokens, credentials, or complete user prompts by
default.

Updates are prepared outside the live snapshot and published atomically.
Retrievals already using the previous snapshot continue against that snapshot
while the new one is prepared.

## 13. Concurrency and retrieval behavior

```text
one shared Octopus engine
    shared models, index, retriever, selector
    many isolated session IDs
```

The default store synchronizes sessions independently. Custom stores must
provide equivalent atomicity across processes. Index publication uses a
snapshot swap so retrieval never observes a partially written local index.

The default V3 pipeline remains:

```text
user query -> deterministic spaCy intent analysis
            -> retrieval_text capability representation
            -> configured embedding retrieval
            -> Min-4 + Bounded Max Gap selection
            -> merge and deduplicate
            -> active context
            -> TTL 8 and active cap 30
```

Restrict a turn before ranking with `allowed_mcp_urls`:

```python
result = octopus.retrieve_result(
    "Search the web",
    session_id="conversation-123",
    allowed_mcp_urls=["https://example.com/web/mcp"],
)
```

The strategy is recall-oriented within bounded limits. Extra plausible
capabilities are preferred over aggressive pruning that could omit a required
capability. Perfect recall is not guaranteed.

## 14. Security boundary

Do not store API keys, JWTs, OAuth tokens, passwords, or conversation history
in index metadata or session state. Capability metadata may identify an
execution endpoint, but authentication belongs to the host execution client.

## 15. Release validation

```bash
uv sync
uv run brown-octopus setup-models
uv run pytest -q -m "not external" tests
uv build
```

Validate a wheel in a clean environment rather than importing the source tree:

```bash
python -m venv clean-env
clean-env\Scripts\activate
pip install dist/brown_octopus-<version>-py3-none-any.whl
brown-octopus --help
brown-octopus --version
python -c "from brown_octopus import Octopus, OctopusIndex; print('import OK')"
```

On Unix, activate with `source clean-env/bin/activate`. Research and
evaluation history remains in `evals/` and `docs/research/`; it is not a
runtime dependency or an installable package resource.
