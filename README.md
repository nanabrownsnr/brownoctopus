# Brown Octopus

Brown Octopus manages the capability context exposed to an AI agent.

Given a user request, it identifies relevant capabilities, retrieves a bounded
set of candidates, and maintains the active capability context for a
conversation. It returns capability definitions to the host application. It
does not call tools, own an LLM, manage credentials, or store conversation
history.

```text
Distribution: brown-octopus
Import:       brown_octopus
Runtime:      Octopus
Index facade: OctopusIndex
```

## Install

Brown Octopus supports Python 3.12 and newer.

```bash
pip install brown-octopus
```

Or with uv:

```bash
uv add brown-octopus
```

Prepare the default runtime models explicitly:

```bash
brown-octopus setup-models
```

This prepares `en_core_web_trf` and `Qwen/Qwen3-Embedding-0.6B`.
`Octopus.initialize()` never downloads models or discovers capabilities
implicitly.

Check the installation:

```bash
brown-octopus doctor
brown-octopus inspect
```

## Quick start

The normal lifecycle is:

```text
CapabilitySource -> OctopusIndex.create() -> persisted index
                                              |
                                              v
                    Octopus.initialize() -> retrieve_result()
```

### 1. Configure a local MCP catalog

Create `data/mcps.json`:

```json
{
  "items": [
    {
      "id": "outlook",
      "name": "Outlook Mail",
      "url": "https://example.com/outlook/mcp"
    }
  ]
}
```

The URL must be a reachable MCP server that the host is authorized to access.

### 2. Build the index

Create `setup_index.py`:

```python
import asyncio
from pathlib import Path

from brown_octopus import OctopusIndex
from brown_octopus.sources import LocalMcpCatalogSource


async def main() -> None:
    index = OctopusIndex.from_sources(
        [LocalMcpCatalogSource(Path("data/mcps.json"))],
        index_path="data/indexes/default",
    )
    report = await index.create()
    print(f"Indexed {report.tool_count} capabilities")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it once after the catalog is configured:

```bash
python setup_index.py
```

`create()` creates and publishes a snapshot only when the index does not already
exist. It discovers capabilities, builds embeddings, and validates the
snapshot atomically. To deliberately rebuild an existing index, use
`create(replace=True)`. It does not execute MCP tools.

### 3. Start the runtime

Create `app.py`:

```python
import asyncio

from brown_octopus import Octopus


async def main() -> None:
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

Run it with `python app.py`. Pass `result.tools` to the agent or harness; the
host decides how to bind and execute the definitions.

## Index and runtime

`OctopusIndex` owns index lifecycle:

- `create()` creates the index only if it is missing;
- `create(replace=True)` deliberately rebuilds the index;
- `update()` synchronizes configured sources;
- `add(source)` adds a source and synchronizes;
- `remove_source(source_id)` removes a configured source and synchronizes;
- `reset()` removes the persisted index;
- `runtime()` returns the runtime facade.

`Octopus` owns runtime retrieval:

- `initialize()` loads models and the current snapshot;
- `retrieve()` returns the active capability context;
- `retrieve_result()` returns the typed result contract;
- session methods maintain conversation-specific capability state.

Sources are optional when opening an index:

```python
index = OctopusIndex(index_path="data/indexes/new-index")
octopus = index.runtime()
await octopus.initialize()

result = octopus.retrieve_result("Reply to Tom's email")
assert result.tools == []
assert result.retrieved_tools == []
assert result.tool_ids == []
```

If the path has no snapshot, runtime initialization starts with an empty
capability universe. Configure a source and call `create()` or `update()` when
the host is ready to provision capabilities. Calling `create()` during every
startup is safe: an existing snapshot is left unchanged.

## Updating capabilities

After changing a configured source:

```python
report = await index.update()
print(report.added)
print(report.changed)
print(report.removed)
print(report.failed_sources)
```

`update()` is a synchronization operation, not an append-only operation. It
rediscovers the configured universe and publishes a complete replacement
snapshot:

```text
new capability                      -> added
changed metadata                    -> changed
unchanged capability                -> retained
authoritatively removed capability  -> removed
temporarily failed source           -> previous capabilities preserved
```

The host decides when updates happen. Brown Octopus has no refresh scheduler.

Add a source to an existing index:

```python
from brown_octopus import McpServerSource

await index.add(
    McpServerSource(
        "https://example.com/word/mcp",
        source_id="word",
    )
)
```

`remove_capability()` remains temporarily for older applications but is
deprecated. Prefer changing the authoritative source or removing the source.

## Retrieval results and filtering

```python
result = octopus.retrieve_result(
    "Send Sarah an email",
    session_id="conversation-123",
)
```

The result provides:

```text
result.retrieved_tools  current-turn selection
result.tools            final active context
result.tool_ids         stable IDs for result.tools
result.session_id       conversation identifier
result.turn             current session turn
result.metadata         timing and pipeline metadata
```

The simpler method is also available:

```python
active_tools = octopus.retrieve(
    "Search the web for the latest NVIDIA news",
    session_id="conversation-123",
)
```

Restrict a turn before ranking with `allowed_mcp_urls`:

```python
result = octopus.retrieve_result(
    "Reply to an email",
    session_id="conversation-123",
    allowed_mcp_urls=["https://example.com/outlook/mcp"],
)
```

## Sessions

One engine can serve many conversations:

```python
first = octopus.retrieve_result("Search the web", session_id="conversation-a")
second = octopus.retrieve_result("Reply to Tom", session_id="conversation-b")
```

Each session has its own turn counter and active capability state. A session
ID represents a conversation or workflow identity, not a user identity.

```python
snapshot = octopus.get_session("conversation-a")
octopus.reset_session("conversation-a")
octopus.delete_session("conversation-b")
```

The default store is process-local. A persistent application can inject a
custom atomic `SessionStore`; see `docs/production.md`.

## Capability sources

Built-in adapters include:

| Adapter | Use it for |
| --- | --- |
| `LocalMcpCatalogSource` | local JSON catalog of MCP servers |
| `McpServerSource` | one HTTP MCP server |
| `McpRegistrySource` | API returning MCP server records |
| `JsonCapabilitySource` | normalized capabilities from JSON |
| `ApiCapabilitySource` | normalized capabilities from an HTTP API |
| custom `CapabilitySource` | application-specific discovery |

Convenience constructor:

```python
index = OctopusIndex.from_mcp_server(
    "https://example.com/outlook/mcp",
    source_id="outlook",
    index_path="data/indexes/outlook",
)
```

Detailed source contracts, pagination, authentication, and field mapping are
documented in `docs/production.md`.

## Embedding providers

The default provider is local Qwen. A compatible local model can be supplied:

```python
from brown_octopus import LocalEmbeddingProvider, OctopusIndex

provider = LocalEmbeddingProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
index = OctopusIndex.from_sources(
    [source],
    index_path="data/indexes/minilm",
    embedding_provider=provider,
)
await index.create()
```

An HTTP provider can target a remote or self-hosted service:

```python
from brown_octopus import HttpEmbeddingProvider, OctopusIndex

provider = HttpEmbeddingProvider(
    url="http://localhost:11434/api/embed",
    model="qwen3-embedding:0.6b",
)
index = OctopusIndex.from_sources(
    [source],
    index_path="data/indexes/http-embeddings",
    embedding_provider=provider,
)
await index.create()
```

Use the same provider when loading that index. Provider dimensions are checked
against index metadata.

## CLI

```bash
brown-octopus --help
brown-octopus --version
brown-octopus version
brown-octopus setup-models
brown-octopus doctor
brown-octopus inspect
brown-octopus inspect --json
```

Help and version do not load models. `doctor` checks package, models, and the
configured index. `inspect` reports configuration and index metadata.

## Runtime context limits

The default active-context policy is:

```text
TTL:       8 turns
Active cap: 30 capabilities
```

These defaults preserve the validated behavior, but applications can tune
them when their context budget requires it:

```python
from brown_octopus import OctopusIndex

index = OctopusIndex(index_path="data/indexes/default")
await index.create()
octopus = index.runtime(
    capability_ttl=3,
    active_cap=20,
)
await octopus.initialize()
```

`capability_ttl` controls how many later turns a retrieved capability can
remain active without being retrieved again. `active_cap` limits the total
number of active capabilities in a session. These settings affect session
context management, not candidate ranking or the V3 selection algorithm.

The same values can be configured through environment variables:

```text
OCTOPUS_CAPABILITY_TTL=3
OCTOPUS_ACTIVE_CAP=20
```

## Architecture boundary

Brown Octopus owns capability discovery adapters, indexing, intent analysis,
retrieval, selection, active capability context, and capability session state.

The host owns the LLM, conversation history, tool binding and execution,
credentials, authentication, authorization, user identity, update timing, and
session lifecycle policy.

Returned MCP metadata such as `mcp_url` and `tool_name` describes execution
context. Brown Octopus does not execute the capability.

## Retrieval behavior

The default V3 pipeline is unchanged:

```text
user request -> deterministic spaCy intent analysis
             -> capability-oriented representation
             -> dense embedding retrieval
             -> Min-4 + Bounded Max Gap
             -> active capability context
             -> TTL 8 and active cap 30
```

The strategy is recall-oriented within bounded limits. It may expose extra
plausible capabilities rather than aggressively pruning a capability that may
be required. It does not guarantee perfect recall.

## Development and research

```bash
uv sync
uv run brown-octopus setup-models
uv run pytest -q -m "not external" tests
uv build
```

Product code, tests, examples, and research/evaluation history are kept
separate. Evaluation material under `evals/` and `docs/research/` is retained
for reproducibility and is not required at runtime.

See [`docs/production.md`](docs/production.md) for the complete integration
reference.

## License

See [`LICENSE`](LICENSE).
