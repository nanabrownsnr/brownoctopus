# Brown Octopus

Brown Octopus is a Python library that manages the capability context exposed
to tool-using AI agents. It analyzes requests, retrieves relevant capability
definitions, and maintains active capabilities across conversation turns.

Brown Octopus does not execute tools, manage credentials, or manage the LLM's
conversation history. Those responsibilities stay with the host application.

## What the package provides

| Part | Responsibility |
| --- | --- |
| `OctopusIndex` | Create and update a persisted capability index |
| `Octopus` | Load the index and retrieve the active capability context |
| `CapabilitySource` | Supply capabilities from MCP, files, APIs, or custom systems |
| `SessionStore` | Persist per-conversation capability state |

The normal application path is:

```text
source -> OctopusIndex.create() -> persisted index
             update() for later synchronization
                                              |
                                              v
                         Octopus.initialize() -> retrieve_result()
```

## Installation

Brown Octopus requires Python 3.12 or newer.

Using `uv`:

```bash
uv add brown-octopus
```

Using `pip`:

```bash
pip install brown-octopus
```

Prepare the runtime models explicitly:

```bash
uv run brown-octopus setup-models
uv run brown-octopus doctor
```

If the environment is already activated, the `uv run` prefix is optional.
Model setup does not happen automatically during application startup.

### Choosing an embedding provider

The default embedding provider is a local SentenceTransformers model using
`Qwen/Qwen3-Embedding-0.6B`. You can replace it with another compatible local
model or an HTTP embedding service. The analyzer remains the deterministic
spaCy analyzer; embedding-provider configuration affects capability indexing
and retrieval only.

Use another local model by name:

```python
from brown_octopus import LocalEmbeddingProvider, OctopusIndex

provider = LocalEmbeddingProvider(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

index = OctopusIndex.from_sources(
    [source],
    index_path="data/indexes/custom-model",
    embedding_provider=provider,
)
await index.create()
```

Or load a model from a local directory:

```python
provider = LocalEmbeddingProvider(
    model_path="C:/models/my-embedding-model",
)
```

For an HTTP service such as Ollama:

```python
from brown_octopus import HttpEmbeddingProvider, OctopusIndex

provider = HttpEmbeddingProvider(
    url="http://localhost:11434/api/embed",
    model="qwen3-embedding:0.6b",
)

index = OctopusIndex.from_sources(
    [source],
    index_path="data/indexes/ollama",
    embedding_provider=provider,
)
await index.create()
```

Pass the same provider when loading the index at runtime:

```python
from brown_octopus import Octopus

octopus = Octopus(
    index_path="data/indexes/ollama",
    embedding_provider=provider,
)
await octopus.initialize()
```

The provider determines the vector dimension. Do not manually resize vectors.
If you change the model or provider, create a new index or rebuild the index
with that provider. Brown Octopus records provider metadata and rejects an
incompatible provider/index combination.

## Quick start

Brown Octopus has two distinct phases:

```text
SETUP
CapabilitySource -> OctopusIndex.create() -> persisted capability index
                    update() for later synchronization

RUNTIME
initialize() -> retrieve capability context -> host agent
```

### Build the capability index

The built-in local source reads MCP server URLs from a catalog with this shape:

```json
{
  "items": [
    {"url": "https://example.com/outlook/mcp"},
    {"url": "https://example.com/word/mcp"}
  ]
}
```

Create `setup_octopus.py`:

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
    print(f"Indexed {report.tool_count} capabilities")
    print(f"Added: {len(report.added)}")
    print(f"Changed: {len(report.changed)}")
    print(f"Removed: {len(report.removed)}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it after `setup-models`:

```bash
uv run python setup_octopus.py
```

The repository also includes the equivalent runnable example:

```bash
python examples/setup_index.py
```

For a registry that returns MCP server records, use the registry-specific
facade. It follows page/limit or cursor pagination, discovers tools from each
server URL, and preserves the execution metadata the host needs:

```python
from brown_octopus import OctopusIndex

index = OctopusIndex.from_mcp_registry(
    "https://registry.example.com/mcp-servers",
    headers={"Authorization": "Bearer <host-token>"},
    index_path="data/indexes/default",
)

report = await index.create()
```

Use `from_api()` instead when the API already returns one normalized capability
record per item. See [`docs/production.md`](docs/production.md) for registry,
API, source, session-store, and deployment examples.

### Use the index at runtime

Create `app.py`:

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
    print("Capabilities exposed to the agent:")
    for capability in result.tools:
        print("-", capability["name"])


if __name__ == "__main__":
    asyncio.run(main())
```

Run it with:

```bash
uv run python app.py
```

`result.tools` is the final active capability context to expose to the agent.
Brown Octopus returns definitions and metadata; the host binds and executes
the tools.

### Restrict retrieval to selected MCPs

Pass an optional list of allowed MCP URLs when a host wants to limit retrieval
to capabilities owned or enabled for that request:

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

The filter is applied before ranking. It is request-scoped and does not modify
the shared index. `None` means no filtering; an empty list means no MCP
capabilities are allowed. The same scope is applied to retained session
capabilities, so a tool from a disallowed MCP is not exposed through
`result.tools`.

MCP URL matching ignores a trailing slash. Capabilities without an `mcp_url`
are excluded while an allowlist is active.

The result still contains complete capability definitions:

```python
print(result.retrieved_tools)
```

```text
[
  {
    'rank': 1,
    'score': 0.91,
    'capability_id': 'outlook-123:send_email',
    'source_id': 'outlook-123',
    'name': 'outlook_send_email',
    'tool_name': 'send_email',
    'mcp_url': 'https://example.com/outlook/mcp',
    'description': 'Send an email.',
    'input_schema': {...}
  }
]
```

`result.retrieved_tools` contains the current-turn selection. `result.tools`
contains the final active context for the session and retains the same tool
metadata.

The HTTP adapter accepts the same option:

```json
{
  "session_id": "conversation-123",
  "query": "Reply to Tom's email",
  "allowed_mcp_urls": [
    "https://example.com/outlook/mcp"
  ]
}
```

## Updating capabilities

When the configured capability universe changes, update the source and run the
index synchronization operation:

```bash
uv run python examples/setup_index.py --update
```

The initial run uses `await index.create()`. Later runs use
`await index.update()` against the same configured sources.

`update()` currently rediscovers all configured sources and rebuilds embeddings
for the complete resulting universe before atomically publishing a new index.
It is not yet an incremental embedding or append-only operation.

Capabilities are compared by stable `capability_id`:

```text
new capability                         -> added
existing ID with changed metadata      -> replaced
unchanged capability                   -> retained
missing from authoritative snapshot    -> removed
temporarily failed source              -> preserved
```

Therefore, to add an MCP while preserving existing capabilities, add it to the
existing `data/mcps.json` catalog and run the update again. Passing a separate
file containing only the new MCP makes that file the configured source
snapshot; it does not automatically append to the previous catalog.

An arbitrary capability JSON file is not scanned automatically. Implement a
custom `CapabilitySource` if your capabilities come from a file, API,
database, registry, or marketplace.

## Capability sources

`LocalMcpCatalogSource` is the built-in/default source. It reads MCP server
locations from the configured catalog and discovers their capabilities.

The built-in setup facades cover three common shapes:

| Constructor | Input |
| --- | --- |
| `OctopusIndex.from_mcp_server()` | One HTTP MCP server URL |
| `OctopusIndex.from_mcp_registry()` | An API returning MCP server records |
| `OctopusIndex.from_file()` / `from_api()` | Already-normalized capability records |

For registry-backed MCPs, `mcp_url` and `tool_name` remain in the returned
capability definition. Brown Octopus uses them as execution metadata; the host
still owns authentication and tool execution.

Brown Octopus is not limited to MCP. Applications can provide a custom source
for an internal API, database, registry, marketplace, or another capability
system.

The source discovers capabilities. Brown Octopus owns indexing and active
capability-context management. The host application owns source credentials,
permissions, update timing, and tool execution.

For complete constructor signatures, method behavior, source contracts, session
store requirements, result fields, and operational examples, see the
[production API guide](docs/production.md#public-api-reference).

### Built-in local MCP source

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
    print(f"Indexed {report.tool_count} capabilities")


if __name__ == "__main__":
    asyncio.run(main())
```

### Custom source

A custom source implements the public `CapabilitySource` contract and returns
a `CapabilityDiscoveryResult` from `discover()`:

```python
from brown_octopus import CapabilityDiscoveryResult


class InternalRegistrySource:
    async def discover(self) -> CapabilityDiscoveryResult:
        return CapabilityDiscoveryResult(
            tools=[
                {
                    "capability_id": "internal:crm:search_customers",
                    "source_id": "internal-crm",
                    "name": "search_customers",
                    "description": "Search the customer database.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            ],
            successful_sources=["internal-crm"],
            failed_sources={},
            authoritative=True,
        )
```

Use it when constructing `OctopusIndex`:

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

Each capability should provide:

```text
capability_id  stable globally unique capability identity
source_id      stable discovery provenance
name           readable capability name
description    text used for retrieval
input_schema   schema supplied to the host agent
```

MCP-specific fields such as `mcp_url`, `mcp_name`, and `tool_name` may be
included when applicable, but they are not required for non-MCP sources.

When `update()` discovers a changed capability universe, it adds new
capabilities, replaces changed metadata, and removes capabilities missing from
an authoritative snapshot. A temporarily failed source should be reported in
`failed_sources`; its previously indexed capabilities are preserved.

## Method results

The main `Octopus` methods return different kinds of results:

```python
report = await index.create()
# initial CapabilityUpdateReport

report = await index.update()
# later synchronization: added, changed, removed, failed_sources, ...

tools = await octopus.initialize()
# list[dict]: the capabilities loaded from the existing index

active_tools = octopus.retrieve("Reply to Tom's email")
# list[dict]: final active capability definitions

result = octopus.retrieve_result("Reply to Tom's email")
# RetrievalResult:
# result.retrieved_tools -> current-turn selection
# result.tools            -> final active context

details = octopus.process("Reply to Tom's email")
# dict containing retrieved_tools, active_tools, tool_ids, session, and timing

session = octopus.get_session("conversation-123")
# dict with session_id, turn, active_state, and active_tool_ids
```

`reset_session()` and `delete_session()` change session state and return
`None`. `update()` and `initialize()` are asynchronous because they perform
index/model I/O; retrieval and session inspection are synchronous.

## Sessions

Use a stable conversation identifier for `session_id`:

```python
result = octopus.retrieve_result(
    "Send Sarah an email",
    session_id="conversation-123",
)
```

One engine can serve many isolated conversations. Models, indexes, and
retrievers are shared; turn counters, TTL state, and active capabilities are
session-specific.

```python
snapshot = octopus.get_session("conversation-123")
octopus.reset_session("conversation-123")
octopus.delete_session("conversation-123")
```

The default `InMemorySessionStore` stores serializable capability state in the
current process. Applications needing persistence can inject a custom store:

```python
octopus = Octopus(session_store=my_store)
```

Custom stores must make session mutation atomic across workers or processes.
Brown Octopus stores capability state, not conversation history.

## Result fields

```text
result.retrieved_tools  capabilities selected for the current request
result.tools            final active context after session/TTL management
result.tool_ids         IDs in result.tools
result.session_id       session identifier
result.turn             current session turn
```

Use `result.tools` for the agent. Use `result.retrieved_tools` when you need
to inspect only the current-turn selection.

## CLI and model storage

```bash
uv run brown-octopus setup-models
uv run brown-octopus doctor
uv run brown-octopus inspect
uv run brown-octopus inspect --json
uv run brown-octopus --version
```

Models are stored outside the consumer virtual environment so operations such
as `uv sync` do not remove them. Set `BROWN_OCTOPUS_MODEL_DIR` to use a custom
location for containers, CI, or shared model volumes.

## Current default pipeline

```text
request
  -> deterministic spaCy operational-intent analysis
  -> capability-oriented retrieval text
  -> Qwen/Qwen3-Embedding-0.6B dense retrieval
  -> Min-4 + Bounded Max Gap selection
  -> merge/deduplicate
  -> active capability context
  -> host agent
```

Current limits are 4 minimum tools per intent, 16 maximum tools per intent,
2% minimum gap, an 8-turn TTL, and a 30-capability active context cap.

## Agent integrations

The integration boundary is:

```text
user message -> Brown Octopus -> result.tools -> LLM.bind_tools(...) -> agent
```

See [`examples/langgraph_app`](examples/langgraph_app/README.md) for a
LangGraph example. LangGraph is not a Brown Octopus runtime dependency.

## Architecture boundaries

Brown Octopus owns capability discovery abstraction, indexing, intent analysis,
retrieval, selection, active capability context, and session capability state.

The host application owns the LLM, conversation history, tool binding and
execution, credentials, authentication, authorization, user identity, update
timing, and session lifecycle policy.

## Development and research

The detailed Python API, source, session-store, and deployment documentation is
in [`docs/production.md`](docs/production.md).

The repository separates the installable library from integration examples and
research history:

```text
src/brown_octopus/  production Python package
tests/              package correctness tests
examples/           host/agent integration examples
docs/               user and deployment documentation
docs/research/      research notes and historical technical reports
evals/              evaluation infrastructure
data/evals/         benchmark datasets
results/            raw and processed evaluation outputs
```

```bash
uv sync
uv run brown-octopus setup-models
uv run pytest -m "not external"
uv build
```

The registry integration test plan is in
[`docs/registry_integration_test_plan.md`](docs/registry_integration_test_plan.md).
It covers live registry discovery, cursor pagination, partial MCP failures,
token refresh, Redis session persistence, and safe read-only execution.

Research and reproducibility materials remain in the repository:

```text
evals/              evaluation infrastructure
data/evals/         benchmark datasets
results/            reports and experiment outputs
docs/research/      research notes and historical reports
```

They are separate from the installable `brown_octopus` runtime package.

## Current status

The default V3 retrieval path is frozen. Brown Octopus has been validated with
MCP discovery, registry-backed indexing, LangGraph capability retrieval,
read-only MCP execution, and synchronous Redis-backed session persistence.
Write-capable MCP execution remains host- and environment-specific and is not
run against production services by default.

## Documentation

- [Production and deployment guide](docs/production.md)
- [Registry integration test plan](docs/registry_integration_test_plan.md)
- [LangGraph example](examples/langgraph_app/README.md)
- [Research and evaluation history](docs/research/)

## License

See [`LICENSE`](LICENSE).
