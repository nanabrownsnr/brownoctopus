# Brown Octopus

Brown Octopus is a Python library that manages the capability context exposed
to tool-using AI agents. It analyzes requests, retrieves relevant capability
definitions, and maintains active capabilities across conversation turns.

Brown Octopus does not execute tools, manage credentials, or manage the LLM's
conversation history. Those responsibilities stay with the host application.

## Installation

Brown Octopus requires Python 3.13 or newer.

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

## Quick start

Brown Octopus has two distinct phases:

```text
SETUP
CapabilitySource -> update() -> persisted capability index

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

from brown_octopus import Octopus
from brown_octopus.sources import LocalMcpCatalogSource


async def main():
    source = LocalMcpCatalogSource("data/mcps.json")
    octopus = Octopus(
        capability_source=source,
        index_path="data/indexes/default",
    )

    report = await octopus.update()
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

## Updating capabilities

When the configured capability universe changes, update the source and run the
same setup script again:

```bash
uv run python setup_octopus.py
```

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

Brown Octopus is not limited to MCP. Applications can provide a custom source
for an internal API, database, registry, marketplace, or another capability
system.

The source discovers capabilities. Brown Octopus owns indexing and active
capability-context management. The host application owns source credentials,
permissions, update timing, and tool execution.

### Built-in local MCP source

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

Use it when constructing `Octopus`:

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

```bash
uv sync
uv run brown-octopus setup-models
uv run pytest -m "not external"
uv build
```

Research and reproducibility materials remain in the repository:

```text
evals/       evaluation infrastructure
data/evals/  benchmark datasets
results/     reports and experiment outputs
```

They are separate from the installable `brown_octopus` runtime package.

## License

See [`LICENSE`](LICENSE).
