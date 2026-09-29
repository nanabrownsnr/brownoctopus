# Brown Octopus Python API

This document describes the public Python API in Brown Octopus 0.4.x. Brown
Octopus manages capability context; it does not execute capabilities or manage
LLM conversation history.

## Basic lifecycle

There are two separate operations:

```text
setup-models       prepare local runtime models (CLI operation)
Octopus.update()   discover capabilities and build/update the index
Octopus.initialize() load the existing index for runtime retrieval
```

Runtime startup does not discover capabilities or download models.

### Build or update an index

The default source reads the local MCP catalog at `data/mcps.json`:

```python
import asyncio

from brown_octopus import Octopus


async def build_index() -> None:
    octopus = Octopus()
    report = await octopus.update()
    print(f"Indexed {report.tool_count} capabilities")


if __name__ == "__main__":
    asyncio.run(build_index())
```

`update()` synchronizes the configured capability source with the persisted
index. It handles added, changed, unchanged, and authoritatively removed
capabilities. A source that reports a temporary failure does not cause its
previous capabilities to be removed.

### Runtime retrieval

```python
import asyncio

from brown_octopus import Octopus


async def main() -> None:
    octopus = Octopus()
    await octopus.initialize()

    result = octopus.retrieve_result(
        "Reply to Tom's email",
        session_id="conversation-123",
    )

    # Current-turn selection only.
    print([tool["name"] for tool in result.retrieved_tools])

    # Final active capability context for the host agent.
    print([tool["name"] for tool in result.tools])


if __name__ == "__main__":
    asyncio.run(main())
```

`result.tools` is the normal value to pass to the host agent. Brown Octopus
returns capability definitions and metadata; the host chooses how to convert
them into the tool format required by LangGraph, OpenAI, another framework, or
a custom agent.

## `Octopus`

### Constructor

```python
Octopus(
    index_path="data/indexes/default",
    catalog_path="data/mcps.json",
    config=None,
    capability_source=None,
    session_store=None,
)
```

Parameters:

`index_path`
: Directory containing the persisted capability index. The default is
  `data/indexes/default`, or the configured value from `OctopusConfig`.

`catalog_path`
: Local MCP catalog used by the default `LocalMcpCatalogSource`.

`config`
: Optional `OctopusConfig` for environment and runtime settings.

`capability_source`
: Optional custom source. If omitted, Brown Octopus uses
  `LocalMcpCatalogSource(catalog_path)`.

`session_store`
: Optional session-state implementation. If omitted, Brown Octopus uses an
  in-process `InMemorySessionStore`.

### Methods

`await initialize()`
: Load the managed runtime models and existing persisted index. It does not
  create an absent index, discover capabilities, or download models.

`await update()`
: Discover the configured capability source, build a new valid snapshot, and
  atomically install it. Existing retrieval continues using the previous valid
  snapshot until publication succeeds.

`retrieve(query, session_id="default", allowed_mcp_urls=None)`
: Retrieve the final active capability definitions as a list.

`retrieve_result(query, session_id="default", allowed_mcp_urls=None)`
: Retrieve a typed `RetrievalResult` containing both current-turn and final
  active capabilities.

`process(query, session_id="default", allowed_mcp_urls=None)`
: Return the underlying serializable pipeline dictionary. This is useful for
  service adapters and diagnostics; most applications should use
  `retrieve_result()`.

`reset_session(session_id)`
: Clear one session's turn and active capability state.

`delete_session(session_id)`
: Delete one session from the configured session store.

`get_session(session_id="default")`
: Return a read-only serializable snapshot of session state, or `None` if the
  session does not exist.

The optional `allowed_mcp_urls` filter restricts retrieval to indexed MCP tools
whose execution URL is in the supplied collection. If it is omitted, all
indexed capabilities are eligible. An empty collection allows no MCP URLs.

## `RetrievalResult`

```python
result.tool_ids
result.tools
result.retrieved_tools
result.active_tools
result.session_id
result.turn
result.metadata
```

`retrieved_tools`
: Capabilities selected for the current request before active-context memory.

`tools`
: Final active capability context after TTL, ordering, and active-cap rules.
Current-turn capabilities appear before retained capabilities.

`active_tools`
: Alias for `tools`.

`tool_ids`
: Stable `capability_id` values for the final active context.

Each capability dictionary preserves the metadata supplied by its source. A
typical MCP-backed definition contains:

```python
{
    "capability_id": "outlook_prod_send_email",
    "source_id": "outlook-prod",
    "name": "outlook_send_email",
    "description": "Send an email.",
    "input_schema": {...},
    "mcp_name": "Outlook",
    "mcp_url": "https://example.com/outlook/mcp",
    "tool_name": "send_email",
}
```

The host decides which fields to send to the LLM. Usually the model-facing
tool definition uses `name`, `description`, and `input_schema`; execution
routing may additionally use `capability_id`, `source_id`, `mcp_url`, and
`tool_name`.

## Capability sources

The built-in source reads MCP URLs from a JSON catalog and discovers the tools
from each server:

```python
from brown_octopus import LocalMcpCatalogSource, Octopus

source = LocalMcpCatalogSource("data/mcps.json")
octopus = Octopus(capability_source=source)
```

Custom sources implement the public protocol:

```python
from brown_octopus import CapabilityDiscoveryResult


class InternalRegistrySource:
    async def discover(self) -> CapabilityDiscoveryResult:
        records = await load_records_from_my_registry()
        return CapabilityDiscoveryResult(
            tools=records,
            successful_sources=["internal-registry"],
            failed_sources={},
            authoritative=True,
        )
```

Capability records should provide a stable `capability_id`, stable discovery
`source_id`, readable `name`, semantic `description`, and `input_schema`.
MCP-specific fields such as `mcp_url` are optional for non-MCP sources.

## Session state

Session IDs are opaque conversation identifiers supplied by the host. They are
not user identities and Brown Octopus does not perform authentication.

One engine shares models, the capability universe, and the embedding index
across sessions. Each session has independent turn, TTL, and active-capability
state.

The default store is in memory:

```python
octopus = Octopus()
```

An application can provide a custom `SessionStore`:

```python
octopus = Octopus(session_store=my_store)
```

The store persists only `SessionState`:

```python
{
    "session_id": "conversation-123",
    "turn": 4,
    "active_state": {
        "outlook_prod_send_email": 4,
    },
}
```

It must not persist models, locks, capability definitions, credentials, or
conversation history. A multi-process implementation must make its `mutate()`
operation atomic using an appropriate transaction or distributed lock.

## Internal compatibility paths

The package still contains lower-level functions used by the evaluation suite,
historical scripts, and compatibility integrations, including
`retrieve_tools()`, `process_turn()`, `rank_tools()`, and the bootstrap module.
The supported application-facing path is `Octopus.initialize()` followed by
`retrieve_result()`.

The default internal pipeline remains:

```text
deterministic intent analysis
    -> Qwen dense candidate retrieval
    -> Min-4 + Bounded Max Gap selection
    -> merge/deduplicate by capability_id
    -> active capability context
```

