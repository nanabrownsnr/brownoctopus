# Brown Octopus

Brown Octopus is a Python library for dynamic capability-context management
for tool-using AI agents. It determines which capability definitions should be
available to an agent on each interaction. It does not execute tools, manage
credentials, or replace the host agent framework.

## Install

```bash
pip install brown-octopus
brown-octopus setup-models
```

`setup-models` explicitly prepares `en_core_web_trf` and
`Qwen/Qwen3-Embedding-0.6B` in Brown Octopus-managed persistent storage.
Normal initialization does not download models.

## Quick start

```python
from brown_octopus import Octopus


async def main():
    octopus = Octopus()
    await octopus.initialize()

    result = octopus.retrieve_result(
        "Reply to Tom's email",
        session_id="conversation-123",
    )

    # Final active capability context for the host agent.
    for capability in result.tools:
        print(capability["name"])
```

Use `await octopus.update()` when the host deliberately wants to discover
capabilities and build an updated index. `initialize()` loads existing local
models and index state; it does not discover or download implicitly.

### Build or update an index

The repository includes a runnable example for building the persisted index
from an MCP catalog:

```bash
brown-octopus setup-models
python examples/setup_index.py
```

By default it reads `data/mcps.json` and writes `data/indexes/default`. Use a
different catalog or output directory when needed:

```bash
python examples/setup_index.py \
  --catalog path/to/my-mcps.json \
  --index path/to/my-index
```

The host can perform the same operation in application code:

```python
from brown_octopus import Octopus

octopus = Octopus(
    catalog_path="path/to/my-mcps.json",
    index_path="path/to/my-index",
)
report = await octopus.update()
print(report.tool_count)
```

An update is not a blind append and it is not an unconditional reset. Brown
Octopus compares capabilities by stable `capability_id`:

```text
new capability       -> added
same ID, new metadata -> changed/replaced
same ID, unchanged    -> retained
missing from an authoritative source snapshot -> removed
temporarily failed source -> previous capabilities preserved
```

So if you add a new MCP entry to the catalog and run the update, its
capabilities are added to the existing universe. If you remove an MCP from an
authoritative catalog, its capabilities are removed. A source failure is not
treated as deletion.

For a standalone capabilities JSON file, implement a `CapabilitySource` that
reads that file and returns a `CapabilityDiscoveryResult`; `update()` uses the
source snapshot and does not automatically scan arbitrary files placed beside
the index.

## Frozen V3 pipeline

```text
user request
    -> deterministic spaCy operational-intent analysis
    -> capability-oriented retrieval text
    -> Qwen dense retrieval
    -> Min-4 + Bounded Max Gap selection
    -> merge and deduplicate
    -> session capability state
    -> final active capability context
    -> host agent
```

```text
Embedding:        Qwen/Qwen3-Embedding-0.6B
Analyzer:         en_core_web_trf
Representation:   capability name + description
MIN_TOOLS:        4 per intent
MAX_TOOLS:        16 per intent
MIN_GAP_PERCENT:  2.0
TTL:              8 turns
Active cap:       30 capabilities
```

Brown Octopus is intentionally recall-oriented for its first production
version. It prefers a bounded set of plausible capabilities over aggressively
pruning and risking a missing required capability. It does not guarantee
perfect recall.

## Results and sessions

```python
result = octopus.retrieve_result(query, session_id="conversation-123")
```

```text
result.retrieved_tools  capabilities selected on this turn
result.tools            final active context after TTL/session management
result.tool_ids         IDs in result.tools
result.session_id       session identifier
result.turn             session turn number
```

Expose `result.tools` to the agent. Brown Octopus returns definitions and
metadata only; the host binds and executes tools.

One engine can serve many isolated conversations. Models, indexes, and
retrievers are shared; turn counters, TTL state, and active capabilities are
session-specific.

```python
octopus.reset_session("conversation-123")
octopus.delete_session("conversation-123")
snapshot = octopus.get_session("conversation-123")
```

The default `InMemorySessionStore` keeps serializable capability state in
process. Applications needing persistence can inject a custom store:

```python
octopus = Octopus(session_store=my_store)
```

Custom stores must make session mutation atomic across workers/processes.
Brown Octopus stores capability state, not conversation history.

## Capability sources and updates

The default source reads the local MCP catalog, but the source abstraction is
provider-neutral. Use a custom source without changing the retrieval engine:

```python
octopus = Octopus(capability_source=my_source)
report = await octopus.update()
```

Capability identity fields are:

```text
capability_id  stable globally unique capability identity
source_id      discovery provenance
mcp_url        execution endpoint when applicable
```

Failed or non-authoritative discovery does not silently delete capabilities
from unavailable sources. New index snapshots are validated and published
atomically.

## Model storage and diagnostics

Override the managed model directory for containers or shared volumes with
`BROWN_OCTOPUS_MODEL_DIR`. Model setup remains explicit:

```bash
brown-octopus setup-models
brown-octopus doctor
brown-octopus inspect
brown-octopus inspect --json
```

`doctor` checks package, model, and index readiness. `inspect` reports the
active index and frozen configuration without exposing credentials.

## Architecture boundary

Brown Octopus owns intent analysis, capability retrieval/selection, index
construction, active capability context, session capability state, and
explicit capability updates.

The host owns the LLM, conversation history, tool binding/execution,
credentials, authentication, authorization, user identity, refresh timing,
and session lifecycle policy.

Brown Octopus is harness-agnostic. A LangGraph integration example is available
at [`examples/langgraph_app`](examples/langgraph_app/README.md); LangGraph is
not a runtime dependency.

## Development

```bash
uv sync
brown-octopus setup-models
uv run pytest -m "not external"
uv build
```

Historical evaluation infrastructure remains under `evals/`, benchmark data
under `data/evals/`, and research outputs under `results/`. These are separate
from the installable `brown_octopus` package.

See [`LICENSE`](LICENSE) for licensing information.
