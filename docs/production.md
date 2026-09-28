# Production integration

Brown Octopus is a capability-context service. It returns tool definitions and
metadata; it never executes a tool and it has no dependency on an agent
harness.

## Runtime pipeline

```text
request
  -> IntentAnalyzer
  -> CandidateRetriever (Qwen dense)
  -> SelectionStrategy (V3 Min-4 + Bounded Max Gap)
  -> merge/dedupe
  -> session-owned ActiveCapabilityContext (TTL and active cap)
  -> RetrievalResult
```

The interfaces live in `brown_octopus.contracts`. The current implementations are:

- `DeterministicIntentAnalyzer` using `en_core_web_trf`;
- `QwenCandidateRetriever` using `Qwen/Qwen3-Embedding-0.6B` and `name + description`;
- `BoundedMaxGapSelectionStrategy` implementing frozen V3;
- `JevSelectionStrategy` is retained as an internal experimental implementation,
  but is not selected by the default `Octopus` pipeline;
- `ActiveCapabilityContext` with TTL 8 and active cap 30.

The default V3 path ranks the indexed universe and applies the frozen Min-4 +
Bounded Max Gap selector per intent. The final exposed context remains dynamic.

## Configuration

Configuration is read from environment variables. Important values are:

```text
OCTOPUS_INDEX_PATH=data/indexes/default
OCTOPUS_CAPABILITY_TTL=8
OCTOPUS_ACTIVE_CAP=30
TYPESAFE_API_KEY=...
```

Jev is not required by the default V3 runtime. The optional Jev dependency and
API key are only needed for internal Jev experiments.

## Service

## CLI and release validation

The installed command is intentionally small and is for setup and diagnostics,
not agent execution:

```bash
brown-octopus --help
brown-octopus --version
brown-octopus version
brown-octopus setup-models
brown-octopus doctor
brown-octopus inspect
brown-octopus inspect --json
```

Model setup is explicit. `initialize()` never downloads models. `doctor`
checks local package/model/index readiness, while `inspect` reports the active
index and frozen V3 configuration without loading models or making network
calls.

For a release validation from a clone:

```bash
python -m pip install --upgrade uv
uv sync --dev
brown-octopus setup-models
uv run pytest
uv build
```

Install the wheel into a separate environment before distributing it. The
source checkout must not be required by the consumer.

Build and run the container after putting the persisted default index and MCP
catalog in the build context:

```bash
docker build -t brown-octopus .
docker run --rm -p 8000:8000 \
  -e TYPESAFE_API_KEY="$TYPESAFE_API_KEY" \
  brown-octopus
```

Endpoints:

- `GET /health` — process health;
- `GET /ready` — model/index readiness;
- `POST /v1/capabilities` with `{"session_id":"...", "query":"..."}`;
- `POST /v1/sessions/{session_id}/reset` — clear conversation state.

Each session has its own turn counter, lock, and active capability context while
the analyzer, Qwen model, index, tool universe, retriever, and V3 selector are
shared by the single `Octopus` engine. Tool execution remains the responsibility
of the consuming harness.

The Python API accepts an optional opaque session ID:

```python
await octopus.initialize()
context = octopus.retrieve_result(
    "send her an email",
    session_id="conversation-123",
)
agent_tools = context.tools
```

`context.retrieved_tools` contains only the current-turn V3 selection;
`context.tools` contains the final active context. Session lifecycle methods are
`reset_session(session_id)`, `delete_session(session_id)`, and
`get_session(session_id)`. Storage is currently in-memory behind
`InMemorySessionStore`.

## Capability sources and updates

The default source is `LocalMcpCatalogSource`, which reads the configured
`data/mcps.json` URL catalog. Hosts can inject any object implementing
`CapabilitySource`:

```python
class CapabilitySource(Protocol):
    async def discover(self) -> CapabilityDiscoveryResult: ...
```

Capability refresh is host-controlled; Brown Octopus does not run a scheduler:

```python
report = await octopus.update()
print(report.added, report.changed, report.removed, report.failed_sources)
```

Updates are prepared outside the live swap, written as a complete versioned
snapshot, then published by atomically replacing a small `current.json`
pointer. Retrievals in progress continue using their existing in-memory
snapshot. If a source fails, its previously known capabilities are retained;
successful sources can add, change, or authoritatively remove tools.

Capability provenance is separate from execution metadata:

- `source_id` is the stable identifier for the discovery source that provided
  a capability;
- `capability_id` is the stable globally unique identity of the capability;
- `mcp_url` is an execution endpoint when applicable, not generic provenance;
- `failed_sources` is keyed by `source_id`, so unavailable sources do not cause
  their previously indexed capabilities to be removed;
- `authoritative=False` means the discovery result is incomplete as a whole,
  so absent capabilities must not be removed.

For a new installation, model assets are bootstrapped explicitly:

```bash
brown-octopus setup-models
```

This prepares `en_core_web_trf` and `Qwen/Qwen3-Embedding-0.6B`. Normal
initialization never downloads them implicitly.

## External session stores

`InMemorySessionStore` is the default. A custom store can be injected with:

```python
octopus = Octopus(session_store=my_session_store)
```

The persisted record is the public serializable `SessionState`:

```python
@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    active_state: dict[str, int] = field(default_factory=dict)
```

`SessionState` contains no locks, runtime context managers, pipelines, models,
tool definitions, credentials, or conversation history. Brown Octopus hydrates
an internal `ActiveCapabilityContext` for each mutation and writes the updated
`active_state` back to the record.

The store boundary includes `mutate(session_id, operation)`. An external
implementation must make that operation atomic—for example, with a Redis
transaction or distributed lock. A separate get/modify/save sequence is not
sufficient for multi-process correctness.

Session records contain only capability-context state: session ID, turn, active
capability IDs, and their last-retrieved turns. Conversation history and
authentication remain outside Brown Octopus.
