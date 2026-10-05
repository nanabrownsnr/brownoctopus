# Brown Octopus — Current Integration Test Plan

This plan validates the current Brown Octopus implementation end to end. Run
the consumer tests against the built wheel, not an editable source checkout,
for the release-acceptance sections.

## Test rules

- Record Python version, operating system, Brown Octopus version, wheel hash,
  index path, embedding provider, and compute device.
- Do not print API keys, JWTs, database passwords, MCP tokens, or complete
  sensitive capability schemas.
- Use disposable test indexes and disposable database names.
- Use approved read-only MCPs for ordinary tests.
- Use a disposable write-capable MCP only for the explicit write workflow.
- Do not change V3 thresholds or tune results during testing.
- Mark tests `SKIPPED` only when the required external service/credential is
  genuinely unavailable, and record the reason.

## 1. Clean wheel installation

Create a new consumer project outside the Brown Octopus repository.

```powershell
uv init
uv add ..\dist\brown_octopus-<version>-py3-none-any.whl
uv sync
```

Verify:

```powershell
python -c "import brown_octopus; print(brown_octopus.__version__)"
python -c "from brown_octopus import Octopus, OctopusIndex"
brown-octopus --help
brown-octopus --version
brown-octopus version
```

Expected:

- the wheel is imported from the consumer `.venv`;
- no repository source path is imported;
- all commands exit with code 0;
- help/version do not load ML models or make network calls.

## 2. Model bootstrap selectors

Use a temporary `BROWN_OCTOPUS_MODEL_DIR`.

### Default behavior

```powershell
brown-octopus setup-models
brown-octopus doctor
```

Expected: both `en_core_web_trf` and `Qwen/Qwen3-Embedding-0.6B` are prepared.

### spaCy-only setup

Use a fresh model directory:

```powershell
brown-octopus setup-models --spacy
```

Expected:

- spaCy model is prepared;
- Qwen is not downloaded by this command;
- the command exits successfully.

### Embedding-only setup

Use another fresh model directory:

```powershell
brown-octopus setup-models --embedding
```

Expected:

- Qwen is prepared;
- spaCy is not downloaded by this command;
- the command exits successfully.

### Selector validation

```powershell
brown-octopus setup-models --spacy --embedding
```

Expected: argparse rejects the mutually exclusive options with a non-zero exit
code and no traceback.

Run setup twice for each mode. The second run must be idempotent and avoid
unnecessary downloads.

## 3. Runtime model behavior

With both models prepared:

```powershell
python -c "from brown_octopus.analyzer import initialize_analyzer; initialize_analyzer(); print('Analyzer OK')"
```

Expected: `Analyzer OK`.

Remove or point to an empty model directory and verify:

```powershell
brown-octopus doctor
```

Expected: actionable failure directing the user to `brown-octopus setup-models`.

Verify `Octopus.initialize()` does not download missing models implicitly.

## 4. Device and GPU behavior

Record:

```powershell
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
brown-octopus inspect --json
```

Cases:

1. CUDA-enabled PyTorch and visible NVIDIA GPU.
   - expected device: `cuda`;
   - local embedding provider receives `device="cuda"`.
2. CPU-only PyTorch or no GPU.
   - expected device: `cpu`;
   - initialization still succeeds.
3. HTTP embedding provider.
   - local GPU is not expected to perform embedding inference;
   - remote provider handles embedding computation.
4. spaCy startup with compatible CuPy installed.
   - verify spaCy attempts GPU initialization;
   - CPU fallback remains valid when CuPy is unavailable.

Do not treat low GPU utilization during a single short query as failure. Check
model load and repeated/batched retrieval as well.

## 5. Embedding providers

### Default local provider

```python
from brown_octopus import LocalEmbeddingProvider

provider = LocalEmbeddingProvider()
provider.healthcheck()
vector = provider.embed("search the web")
```

Verify the vector is numeric, normalized, and has the model's expected
dimension.

### Custom local model

Use a compatible SentenceTransformers model or local model path. Verify:

- index creation records provider identity and dimension;
- runtime rejects an incompatible provider/index combination;
- a matching provider initializes successfully.

### HTTP provider

Use a local HTTP mock first, then an approved real endpoint if available:

```python
from brown_octopus import HttpEmbeddingProvider

provider = HttpEmbeddingProvider(
    url="http://localhost:8000/api/embed",
    model="test-embedding",
)
```

Verify:

- request model and input are correct;
- `embeddings` responses are accepted;
- OpenAI-style `data` responses are accepted;
- inconsistent dimensions fail clearly;
- timeout/unreachable endpoint fails clearly;
- credentials are never logged.

## 6. OctopusIndex facade

Use a disposable index path and a deterministic local/mock source.

### Initial creation

```python
index = OctopusIndex.from_sources([source], index_path=path)
report = await index.create()
```

Verify:

- capabilities are discovered;
- embeddings and metadata are persisted;
- `report.tool_count` is correct;
- a separate runtime can initialize from the persisted index.

### Empty index

```python
index = OctopusIndex.from_sources([], index_path=path)
await index.create()
octopus = index.runtime()
await octopus.initialize()
result = octopus.retrieve_result("search the web")
```

Expected:

```text
result.retrieved_tools == []
result.tools == []
result.tool_ids == []
```

### Synchronization

Verify:

- new capability is added;
- changed capability is replaced;
- removed capability disappears from authoritative snapshots;
- unchanged capability remains;
- failed/non-authoritative sources preserve prior capabilities.

### Source lifecycle

Test:

```python
await index.add(source)
await index.remove_source(source_id)
await index.remove_capability(capability_id)
index.reset()
```

Verify that each operation affects only the intended source/index and that
reset does not delete sessions, models, or unrelated indexes.

## 7. Capability sources

Test each supported source:

- `McpServerSource` against an approved read-only MCP;
- `LocalMcpCatalogSource` / JSON catalog;
- `JsonCapabilitySource`;
- `ApiCapabilitySource` using a local HTTP mock;
- `McpRegistrySource` against a local mock and, if approved, the live registry;
- `CompositeCapabilitySource` with multiple sources;
- custom `CapabilitySource` returning `CapabilityDiscoveryResult`.

Verify normalized fields:

```text
capability_id
source_id
name
description
input_schema
mcp_url when applicable
tool_name when applicable
```

For registry sources verify cursor pagination, token refresh, repeated 401
handling, unreachable registry behavior, and partial MCP failure preservation.

## 8. Runtime retrieval and V3 invariants

Create a known test index and verify:

- `retrieval_text` is used when present;
- fallback to `text`, then original query, works;
- `result.retrieved_tools` contains current-turn selection only;
- `result.tools` contains final active context;
- `result.tool_ids` matches `result.tools`;
- current-turn tools appear before retained tools;
- stable `capability_id` deduplication remains intact;
- allowed MCP URL filtering happens before candidate ranking;
- V3 Min-4 and Bounded Max Gap behavior is unchanged;
- TTL remains 8 turns;
- active cap remains 30.

Do not compare only tool names when validating identity. Use full
`capability_id` values.

## 9. Sessions and Redis

### In-memory store

Verify:

- same-session continuity;
- different-session isolation;
- independent turn counters;
- TTL progression per session;
- independent cap/eviction state;
- reset affects only one session;
- delete affects only one session;
- same-session concurrent retrieval is serialized safely;
- different sessions can progress concurrently.

### Redis/custom store

Use a disposable Redis instance and a synchronous `SessionStore` adapter.

Verify across two Octopus instances and separate processes:

- session continuation;
- isolation;
- atomic same-session turns;
- reset/delete visibility;
- process restart persistence;
- TTL 8 and active cap 30;
- no locks/model objects/conversation history are persisted.

Record that distributed atomicity is the responsibility of the supplied store.

## 10. Local retrieval backend

With `Octopus()` or `LocalIndexStore`, verify:

- the runtime selects `LocalCandidateRetriever`;
- `QwenCandidateRetriever` remains importable as a compatibility alias;
- local cosine ranking returns ranked tools with `rank` and `score`;
- custom local embedding providers work;
- no database dependency is imported or required.

## 11. Postgres snapshot and pgvector retrieval

### Snapshot persistence

Run Postgres in Docker or use an approved test database. Verify:

- `PostgresIndexStore` creates its schema;
- `create()` writes a complete snapshot;
- a second process/instance loads the same snapshot;
- metadata, tools, embeddings, and current pointer survive restart;
- `reset()` removes only the configured `index_name`.

### Plain Postgres fallback

Use Postgres without pgvector. Verify:

- snapshot storage succeeds;
- `supports_native_vector_search` is false;
- runtime uses local retrieval.

### pgvector

Use `pgvector/pgvector` in Docker. Verify:

- pgvector is detected;
- vectors are stored in the vector table;
- runtime selects `VectorSearchCandidateRetriever`;
- cosine-distance search returns the expected ranked capability;
- `allowed_mcp_urls` is applied in the database query;
- V3 selection receives the same candidate shape as local retrieval;
- runtime does not need to load the embedding matrix into local memory;
- a failed database query returns a clear error and does not corrupt the
  persisted snapshot.

## 12. Mongo snapshot and Atlas vector retrieval

### Ordinary MongoDB

Use a disposable local MongoDB container. Verify:

- snapshot create/load/reset works;
- a second instance can initialize from the same store;
- without `vector_search_index`, local retrieval is used.

### MongoDB Atlas Vector Search

Against an approved Atlas test deployment:

```python
MongoIndexStore(
    connection_url=mongodb_uri,
    index_name="capabilities",
    vector_search_index="brown-octopus-vectors",
)
```

Verify:

- the Atlas index targets the `embedding` field;
- its dimension matches the configured embedding provider;
- `$vectorSearch` returns ranked capability documents;
- URL filtering is included in the vector-search filter;
- runtime selects native vector retrieval;
- missing/misconfigured Atlas vector indexes produce an actionable error.

Do not claim native Mongo vector retrieval passed using only a standard local
MongoDB container; local Mongo does not provide Atlas `$vectorSearch`.

## 13. LangGraph integration

Use a separate consumer application. Verify:

```text
user message
    ↓
Brown Octopus retrieve_result()
    ↓
result.tools
    ↓
LLM.bind_tools(result.tools)
    ↓
model tool selection
    ↓
host-owned tool execution
```

Test:

- single-capability request;
- multi-capability request;
- follow-up turn using active capability state;
- inspection of exposed capabilities per turn;
- host executes the selected MCP tool;
- Brown Octopus does not execute tools or receive credentials.

For a disposable approved Word MCP, test:

1. retrieve `create_word_document` from the original compound request;
2. bind the returned capability to the model;
3. host executes document creation;
4. host adds content;
5. host verifies the document;
6. host deletes the test document.

Run write-capable tests only with explicit approval and a disposable target.

## 14. Failure and security checks

Verify:

- missing models produce actionable setup instructions;
- missing/corrupt indexes produce Brown Octopus errors;
- provider dimension mismatch is rejected;
- database outage does not destroy a prior snapshot;
- partial source failure preserves prior source capabilities;
- expired tokens are refreshed only by the host callback/source;
- repeated authorization failure does not leak tokens;
- database URLs, API keys, JWTs, and MCP credentials are not logged;
- normal initialize/retrieve does not perform unexpected network discovery or
  model downloads.

## 15. Final release validation

From a clean consumer environment:

```powershell
uv add ..\dist\brown_octopus-<version>-py3-none-any.whl
uv sync
uv run brown-octopus --help
uv run brown-octopus --version
uv run brown-octopus doctor
uv run brown-octopus inspect --json
uv run pytest -q -m "not external"
```

Validate the wheel and sdist metadata, inspect wheel contents, and confirm no
research datasets, caches, credentials, or developer artifacts are included.

Final report must include:

- package version and artifact hashes;
- Python/OS matrix;
- passed/failed/skipped counts;
- provider and compute device;
- index backend and index name;
- database/MCP tests actually run;
- known limitations and skipped-test reasons;
- confirmation that V3 parameters were unchanged.
