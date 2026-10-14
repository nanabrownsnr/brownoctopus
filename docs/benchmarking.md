# Retrieval and indexing benchmarks

Brown Octopus keeps benchmark tooling outside the runtime package. The
commands below measure the stages that matter for context delivery without
changing the production selector or requiring an LLM judge.

## Synthetic retrieval scaling

This isolates pipeline and context costs while varying the capability-universe
size and query length:

```powershell
cmd /c "uv run python scripts\benchmark_retrieval.py --sizes 10 100 1000 10000 --repeats 5 --json > artifacts\retrieval\baseline.json"
uv run python scripts\plot_retrieval_benchmark.py artifacts/retrieval/baseline.json --output artifacts/retrieval/baseline.html
```

The synthetic candidate retriever deliberately scans the complete universe. It
is therefore a scaling baseline, not a claim about dense-vector backend
performance.

## Reviewed relevance evaluation

The reviewed task set is `data/evals/octopus_capability_benchmark.json`. A
rankings CSV can be evaluated against required and supporting capabilities:

```powershell
uv run python -m scripts.evaluate_retrieval_policies `
  --rankings data/experiments/adaptive_cutoff/simple_policy_rankings.csv `
  --output artifacts/retrieval/policy-evaluation.json
```

The evaluator reports required recall, complete-required rate, supporting
recall, selected tool count, context reduction, payload characters, and
missing rankings. Missing rankings are reported explicitly; they are not
treated as successful or silently dropped.

`gap_cluster` is an experiment policy only. The production bounded-max-gap
selector remains the default until a complete relevance run demonstrates an
improvement without unacceptable context growth.

## Real runtime retrieval

This loads the actual analyzer, embedding provider, persisted index, and local
or PostgreSQL-backed store:

```powershell
uv run python -m scripts.benchmark_real_retrieval `
  --index-path data/indexes/default `
  --limit 20 `
  --output artifacts/retrieval/real-baseline.json
```

The output includes median analysis, retrieval, selection, context-update and
total latency, query characters, active payload characters, Python peak memory,
provider, device, backend, and per-query samples. Run it in an environment
where the configured spaCy and embedding models are already installed; this
command does not perform model setup.

For PostgreSQL/pgvector, pass the connection URL and index name:

```powershell
uv run python -m scripts.benchmark_real_retrieval `
  --postgres-url $env:PG_VECTOR_URL `
  --index-name default `
  --limit 20
```

These measurements are diagnostic artifacts. They do not enable a new
selection policy, alter snapshots, or contact MCP servers.
