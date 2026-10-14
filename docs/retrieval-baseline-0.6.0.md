# Brown Octopus 0.6.0 retrieval baseline

This is the first baseline from the reviewed retrieval harness. It measures
the current production pipeline; it is not a comparison against another
implementation on an identical run.

## Run

```powershell
uv run python -m scripts.run_retrieval_baseline `
  --output artifacts\retrieval\current-baseline.json
```

The run used the persisted default index, the configured spaCy analyzer, the
Qwen embedding provider, and the production `brown_octopus_v3` selector. It
evaluated 60 reviewed turns across single-capability, capability-family,
compound, hidden-dependency, and multi-turn tasks.

## Current results

| Metric | Result |
|---|---:|
| Required tools completely retrieved | 80.00% of turns |
| Recall@1 | 45.00% |
| Recall@4 | 74.17% |
| Recall@8 | 80.00% |
| Recall@16 | 80.83% |
| nDCG@1 | 49.44% |
| nDCG@4 | 59.68% |
| nDCG@8 | 62.27% |
| nDCG@16 | 62.86% |
| Precision@1 | 51.67% |
| Precision@4 | 32.08% |
| Precision@8 | 26.11% |
| Precision@16 | 24.25% |
| Average current-turn selection | 7.15 capabilities |
| Average active-context selection | 8.00 capabilities |
| Average active payload | 7,160.6 characters |
| Average total retrieval latency | 541.7 ms |

The diagnostic pass also recorded the candidate pool before final selection:

| Candidate-stage metric | Result |
|---|---:|
| Required tools present somewhere in candidate pools | 100.00% |
| Candidate-ranked Recall@8 | 87.50% |
| Candidate-ranked Recall@16 | 91.67% |
| Turns with a required tool present but not selected | 12 of 60 |

## Interpretation

The baseline shows the central trade-off clearly: recall improves as the
allowed result set grows, while precision falls. The candidate-stage diagnostic
is more important than the first selected-result table: all required tools
entered a candidate pool, but 12 turns lost at least one required tool during
final selection. This makes dynamic K and score-gap analysis a justified next
experiment. It also means that changing the embedding model is not the first
intervention to test.

Candidate-ranked Recall@16 reaches 91.67%, while the current selected-result
Recall@16 is 80.83%. The difference is selection loss, not candidate coverage.
This does not prove that simply increasing K is correct; it gives us a precise
target for testing whether a larger or score-aware cutoff recovers those tools
without making payload size and precision unacceptable.

The next selector experiment should use exactly these same 60 turns and the
same retrieved rankings. It should report required recall, complete-required
rate, nDCG, selected count, and payload characters side by side with this
baseline. No policy should replace `brown_octopus_v3` until it demonstrates a
quality improvement at equal or lower context cost.

## Limits of comparison

Published BM25, dense, and intent-plus-dense scores are useful external
reference points, but their datasets, tools, embedding models, and evaluation
protocols differ. They must not be presented as a direct head-to-head result
against this baseline. Brown Octopus comparisons become meaningful when the
same reviewed tasks, relevance labels, and candidate universe are used.

