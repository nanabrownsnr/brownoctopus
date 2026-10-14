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

## Interpretation

The baseline shows the central trade-off clearly: recall improves as the
allowed result set grows, while precision falls. Recall gains from K=8 to K=16
are very small in this run, which makes the 8–16 region the first place to
investigate a score-gap or clustering-based cutoff. This is a hypothesis, not a
conclusion: the task set is reviewed but still limited, and the baseline uses
the current index and model configuration.

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

