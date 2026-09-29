# Milestone 4 selection methods

Milestone 4 changes only the post-ranking selection rule. Intent decomposition,
Qwen embeddings, tool metadata, merge/deduplication, active memory, and the
122-tool universe remain frozen.

## Methods

### Bounded Max Gap

The current production selector. It searches the bounded ranks 1–17, measures
adjacent score gaps relative to the top score, and cuts after the strongest gap
when it reaches the frozen 2% minimum. It exposes at most 16 tools per intent.

### Query-relative score retention

This keeps the longest prefix whose score is at least `alpha * top_score`, with
the same maximum of 16 tools. `alpha` is selected on the development split only
using the preregistered rule in the experiment metadata.

Score thresholds are a standard retrieval control: they make the relevance bar
query-relative instead of imposing one absolute score across all queries.

### Ranked-score knee

This applies a parameter-free knee rule to the bounded score curve. Scores in
ranks 1–17 are min-max normalized; the selected cutoff is the point with the
largest vertical distance from the straight line joining the endpoints, then
clamped to the 16-tool exposure bound.

This is a bounded, offline form of the Kneedle family of knee-point methods.
The original method defines knees using curvature and normalized curve distance:
[Satopaa et al., Finding a “Kneedle” in a Haystack](https://doi.org/10.1109/ICDCSW.2011.20).

### Deferred: Extreme-Value truncation

`Surprise: Result List Truncation via Extreme Value Theory` is directly relevant
to this problem. It fits a Generalized Pareto distribution to ranked-score
tails and turns the scores into calibrated surprise values. It is deferred from
this milestone because fitting and validating that distribution requires a
larger calibration protocol than the current 50-task study:
[Bahri et al., Surprise](https://arxiv.org/abs/2010.09797).

## Development/held-out protocol

The frozen 50-task order is split deterministically and stratified:

- six tasks per category for development (30 total),
- four tasks per category held out for final comparison (20 total).

The relative-threshold candidate grid is fixed before evaluation. The selected
threshold is the candidate with the highest development complete-required rate;
ties are broken by the fewest mean exposed tools per turn, then highest
supporting recall. This is a development calibration, not a change to Octopus.

All held-out results are reported separately and are not used to select a
threshold.
