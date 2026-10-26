## Decision
Determining how many transactions should be reviewed manually for mule activity by the five analysts without overwhelming them. We need to set risk-score thresholds that balance fraud detection with analyst capacity.

## Metric
We measure performance using Precision, Recall, and an Expected Cost framework. This allows us to trade off false positives (analyst review time) against false negatives (uncaught fraud losses) based on business cost parameters.

## Baseline
A rule-based static heuristic identifying obvious mule indicators (such as immediate post-creation transfers or known blacklisted accounts), serving as the minimum benchmark before applying model scores.

## Classical ML or LLM
Using tabular machine learning algorithms like Gradient Boosted Trees or Logistic Regression on engineered transaction traits. This provides fast, interpretable risk scoring suitable for real-time transfer auditing without non-deterministic LLM overhead.