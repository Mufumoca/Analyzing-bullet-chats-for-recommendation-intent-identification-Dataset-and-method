# Uncertainty-gate artifacts

`uncertainty_gate.json` records a fixed 1,000-train/400-test experiment. A 20%
stratified training-internal validation split chooses the symmetric probability
margin; the untouched test labels are evaluated once. At the selected margin
`p in [0.28, 0.72]`, Qwen3-8B direct predictions replace 38.5% of test rows.

The test Macro-F1 is 82.29% for the TF-IDF base, 74.70% for Qwen direct, and
83.71% for the gate (+1.42 pp). The 5,000-sample paired row bootstrap CI is
[-2.61, +5.37] pp, so this is an exploratory routing result. Validation and
test prediction CSVs retain row-level probabilities, replacement flags, and
cache provenance. Both splits were regenerated with the same Qwen3-8B settings
(`think=false`, temperature 0, `num_predict=8`, `num_ctx=512`); the primary
result reuses zero records from the older direct-classification cache.
