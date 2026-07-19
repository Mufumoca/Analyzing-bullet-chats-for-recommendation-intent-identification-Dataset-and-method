# BC4RII local reproduction and improvement project

This workspace contains an RTX 4060-scale study of Zhu et al. (Artificial
Intelligence, 2026). The current full method uses a fixed 70-labelled-examples-
per-class split, domain-adaptive pretraining (DAPT), R-Drop, a same-label
five-model ensemble, and an evidence-constrained selective Qwen3-8B verifier.

- Authoritative Chinese report: `experiment/report/improvement_comparison/提升数据与原论文逐项对比.pdf`
- Authoritative LaTeX source: `experiment/report/improvement_comparison/improvement_comparison.tex`
- Strict protocol registry: `experiment/config/same_budget_protocol_v1.json`
- Strict analysis: `experiment/results/same_budget/strict70_analysis.json`
- Selective-Qwen result: `experiment/results/same_budget/strict70_qwen_gate_result.json`
- Selective-Qwen analysis: `experiment/results/same_budget/strict70_qwen_gate_analysis.json`
- Reproduction guide: `experiment/README.md`
- Official repository snapshot: `bc4rii_source/`

Strict fixed-split results on the 9,069-row RecDY test set:

- DAPT + R-Drop, five-initialization mean: 80.01 +/- 0.62 Macro-F1.
- Same-label five-model ensemble: 80.60 Macro-F1.
- Ensemble + selective Qwen3-8B verifier: 80.94 Macro-F1.
- Qwen adds a +0.34-point estimate over the local ensemble; row bootstrap CI
  is [-0.26, +0.92] and live-room cluster CI is [-0.37, +1.54], so this
  incremental gain is not statistically robust.
- Paper Table 5 reported 79.23 F1; the full local system is descriptively
  +1.71 percentage points above it.
- Relative to the local same-protocol baseline ensemble (79.15), the gain is
  +1.45 points. Row bootstrap CI is [+0.85, +2.06], and live-room cluster
  bootstrap CI is [+0.77, +2.45].

The comparison keeps the human-label budget fixed, but it does not keep total
data or compute fixed: DAPT reads 21,159 unlabelled training comments, the
ensemble uses about five times single-model compute, and Qwen routes 1,594 of
9,069 test rows through two or three verifier views. The paper does not state
its F1 averaging convention, and this project does not reproduce the complete
SPT-RII soft-prompt/dynamic-verbalizer pipeline. Cross-paper deltas are therefore
descriptive rather than a claim of an exact reproduction.
