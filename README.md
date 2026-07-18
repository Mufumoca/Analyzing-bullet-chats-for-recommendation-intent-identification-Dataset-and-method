# BC4RII local reproduction project

This workspace contains a scaled reproduction of the explanation-augmentation
component from Zhu et al. (Artificial Intelligence, 2026) for an RTX 4060 8 GB
GPU.

- Final Chinese report: `experiment/report/report.pdf`
- LaTeX source: `experiment/report/report.tex`
- Reproduction guide: `experiment/README.md`
- Official repository snapshot: `bc4rii_source/`
- Experiment code, data, logs, predictions, figures, and source data:
  `experiment/`
- Includes a full-RecDY, Table-5 shot-matched comparison (50/60/70 examples
  per class; fixed 9,069-row test; three seeds) and a validation-only selective
  Qwen second-opinion gate. The latter is exploratory because its test CI
  crosses zero.
- The paper comparison separates independent-run mean +/- SD from a single
  three-seed probability-ensemble point, so the reported estimators are not
  conflated.

The experiment is intentionally scoped to explanation augmentation. It is not
an equivalent reproduction of the complete SPT-RII soft-prompt and dynamic
verbalizer pipeline.
