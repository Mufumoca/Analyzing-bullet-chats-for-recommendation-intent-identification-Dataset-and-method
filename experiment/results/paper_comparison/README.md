# Paper-matched comparison artifacts

`paper_matched_summary.csv` contains three single-seed runs plus the predeclared
three-seed probability ensemble for each 50/60/70-shot setting. The fixed test
file has 9,069 RecDY rows; training and validation each contain `shot` examples
per class and are disjoint. No test labels are used for model/threshold
selection.

`paper_reference_table5.json` records the exact RecDY SPT-RII values transcribed
from Table 5 (PDF page 14). `paper_vs_local_comparison.csv` joins those values
with local independent-run means (the apples-to-apples mean +/- SD comparison),
separately marked three-seed probability-ensemble points, and an explicitly
oracle-only official-background reference.

The local model is Chinese-RoBERTa (with an optional character TF-IDF fusion),
not the paper's full soft-prompt/BiLSTM/dynamic-verbalizer SPT-RII. Results are
therefore descriptive protocol-matched comparisons, not claims of exact method
reproduction. The ensemble rows are single engineering points and must not be
read as independent-run means.
