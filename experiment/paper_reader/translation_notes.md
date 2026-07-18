# Translation and Grounding Notes

## Mode

This is a draft-mode full-paper reader generated from selectable PDF text. The
English source is retained page by page and every substantive block has a stable
`S####` anchor. The extracted PDF contains ligature and two-column reading-order
artifacts; those are preserved in the source text and should be checked against
the page image when a quotation matters.

## Translation policy

The abstract, task definition, method overview, explanation-window statement and
Table 5/6 experimental facts have checked Chinese translations. Other blocks are
marked `中文译文待核对` rather than silently inventing a translation. This keeps
the reader source-grounded and makes the missing translation visible.

## Table/figure extraction

`assets/table5_recdy.png` is an approximate tight crop of PDF page 14. It is
linked from the Table 5 card in `paper.md` and its source pointer is `T001` in
`source_map.json`. No other image asset is used by the reader bundle.

## Protocol note for this project

The local experiment matches the paper's RecDY test size and per-class shot
count in a separate substitute-model run, but does not implement the paper's
soft prompt, BiLSTM prompt interaction, or dynamic verbalizer. Official
explanations are therefore kept as an oracle-only reference in the report.
