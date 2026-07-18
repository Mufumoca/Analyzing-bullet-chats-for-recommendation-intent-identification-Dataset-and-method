"""Create source data and publication figures for the paper-matched comparison.

The reported paper values are transcribed from RecDY Table 5.  Local values are
read from the frozen full-test experiment outputs.  The script deliberately
keeps the official-background branch as an oracle-only reference; it is not
included in the clean primary comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Required editable-text settings for the Python publication-figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment" / "results" / "paper_comparison"
FIGURES = ROOT / "experiment" / "figures"


PAPER = {
    50: {"macro_f1": 0.7784, "sd": 0.0032},
    60: {"macro_f1": 0.7892, "sd": 0.0056},
    70: {"macro_f1": 0.7923, "sd": 0.0031},
}


def read_summary() -> pd.DataFrame:
    summary = pd.read_csv(RESULTS / "paper_matched_summary.csv", encoding="utf-8-sig")
    summary["shot"] = summary["shot"].astype(int)
    return summary


def build_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shot in sorted(PAPER):
        paper = PAPER[shot]
        ensemble = summary[(summary["shot"] == shot) & (summary["seed"].astype(str) == "ensemble")].iloc[0]
        seed_rows = summary[(summary["shot"] == shot) & (summary["seed"].astype(str) != "ensemble")]
        content_values = seed_rows["content_macro_f1"].astype(float).to_numpy()
        clean_values = seed_rows["macro_f1"].astype(float).to_numpy()
        oracle_values = seed_rows["oracle_macro_f1"].astype(float).to_numpy()
        content_mean = float(np.mean(content_values))
        content_sd = float(np.std(content_values, ddof=1))
        clean_mean = float(np.mean(clean_values))
        clean_sd = float(np.std(clean_values, ddof=1))
        rows.extend(
            [
                {
                    "shot_per_class": shot,
                    "source": "paper",
                    "method": "SPT-RII (reported Table 5)",
                    "macro_f1": paper["macro_f1"],
                    "sd": paper["sd"],
                    "delta_vs_paper_pp": 0.0,
                    "role": "reported_reference",
                },
                {
                    "shot_per_class": shot,
                    "source": "local",
                    "method": "RoBERTa content, independent-run mean",
                    "macro_f1": content_mean,
                    "sd": content_sd,
                    "delta_vs_paper_pp": 100 * (content_mean - paper["macro_f1"]),
                    "role": "clean_primary_mean",
                },
                {
                    "shot_per_class": shot,
                    "source": "local",
                    "method": "RoBERTa content, 3-seed probability ensemble",
                    "macro_f1": float(ensemble["content_macro_f1"]),
                    "sd": float("nan"),
                    "delta_vs_paper_pp": 100 * (float(ensemble["content_macro_f1"]) - paper["macro_f1"]),
                    "role": "clean_primary_ensemble",
                },
                {
                    "shot_per_class": shot,
                    "source": "local",
                    "method": "Content + char-TF-IDF validation-fusion mean",
                    "macro_f1": clean_mean,
                    "sd": clean_sd,
                    "delta_vs_paper_pp": 100 * (clean_mean - paper["macro_f1"]),
                    "role": "clean_secondary_mean",
                },
                {
                    "shot_per_class": shot,
                    "source": "local",
                    "method": "Content + char-TF-IDF validation-fusion ensemble",
                    "macro_f1": float(ensemble["macro_f1"]),
                    "sd": float("nan"),
                    "delta_vs_paper_pp": 100 * (float(ensemble["macro_f1"]) - paper["macro_f1"]),
                    "role": "clean_secondary_ensemble",
                },
                {
                    "shot_per_class": shot,
                    "source": "local",
                    "method": "Official-background fusion (oracle only)",
                    "macro_f1": float(ensemble["oracle_macro_f1"]),
                    "sd": float(np.std(oracle_values, ddof=1)),
                    "delta_vs_paper_pp": 100 * (float(ensemble["oracle_macro_f1"]) - paper["macro_f1"]),
                    "role": "oracle_only",
                },
            ]
        )
    return pd.DataFrame(rows)


def write_assets(comparison: pd.DataFrame, summary: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(RESULTS / "paper_vs_local_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(FIGURES / "source_data_fig6.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "paper_reference_table5.json").write_text(
        json.dumps(
            {
                "paper": "Zhu et al., Artificial Intelligence 355 (2026) 104528",
                "doi": "10.1016/j.artint.2026.104528",
                "table": "Table 5",
                "pdf_page": 14,
                "platform": "RecDY",
                "metric": "Macro-F1",
                "shot_definition": "examples per class in train and validation",
                "values": PAPER,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Keep the exact per-seed points in a separate source-data file for audit.
    seed_rows = summary[summary["seed"].astype(str) != "ensemble"].copy()
    seed_rows.to_csv(RESULTS / "paper_matched_per_seed_source.csv", index=False, encoding="utf-8-sig")

    # Report macros and a compact table are generated from the same source CSV
    # as the figure, so a later rerun cannot silently desynchronise prose/table.
    macro_lines = ["% Generated by make_paper_comparison_assets.py."]
    table_lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"shot & 方法 & Macro-F1 & SD & 相对论文 pp\\",
        r"\midrule",
    ]
    shot_names = {50: "Fifty", 60: "Sixty", 70: "Seventy"}
    macro_names = {
        "reported_reference": "PaperSPTRII",
        "clean_primary_mean": "PaperMatchedContentMean",
        "clean_primary_ensemble": "PaperMatchedContentEnsemble",
        "clean_secondary_mean": "PaperMatchedFusionMean",
        "clean_secondary_ensemble": "PaperMatchedFusionEnsemble",
        "oracle_only": "PaperMatchedOracle",
    }
    labels = {
        "reported_reference": "原论文 SPT-RII",
        "clean_primary_mean": "本地 RoBERTa 内容独立运行均值",
        "clean_primary_ensemble": "本地 RoBERTa 内容三种子集成",
        "clean_secondary_mean": "本地内容+TF--IDF 独立运行均值",
        "clean_secondary_ensemble": "本地内容+TF--IDF 三种子集成",
        "oracle_only": "官方释义融合（仅上限参考）",
    }
    for role, macro_prefix in macro_names.items():
        for shot in sorted(PAPER):
            row = comparison[(comparison["role"] == role) & (comparison["shot_per_class"] == shot)].iloc[0]
            # TeX control sequence names cannot end in digits; use words.
            suffix = shot_names[shot]
            macro_lines.append(rf"\newcommand{{\{macro_prefix}{suffix}}}{{{100 * row['macro_f1']:.2f}}}")
            macro_lines.append(rf"\newcommand{{\{macro_prefix}Delta{suffix}}}{{{row['delta_vs_paper_pp']:+.2f}}}")
            if np.isfinite(float(row["sd"])):
                macro_lines.append(rf"\newcommand{{\{macro_prefix}SD{suffix}}}{{{100 * row['sd']:.2f}}}")
            if role in {
                "reported_reference",
                "clean_primary_mean",
                "clean_primary_ensemble",
                "clean_secondary_mean",
                "clean_secondary_ensemble",
            }:
                sd_text = f"{100 * row['sd']:.2f}" if np.isfinite(float(row["sd"])) else "--"
                table_lines.append(
                    f"{shot} & {labels[role]} & {100 * row['macro_f1']:.2f} & {sd_text} & {row['delta_vs_paper_pp']:+.2f}\\\\"
                )
    table_lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{5}{l}{\footnotesize 官方释义融合仅作标签近似信息的上限参考，不纳入主结论。}\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (ROOT / "experiment" / "report" / "paper_comparison_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    (ROOT / "experiment" / "report" / "table_paper_comparison.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.05, 1.25]})
    ax = axes[0]
    shots = np.array(sorted(PAPER))
    width = 0.21
    x = np.arange(len(shots))
    role_specs = [
        ("reported_reference", "Paper SPT-RII mean", "#606060", 0),
        ("clean_primary_mean", "Local content mean", "#484878", 1),
        ("clean_secondary_mean", "Local clean-fusion mean", "#42949E", 2),
    ]
    label_offsets = {
        "reported_reference": 0.45,
        "clean_primary_mean": 0.35,
        "clean_secondary_mean": 0.08,
    }
    for role, label, color, offset_index in role_specs:
        rows = comparison[comparison["role"] == role].set_index("shot_per_class").loc[shots]
        offset = (offset_index - 1) * width
        bars = ax.bar(x + offset, rows["macro_f1"] * 100, width=width, color=color, label=label, zorder=2)
        if role in {"reported_reference", "clean_primary_mean", "clean_secondary_mean"}:
            ax.errorbar(x + offset, rows["macro_f1"] * 100, yerr=rows["sd"] * 100, fmt="none", ecolor="#272727", capsize=2, lw=0.8, zorder=3)
        for bar, value in zip(bars, rows["macro_f1"] * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offsets[role],
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.2,
            )
    content_ensemble = comparison[comparison["role"] == "clean_primary_ensemble"].set_index("shot_per_class").loc[shots]
    fusion_ensemble = comparison[comparison["role"] == "clean_secondary_ensemble"].set_index("shot_per_class").loc[shots]
    ax.scatter(x, content_ensemble["macro_f1"] * 100, marker="D", s=22, facecolor="white", edgecolor="#484878", linewidth=1.0, label="Content ensemble point", zorder=4)
    ax.scatter(x + width, fusion_ensemble["macro_f1"] * 100, marker="s", s=22, facecolor="white", edgecolor="#42949E", linewidth=1.0, label="Clean-fusion ensemble point", zorder=4)
    ax.set_xticks(x, [f"{shot}" for shot in shots])
    ax.set_xlabel("Examples per class (train = validation)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_ylim(74, 85)
    ax.set_title("Full RecDY test (n = 9,069)", fontsize=8.5, pad=7)
    ax.grid(axis="y", color="#D8D8D8", lw=0.5, zorder=0)
    ax.legend(loc="upper left", fontsize=6.3, handlelength=1.2)
    ax.text(-0.12, 1.03, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    ax = axes[1]
    y_positions = np.arange(len(shots))
    paper_rows = comparison[comparison["role"] == "reported_reference"].set_index("shot_per_class").loc[shots]
    content_rows = comparison[comparison["role"] == "clean_primary_mean"].set_index("shot_per_class").loc[shots]
    fusion_rows = comparison[comparison["role"] == "clean_secondary_mean"].set_index("shot_per_class").loc[shots]
    content_ensemble = comparison[comparison["role"] == "clean_primary_ensemble"].set_index("shot_per_class").loc[shots]
    fusion_ensemble = comparison[comparison["role"] == "clean_secondary_ensemble"].set_index("shot_per_class").loc[shots]
    oracle_rows = comparison[comparison["role"] == "oracle_only"].set_index("shot_per_class").loc[shots]
    ax.errorbar(paper_rows["macro_f1"] * 100, y_positions + 0.18, xerr=paper_rows["sd"] * 100, fmt="o", ms=4, color="#606060", label="Paper mean ± SD", capsize=2, lw=0.8)
    ax.errorbar(content_rows["macro_f1"] * 100, y_positions, xerr=content_rows["sd"] * 100, fmt="o", ms=4, color="#484878", label="Local independent mean ± SD", capsize=2, lw=0.8)
    ax.errorbar(fusion_rows["macro_f1"] * 100, y_positions - 0.18, xerr=fusion_rows["sd"] * 100, fmt="o", ms=4, color="#42949E", label="Local clean-fusion mean ± SD", capsize=2, lw=0.8)
    ax.scatter(content_ensemble["macro_f1"] * 100, y_positions + 0.04, marker="D", s=24, facecolor="white", edgecolor="#484878", linewidth=1.0, label="Content ensemble point", zorder=4)
    ax.scatter(fusion_ensemble["macro_f1"] * 100, y_positions - 0.04, marker="s", s=24, facecolor="white", edgecolor="#42949E", linewidth=1.0, label="Clean-fusion ensemble point", zorder=4)
    ax.scatter(oracle_rows["macro_f1"] * 100, y_positions - 0.42, marker="x", s=26, color="#C56A83", label="Oracle-only reference")
    for yi, row in zip(y_positions, content_ensemble.itertuples()):
        ax.text(float(row.macro_f1) * 100 + 0.18, yi + 0.02, f"ensemble {100 * row.macro_f1 - 100 * paper_rows.loc[row.Index, 'macro_f1']:+.2f} pp", fontsize=5.8, va="center", color="#484878")
    ax.set_yticks(y_positions, [f"{shot}-shot" for shot in shots])
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_xlim(74, 85)
    ax.set_title("Independent-run variability vs ensemble point", fontsize=8.5, pad=7)
    ax.grid(axis="x", color="#D8D8D8", lw=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=6.1, handlelength=1.2)
    ax.text(-0.10, 1.03, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")

    fig.suptitle("Paper-matched comparison on the fixed RecDY test set", fontsize=10, y=1.01)
    fig.tight_layout(pad=1.0, w_pad=1.8)
    base = FIGURES / "fig6_paper_comparison"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    (FIGURES / "fig6_qa.txt").write_text(
        "Core conclusion: independent-run means are the apples-to-apples comparison with Table 5; a predeclared three-seed probability ensemble is shown separately and exceeds the reported point at 70-shot. This is a protocol-matched descriptive comparison, not an exact SPT-RII reproduction.\n"
        "Archetype: quantitative grid. Backend: Python/matplotlib.\n"
        "Primary evidence: paper vs local independent-run means and separately marked probability ensembles on the unchanged 9,069-row test.\n"
        "Supporting evidence: per-seed SD and oracle-only marker.\n"
        "Statistics: paper and local independent-run SD are repeat/sample SD (n=3); ensemble markers are single predeclared points with no SD; no test-label tuning.\n"
        "Source data: paper_vs_local_comparison.csv and paper_matched_per_seed_source.csv.\n"
        "Review risk: local model is RoBERTa/TF-IDF fusion, not SPT-RII soft prompt + dynamic verbalizer; oracle marker uses official explanations and is excluded from the clean claim.\n",
        encoding="utf-8",
    )


def main() -> None:
    summary = read_summary()
    comparison = build_comparison(summary)
    write_assets(comparison, summary)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
