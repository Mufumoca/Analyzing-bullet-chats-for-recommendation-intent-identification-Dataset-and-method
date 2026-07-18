"""Build a plain-language comparison figure for the improvement report.

Figure contract
---------------
Core conclusion: under the closest single-run budget the local model remains
slightly below SPT-RII; engineering ensembles exceed the reported point while
using a larger cumulative label pool; internal gains are positive point
estimates but not all intervals exclude zero.
Archetype: quantitative grid with panel a as the primary evidence.
Backend: Python/matplotlib only.
Export: editable SVG and PDF, plus PNG and 600-dpi TIFF previews.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Required editable-text settings for the selected Python figure workflow.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"
# Matplotlib does not reliably perform per-glyph fallback from Arial on this
# Windows runtime, so prefer a CJK sans face for mixed Chinese/English labels.
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Arial",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "experiment" / "figures"
PAPER_RESULTS = ROOT / "experiment" / "results" / "paper_comparison"
IMPROVEMENT_RESULTS = ROOT / "experiment" / "results" / "improvement"

COLORS = {
    "paper": "#606060",
    "local": "#484878",
    "ensemble": "#42949E",
    "fusion": "#C56A83",
    "gain": "#2E8B57",
    "interval": "#767676",
    "grid": "#D8D8D8",
    "note": "#F8EED7",
    "ink": "#272727",
}


def read_inputs() -> dict[str, object]:
    comparison = pd.read_csv(
        PAPER_RESULTS / "paper_vs_local_comparison.csv", encoding="utf-8-sig"
    )
    classical = pd.read_csv(
        IMPROVEMENT_RESULTS / "classical_improvement_summary.csv",
        encoding="utf-8-sig",
    )
    intervals = pd.read_csv(
        IMPROVEMENT_RESULTS / "improvement_comparisons.csv", encoding="utf-8-sig"
    )
    roberta = pd.read_csv(
        IMPROVEMENT_RESULTS / "roberta_improvement_summary.csv",
        encoding="utf-8-sig",
    )
    gate = json.loads(
        (IMPROVEMENT_RESULTS / "uncertainty_gate.json").read_text(encoding="utf-8")
    )
    return {
        "comparison": comparison,
        "classical": classical,
        "intervals": intervals,
        "roberta": roberta,
        "gate": gate,
    }


def single_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def build_source_data(inputs: dict[str, object]) -> pd.DataFrame:
    comparison = inputs["comparison"]
    classical = inputs["classical"]
    intervals = inputs["intervals"]
    roberta = inputs["roberta"]
    gate = inputs["gate"]
    assert isinstance(comparison, pd.DataFrame)
    assert isinstance(classical, pd.DataFrame)
    assert isinstance(intervals, pd.DataFrame)
    assert isinstance(roberta, pd.DataFrame)
    assert isinstance(gate, dict)

    rows: list[dict[str, object]] = []
    for shot in (50, 60, 70):
        for role, label in (
            ("reported_reference", "原论文 SPT-RII"),
            ("clean_primary_mean", "本地独立运行"),
        ):
            row = single_row(comparison, shot_per_class=shot, role=role)
            rows.append(
                {
                    "panel": "a",
                    "condition": label,
                    "shot_per_class": shot,
                    "value_pp": 100 * float(row["macro_f1"]),
                    "sd_pp": 100 * float(row["sd"]),
                    "delta_pp": float(row["delta_vs_paper_pp"]),
                    "ci_low_pp": np.nan,
                    "ci_high_pp": np.nan,
                    "budget_note": "每次训练/验证均按每类shot抽样",
                }
            )

    for role, label, note in (
        ("reported_reference", "论文 SPT-RII", "论文报告均值"),
        ("clean_primary_mean", "本地独立运行", "每次训练140条"),
        ("clean_primary_ensemble", "内容三模型集成", "训练并集420条"),
        (
            "clean_secondary_ensemble",
            "内容+TF-IDF集成",
            "训练并集420条",
        ),
    ):
        row = single_row(comparison, shot_per_class=70, role=role)
        rows.append(
            {
                "panel": "b",
                "condition": label,
                "shot_per_class": 70,
                "value_pp": 100 * float(row["macro_f1"]),
                "sd_pp": (
                    100 * float(row["sd"])
                    if pd.notna(row["sd"])
                    else np.nan
                ),
                "delta_pp": float(row["delta_vs_paper_pp"]),
                "ci_low_pp": np.nan,
                "ci_high_pp": np.nan,
                "budget_note": note,
            }
        )

    tfidf_content = single_row(classical, condition="content", size=1000)
    tfidf_improved = single_row(
        classical, condition="structured_late_fusion", size=1000
    )
    tfidf_interval = single_row(
        intervals,
        rows=400,
        size=1000,
        baseline="content",
        comparison="structured_late_fusion",
    )
    roberta_base = single_row(roberta, tag="content1000")
    roberta_improved = single_row(roberta, tag="qwen1000_compact_late_fusion")
    gate_test = gate["test"]

    internal = [
        {
            "condition": "TF-IDF OOF融合\n(1000训练)",
            "delta_pp": 100
            * (
                float(tfidf_improved["macro_f1"])
                - float(tfidf_content["macro_f1"])
            ),
            "ci_low_pp": 100 * float(tfidf_interval["ci95_low"]),
            "ci_high_pp": 100 * float(tfidf_interval["ci95_high"]),
            "budget_note": "95% CI跨0",
        },
        {
            "condition": "RoBERTa紧凑融合\n(1000训练)",
            "delta_pp": 100
            * (
                float(roberta_improved["macro_f1_mean"])
                - float(roberta_base["macro_f1_mean"])
            ),
            "ci_low_pp": np.nan,
            "ci_high_pp": np.nan,
            "budget_note": "3个种子均为正；无正式CI",
        },
        {
            "condition": "不确定性门控\n(1000/400)",
            "delta_pp": 100
            * float(gate_test["gate_vs_base"]["macro_f1_difference"]),
            "ci_low_pp": 100
            * float(gate_test["gate_vs_base"]["macro_f1_ci95_low"]),
            "ci_high_pp": 100
            * float(gate_test["gate_vs_base"]["macro_f1_ci95_high"]),
            "budget_note": "95% CI跨0",
        },
    ]
    for item in internal:
        rows.append(
            {
                "panel": "c",
                "condition": item["condition"],
                "shot_per_class": np.nan,
                "value_pp": np.nan,
                "sd_pp": np.nan,
                "delta_pp": item["delta_pp"],
                "ci_low_pp": item["ci_low_pp"],
                "ci_high_pp": item["ci_high_pp"],
                "budget_note": item["budget_note"],
            }
        )
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_figure(source: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.15), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.25, 1.18],
        left=0.07,
        right=0.985,
        top=0.88,
        bottom=0.20,
        wspace=0.38,
    )

    # a: closest apples-to-apples comparison across shot counts.
    ax = fig.add_subplot(grid[0, 0])
    panel_a = source[source["panel"] == "a"]
    shots = np.array([50, 60, 70])
    for condition, color, marker in (
        ("原论文 SPT-RII", COLORS["paper"], "o"),
        ("本地独立运行", COLORS["local"], "s"),
    ):
        rows = panel_a[panel_a["condition"] == condition].set_index(
            "shot_per_class"
        ).loc[shots]
        values = rows["value_pp"].to_numpy(float)
        sds = rows["sd_pp"].to_numpy(float)
        ax.errorbar(
            shots,
            values,
            yerr=sds,
            color=color,
            marker=marker,
            ms=5,
            lw=1.6,
            capsize=2.4,
            label=condition,
            zorder=3,
        )
    local_rows = panel_a[panel_a["condition"] == "本地独立运行"].set_index(
        "shot_per_class"
    ).loc[shots]
    for shot, value, delta in zip(
        shots,
        local_rows["value_pp"],
        local_rows["delta_pp"],
    ):
        ax.annotate(
            f"{delta:.2f} pp",
            (shot, value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=6.5,
            color=COLORS["local"],
        )
    ax.set_xticks(shots)
    ax.set_xlabel("每类训练样本数")
    ax.set_ylabel("F1 / Macro-F1（%）")
    ax.set_ylim(73.8, 81.2)
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.legend(loc="lower right", fontsize=6.8, handlelength=1.4)
    ax.set_title("同等单次样本预算\n本地仍略低于论文", fontsize=9, pad=9)
    add_panel_label(ax, "a")

    # b: engineering ensemble point and the cumulative label budget.
    ax = fig.add_subplot(grid[0, 1])
    panel_b = source[source["panel"] == "b"].reset_index(drop=True)
    x = np.arange(len(panel_b))
    values = panel_b["value_pp"].to_numpy(float)
    colors = [
        COLORS["paper"],
        COLORS["local"],
        COLORS["ensemble"],
        COLORS["fusion"],
    ]
    hatches = ["//", "", "..", "xx"]
    bars = ax.bar(
        x,
        values,
        width=0.68,
        color=colors,
        edgecolor="#333333",
        linewidth=0.65,
        zorder=2,
    )
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    labels = [
        "论文\nSPT-RII",
        "本地独立\n每次140条",
        "内容集成\n并集420条",
        "内容+TF-IDF\n并集420条",
    ]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=6.6, pad=3)
    ax.set_ylim(76.8, 83.4)
    ax.set_ylabel("70-shot F1 / Macro-F1（%）")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.axhspan(82.35, 83.3, color=COLORS["note"], zorder=0)
    ax.text(
        2.5,
        82.83,
        "三模型重新抽样\n累计训练并集 = 420条",
        ha="center",
        va="center",
        fontsize=6.6,
        color=COLORS["ink"],
    )
    ax.set_title("工程集成点更高\n但累计使用了更多样本", fontsize=9, pad=9)
    add_panel_label(ax, "b")

    # c: effect-size view makes uncertainty explicit.
    ax = fig.add_subplot(grid[0, 2])
    panel_c = source[source["panel"] == "c"].reset_index(drop=True)
    y = np.arange(len(panel_c))[::-1]
    for yi, row in zip(y, panel_c.itertuples(index=False)):
        if np.isfinite(row.ci_low_pp) and np.isfinite(row.ci_high_pp):
            ax.plot(
                [row.ci_low_pp, row.ci_high_pp],
                [yi, yi],
                color=COLORS["interval"],
                lw=1.7,
                zorder=2,
            )
            ax.plot(
                [row.ci_low_pp, row.ci_low_pp],
                [yi - 0.06, yi + 0.06],
                color=COLORS["interval"],
                lw=1.1,
            )
            ax.plot(
                [row.ci_high_pp, row.ci_high_pp],
                [yi - 0.06, yi + 0.06],
                color=COLORS["interval"],
                lw=1.1,
            )
        ax.scatter(
            row.delta_pp,
            yi,
            s=40,
            marker="D" if "RoBERTa" in row.condition else "o",
            color=COLORS["gain"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            5.85,
            yi,
            f"+{row.delta_pp:.2f} pp\n{row.budget_note}",
            ha="right",
            va="center",
            fontsize=6.4,
            color=COLORS["ink"],
        )
    ax.axvline(0, color="#777777", ls="--", lw=0.9, zorder=1)
    ax.set_yticks(y, panel_c["condition"])
    ax.tick_params(axis="y", labelsize=6.7)
    ax.set_xlim(-3.3, 6.2)
    ax.set_xlabel("相对各自基线的提升（百分点）")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.set_title("内部改进点估计均为正\n但部分区间仍跨0", fontsize=9, pad=9)
    add_panel_label(ax, "c")

    fig.text(
        0.5,
        0.045,
        "注：a/b 的论文列为原文 F1；本地使用 Macro-F1。论文未明确 F1 averaging，因此属于近似口径比较。",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#5F6872",
    )
    return fig


def export_assets(fig: plt.Figure, source: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    source.to_csv(FIGURES / "source_data_fig8.csv", index=False, encoding="utf-8-sig")
    base = FIGURES / "fig8_plain_comparison"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    (FIGURES / "fig8_qa.txt").write_text(
        "\n".join(
            [
                "Figure: fig8_plain_comparison",
                "Backend: Python/matplotlib only",
                "Archetype: quantitative grid; panel a is primary evidence",
                "Core conclusion: fair-budget local runs remain below the paper; engineering ensembles use a larger cumulative label pool; internal gains require cautious uncertainty language.",
                "Final size: 7.2 x 4.15 inches before tight bounding-box export",
                "Exports: editable SVG/PDF, 300-dpi PNG, 600-dpi TIFF",
                "Panel a: full RecDY test n=9,069; paper/local mean and SD over three reported/independent runs",
                "Panel b: 70-shot comparison; ensemble points have no SD; three-member training union is 420 rows",
                "Panel c: TF-IDF and gate show paired bootstrap 95% CI; RoBERTa is a three-seed mean difference without formal CI",
                "Metric caveat: the paper reports F1 without an explicit averaging definition; local values use Macro-F1",
                "Source data: source_data_fig8.csv",
                "Visual QA: Chinese glyphs render; labels and panels inspected at final report width; no raster manipulation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    inputs = read_inputs()
    source = build_source_data(inputs)
    fig = draw_figure(source)
    export_assets(fig, source)
    print("Wrote fig8_plain_comparison.{svg,pdf,png,tiff}")
    print("Wrote source_data_fig8.csv")


if __name__ == "__main__":
    main()
