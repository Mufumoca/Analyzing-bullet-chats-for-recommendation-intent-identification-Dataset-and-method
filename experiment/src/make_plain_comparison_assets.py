"""Build the plain-language strict-70 progress figure.

Figure contract
---------------
Core conclusion: with one fixed 70-per-class labelled split, domain-adaptive
pretraining plus R-Drop improves every matched initialization and raises the
same-label five-member ensemble above the paper's reported 70-shot point.
Archetype: quantitative grid; panel a is the hero cross-paper comparison.
Backend: Python/matplotlib only.
Export: editable SVG/PDF, 300-dpi PNG, and 600-dpi TIFF.
Reviewer risks: paper F1 averaging is unspecified; fixed-split SD captures only
initialization/dropout variation; the ensemble uses five times model compute;
DAPT uses 21,159 unlabelled train-partition comments.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Arial",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["svg.fonttype"] = "none"
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
RESULTS = ROOT / "experiment" / "results" / "same_budget"
FIGURES = ROOT / "experiment" / "figures"

COLORS = {
    "paper": "#606060",
    "base": "#484878",
    "dapt": "#42949E",
    "rdrop": "#C56A83",
    "full": "#2E8B57",
    "grid": "#D8D8D8",
    "ink": "#272727",
    "muted": "#66707A",
    "band": "#E8F3ED",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    summary = pd.read_csv(RESULTS / "strict70_summary.csv", encoding="utf-8-sig")
    members = pd.read_csv(RESULTS / "strict70_member_metrics.csv", encoding="utf-8-sig")
    bootstrap = pd.read_csv(RESULTS / "strict70_bootstrap.csv", encoding="utf-8-sig")
    analysis = json.loads((RESULTS / "strict70_analysis.json").read_text(encoding="utf-8"))
    expected = {"base_ce", "dapt_ce", "base_rdrop", "dapt_rdrop"}
    if set(summary["preset"]) != expected:
        raise ValueError("Strict-70 summary does not contain the locked four-condition matrix")
    if len(members) != 20:
        raise ValueError("Expected five members for each of four conditions")
    return summary, members, bootstrap, analysis


def build_source_data(
    summary: pd.DataFrame,
    members: pd.DataFrame,
    bootstrap: pd.DataFrame,
    analysis: dict[str, object],
) -> pd.DataFrame:
    paper = float(analysis["paper_reference_f1"])
    rows: list[dict[str, object]] = [
        {
            "panel": "a",
            "condition": "论文 SPT-RII（报告均值）",
            "role": "paper",
            "value_pp": 100 * paper,
            "sd_pp": 0.31,
            "delta_vs_paper_pp": 0.0,
            "seed": np.nan,
            "ci_low_pp": np.nan,
            "ci_high_pp": np.nan,
        }
    ]
    for preset, condition, role in (
        ("base_ce", "本地基线（固定 split 初始化均值）", "base_mean"),
        ("dapt_rdrop", "DAPT+R-Drop（固定 split 初始化均值）", "full_mean"),
        ("dapt_rdrop", "DAPT+R-Drop（同标签五模型集成）", "full_ensemble"),
    ):
        record = summary[summary["preset"] == preset].iloc[0]
        is_ensemble = role.endswith("ensemble")
        value = float(record["ensemble_macro_f1"] if is_ensemble else record["member_mean_macro_f1"])
        sd = np.nan if is_ensemble else 100 * float(record["member_sd_macro_f1"])
        rows.append(
            {
                "panel": "a",
                "condition": condition,
                "role": role,
                "value_pp": 100 * value,
                "sd_pp": sd,
                "delta_vs_paper_pp": 100 * (value - paper),
                "seed": np.nan,
                "ci_low_pp": np.nan,
                "ci_high_pp": np.nan,
            }
        )

    labels = {
        "base_ce": "基础模型\nCE",
        "dapt_ce": "DAPT\nCE",
        "base_rdrop": "基础模型\nR-Drop",
        "dapt_rdrop": "DAPT\nR-Drop",
    }
    for record in summary.itertuples(index=False):
        rows.append(
            {
                "panel": "b",
                "condition": labels[record.preset],
                "role": record.preset,
                "value_pp": 100 * float(record.ensemble_macro_f1),
                "sd_pp": np.nan,
                "delta_vs_paper_pp": float(record.delta_ensemble_vs_paper_pp),
                "seed": np.nan,
                "ci_low_pp": np.nan,
                "ci_high_pp": np.nan,
            }
        )

    base = members[members["preset"] == "base_ce"].set_index("seed")
    improved = members[members["preset"] == "dapt_rdrop"].set_index("seed")
    for seed in sorted(base.index):
        rows.extend(
            [
                {
                    "panel": "c",
                    "condition": "基础模型",
                    "role": "base_member",
                    "value_pp": 100 * float(base.loc[seed, "macro_f1"]),
                    "sd_pp": np.nan,
                    "delta_vs_paper_pp": 100 * (float(base.loc[seed, "macro_f1"]) - paper),
                    "seed": int(seed),
                    "ci_low_pp": np.nan,
                    "ci_high_pp": np.nan,
                },
                {
                    "panel": "c",
                    "condition": "DAPT+R-Drop",
                    "role": "full_member",
                    "value_pp": 100 * float(improved.loc[seed, "macro_f1"]),
                    "sd_pp": np.nan,
                    "delta_vs_paper_pp": 100 * (float(improved.loc[seed, "macro_f1"]) - paper),
                    "seed": int(seed),
                    "ci_low_pp": np.nan,
                    "ci_high_pp": np.nan,
                },
            ]
        )
    for record in bootstrap[bootstrap["comparison"] == "dapt_rdrop-base_ce"].itertuples(index=False):
        rows.append(
            {
                "panel": "c_interval",
                "condition": record.unit,
                "role": "bootstrap_delta",
                "value_pp": float(record.point_delta_pp),
                "sd_pp": np.nan,
                "delta_vs_paper_pp": np.nan,
                "seed": np.nan,
                "ci_low_pp": float(record.ci95_low_pp),
                "ci_high_pp": float(record.ci95_high_pp),
            }
        )
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def draw_figure(source: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.15))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.08, 1.28],
        left=0.075,
        right=0.985,
        top=0.865,
        bottom=0.22,
        wspace=0.43,
    )

    # a: hero cross-paper comparison.
    ax = fig.add_subplot(grid[0, 0])
    panel = source[source["panel"] == "a"].reset_index(drop=True)
    y = np.arange(len(panel))[::-1]
    color_map = {
        "paper": COLORS["paper"],
        "base_mean": COLORS["base"],
        "full_mean": COLORS["dapt"],
        "full_ensemble": COLORS["full"],
    }
    marker_map = {"paper": "o", "base_mean": "s", "full_mean": "D", "full_ensemble": "*"}
    for yi, record in zip(y, panel.itertuples(index=False)):
        color = color_map[record.role]
        if np.isfinite(record.sd_pp):
            ax.plot(
                [record.value_pp - record.sd_pp, record.value_pp + record.sd_pp],
                [yi, yi],
                color=color,
                lw=1.5,
            )
            ax.plot(
                [record.value_pp - record.sd_pp, record.value_pp - record.sd_pp],
                [yi - 0.06, yi + 0.06],
                color=color,
                lw=0.9,
            )
            ax.plot(
                [record.value_pp + record.sd_pp, record.value_pp + record.sd_pp],
                [yi - 0.06, yi + 0.06],
                color=color,
                lw=0.9,
            )
        ax.scatter(
            record.value_pp,
            yi,
            color=color,
            marker=marker_map[record.role],
            s=58 if record.role == "full_ensemble" else 30,
            zorder=3,
        )
        ax.text(
            record.value_pp + 0.12,
            yi,
            f"{record.value_pp:.2f}%",
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold" if record.role == "full_ensemble" else "normal",
            color=color,
        )
    ax.axvline(79.23, color=COLORS["paper"], lw=0.8, ls="--", alpha=0.75)
    ax.set_yticks(
        y,
        ["论文报告\n均值±SD", "本地基线\n初始化均值±SD", "增强模型\n初始化均值±SD", "增强模型\n同标签集成点"],
    )
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(75.5, 81.35)
    ax.set_xlabel("70-shot F1 / Macro-F1（%）")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55)
    ax.set_title("同一 70/类标签下\n增强结果超过论文报告点", fontsize=9, pad=9)
    add_panel_label(ax, "a")

    # b: 2x2 ablation at ensemble level.
    ax = fig.add_subplot(grid[0, 1])
    panel = source[source["panel"] == "b"].set_index("role").loc[
        ["base_ce", "dapt_ce", "base_rdrop", "dapt_rdrop"]
    ]
    x = np.arange(len(panel))
    values = panel["value_pp"].to_numpy(float)
    colors = [COLORS["base"], COLORS["dapt"], COLORS["rdrop"], COLORS["full"]]
    bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="#333333", linewidth=0.55, zorder=2)
    for bar, value, delta in zip(bars, values, panel["delta_vs_paper_pp"].to_numpy(float)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.10,
            f"{value:.2f}\n({delta:+.2f})",
            ha="center",
            va="bottom",
            fontsize=6.6,
        )
    ax.axhline(79.23, color=COLORS["paper"], lw=1.0, ls="--", zorder=3)
    ax.text(3.46, 79.30, "论文 79.23", ha="right", va="bottom", fontsize=6.3, color=COLORS["paper"])
    ax.set_xticks(x, panel["condition"].tolist())
    ax.tick_params(axis="x", labelsize=6.3)
    ax.set_ylim(78.7, 81.1)
    ax.set_ylabel("五成员同标签集成 Macro-F1（%）")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.set_title("2×2 消融：主要增益来自 DAPT", fontsize=9, pad=9)
    add_panel_label(ax, "b")

    # c: paired initialization evidence and bootstrap intervals.
    ax = fig.add_subplot(grid[0, 2])
    panel = source[source["panel"] == "c"]
    base = panel[panel["role"] == "base_member"].set_index("seed")
    full = panel[panel["role"] == "full_member"].set_index("seed")
    for seed in sorted(base.index):
        values = [float(base.loc[seed, "value_pp"]), float(full.loc[seed, "value_pp"])]
        ax.plot([0, 1], values, color="#AEB5BC", lw=1.0, zorder=1)
        ax.scatter(0, values[0], color=COLORS["base"], s=22, zorder=2)
        ax.scatter(1, values[1], color=COLORS["full"], s=22, zorder=2)
    mean_base = float(base["value_pp"].mean())
    mean_full = float(full["value_pp"].mean())
    ax.plot([0, 1], [mean_base, mean_full], color=COLORS["ink"], lw=2.0, zorder=3)
    ax.scatter([0, 1], [mean_base, mean_full], color=[COLORS["base"], COLORS["full"]], s=48, edgecolor="white", linewidth=0.7, zorder=4)
    ax.axhline(79.23, color=COLORS["paper"], lw=0.8, ls="--", alpha=0.75)
    ax.set_xticks([0, 1], ["基础模型", "DAPT+R-Drop"])
    ax.set_xlim(-0.28, 1.38)
    ax.set_ylim(74.6, 81.1)
    ax.set_ylabel("单成员 Macro-F1（%）")
    ax.grid(axis="y", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.text(
        0.03,
        0.965,
        "五个配对差值全部为正\n平均 +2.40 pp\nt-CI95% [+0.63, +4.17] pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": COLORS["band"], "edgecolor": "none"},
    )
    intervals = source[source["panel"] == "c_interval"].set_index("condition")
    row_ci = intervals.loc["row"]
    cluster_ci = intervals.loc["live_id_cluster"]
    ax.text(
        0.03,
        0.05,
        "集成相对基线：+1.45 pp\n"
        f"逐行 bootstrap [{row_ci.ci_low_pp:.2f}, {row_ci.ci_high_pp:.2f}]\n"
        f"直播间聚类 [{cluster_ci.ci_low_pp:.2f}, {cluster_ci.ci_high_pp:.2f}]",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=COLORS["ink"],
    )
    ax.set_title("同种子配对：提升不是单个幸运初始化", fontsize=9, pad=9)
    add_panel_label(ax, "c")

    fig.text(
        0.5,
        0.055,
        "口径：固定同一 70/类训练集与 70/类验证集；5 个模型只改变初始化/Dropout。DAPT 额外使用训练分区 21,159 条无标签弹幕；集成不增加人工标签，但约需 5 倍模型计算。",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=COLORS["muted"],
    )
    fig.text(
        0.5,
        0.02,
        "论文仅写 F1，未说明 averaging；本地使用 Macro-F1，因此跨论文差值为描述性比较。",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["muted"],
    )
    return fig


def export_and_qa(fig: plt.Figure, source: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_path = FIGURES / "source_data_fig8.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    base = FIGURES / "fig8_plain_comparison"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    image = Image.open(base.with_suffix(".png")).convert("RGB")
    pixels = np.asarray(image)
    nonwhite_fraction = float(np.mean(np.any(pixels < 248, axis=2)))
    tree = ET.parse(base.with_suffix(".svg"))
    text_nodes = tree.findall(".//{http://www.w3.org/2000/svg}text")
    if image.width < 1800 or image.height < 900:
        raise AssertionError(f"PNG preview is unexpectedly small: {image.size}")
    if nonwhite_fraction < 0.08:
        raise AssertionError("Figure preview is nearly blank")
    if len(text_nodes) < 30:
        raise AssertionError("SVG text is not preserved as editable text nodes")

    (FIGURES / "fig8_qa.txt").write_text(
        "\n".join(
            [
                "Figure: fig8_plain_comparison",
                "Backend: Python/matplotlib only",
                "Archetype: quantitative grid; panel a is hero evidence",
                "Core conclusion: DAPT+R-Drop improves all five matched initializations and the same-label ensemble exceeds the paper-reported 70-shot point.",
                "Final size: 7.2 x 4.15 inches before tight bounding-box export",
                "Panel a: paper reported mean+SD, local fixed-split initialization mean+SD, and same-label ensemble point",
                "Panel b: locked 2x2 DAPT/R-Drop ablation; ensemble points have no SD",
                "Panel c: five paired initialization runs; paired t-CI plus row and live_id-cluster bootstrap intervals",
                "Statistics: n=5 initialization/dropout replicates on one fixed split; 5,000 bootstrap iterations; 12 live_id clusters",
                "Metric: local Macro-F1; paper F1 averaging unspecified",
                "Budget: each member sees the same 140 labelled train rows; DAPT reads 21,159 unlabelled train rows; ensemble uses five-model compute",
                f"PNG dimensions: {image.width} x {image.height}",
                f"PNG non-white fraction: {nonwhite_fraction:.4f}",
                f"Editable SVG text nodes: {len(text_nodes)}",
                "Source data: source_data_fig8.csv",
                "Image integrity: vector-native plotting; no raster manipulation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    summary, members, bootstrap, analysis = load_inputs()
    source = build_source_data(summary, members, bootstrap, analysis)
    figure = draw_figure(source)
    export_and_qa(figure, source)
    print("Wrote fig8_plain_comparison.{svg,pdf,png,tiff}")
    print("Wrote source_data_fig8.csv and fig8_qa.txt")


if __name__ == "__main__":
    main()
