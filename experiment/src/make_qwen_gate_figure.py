"""Build the selective-Qwen comparison figure from locked result files.

Figure contract
---------------
Core conclusion: under one fixed 70-per-class human-label split, an
evidence-constrained selective Qwen3-8B verifier raises the local point
estimate from 80.60% to 80.94% Macro-F1, while paired bootstrap intervals for
the +0.34 pp increment cross zero.
Archetype: asymmetric quantitative grid; panel a is the hero comparison.
Target/output: double-column report figure, 183 x 107 mm.
Backend: Python/matplotlib only.
Panel map: a, performance points; b, routing coverage; c, corrected versus
harmed predictions; d, paired-bootstrap intervals.
Evidence hierarchy: a is the main result; b-c explain the mechanism; d is the
statistical boundary.
Statistics: 9,069 fixed test rows; 5,000 paired row and live_id-cluster
bootstrap resamples.
Reviewer risks: the paper does not state F1 averaging; Paper Table 6 uses a
different 1,000-sample PEFT role; the local +0.34 pp interval crosses zero;
the public test set was observed in earlier project experiments.
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


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiment" / "results" / "same_budget"
FIGURES = ROOT / "experiment" / "figures"

RESULT_FILE = RESULTS / "strict70_qwen_gate_result.json"
ANALYSIS_FILE = RESULTS / "strict70_qwen_gate_analysis.json"

COLORS = {
    "paper_qwen": "#8A8A8A",
    "paper_spt": "#4F5862",
    "student": "#287D8E",
    "full": "#2D7C55",
    "harm": "#A84A59",
    "grid": "#D8DDE1",
    "ink": "#27313B",
    "muted": "#626C76",
    "light_green": "#E3F1EA",
    "light_gold": "#F8EED7",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.titlesize": 8.6,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_locked_results() -> tuple[dict[str, object], dict[str, object]]:
    result = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS_FILE.read_text(encoding="utf-8"))
    if result["status"] != "completed":
        raise ValueError("Selective-Qwen result is not complete")
    if result["protocol_hash"] != analysis["protocol_verification"]["protocol_hash"]:
        raise ValueError("Result and analysis protocol hashes differ")
    if not analysis["protocol_verification"]["selection_hash_matches"]:
        raise ValueError("Locked validation selection hash was not verified")
    return result, analysis


def build_source_data(result: dict[str, object], analysis: dict[str, object]) -> pd.DataFrame:
    paper = analysis["paper_references"]
    base = analysis["base_metrics"]
    full = analysis["qwen_gate_metrics"]
    routing = analysis["routing"]
    bootstrap = analysis["paired_bootstrap"]
    total = int(result["test_rows"])

    rows: list[dict[str, object]] = []
    comparisons = [
        ("论文 Table 6\nQwen3:8B PEFT", "paper_qwen", 100 * paper["qwen3_8b_table6_peft_f1"], "功能对照；1,000 样本"),
        ("论文 Table 5\nSPT-RII 70-shot", "paper_spt", 100 * paper["spt_rii_table5_70_shot_f1"], "跨论文描述性对照"),
        ("本地 DAPT+R-Drop\n同标签集成", "student", 100 * base["macro_f1"], "同一测试集、无推理期 LLM"),
        ("本地完整方法\n+选择性 Qwen", "full", 100 * full["macro_f1"], "同一测试集、同人工标签"),
    ]
    for label, role, value, note in comparisons:
        rows.append(
            {
                "panel": "a",
                "label": label,
                "role": role,
                "value": value,
                "low": np.nan,
                "high": np.nan,
                "count": np.nan,
                "denominator": np.nan,
                "note": note,
            }
        )

    for label, key in (
        ("进入 Qwen", "candidate_rows"),
        ("形成有效共识", "consensus_rows"),
        ("最终改变预测", "changed_rows"),
    ):
        count = int(routing[key])
        rows.append(
            {
                "panel": "b",
                "label": label,
                "role": key,
                "value": 100 * count / total,
                "low": np.nan,
                "high": np.nan,
                "count": count,
                "denominator": total,
                "note": "占完整测试集比例",
            }
        )

    for label, key, sign in (
        ("纠正基线错误", "corrected_rows", 1),
        ("破坏原正确项", "harmed_rows", -1),
        ("净增加正确项", "net_corrected_rows", 1),
    ):
        count = int(routing[key])
        rows.append(
            {
                "panel": "c",
                "label": label,
                "role": key,
                "value": sign * count,
                "low": np.nan,
                "high": np.nan,
                "count": count,
                "denominator": total,
                "note": "changed rows accounting",
            }
        )

    point = float(analysis["delta_vs_local_base_pp"])
    for label, key in (("逐行重采样", "row"), ("按直播间聚类", "live_id_cluster")):
        record = bootstrap[key]
        rows.append(
            {
                "panel": "d",
                "label": label,
                "role": key,
                "value": point,
                "low": 100 * float(record["delta_ci95"][0]),
                "high": 100 * float(record["delta_ci95"][1]),
                "count": int(record.get("clusters", result["test_rows"])),
                "denominator": int(record["iterations"]),
                "note": f"P(delta<=0)={record['probability_delta_le_zero']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=9.2, va="bottom")


def draw_figure(source: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(7.2, 4.2))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[1.1, 1.1, 0.95, 0.95, 1.05, 1.05],
        height_ratios=[1, 1],
        left=0.075,
        right=0.985,
        bottom=0.19,
        top=0.89,
        wspace=0.92,
        hspace=0.72,
    )

    # a: performance points. The paper-Qwen point is intentionally muted
    # because it uses a different role and supervision budget.
    ax = fig.add_subplot(grid[:, 0:2])
    panel = source[source["panel"] == "a"].reset_index(drop=True)
    y = np.arange(len(panel))[::-1]
    markers = {"paper_qwen": "X", "paper_spt": "o", "student": "s", "full": "*"}
    colors = {key: COLORS[key] for key in markers}
    ax.axvspan(80.597, 80.937, color=COLORS["light_green"], alpha=0.9, zorder=0)
    for yi, row in zip(y, panel.itertuples(index=False)):
        ax.hlines(yi, 74.2, row.value, color="#CFD4D8", lw=1.0, zorder=1)
        ax.scatter(
            row.value,
            yi,
            marker=markers[row.role],
            s=70 if row.role == "full" else 34,
            color=colors[row.role],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            row.value + 0.12,
            yi,
            f"{row.value:.2f}%",
            va="center",
            color=colors[row.role],
            fontweight="bold" if row.role == "full" else "normal",
            fontsize=6.8,
        )
    ax.set_yticks(y, panel["label"].tolist())
    ax.set_xlim(74.0, 82.25)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("F1 / Macro-F1（%）")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.set_title("大模型受控参与后\n点估计进一步提高", pad=7)
    ax.text(
        0.03,
        0.02,
        "相对本地 +0.34 pp\n相对论文 Table 5 +1.71 pp",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.32", "facecolor": COLORS["light_green"], "edgecolor": "none"},
    )
    add_panel_label(ax, "a")

    # b: how selectively the LLM is invoked.
    ax = fig.add_subplot(grid[0, 2:4])
    panel = source[source["panel"] == "b"].reset_index(drop=True)
    y = np.arange(len(panel))[::-1]
    bars = ax.barh(y, panel["value"], color=[COLORS["student"], "#4F9CA7", "#8BC1B2"], height=0.56)
    for bar, row in zip(bars, panel.itertuples(index=False)):
        ax.text(
            row.value + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.count):,}\n{row.value:.2f}%",
            va="center",
            fontsize=6.2,
        )
    ax.set_yticks(y, panel["label"].tolist())
    ax.set_xlim(0, 22.5)
    ax.set_xlabel("占 9,069 条测试集（%）")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.set_title("Qwen 只复核低置信样本", pad=5)
    add_panel_label(ax, "b", x=-0.21, y=1.08)

    # c: gains and harms among changed predictions.
    ax = fig.add_subplot(grid[1, 2:4])
    panel = source[source["panel"] == "c"].set_index("role")
    specs = [
        ("corrected_rows", COLORS["full"]),
        ("harmed_rows", COLORS["harm"]),
        ("net_corrected_rows", COLORS["paper_spt"]),
    ]
    y = np.arange(3)[::-1]
    values = [float(panel.loc[key, "value"]) for key, _ in specs]
    bars = ax.barh(y, values, color=[color for _, color in specs], height=0.55)
    for bar, value in zip(bars, values):
        if value < 0:
            text_x = value + 10
            text_ha = "left"
            text_color = "white"
        else:
            text_x = value + 10
            text_ha = "left"
            text_color = COLORS["ink"]
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.0f}",
            ha=text_ha,
            va="center",
            fontsize=6.6,
            color=text_color,
            fontweight="bold" if abs(value) == 50 else "normal",
        )
    ax.axvline(0, color=COLORS["ink"], lw=0.75)
    ax.set_yticks(y, ["纠正错误", "造成新错", "净增正确"])
    ax.set_xlim(-370, 420)
    ax.set_xlabel("预测条数（正为收益，负为损失）")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.set_title("349 条改对，299 条改错", pad=5)
    add_panel_label(ax, "c", x=-0.21, y=1.08)

    # d: uncertainty of the incremental local gain.
    ax = fig.add_subplot(grid[:, 4:6])
    panel = source[source["panel"] == "d"].reset_index(drop=True)
    y = np.arange(len(panel))[::-1]
    ax.axvline(0, color=COLORS["paper_spt"], lw=1.0, ls="--", zorder=0)
    for yi, row, color in zip(y, panel.itertuples(index=False), [COLORS["student"], COLORS["full"]]):
        ax.plot([row.low, row.high], [yi, yi], color=color, lw=2.0)
        ax.plot([row.low, row.low], [yi - 0.07, yi + 0.07], color=color, lw=1.0)
        ax.plot([row.high, row.high], [yi - 0.07, yi + 0.07], color=color, lw=1.0)
        ax.scatter(row.value, yi, color=color, s=32, zorder=2)
        ax.text(
            row.high + 0.06,
            yi,
            f"[{row.low:+.2f}, {row.high:+.2f}]",
            va="center",
            fontsize=6.3,
            color=color,
        )
    ax.set_yticks(y, ["逐行重采样", "直播间聚类"])
    ax.set_xlim(-0.55, 1.78)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlabel("相对本地基线的 Macro-F1 差值（pp）")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55, zorder=0)
    ax.set_title("+0.34 pp 的区间仍跨 0", pad=7)
    ax.text(
        0.03,
        0.06,
        "点估计为正，但尚不足以称统计显著。\n逐行 P(Delta<=0)=0.135；聚类为 0.251。",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.3,
        bbox={"boxstyle": "round,pad=0.32", "facecolor": COLORS["light_gold"], "edgecolor": "none"},
    )
    add_panel_label(ax, "d")

    fig.text(
        0.5,
        0.08,
        "本地两行使用相同 70/类训练与验证标签、相同 9,069 条测试集；Qwen3:8B Q4_K_M 仅处理 |p-0.5|<=0.32 的样本，三视角至少两票一致且证据为当前弹幕原文子串时才允许覆盖。",
        ha="center",
        va="center",
        fontsize=6.15,
        color=COLORS["muted"],
    )
    fig.text(
        0.5,
        0.035,
        "论文未说明 F1 averaging；Table 6 的 Qwen 为 1,000 样本 PEFT 独立分类器。跨论文差值仅作描述性/功能性对照，不等同于公平消融。",
        ha="center",
        va="center",
        fontsize=6.15,
        color=COLORS["muted"],
    )
    return fig


def export_and_qa(fig: plt.Figure, source: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_path = FIGURES / "source_data_fig9.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig")
    base = FIGURES / "fig9_qwen_gate_comparison"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    image = Image.open(base.with_suffix(".png")).convert("RGB")
    pixels = np.asarray(image)
    nonwhite_fraction = float(np.mean(np.any(pixels < 248, axis=2)))
    text_nodes = ET.parse(base.with_suffix(".svg")).findall(".//{http://www.w3.org/2000/svg}text")
    if image.width < 1800 or image.height < 950:
        raise AssertionError(f"PNG preview is unexpectedly small: {image.size}")
    if nonwhite_fraction < 0.08:
        raise AssertionError("Figure preview is nearly blank")
    if len(text_nodes) < 35:
        raise AssertionError("SVG text is not preserved as editable text")

    qa_lines = [
        "Figure: fig9_qwen_gate_comparison",
        "Backend: Python/matplotlib only",
        "Archetype: asymmetric quantitative grid; panel a is hero evidence",
        "Core conclusion: selective evidence-constrained Qwen raises the local Macro-F1 point estimate from 80.60% to 80.94%, but both paired-bootstrap intervals for the +0.34 pp increment cross zero.",
        "Cross-paper descriptive comparison: 80.94% is +1.71 pp above Paper Table 5 SPT-RII 70-shot F1 (79.23%); paper F1 averaging is unspecified.",
        "Final size: 7.2 x 4.2 inches (approximately 183 x 107 mm) before tight bounding-box export",
        "Panel a: Paper Table 6 Qwen PEFT, Paper Table 5 SPT-RII, local no-LLM student, local selective-Qwen full method",
        "Panel b: 9,069 test rows; 1,594 routed; 1,567 consensus; 648 changed",
        "Panel c: 349 corrected, 299 harmed, net +50 correct rows",
        "Panel d: 5,000 paired row bootstrap and 5,000 live_id-cluster bootstrap resamples; 12 live_id clusters",
        "Metric: local Macro-F1; paper F1 averaging unspecified",
        "Split: fixed 140-row train, 140-row validation, and 9,069-row test; validation-only margin selection",
        "Baseline: same-label five-model DAPT+R-Drop ensemble",
        "LLM: qwen3:8b Q4_K_M; evidence-constrained selective verifier; no PEFT",
        f"PNG dimensions: {image.width} x {image.height}",
        f"PNG non-white fraction: {nonwhite_fraction:.4f}",
        f"Editable SVG text nodes: {len(text_nodes)}",
        "Source data: source_data_fig9.csv",
        "Image integrity: vector-native plotting; no raster manipulation",
        "Reviewer risk: the +0.34 pp interval crosses zero; cross-paper differences are descriptive; Table 6 is not a matched-budget comparison.",
    ]
    (FIGURES / "fig9_qa.txt").write_text("\n".join(qa_lines) + "\n", encoding="utf-8")


def main() -> None:
    result, analysis = load_locked_results()
    source = build_source_data(result, analysis)
    figure = draw_figure(source)
    export_and_qa(figure, source)
    print("Wrote fig9_qwen_gate_comparison.{svg,pdf,png,tiff}")
    print("Wrote source_data_fig9.csv and fig9_qa.txt")


if __name__ == "__main__":
    main()
