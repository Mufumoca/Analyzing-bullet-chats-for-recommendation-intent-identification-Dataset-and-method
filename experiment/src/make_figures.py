from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

from common import DATA_DIR, FIGURES_DIR, LOGS_DIR, RESULTS_DIR, bootstrap_ci, ensure_dirs


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Microsoft YaHei", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7.5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7


COLORS = {
    "content": "#484878",
    "context": "#A8A8A8",
    "qwen": "#C56A83",
    "official": "#42949E",
    "direct": "#8A6FA8",
    "positive": "#2E9E44",
    "negative": "#D04A45",
    "ink": "#272727",
    "muted": "#767676",
    "light": "#D8D8D8",
}

LABELS = {
    "content": "Current chat",
    "content_context": "+ raw context",
    "context": "+ raw context",
    "content_qwen": "+ Qwen explanation",
    "qwen": "+ Qwen explanation",
    "official": "+ official explanation",
    "content_official": "+ official explanation",
    "official_only": "Official explanation only",
    "qwen_direct": "Qwen direct",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def export_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.tiff", dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, values, digits: int = 3, upper_errors=None) -> None:
    if upper_errors is None:
        upper_errors = np.zeros(len(values), dtype=float)
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for bar, value, upper_error in zip(bars, values, upper_errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + upper_error + 0.015 * y_range,
            f"{value:.{digits}f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def figure_workflow() -> None:
    stats = load_json(DATA_DIR / "dataset_stats.json")
    environment = load_json(RESULTS_DIR / "environment.json")
    qwen_train = load_json(RESULTS_DIR / "qwen_generation_train.json")
    qwen_test = load_json(RESULTS_DIR / "qwen_generation_test.json")
    qwen_direct = load_json(RESULTS_DIR / "qwen_direct.json")
    roberta = pd.read_csv(RESULTS_DIR / "roberta_summary.csv", encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 5.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.42, wspace=0.34)
    ax_flow = fig.add_subplot(grid[0, :])
    ax_scale = fig.add_subplot(grid[1, 0])
    ax_memory = fig.add_subplot(grid[1, 1])

    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    box_specs = [
        (0.01, 0.31, 0.16, 0.42, "Released RecDY\n30,228 rows", "#E7E8F3"),
        (0.215, 0.31, 0.16, 0.42, "Fixed scales\n2,000 / 400", "#E7E8F3"),
        (0.42, 0.31, 0.16, 0.42, "Previous 5 chats\nchronological", "#ECECEC"),
        (0.625, 0.31, 0.16, 0.42, "Qwen3-8B Q4\none-sentence gloss", "#F3DDE4"),
        (0.83, 0.31, 0.16, 0.42, "TF-IDF / RoBERTa\n+ direct Qwen", "#DDEEEF"),
    ]
    for x, y, width, height, text, color in box_specs:
        ax_flow.add_patch(Rectangle((x, y), width, height, facecolor=color, edgecolor="#5A5A5A", lw=0.8))
        ax_flow.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=7.2)
    for index in range(len(box_specs) - 1):
        x_start = box_specs[index][0] + box_specs[index][2]
        x_end = box_specs[index + 1][0]
        ax_flow.add_patch(
            FancyArrowPatch(
                (x_start + 0.008, 0.52),
                (x_end - 0.008, 0.52),
                arrowstyle="-|>",
                mutation_scale=9,
                lw=0.9,
                color=COLORS["ink"],
            )
        )
    ax_flow.text(
        0.5,
        0.93,
        "Scaled explanation-augmentation experiment on a single 8 GB GPU",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    ax_flow.text(
        0.5,
        0.11,
        "No label or official explanation is passed to Qwen; test evaluation occurs once after training.",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    add_panel_label(ax_flow, "a")

    scale_labels = ["Released split", "Main subset", "Qwen subset"]
    train_values = [stats["platforms"]["douyin"]["train"]["rows"], 2000, 400]
    test_values = [stats["platforms"]["douyin"]["test"]["rows"], 2000, 400]
    y = np.arange(len(scale_labels))
    ax_scale.barh(y + 0.17, train_values, height=0.30, color=COLORS["content"], label="Train")
    ax_scale.barh(y - 0.17, test_values, height=0.30, color="#9BA4CB", label="Test")
    ax_scale.set_xscale("log")
    ax_scale.set_yticks(y, scale_labels)
    ax_scale.invert_yaxis()
    ax_scale.set_xlabel("Rows (log scale)")
    ax_scale.legend(loc="lower right", fontsize=6.5)
    ax_scale.set_title("Data scales")
    add_panel_label(ax_scale, "b")

    roberta_peak = roberta["peak_torch_memory_mb_mean"].max() / 1024
    memory_values = [
        max(qwen_train["peak_gpu_memory_mb"], qwen_test["peak_gpu_memory_mb"]) / 1024,
        qwen_direct["peak_gpu_memory_mb"] / 1024,
        roberta_peak,
    ]
    memory_labels = ["Qwen explanation", "Qwen direct", "RoBERTa training"]
    colors = [COLORS["qwen"], COLORS["direct"], COLORS["official"]]
    bars = ax_memory.bar(np.arange(3), memory_values, color=colors, edgecolor="white", width=0.68)
    ax_memory.axhline(
        environment["gpu"]["memory_mb"] / 1024,
        color=COLORS["negative"],
        ls="--",
        lw=1.0,
    )
    ax_memory.set_xticks(np.arange(3), memory_labels, rotation=18, ha="right")
    ax_memory.set_ylabel("Peak memory (GiB)")
    ax_memory.set_ylim(0, 8.7)
    annotate_bars(ax_memory, bars, memory_values, digits=2)
    ax_memory.text(2.45, 8.10, "8 GB device limit", ha="right", va="bottom", fontsize=6.2)
    ax_memory.set_title("Observed GPU envelope\nQwen: NVML total; RoBERTa: PyTorch allocated", fontsize=7.5)
    add_panel_label(ax_memory, "c")

    source = pd.DataFrame(
        {
            "stage": memory_labels,
            "peak_memory_gib": memory_values,
            "measurement": ["NVML total", "NVML total", "PyTorch allocated"],
        }
    )
    source.to_csv(FIGURES_DIR / "source_data_fig1.csv", index=False, encoding="utf-8-sig")
    fig.suptitle("Experimental design and resource boundary", fontsize=10.5, fontweight="bold", y=0.995)
    export_figure(fig, "fig1_workflow")


def figure_classical() -> None:
    classical = pd.read_csv(RESULTS_DIR / "classical_summary.csv", encoding="utf-8-sig")
    paired = pd.read_csv(RESULTS_DIR / "paired_bootstrap.csv", encoding="utf-8-sig")
    direct = load_json(RESULTS_DIR / "qwen_direct.json")
    direct_predictions = pd.read_csv(RESULTS_DIR / "predictions_qwen_direct.csv", encoding="utf-8-sig")
    direct_low, direct_high = bootstrap_ci(
        direct_predictions["label"].astype(int),
        direct_predictions["prediction"].astype(int),
        metric="macro_f1",
        iterations=1000,
    )

    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.88], hspace=0.46, wspace=0.34)
    ax_full = fig.add_subplot(grid[0, 0])
    ax_qwen = fig.add_subplot(grid[0, 1])
    ax_delta = fig.add_subplot(grid[1, :])

    full_conditions = ["content", "official_only", "content_official"]
    full_rows = classical[(classical["suite"] == "full")].set_index("condition").loc[full_conditions]
    full_values = full_rows["macro_f1"].to_numpy()
    full_low = full_rows["macro_f1_ci95_low"].to_numpy()
    full_high = full_rows["macro_f1_ci95_high"].to_numpy()
    full_colors = [COLORS["content"], "#93C6CB", COLORS["official"]]
    x = np.arange(len(full_conditions))
    bars = ax_full.bar(
        x,
        full_values,
        yerr=np.vstack([full_values - full_low, full_high - full_values]),
        color=full_colors,
        capsize=3,
        width=0.68,
    )
    ax_full.set_xticks(x, [LABELS[value] for value in full_conditions], rotation=18, ha="right")
    ax_full.set_ylabel("Macro-F1")
    ax_full.set_ylim(0.82, 0.91)
    ax_full.set_title("Full released RecDY (21,159 / 9,069)")
    annotate_bars(ax_full, bars, full_values, upper_errors=full_high - full_values)
    add_panel_label(ax_full, "a")

    qwen_conditions = ["content", "content_context", "content_qwen", "content_official"]
    qwen_rows = classical[(classical["suite"] == "qwen")].set_index("condition").loc[qwen_conditions]
    qwen_values = qwen_rows["macro_f1"].tolist() + [direct["macro_f1"]]
    qwen_low = qwen_rows["macro_f1_ci95_low"].tolist() + [direct_low]
    qwen_high = qwen_rows["macro_f1_ci95_high"].tolist() + [direct_high]
    qwen_names = qwen_conditions + ["qwen_direct"]
    qwen_colors = [
        COLORS["content"],
        COLORS["context"],
        COLORS["qwen"],
        COLORS["official"],
        COLORS["direct"],
    ]
    x = np.arange(len(qwen_names))
    bars = ax_qwen.bar(
        x,
        qwen_values,
        yerr=np.vstack([np.asarray(qwen_values) - qwen_low, np.asarray(qwen_high) - qwen_values]),
        color=qwen_colors,
        capsize=2.5,
        width=0.70,
    )
    ax_qwen.set_xticks(x, [LABELS[value] for value in qwen_names], rotation=24, ha="right")
    ax_qwen.set_ylabel("Macro-F1")
    ax_qwen.set_ylim(0.55, 0.88)
    ax_qwen.set_title("Fixed Qwen subset (400 / 400)")
    annotate_bars(
        ax_qwen,
        bars,
        qwen_values,
        upper_errors=np.asarray(qwen_high) - np.asarray(qwen_values),
    )
    add_panel_label(ax_qwen, "b")

    forest_specs = [
        ("Full: + official explanation", "full", "content_official"),
        ("400: + raw context", "qwen", "content_context"),
        ("400: + Qwen explanation", "qwen", "content_qwen"),
        ("400: + official explanation", "qwen", "content_official"),
    ]
    estimates = []
    lows = []
    highs = []
    labels = []
    colors = []
    for label, suite, comparison in forest_specs:
        row = paired[
            (paired["suite"] == suite)
            & (paired["comparison"] == comparison)
            & (paired["metric"] == "macro_f1")
        ].iloc[0]
        labels.append(label)
        estimates.append(row["difference"] * 100)
        lows.append(row["ci95_low"] * 100)
        highs.append(row["ci95_high"] * 100)
        colors.append(COLORS["positive"] if row["difference"] >= 0 else COLORS["negative"])
    y = np.arange(len(labels))[::-1]
    for yi, estimate, low, high, color in zip(y, estimates, lows, highs, colors):
        ax_delta.plot([low, high], [yi, yi], color=color, lw=1.5)
        ax_delta.scatter(estimate, yi, s=24, color=color, zorder=3)
    ax_delta.axvline(0, color=COLORS["muted"], ls="--", lw=0.9)
    ax_delta.set_yticks(y, labels)
    ax_delta.set_xlabel("Paired Macro-F1 difference vs current chat (percentage points; 95% row bootstrap CI)")
    ax_delta.set_xlim(-24, 10)
    ax_delta.set_title("Compressed explanations avoid the severe raw-context degradation")
    add_panel_label(ax_delta, "c")

    source = pd.DataFrame(
        {
            "comparison": labels,
            "difference_pp": estimates,
            "ci95_low_pp": lows,
            "ci95_high_pp": highs,
        }
    )
    source.to_csv(FIGURES_DIR / "source_data_fig2.csv", index=False, encoding="utf-8-sig")
    fig.suptitle("Classical text classifier: performance and paired uncertainty", fontsize=10.5, fontweight="bold", y=0.995)
    export_figure(fig, "fig2_classical_results")


def figure_roberta() -> None:
    summary = pd.read_csv(RESULTS_DIR / "roberta_summary.csv", encoding="utf-8-sig")
    runs = pd.read_csv(RESULTS_DIR / "roberta_runs.csv", encoding="utf-8-sig")

    fig = plt.figure(figsize=(7.2, 4.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.86], hspace=0.47, wspace=0.34)
    ax_main = fig.add_subplot(grid[0, 0])
    ax_small = fig.add_subplot(grid[0, 1])
    ax_delta = fig.add_subplot(grid[1, :])

    main_conditions = ["content", "official"]
    main = summary[(summary["suite"] == "main")].set_index("condition").loc[main_conditions]
    values = main["macro_f1_mean"].to_numpy()
    errors = main["macro_f1_sd"].to_numpy()
    x = np.arange(len(main_conditions))
    bars = ax_main.bar(
        x,
        values,
        yerr=errors,
        color=[COLORS["content"], COLORS["official"]],
        capsize=4,
        width=0.62,
    )
    ax_main.set_xticks(x, [LABELS[value] for value in main_conditions], rotation=12, ha="right")
    ax_main.set_ylabel("Macro-F1 (mean ± SD)")
    ax_main.set_ylim(0.84, 0.90)
    ax_main.set_title("2,000 / 2,000; n=3 seeds")
    annotate_bars(ax_main, bars, values, upper_errors=errors)
    add_panel_label(ax_main, "a")

    small_conditions = ["content", "context", "qwen", "official"]
    small = summary[(summary["suite"] == "qwen")].set_index("condition").loc[small_conditions]
    values = small["macro_f1_mean"].to_numpy()
    errors = small["macro_f1_sd"].to_numpy()
    x = np.arange(len(small_conditions))
    bars = ax_small.bar(
        x,
        values,
        yerr=errors,
        color=[COLORS["content"], COLORS["context"], COLORS["qwen"], COLORS["official"]],
        capsize=3,
        width=0.68,
    )
    ax_small.set_xticks(x, [LABELS[value] for value in small_conditions], rotation=22, ha="right")
    ax_small.set_ylabel("Macro-F1 (mean ± SD)")
    ax_small.set_ylim(0.58, 0.88)
    ax_small.set_title("400 / 400; n=3 seeds")
    annotate_bars(ax_small, bars, values, upper_errors=errors)
    add_panel_label(ax_small, "b")

    comparison_specs = [
        ("Main\nofficial", "main", "official", COLORS["official"]),
        ("400\nraw context", "qwen", "context", COLORS["context"]),
        ("400\nQwen", "qwen", "qwen", COLORS["qwen"]),
        ("400\nofficial", "qwen", "official", COLORS["official"]),
    ]
    for position, (label, suite, condition, color) in enumerate(comparison_specs):
        baseline = runs[(runs["suite"] == suite) & (runs["condition"] == "content")].set_index("seed")
        comparison = runs[(runs["suite"] == suite) & (runs["condition"] == condition)].set_index("seed")
        paired = comparison["macro_f1"] - baseline["macro_f1"]
        jitter = np.linspace(-0.06, 0.06, len(paired))
        ax_delta.scatter(
            np.full(len(paired), position) + jitter,
            paired.to_numpy() * 100,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            s=30,
            zorder=3,
        )
        ax_delta.plot(
            [position - 0.16, position + 0.16],
            [paired.mean() * 100, paired.mean() * 100],
            color=COLORS["ink"],
            lw=1.3,
        )
    ax_delta.axhline(0, color=COLORS["muted"], ls="--", lw=0.9)
    ax_delta.set_xticks(np.arange(len(comparison_specs)), [value[0] for value in comparison_specs])
    ax_delta.set_ylabel("Paired Macro-F1 difference (pp)")
    ax_delta.set_title("Seed-wise effect relative to current-chat input")
    ax_delta.set_ylim(-26, 5)
    add_panel_label(ax_delta, "c")

    source = runs[["suite", "condition", "seed", "macro_f1", "elapsed_seconds", "peak_torch_memory_mb"]]
    source.to_csv(FIGURES_DIR / "source_data_fig3.csv", index=False, encoding="utf-8-sig")
    fig.suptitle("Chinese-RoBERTa: explanation gains depend on training scale", fontsize=10.5, fontweight="bold", y=0.995)
    export_figure(fig, "fig3_roberta_results")


def read_jsonl(path: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    latest = {str(record["row_id"]): record for record in records}
    return pd.DataFrame(latest.values())


def figure_efficiency() -> None:
    generation = pd.concat(
        [
            read_jsonl(LOGS_DIR / "qwen_generation_train.jsonl").assign(stage="Explanation"),
            read_jsonl(LOGS_DIR / "qwen_generation_test.jsonl").assign(stage="Explanation"),
        ],
        ignore_index=True,
    )
    direct_predictions = pd.read_csv(RESULTS_DIR / "predictions_qwen_direct.csv", encoding="utf-8-sig")
    direct_predictions["stage"] = "Direct classification"
    qwen_train = load_json(RESULTS_DIR / "qwen_generation_train.json")
    qwen_test = load_json(RESULTS_DIR / "qwen_generation_test.json")
    qwen_direct = load_json(RESULTS_DIR / "qwen_direct.json")
    roberta = pd.read_csv(RESULTS_DIR / "roberta_summary.csv", encoding="utf-8-sig")
    explanation_lengths = pd.concat(
        [
            pd.read_csv(DATA_DIR / "douyin_train_qwen_generated.csv", encoding="utf-8-sig"),
            pd.read_csv(DATA_DIR / "douyin_test_qwen_generated.csv", encoding="utf-8-sig"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3))
    ax_latency, ax_wall, ax_memory, ax_length = axes.flat

    latency_data = [
        generation["generation_seconds"].to_numpy(),
        direct_predictions["inference_seconds"].to_numpy(),
    ]
    box = ax_latency.boxplot(
        latency_data,
        tick_labels=["Explanation", "Direct class."],
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], [COLORS["qwen"], COLORS["direct"]]):
        patch.set_facecolor(color)
    ax_latency.set_ylabel("Seconds per chat")
    ax_latency.set_yscale("log")
    ax_latency.set_title("Qwen latency distribution")
    add_panel_label(ax_latency, "a")

    qwen_group = roberta[roberta["suite"] == "qwen"]
    main_group = roberta[roberta["suite"] == "main"]
    wall_labels = ["800 explanations", "400 direct", "RoBERTa 400", "RoBERTa 2,000"]
    wall_values = [
        qwen_train["wall_seconds"] + qwen_test["wall_seconds"],
        qwen_direct["wall_seconds"],
        qwen_group["elapsed_seconds_mean"].mean(),
        main_group["elapsed_seconds_mean"].mean(),
    ]
    bars = ax_wall.barh(
        np.arange(4),
        wall_values,
        color=[COLORS["qwen"], COLORS["direct"], "#87BFC4", COLORS["official"]],
    )
    ax_wall.set_yticks(np.arange(4), wall_labels)
    ax_wall.invert_yaxis()
    ax_wall.set_xscale("log")
    ax_wall.set_xlabel("Wall time per recorded stage (s; log scale)")
    ax_wall.set_title("End-to-end stage time")
    for bar, value in zip(bars, wall_values):
        ax_wall.text(value * 1.05, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=6.2)
    add_panel_label(ax_wall, "b")

    memory_labels = ["Qwen explain", "Qwen direct", "RoBERTa 400", "RoBERTa 2,000"]
    memory_values = [
        max(qwen_train["peak_gpu_memory_mb"], qwen_test["peak_gpu_memory_mb"]) / 1024,
        qwen_direct["peak_gpu_memory_mb"] / 1024,
        qwen_group["peak_torch_memory_mb_mean"].max() / 1024,
        main_group["peak_torch_memory_mb_mean"].max() / 1024,
    ]
    bars = ax_memory.bar(
        np.arange(4),
        memory_values,
        color=[COLORS["qwen"], COLORS["direct"], "#87BFC4", COLORS["official"]],
        width=0.68,
    )
    ax_memory.axhline(8.0, color=COLORS["negative"], ls="--", lw=0.9)
    ax_memory.set_xticks(np.arange(4), memory_labels, rotation=20, ha="right")
    ax_memory.set_ylabel("Peak memory (GiB)")
    ax_memory.set_ylim(0, 8.7)
    annotate_bars(ax_memory, bars, memory_values, digits=2)
    ax_memory.set_title(
        "All stages fit the 8 GB device\nQwen: NVML total; RoBERTa: PyTorch allocated",
        fontsize=7.5,
    )
    add_panel_label(ax_memory, "c")

    length_data = [
        explanation_lengths["content"].astype(str).str.len().to_numpy(),
        explanation_lengths["qwen_background"].astype(str).str.len().to_numpy(),
        explanation_lengths["official_background"].astype(str).str.len().to_numpy(),
    ]
    box = ax_length.boxplot(
        length_data,
        tick_labels=["Chat", "Qwen gloss", "Official gloss"],
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], [COLORS["content"], COLORS["qwen"], COLORS["official"]]):
        patch.set_facecolor(color)
    ax_length.set_ylabel("Chinese characters")
    ax_length.set_title("Explanation compression length")
    add_panel_label(ax_length, "d")

    source = pd.DataFrame(
        {
            "stage": wall_labels,
            "wall_seconds": wall_values,
            "peak_memory_gib": memory_values,
        }
    )
    source.to_csv(FIGURES_DIR / "source_data_fig4.csv", index=False, encoding="utf-8-sig")
    fig.suptitle("Runtime, memory and generated-text characteristics", fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(pad=1.5)
    export_figure(fig, "fig4_efficiency")


def main() -> None:
    ensure_dirs()
    figure_workflow()
    figure_classical()
    figure_roberta()
    figure_efficiency()
    print(f"Figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
