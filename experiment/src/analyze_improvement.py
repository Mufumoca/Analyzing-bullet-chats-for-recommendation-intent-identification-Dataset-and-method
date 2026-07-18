from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import RESULTS_DIR, dump_json
from improvement_common import IMPROVEMENT_RESULTS_DIR, SEED, ensure_improvement_dirs


BOOTSTRAP_ITERATIONS = 5000
CLUSTER_BOOTSTRAP_ITERATIONS = 1000


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def paired_bootstrap(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, object]:
    if set(baseline["row_id"]) != set(comparison["row_id"]):
        raise ValueError("Paired predictions do not contain the same row_id set")
    ordered = baseline[["row_id", "label", "prediction"]].merge(
        comparison[["row_id", "prediction"]].rename(columns={"prediction": "comparison_prediction"}),
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    y = ordered["label"].to_numpy(dtype=int)
    base = ordered["prediction"].to_numpy(dtype=int)
    comp = ordered["comparison_prediction"].to_numpy(dtype=int)
    point = macro_f1(y, comp) - macro_f1(y, base)
    rng = np.random.default_rng(SEED)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(y), len(y))
        values[index] = macro_f1(y[sample], comp[sample]) - macro_f1(y[sample], base[sample])
    return {
        "rows": int(len(y)),
        "difference": float(point),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "iterations": iterations,
        "fraction_above_zero": float(np.mean(values > 0)),
    }


def cluster_bootstrap(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    iterations: int = CLUSTER_BOOTSTRAP_ITERATIONS,
) -> dict[str, object]:
    merged = baseline[["row_id", "live_id", "label", "prediction"]].merge(
        comparison[["row_id", "prediction"]].rename(columns={"prediction": "comparison_prediction"}),
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    group_indices = [group.index.to_numpy() for _, group in merged.groupby("live_id", sort=True)]
    y = merged["label"].to_numpy(dtype=int)
    base = merged["prediction"].to_numpy(dtype=int)
    comp = merged["comparison_prediction"].to_numpy(dtype=int)
    point = macro_f1(y, comp) - macro_f1(y, base)
    rng = np.random.default_rng(SEED + 1)
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.integers(0, len(group_indices), len(group_indices))
        sample_indices = np.concatenate([group_indices[item] for item in selected])
        values[index] = macro_f1(y[sample_indices], comp[sample_indices]) - macro_f1(
            y[sample_indices], base[sample_indices]
        )
    return {
        "clusters": len(group_indices),
        "difference": float(point),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
        "iterations": iterations,
    }


def load_prediction(size: int, condition: str) -> pd.DataFrame:
    path = IMPROVEMENT_RESULTS_DIR / f"predictions_classical_{size}_{condition}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def summarize_roberta() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for path in sorted(IMPROVEMENT_RESULTS_DIR.glob("roberta_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "macro_f1" not in record and "test_fused_metrics" in record:
            record = {
                **record,
                "macro_f1": record["test_fused_metrics"]["macro_f1"],
                "accuracy": record["test_fused_metrics"]["accuracy"],
                "elapsed_seconds": sum(
                    branch.get("elapsed_seconds", 0.0)
                    for branch in record.get("test_branch_runs", {}).values()
                ),
                "peak_torch_memory_mb": max(
                    (branch.get("peak_torch_memory_mb", 0.0)
                     for branch in record.get("test_branch_runs", {}).values()),
                    default=0.0,
                ),
                "fusion_run": True,
            }
        else:
            record["fusion_run"] = False
        records.append(record)
    if not records:
        return pd.DataFrame(), pd.DataFrame()
    runs = pd.DataFrame(records)
    summary = (
        runs.groupby("tag", as_index=False)
        .agg(
            seeds=("seed", lambda values: ",".join(str(int(value)) for value in sorted(values))),
            runs=("seed", "count"),
            train_rows=("train_rows", "first"),
            test_rows=("test_rows", "first"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            peak_torch_memory_mb_mean=("peak_torch_memory_mb", "mean"),
            fusion_runs=("fusion_run", "sum"),
        )
    )
    comparisons: list[dict[str, object]] = []

    def prediction_path(tag: str, seed: int) -> Path:
        return IMPROVEMENT_RESULTS_DIR / f"predictions_roberta_{tag}_seed{seed}.csv"

    def paired_row(base_path: Path, comp_path: Path) -> dict[str, object]:
        base = pd.read_csv(base_path, encoding="utf-8-sig")
        comp = pd.read_csv(comp_path, encoding="utf-8-sig")
        merged = base[["row_id", "label", "prediction"]].merge(
            comp[["row_id", "prediction"]].rename(columns={"prediction": "comparison_prediction"}),
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
        y = merged["label"].to_numpy(dtype=int)
        base_pred = merged["prediction"].to_numpy(dtype=int)
        comp_pred = merged["comparison_prediction"].to_numpy(dtype=int)
        return {
            "rows": len(merged),
            "difference": macro_f1(y, comp_pred) - macro_f1(y, base_pred),
        }

    for seed in (100, 101, 102):
        # Existing 400-row baselines are frozen; the new structured run is
        # compared on exactly the same row ids and seed.
        old_content = RESULTS_DIR / f"predictions_roberta_qwen_content_seed{seed}.csv"
        old_qwen = RESULTS_DIR / f"predictions_roberta_qwen_qwen_seed{seed}.csv"
        new_400 = prediction_path("qwen400_structured_concat", seed)
        if old_content.exists() and new_400.exists():
            row = paired_row(old_content, new_400)
            row.update({"comparison": "structured400_vs_content400", "seed": seed})
            comparisons.append(row)
        if old_qwen.exists() and new_400.exists():
            row = paired_row(old_qwen, new_400)
            row.update({"comparison": "structured400_vs_old_qwen400", "seed": seed})
            comparisons.append(row)
        content_1000 = prediction_path("content1000", seed)
        new_1000 = prediction_path("qwen1000_structured_concat", seed)
        if content_1000.exists() and new_1000.exists():
            row = paired_row(content_1000, new_1000)
            row.update({"comparison": "structured1000_vs_content1000", "seed": seed})
            comparisons.append(row)
        compact_400 = prediction_path("qwen400_compact_concat", seed)
        fusion_400 = prediction_path("qwen400_compact_late_fusion", seed)
        if old_content.exists() and compact_400.exists():
            row = paired_row(old_content, compact_400)
            row.update({"comparison": "compact400_vs_content400", "seed": seed})
            comparisons.append(row)
        if compact_400.exists() and fusion_400.exists():
            row = paired_row(compact_400, fusion_400)
            row.update({"comparison": "late_fusion400_vs_compact400", "seed": seed})
            comparisons.append(row)
        if old_content.exists() and fusion_400.exists():
            row = paired_row(old_content, fusion_400)
            row.update({"comparison": "late_fusion400_vs_content400", "seed": seed})
            comparisons.append(row)
        compact_1000 = prediction_path("qwen1000_compact_concat", seed)
        fusion_1000 = prediction_path("qwen1000_compact_late_fusion", seed)
        if content_1000.exists() and compact_1000.exists():
            row = paired_row(content_1000, compact_1000)
            row.update({"comparison": "compact1000_vs_content1000", "seed": seed})
            comparisons.append(row)
        if compact_1000.exists() and fusion_1000.exists():
            row = paired_row(compact_1000, fusion_1000)
            row.update({"comparison": "late_fusion1000_vs_compact1000", "seed": seed})
            comparisons.append(row)
        if content_1000.exists() and fusion_1000.exists():
            row = paired_row(content_1000, fusion_1000)
            row.update({"comparison": "late_fusion1000_vs_content1000", "seed": seed})
            comparisons.append(row)
    paired = pd.DataFrame(comparisons)
    return summary, paired


def compare(size: int, baseline_name: str, comparison_name: str, label: str) -> dict[str, object]:
    baseline = load_prediction(size, baseline_name)
    comparison = load_prediction(size, comparison_name)
    row = paired_bootstrap(baseline, comparison)
    cluster = cluster_bootstrap(baseline, comparison)
    row.update({"size": size, "baseline": baseline_name, "comparison": comparison_name, "label": label})
    row["cluster_ci95_low"] = cluster["ci95_low"]
    row["cluster_ci95_high"] = cluster["ci95_high"]
    row["cluster_count"] = cluster["clusters"]
    return row


def main() -> None:
    ensure_improvement_dirs()
    comparisons = [
        (400, "content", "old_concat", "old prompt: explanation vs current"),
        (400, "old_concat", "structured_concat", "prompt change: structured vs old"),
        (400, "old_concat", "old_late_fusion", "fusion change: late fusion vs concat"),
        (400, "old_concat", "structured_late_fusion", "primary system: structured + late fusion vs old concat"),
        (400, "content", "old_late_fusion", "old prompt + late fusion vs current"),
        (400, "content", "structured_late_fusion", "new prompt + late fusion vs current"),
        (1000, "content", "structured_concat", "1,000 train: structured explanation vs current"),
        (1000, "content", "structured_late_fusion", "1,000 train: structured late fusion vs current"),
    ]
    rows = [compare(*spec) for spec in comparisons if (IMPROVEMENT_RESULTS_DIR / f"predictions_classical_{spec[0]}_{spec[1]}.csv").exists() and (IMPROVEMENT_RESULTS_DIR / f"predictions_classical_{spec[0]}_{spec[2]}.csv").exists()]
    frame = pd.DataFrame(rows)
    frame.to_csv(IMPROVEMENT_RESULTS_DIR / "improvement_comparisons.csv", index=False, encoding="utf-8-sig")

    did_specs = [
        (400, "old_concat", "structured_concat", "difference_in_differences_concat"),
        (400, "old_late_fusion", "structured_late_fusion", "difference_in_differences_late_fusion"),
    ]
    did_rows = [
        compare(*spec)
        for spec in did_specs
        if (IMPROVEMENT_RESULTS_DIR / f"predictions_classical_{spec[0]}_{spec[1]}.csv").exists()
        and (IMPROVEMENT_RESULTS_DIR / f"predictions_classical_{spec[0]}_{spec[2]}.csv").exists()
    ]
    if did_rows:
        pd.DataFrame(did_rows).to_csv(
            IMPROVEMENT_RESULTS_DIR / "difference_in_differences.csv",
            index=False,
            encoding="utf-8-sig",
        )

    generation: list[dict[str, object]] = []
    for path in sorted(IMPROVEMENT_RESULTS_DIR.glob("qwen_generation_*.json")):
        generation.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "comparisons": rows,
        "difference_in_differences": did_rows,
        "generation": generation,
    }
    roberta_summary, roberta_paired = summarize_roberta()
    if not roberta_summary.empty:
        roberta_summary.to_csv(
            IMPROVEMENT_RESULTS_DIR / "roberta_improvement_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        roberta_paired.to_csv(
            IMPROVEMENT_RESULTS_DIR / "roberta_improvement_paired.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary["roberta_summary"] = roberta_summary.to_dict(orient="records")
        summary["roberta_paired"] = roberta_paired.to_dict(orient="records")
    dump_json(IMPROVEMENT_RESULTS_DIR / "improvement_analysis.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
