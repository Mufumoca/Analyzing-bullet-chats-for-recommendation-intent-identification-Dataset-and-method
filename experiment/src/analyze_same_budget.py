"""Verify and statistically analyze the strict 70-shot factorial experiment."""

from __future__ import annotations

import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from common import DATA_DIR, RESULTS_DIR, dump_json


RESULTS = RESULTS_DIR / "same_budget"
PRESETS = ("base_ce", "dapt_ce", "base_rdrop", "dapt_rdrop")
TAGS = {
    "base_ce": "strict70_base_ce_5seed_v1",
    "dapt_ce": "strict70_dapt_ce_5seed_v1",
    "base_rdrop": "strict70_base_rdrop_5seed_v1",
    "dapt_rdrop": "strict70_dapt_rdrop_5seed_v1",
}
LABELS = {
    "base_ce": "Base RoBERTa + CE",
    "dapt_ce": "DAPT RoBERTa + CE",
    "base_rdrop": "Base RoBERTa + R-Drop",
    "dapt_rdrop": "DAPT RoBERTa + R-Drop",
}
PAPER_F1 = 0.7923
EXPECTED_TRAIN_SIGNATURE = "f493253c6d9b2b97c430b860300fd4c0969288e416300b5061e23175d805ec7b"
EXPECTED_VALIDATION_SIGNATURE = "0725f0801089f5c333bf827cda0fe21ab497f74f76b55379f3a4033999c78acd"
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 20260719


def macro_f1_fast(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    predictions = predictions.astype(bool, copy=False)
    tp = int(np.count_nonzero(labels & predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    positive_denominator = 2 * tp + fp + fn
    negative_denominator = 2 * tn + fp + fn
    positive_f1 = 2 * tp / positive_denominator if positive_denominator else 0.0
    negative_f1 = 2 * tn / negative_denominator if negative_denominator else 0.0
    return 0.5 * (positive_f1 + negative_f1)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def row_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    improved: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    baseline_scores = np.empty(iterations, dtype=float)
    improved_scores = np.empty(iterations, dtype=float)
    n = len(labels)
    for iteration in range(iterations):
        indices = rng.integers(0, n, size=n)
        sampled_labels = labels[indices]
        baseline_scores[iteration] = macro_f1_fast(sampled_labels, baseline[indices])
        improved_scores[iteration] = macro_f1_fast(sampled_labels, improved[indices])
    deltas = improved_scores - baseline_scores
    baseline_ci = percentile_interval(baseline_scores)
    improved_ci = percentile_interval(improved_scores)
    delta_ci = percentile_interval(deltas)
    return {
        "unit": "row",
        "iterations": iterations,
        "seed": seed,
        "baseline_ci95": list(baseline_ci),
        "improved_ci95": list(improved_ci),
        "delta_ci95": list(delta_ci),
        "probability_delta_le_zero": float(np.mean(deltas <= 0)),
    }


def cluster_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    improved: np.ndarray,
    clusters: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    unique_clusters = np.unique(clusters)
    indices_by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    baseline_scores = np.empty(iterations, dtype=float)
    improved_scores = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        indices = np.concatenate([indices_by_cluster[cluster] for cluster in sampled_clusters])
        sampled_labels = labels[indices]
        baseline_scores[iteration] = macro_f1_fast(sampled_labels, baseline[indices])
        improved_scores[iteration] = macro_f1_fast(sampled_labels, improved[indices])
        deltas[iteration] = improved_scores[iteration] - baseline_scores[iteration]
    return {
        "unit": "live_id_cluster",
        "clusters": int(len(unique_clusters)),
        "iterations": iterations,
        "seed": seed,
        "baseline_ci95": list(percentile_interval(baseline_scores)),
        "improved_ci95": list(percentile_interval(improved_scores)),
        "delta_ci95": list(percentile_interval(deltas)),
        "probability_delta_le_zero": float(np.mean(deltas <= 0)),
    }


def load_result(preset: str) -> tuple[dict[str, Any], pd.DataFrame]:
    tag = TAGS[preset]
    result_path = RESULTS / f"{tag}_result.json"
    prediction_path = RESULTS / f"{tag}_predictions.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path, encoding="utf-8-sig", dtype={"row_id": str, "live_id": str})
    return result, predictions


def verify_protocols(results: dict[str, dict[str, Any]], predictions: dict[str, pd.DataFrame]) -> dict[str, Any]:
    reference = results["base_ce"]
    invariant_keys = (
        "protocol_registry_sha256",
        "train_signature",
        "validation_signature",
        "test_input_signature",
        "shot_per_class_train",
        "shot_per_class_validation",
        "split_seed",
        "member_seeds",
        "train_rows",
        "validation_rows",
        "test_rows",
        "view",
        "threshold",
        "test_tuning_evaluations",
    )
    mismatches: list[str] = []
    for preset, result in results.items():
        if result.get("status") != "completed":
            mismatches.append(f"{preset}: incomplete status")
        for key in invariant_keys:
            if result.get(key) != reference.get(key):
                mismatches.append(f"{preset}: {key} differs")
        if result["train_signature"] != EXPECTED_TRAIN_SIGNATURE:
            mismatches.append(f"{preset}: unexpected train signature")
        if result["validation_signature"] != EXPECTED_VALIDATION_SIGNATURE:
            mismatches.append(f"{preset}: unexpected validation signature")
        if result["input_role"] != "main_label_free_input":
            mismatches.append(f"{preset}: non-main input role")
        if len(set(result["train_row_ids"])) != 140 or len(set(result["validation_row_ids"])) != 140:
            mismatches.append(f"{preset}: row-id count mismatch")
        if set(result["train_row_ids"]) & set(result["validation_row_ids"]):
            mismatches.append(f"{preset}: train/validation overlap")
    reference_ids = predictions["base_ce"]["row_id"].tolist()
    for preset, frame in predictions.items():
        if frame["row_id"].tolist() != reference_ids:
            mismatches.append(f"{preset}: prediction row order differs")
        expected_probability_columns = {f"probability_seed{seed}" for seed in reference["member_seeds"]}
        if not expected_probability_columns.issubset(frame.columns):
            mismatches.append(f"{preset}: missing member probabilities")
    if mismatches:
        raise AssertionError("Protocol verification failed: " + "; ".join(mismatches))
    return {
        "passed": True,
        "invariant_keys": list(invariant_keys),
        "train_signature": EXPECTED_TRAIN_SIGNATURE,
        "validation_signature": EXPECTED_VALIDATION_SIGNATURE,
        "member_seeds": reference["member_seeds"],
        "test_rows": reference["test_rows"],
    }


def exact_content_overlap() -> dict[str, Any]:
    train = pd.read_csv(DATA_DIR / "douyin_train_full.csv", encoding="utf-8-sig", usecols=["content"], dtype=str)
    test = pd.read_csv(DATA_DIR / "douyin_test_full.csv", encoding="utf-8-sig", usecols=["content"], dtype=str)
    train_values = set(train["content"].fillna("").astype(str))
    repeated_mask = test["content"].fillna("").astype(str).isin(train_values)
    return {
        "train_unique_content": len(train_values),
        "test_rows_with_exact_train_content": int(repeated_mask.sum()),
        "test_fraction_with_exact_train_content": float(repeated_mask.mean()),
        "interpretation": "Published train/test partitions contain repeated surface strings; DAPT reads only train text, but the overlap limits claims of fully novel-text generalization.",
    }


def main() -> None:
    loaded = {preset: load_result(preset) for preset in PRESETS}
    results = {preset: value[0] for preset, value in loaded.items()}
    prediction_frames = {preset: value[1] for preset, value in loaded.items()}
    protocol_verification = verify_protocols(results, prediction_frames)

    test_labels = pd.read_csv(
        DATA_DIR / "douyin_test_full.csv",
        encoding="utf-8-sig",
        usecols=["row_id", "label"],
        dtype={"row_id": str},
    )
    if test_labels["row_id"].tolist() != prediction_frames["base_ce"]["row_id"].tolist():
        raise AssertionError("Test labels do not align with predictions")
    labels = test_labels["label"].to_numpy(dtype=int)
    live_ids = prediction_frames["base_ce"]["live_id"].astype(str).to_numpy()

    summary_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for preset in PRESETS:
        result = results[preset]
        metrics = result["ensemble_test_metrics"]
        summary_rows.append(
            {
                "preset": preset,
                "method": LABELS[preset],
                "dapt": int(preset.startswith("dapt")),
                "rdrop": int(preset.endswith("rdrop")),
                "member_mean_macro_f1": result["member_summary_test_macro_f1"]["mean"],
                "member_sd_macro_f1": result["member_summary_test_macro_f1"]["sd"],
                "ensemble_macro_f1": metrics["macro_f1"],
                "ensemble_macro_precision": metrics["macro_precision"],
                "ensemble_macro_recall": metrics["macro_recall"],
                "ensemble_accuracy": metrics["accuracy"],
                "delta_member_mean_vs_paper_pp": 100 * (result["member_summary_test_macro_f1"]["mean"] - PAPER_F1),
                "delta_ensemble_vs_paper_pp": 100 * (metrics["macro_f1"] - PAPER_F1),
                "delta_ensemble_vs_base_pp": 100 * (metrics["macro_f1"] - results["base_ce"]["ensemble_test_metrics"]["macro_f1"]),
                "member_models": len(result["member_seeds"]),
                "human_train_labels": result["train_rows"],
                "human_validation_labels": result["validation_rows"],
                "unlabeled_dapt_rows": 21159 if preset.startswith("dapt") else 0,
                "peak_torch_memory_mb": max(record["peak_torch_memory_mb"] for record in result["member_results"]),
                "summed_member_elapsed_seconds": sum(record["elapsed_seconds"] for record in result["member_results"]),
            }
        )
        for record in result["member_fixed_threshold_test_metrics"]:
            member_rows.append(
                {
                    "preset": preset,
                    "method": LABELS[preset],
                    **record,
                    "delta_vs_paper_pp": 100 * (record["macro_f1"] - PAPER_F1),
                }
            )

    summary = pd.DataFrame(summary_rows)
    members = pd.DataFrame(member_rows)
    base_members = members[members["preset"] == "base_ce"].set_index("seed")
    paired_rows: list[dict[str, Any]] = []
    paired_statistics: dict[str, Any] = {}
    for preset in ("dapt_ce", "base_rdrop", "dapt_rdrop"):
        current = members[members["preset"] == preset].set_index("seed")
        deltas = 100 * (current.loc[base_members.index, "macro_f1"] - base_members["macro_f1"])
        for seed, delta in deltas.items():
            paired_rows.append({"comparison": f"{preset}-base_ce", "seed": int(seed), "delta_macro_f1_pp": float(delta)})
        mean = float(deltas.mean())
        sd = float(deltas.std(ddof=1))
        sem = sd / math.sqrt(len(deltas))
        critical = float(stats.t.ppf(0.975, df=len(deltas) - 1))
        t_statistic, t_pvalue = stats.ttest_rel(
            current.loc[base_members.index, "macro_f1"].to_numpy(),
            base_members["macro_f1"].to_numpy(),
        )
        paired_statistics[preset] = {
            "comparison": f"{preset} minus base_ce",
            "n_initializations": int(len(deltas)),
            "mean_delta_pp": mean,
            "sd_delta_pp": sd,
            "t_ci95_pp": [mean - critical * sem, mean + critical * sem],
            "paired_t_statistic": float(t_statistic),
            "paired_t_pvalue": float(t_pvalue),
            "all_seed_deltas_positive": bool((deltas > 0).all()),
            "exact_two_sided_sign_test_pvalue": float(stats.binomtest(int((deltas > 0).sum()), len(deltas), 0.5).pvalue),
        }

    base_prediction = (prediction_frames["base_ce"]["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_results: dict[str, Any] = {}
    for offset, preset in enumerate(("dapt_ce", "base_rdrop", "dapt_rdrop")):
        improved_prediction = (prediction_frames[preset]["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
        row_result = row_bootstrap(
            labels,
            base_prediction,
            improved_prediction,
            BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_SEED + offset,
        )
        cluster_result = cluster_bootstrap(
            labels,
            base_prediction,
            improved_prediction,
            live_ids,
            BOOTSTRAP_ITERATIONS,
            BOOTSTRAP_SEED + 100 + offset,
        )
        point_delta = 100 * (
            results[preset]["ensemble_test_metrics"]["macro_f1"]
            - results["base_ce"]["ensemble_test_metrics"]["macro_f1"]
        )
        bootstrap_results[preset] = {
            "comparison": f"{preset} minus base_ce ensemble",
            "point_delta_pp": point_delta,
            "row_bootstrap": row_result,
            "cluster_bootstrap": cluster_result,
        }
        for result in (row_result, cluster_result):
            bootstrap_rows.append(
                {
                    "comparison": f"{preset}-base_ce",
                    "unit": result["unit"],
                    "point_delta_pp": point_delta,
                    "ci95_low_pp": 100 * result["delta_ci95"][0],
                    "ci95_high_pp": 100 * result["delta_ci95"][1],
                    "probability_delta_le_zero": result["probability_delta_le_zero"],
                    "iterations": result["iterations"],
                }
            )

    summary.to_csv(RESULTS / "strict70_summary.csv", index=False, encoding="utf-8-sig")
    members.to_csv(RESULTS / "strict70_member_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(paired_rows).to_csv(RESULTS / "strict70_paired_deltas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bootstrap_rows).to_csv(RESULTS / "strict70_bootstrap.csv", index=False, encoding="utf-8-sig")

    payload = {
        "paper_reference_f1": PAPER_F1,
        "paper_metric_note": "Paper Table 5 reports F1 without specifying averaging; local values are Macro-F1.",
        "protocol_verification": protocol_verification,
        "summary": summary.to_dict(orient="records"),
        "paired_initialization_statistics": paired_statistics,
        "ensemble_bootstrap": bootstrap_results,
        "exact_content_overlap_audit": exact_content_overlap(),
        "interpretation_limits": [
            "Fixed-split member SD measures initialization/dropout variability, not data-resampling variability.",
            "The five-member ensemble keeps the same human labels but uses approximately five times the fine-tuning and inference compute of one member.",
            "Cross-paper deltas are descriptive because the original per-example predictions and F1 averaging convention are unavailable.",
            "The public RecDY test set was observed in prior project experiments, so this is preregistered-style exploratory validation rather than formally blind confirmation.",
        ],
    }
    dump_json(RESULTS / "strict70_analysis.json", payload)
    print(summary.to_string(index=False))
    print(json.dumps(paired_statistics["dapt_rdrop"], ensure_ascii=False, indent=2))
    print(json.dumps(bootstrap_results["dapt_rdrop"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
