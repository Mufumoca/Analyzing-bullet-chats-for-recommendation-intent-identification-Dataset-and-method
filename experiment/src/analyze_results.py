from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import DATA_DIR, RESULTS_DIR, dump_json, ensure_dirs


SEED = 20260718
BOOTSTRAP_ITERATIONS = 5000


def metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_pos":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(metric)


def load_prediction(suite: str, condition: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"predictions_classical_{suite}_{condition}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame[["row_id", "label", "prediction"]].rename(
        columns={"prediction": f"prediction_{condition}"}
    )


def paired_bootstrap(
    suite: str,
    baseline: str,
    comparison: str,
    metric: str,
) -> dict[str, object]:
    left = load_prediction(suite, baseline)
    right = load_prediction(suite, comparison)
    merged = left.merge(right, on=["row_id", "label"], how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(f"Unpaired predictions for {suite}: {baseline} vs {comparison}")
    y_true = merged["label"].to_numpy(dtype=int)
    pred_a = merged[f"prediction_{baseline}"].to_numpy(dtype=int)
    pred_b = merged[f"prediction_{comparison}"].to_numpy(dtype=int)
    point = metric_score(y_true, pred_b, metric) - metric_score(y_true, pred_a, metric)
    rng = np.random.default_rng(SEED)
    differences = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    for iteration in range(BOOTSTRAP_ITERATIONS):
        indices = rng.integers(0, len(y_true), size=len(y_true))
        differences[iteration] = metric_score(
            y_true[indices], pred_b[indices], metric
        ) - metric_score(y_true[indices], pred_a[indices], metric)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "suite": suite,
        "baseline": baseline,
        "comparison": comparison,
        "metric": metric,
        "rows": int(len(merged)),
        "difference": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "fraction_bootstrap_difference_above_zero": float(np.mean(differences > 0)),
    }


def summarize_roberta() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    for path in sorted(RESULTS_DIR.glob("roberta_*_seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append({key: value for key, value in record.items() if key != "history"})
    runs = pd.DataFrame(records)
    if runs.empty:
        raise ValueError("No RoBERTa result files found")
    metrics = [
        "accuracy",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "elapsed_seconds",
        "peak_torch_memory_mb",
    ]
    rows: list[dict[str, object]] = []
    for (suite, condition), group in runs.groupby(["suite", "condition"], sort=True):
        row: dict[str, object] = {
            "suite": suite,
            "condition": condition,
            "seeds": ",".join(str(value) for value in sorted(group["seed"].astype(int))),
            "runs": int(len(group)),
            "train_rows": int(group["train_rows"].iloc[0]),
            "test_rows": int(group["test_rows"].iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return runs, pd.DataFrame(rows)


def roberta_seed_differences(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("main", "content", "official"),
        ("qwen", "content", "context"),
        ("qwen", "content", "official"),
        ("qwen", "content", "qwen"),
    ]
    rows: list[dict[str, object]] = []
    for suite, baseline, comparison in comparisons:
        left = runs[(runs["suite"] == suite) & (runs["condition"] == baseline)][
            ["seed", "macro_f1"]
        ].rename(columns={"macro_f1": "baseline_macro_f1"})
        right = runs[(runs["suite"] == suite) & (runs["condition"] == comparison)][
            ["seed", "macro_f1"]
        ].rename(columns={"macro_f1": "comparison_macro_f1"})
        paired = left.merge(right, on="seed", how="inner", validate="one_to_one")
        if paired.empty:
            continue
        differences = paired["comparison_macro_f1"] - paired["baseline_macro_f1"]
        rows.append(
            {
                "suite": suite,
                "baseline": baseline,
                "comparison": comparison,
                "paired_seeds": int(len(paired)),
                "macro_f1_difference_mean": float(differences.mean()),
                "macro_f1_difference_sd": float(differences.std(ddof=1)) if len(paired) > 1 else 0.0,
                "differences_by_seed": json.dumps(
                    dict(zip(paired["seed"].astype(str), differences)), ensure_ascii=False
                ),
            }
        )
    return pd.DataFrame(rows)


def context_cross_split_audit(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat(
        [train.assign(origin_split="train"), test.assign(origin_split="test")],
        ignore_index=True,
    ).sort_values("all_sequence_index")
    rows: list[dict[str, object]] = []
    for _, group in combined.groupby("live_id", sort=False):
        group = group.sort_values("all_sequence_index")
        prior_splits: list[str] = []
        for row in group.itertuples(index=False):
            context_splits = prior_splits[-5:]
            cross_count = sum(value != row.origin_split for value in context_splits)
            rows.append(
                {
                    "row_id": row.row_id,
                    "origin_split": row.origin_split,
                    "context_items": len(context_splits),
                    "cross_split_context_items": cross_count,
                }
            )
            prior_splits.append(row.origin_split)
    return pd.DataFrame(rows)


def data_audit() -> dict[str, object]:
    train = pd.read_csv(DATA_DIR / "douyin_train_full.csv", encoding="utf-8-sig", dtype={"live_id": str})
    test = pd.read_csv(DATA_DIR / "douyin_test_full.csv", encoding="utf-8-sig", dtype={"live_id": str})
    key_columns = ["live_id", "species", "content", "label", "official_background"]
    train_key_set = set(train[key_columns].astype(str).itertuples(index=False, name=None))
    test_keys = list(test[key_columns].astype(str).itertuples(index=False, name=None))
    cross = context_cross_split_audit(train, test)
    qwen_train_ids = set(
        pd.read_csv(DATA_DIR / "douyin_train_qwen.csv", encoding="utf-8-sig")["row_id"].astype(str)
    )
    qwen_test_ids = set(
        pd.read_csv(DATA_DIR / "douyin_test_qwen.csv", encoding="utf-8-sig")["row_id"].astype(str)
    )

    def cross_summary(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "rows": int(len(frame)),
            "rows_with_cross_split_context": int((frame["cross_split_context_items"] > 0).sum()),
            "fraction_with_cross_split_context": float(
                (frame["cross_split_context_items"] > 0).mean()
            ),
            "mean_cross_split_context_items": float(frame["cross_split_context_items"].mean()),
        }

    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "shared_live_ids": int(len(set(train["live_id"]) & set(test["live_id"]))),
        "train_live_ids": int(train["live_id"].nunique()),
        "test_live_ids": int(test["live_id"].nunique()),
        "test_content_seen_in_train_rows": int(test["content"].isin(set(train["content"])).sum()),
        "test_content_seen_in_train_fraction": float(test["content"].isin(set(train["content"])).mean()),
        "test_exact_five_field_seen_in_train_rows": int(sum(key in train_key_set for key in test_keys)),
        "test_exact_five_field_seen_in_train_fraction": float(
            np.mean([key in train_key_set for key in test_keys])
        ),
        "full_train_context": cross_summary(cross[cross["origin_split"] == "train"]),
        "full_test_context": cross_summary(cross[cross["origin_split"] == "test"]),
        "qwen_train_context": cross_summary(cross[cross["row_id"].isin(qwen_train_ids)]),
        "qwen_test_context": cross_summary(cross[cross["row_id"].isin(qwen_test_ids)]),
    }


def explanation_stats() -> dict[str, object]:
    frames = []
    for split in ("train", "test"):
        frame = pd.read_csv(
            DATA_DIR / f"douyin_{split}_qwen_generated.csv", encoding="utf-8-sig"
        )
        frame["split"] = split
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    result: dict[str, object] = {"rows": int(len(combined))}
    for column in ("content", "official_background", "qwen_background"):
        lengths = combined[column].fillna("").astype(str).str.len()
        result[column] = {
            "mean_chars": float(lengths.mean()),
            "median_chars": float(lengths.median()),
            "p10_chars": float(lengths.quantile(0.10)),
            "p90_chars": float(lengths.quantile(0.90)),
        }
    return result


def main() -> None:
    ensure_dirs()
    classical_rows: list[dict[str, object]] = []
    for suite in ("full", "main", "qwen"):
        classical_rows.extend(
            json.loads((RESULTS_DIR / f"classical_{suite}.json").read_text(encoding="utf-8"))
        )
    classical = pd.DataFrame(classical_rows)
    classical.to_csv(RESULTS_DIR / "classical_summary.csv", index=False, encoding="utf-8-sig")

    bootstrap_jobs = [
        ("full", "content", "content_official"),
        ("main", "content", "content_official"),
        ("qwen", "content", "content_context"),
        ("qwen", "content", "content_official"),
        ("qwen", "content", "content_qwen"),
    ]
    paired_rows = [
        paired_bootstrap(suite, baseline, comparison, metric)
        for suite, baseline, comparison in bootstrap_jobs
        for metric in ("accuracy", "f1_pos", "macro_f1")
    ]
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(RESULTS_DIR / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")

    roberta_runs, roberta_summary = summarize_roberta()
    roberta_runs.to_csv(RESULTS_DIR / "roberta_runs.csv", index=False, encoding="utf-8-sig")
    roberta_summary.to_csv(RESULTS_DIR / "roberta_summary.csv", index=False, encoding="utf-8-sig")
    roberta_differences = roberta_seed_differences(roberta_runs)
    roberta_differences.to_csv(
        RESULTS_DIR / "roberta_seed_differences.csv", index=False, encoding="utf-8-sig"
    )

    audit = data_audit()
    explanations = explanation_stats()
    dump_json(RESULTS_DIR / "data_audit.json", audit)
    dump_json(RESULTS_DIR / "explanation_stats.json", explanations)
    dump_json(
        RESULTS_DIR / "analysis_manifest.json",
        {
            "seed": SEED,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "classical_rows": int(len(classical)),
            "paired_comparisons": int(len(paired)),
            "roberta_runs": int(len(roberta_runs)),
            "roberta_groups": int(len(roberta_summary)),
        },
    )
    print(roberta_summary.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
