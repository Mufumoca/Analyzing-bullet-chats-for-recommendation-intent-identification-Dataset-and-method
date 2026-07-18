from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from common import DATA_DIR, RESULTS_DIR, bootstrap_ci, classification_metrics, dump_json
from improvement_common import IMPROVEMENT_DATA_DIR, IMPROVEMENT_RESULTS_DIR, SEED, ensure_improvement_dirs, metrics


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_pair(size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_csv(IMPROVEMENT_DATA_DIR / f"douyin_train_improvement{size}.csv")
    test = load_csv(IMPROVEMENT_DATA_DIR / "douyin_test_improvement400.csv")
    if size == 400:
        old_train = load_csv(DATA_DIR / "douyin_train_qwen_generated.csv")
        old_test = load_csv(DATA_DIR / "douyin_test_qwen_generated.csv")
        train = train.merge(
            old_train[["row_id", "qwen_background"]].rename(columns={"qwen_background": "qwen_background_old"}),
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        test = test.merge(
            old_test[["row_id", "qwen_background"]].rename(columns={"qwen_background": "qwen_background_old"}),
            on="row_id",
            how="left",
            validate="one_to_one",
        )
    structured_train = load_csv(
        IMPROVEMENT_DATA_DIR / f"douyin_train_improvement{size}_structured_v2.csv"
    )
    structured_test = load_csv(
        IMPROVEMENT_DATA_DIR / "douyin_test_improvement400_structured_v2.csv"
    )
    train = train.merge(
        structured_train[
            ["row_id", "qwen_background_structured", "valid_structure", "input_hash", "prompt_hash"]
        ],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    test = test.merge(
        structured_test[
            ["row_id", "qwen_background_structured", "valid_structure", "input_hash", "prompt_hash"]
        ],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if set(train["row_id"]) & set(test["row_id"]):
        raise ValueError("Improvement train/test overlap")
    return train, test


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 4),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=4.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def texts(frame: pd.DataFrame, field: str | None) -> list[str]:
    content = frame["content"].fillna("").astype(str)
    if field is None:
        return content.tolist()
    explanation = frame[field].fillna("").astype(str)
    return (content + " [SEP] " + explanation).tolist()


def raw_context_texts(frame: pd.DataFrame) -> list[str]:
    contexts = frame["context_json"].map(json.loads).map(lambda items: "；".join(items))
    return (frame["content"].fillna("").astype(str) + " [SEP] " + contexts).tolist()


def fit_concat(train: pd.DataFrame, test: pd.DataFrame, field: str | None) -> tuple[dict[str, object], pd.DataFrame]:
    pipeline = make_pipeline()
    pipeline.fit(texts(train, field), train["label"].astype(int))
    predictions = pipeline.predict(texts(test, field))
    probabilities = pipeline.predict_proba(texts(test, field))[:, 1]
    result = {"condition": "content" if field is None else field, **classification_metrics(test["label"], predictions)}
    result["macro_f1_ci95_low"], result["macro_f1_ci95_high"] = bootstrap_ci(
        test["label"], predictions, metric="macro_f1", iterations=5000
    )
    output = test[["row_id", "live_id", "label", "content"]].copy()
    output["prediction"] = predictions
    output["positive_probability"] = probabilities
    return result, output


def fit_raw_context(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    pipeline = make_pipeline()
    pipeline.fit(raw_context_texts(train), train["label"].astype(int))
    predictions = pipeline.predict(raw_context_texts(test))
    probabilities = pipeline.predict_proba(raw_context_texts(test))[:, 1]
    result = {"condition": "raw_context", **classification_metrics(test["label"], predictions)}
    result["macro_f1_ci95_low"], result["macro_f1_ci95_high"] = bootstrap_ci(
        test["label"], predictions, metric="macro_f1", iterations=5000
    )
    output = test[["row_id", "live_id", "label", "content"]].copy()
    output["prediction"] = predictions
    output["positive_probability"] = probabilities
    return result, output


def choose_fusion(train: pd.DataFrame, field: str) -> dict[str, object]:
    y = train["label"].to_numpy(dtype=int)
    content_text = texts(train, None)
    explanation_text = train[field].fillna("").astype(str).tolist()
    oof_content = np.zeros(len(train), dtype=float)
    oof_explanation = np.zeros(len(train), dtype=float)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_indices, valid_indices) in enumerate(splitter.split(content_text, y)):
        content_model = make_pipeline()
        explanation_model = make_pipeline()
        content_model.fit([content_text[i] for i in fit_indices], y[fit_indices])
        explanation_model.fit([explanation_text[i] for i in fit_indices], y[fit_indices])
        oof_content[valid_indices] = content_model.predict_proba(
            [content_text[i] for i in valid_indices]
        )[:, 1]
        oof_explanation[valid_indices] = explanation_model.predict_proba(
            [explanation_text[i] for i in valid_indices]
        )[:, 1]

    candidates: list[tuple[float, float, float]] = []
    for alpha in np.arange(0.0, 1.0001, 0.05):
        fused = alpha * oof_content + (1.0 - alpha) * oof_explanation
        for threshold in np.arange(0.40, 0.6001, 0.01):
            prediction = (fused >= threshold).astype(int)
            score = float(metrics(y, prediction)["macro_f1"])
            candidates.append((score, float(alpha), float(threshold)))
    candidates.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), abs(item[2] - 0.5)))
    score, alpha, threshold = candidates[0]
    return {
        "field": field,
        "folds": 5,
        "selection_seed": SEED,
        "alpha_content": alpha,
        "alpha_explanation": 1.0 - alpha,
        "threshold": threshold,
        "oof_macro_f1": score,
        "grid_alpha": [round(float(value), 2) for value in np.arange(0.0, 1.0001, 0.05)],
        "grid_threshold": [round(float(value), 2) for value in np.arange(0.40, 0.6001, 0.01)],
    }


def fit_late_fusion(
    train: pd.DataFrame, test: pd.DataFrame, field: str
) -> tuple[dict[str, object], pd.DataFrame, dict[str, object]]:
    selection = choose_fusion(train, field)
    content_model = make_pipeline()
    explanation_model = make_pipeline()
    content_model.fit(texts(train, None), train["label"].astype(int))
    explanation_model.fit(train[field].fillna("").astype(str).tolist(), train["label"].astype(int))
    content_probability = content_model.predict_proba(texts(test, None))[:, 1]
    explanation_probability = explanation_model.predict_proba(
        test[field].fillna("").astype(str).tolist()
    )[:, 1]
    fused_probability = (
        selection["alpha_content"] * content_probability
        + selection["alpha_explanation"] * explanation_probability
    )
    predictions = (fused_probability >= selection["threshold"]).astype(int)
    result = {
        "condition": f"late_fusion_{field}",
        **classification_metrics(test["label"], predictions),
        "alpha_content": selection["alpha_content"],
        "alpha_explanation": selection["alpha_explanation"],
        "threshold": selection["threshold"],
        "oof_macro_f1": selection["oof_macro_f1"],
    }
    result["macro_f1_ci95_low"], result["macro_f1_ci95_high"] = bootstrap_ci(
        test["label"], predictions, metric="macro_f1", iterations=5000
    )
    output = test[["row_id", "live_id", "label", "content"]].copy()
    output["prediction"] = predictions
    output["positive_probability"] = fused_probability
    return result, output, selection


def run_size(size: int) -> list[dict[str, object]]:
    train, test = load_pair(size)
    results: list[dict[str, object]] = []
    predictions: dict[str, pd.DataFrame] = {}
    conditions: list[tuple[str, str | None]] = [("content", None), ("structured_concat", "qwen_background_structured")]
    if size == 400:
        conditions.extend([("old_concat", "qwen_background_old")])
    for name, field in conditions:
        result, output = fit_concat(train, test, field)
        result["size"] = size
        result["condition"] = name
        results.append(result)
        predictions[name] = output
    if size == 400:
        result, output = fit_raw_context(train, test)
        result["size"] = size
        results.append(result)
        predictions["raw_context"] = output
        for field, name in (("qwen_background_old", "old"), ("qwen_background_structured", "structured")):
            result, output, selection = fit_late_fusion(train, test, field)
            result["size"] = size
            result["condition"] = f"{name}_late_fusion"
            result["selection"] = selection
            results.append(result)
            predictions[f"{name}_late_fusion"] = output
    else:
        result, output, selection = fit_late_fusion(train, test, "qwen_background_structured")
        result["size"] = size
        result["condition"] = "structured_late_fusion"
        result["selection"] = selection
        results.append(result)
        predictions["structured_late_fusion"] = output

    for name, output in predictions.items():
        output.to_csv(
            IMPROVEMENT_RESULTS_DIR / f"predictions_classical_{size}_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[400, 1000], choices=[400, 1000])
    args = parser.parse_args()
    ensure_improvement_dirs()
    all_results: list[dict[str, object]] = []
    for size in args.sizes:
        results = run_size(size)
        all_results.extend(results)
        dump_json(IMPROVEMENT_RESULTS_DIR / f"classical_improvement_{size}.json", results)
    pd.DataFrame(all_results).to_csv(
        IMPROVEMENT_RESULTS_DIR / "classical_improvement_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(all_results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
