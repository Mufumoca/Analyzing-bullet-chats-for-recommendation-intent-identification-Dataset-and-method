from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import (
    DATA_DIR,
    RESULTS_DIR,
    Timer,
    bootstrap_ci,
    classification_metrics,
    dump_json,
    ensure_dirs,
)


SEED = 20260718


def condition_text(frame: pd.DataFrame, condition: str) -> list[str]:
    content = frame["content"].fillna("").astype(str)
    official = frame["official_background"].fillna("").astype(str)
    if condition == "content":
        return content.tolist()
    if condition == "official_only":
        return official.tolist()
    if condition == "content_official":
        return (content + " [SEP] " + official).tolist()
    if condition == "content_context":
        context = frame["context_json"].map(json.loads).map(lambda items: "；".join(items))
        return (content + " [SEP] " + context).tolist()
    if condition in {"qwen_only", "content_qwen"}:
        qwen = frame["qwen_background"].fillna("").astype(str)
        if condition == "qwen_only":
            return qwen.tolist()
        return (content + " [SEP] " + qwen).tolist()
    raise ValueError(f"Unknown condition: {condition}")


def build_pipeline() -> Pipeline:
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


def run_one(train: pd.DataFrame, test: pd.DataFrame, condition: str, suite: str) -> dict[str, object]:
    pipeline = build_pipeline()
    with Timer() as train_timer:
        pipeline.fit(condition_text(train, condition), train["label"].astype(int))
    with Timer() as inference_timer:
        predictions = pipeline.predict(condition_text(test, condition))
    probabilities = pipeline.predict_proba(condition_text(test, condition))[:, 1]
    metrics = classification_metrics(test["label"], predictions)
    f1_low, f1_high = bootstrap_ci(test["label"], predictions, metric="f1")
    macro_f1_low, macro_f1_high = bootstrap_ci(test["label"], predictions, metric="macro_f1")
    acc_low, acc_high = bootstrap_ci(test["label"], predictions, metric="accuracy")
    output = test[["row_id", "label", "content", "official_background"]].copy()
    if "qwen_background" in test:
        output["qwen_background"] = test["qwen_background"]
    output["prediction"] = predictions
    output["positive_probability"] = probabilities
    output.to_csv(
        RESULTS_DIR / f"predictions_classical_{suite}_{condition}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return {
        "suite": suite,
        "condition": condition,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        **metrics,
        "f1_ci95_low": f1_low,
        "f1_ci95_high": f1_high,
        "macro_f1_ci95_low": macro_f1_low,
        "macro_f1_ci95_high": macro_f1_high,
        "accuracy_ci95_low": acc_low,
        "accuracy_ci95_high": acc_high,
        "train_seconds": train_timer.seconds,
        "inference_seconds": inference_timer.seconds,
    }


def load_suite(suite: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if suite not in {"full", "main", "qwen"}:
        raise ValueError(suite)
    train = pd.read_csv(DATA_DIR / f"douyin_train_{suite}.csv", encoding="utf-8-sig")
    test = pd.read_csv(DATA_DIR / f"douyin_test_{suite}.csv", encoding="utf-8-sig")
    conditions = ["content", "official_only", "content_official"]
    if suite == "qwen":
        generated_train = DATA_DIR / "douyin_train_qwen_generated.csv"
        generated_test = DATA_DIR / "douyin_test_qwen_generated.csv"
        if generated_train.exists() and generated_test.exists():
            train = pd.read_csv(generated_train, encoding="utf-8-sig")
            test = pd.read_csv(generated_test, encoding="utf-8-sig")
            conditions.extend(["content_context", "qwen_only", "content_qwen"])
    return train, test, conditions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("full", "main", "qwen"), default="full")
    args = parser.parse_args()
    ensure_dirs()
    train, test, conditions = load_suite(args.suite)
    results = [run_one(train, test, condition, args.suite) for condition in conditions]

    majority_prediction = [int(train["label"].mode().iloc[0])] * len(test)
    majority_metrics = classification_metrics(test["label"], majority_prediction)
    results.insert(
        0,
        {
            "suite": args.suite,
            "condition": "majority",
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            **majority_metrics,
            "train_seconds": 0.0,
            "inference_seconds": 0.0,
        },
    )
    path = RESULTS_DIR / f"classical_{args.suite}.json"
    dump_json(path, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
