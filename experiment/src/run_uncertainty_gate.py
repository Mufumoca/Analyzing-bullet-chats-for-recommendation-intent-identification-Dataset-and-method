"""Validation-only uncertainty gate with a Qwen direct-classification second opinion.

The gate is deliberately an independent experiment.  It reads the frozen
1,000/400 improvement split, trains a content-only character TF-IDF baseline,
uses a stratified validation split to choose the uncertainty interval, and
evaluates the selected rule once on the untouched 400-row test set.

Qwen calls are checkpointed in ``logs/improvement``.  The default path uses a
single explicit no-think Qwen protocol for both validation and test. Reusing
the older ``classify_qwen.py`` cache is an opt-in compatibility mode and is
never part of the primary result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import StratifiedShuffleSplit

from common import DATA_DIR, bootstrap_ci, classification_metrics
from improvement_common import (
    IMPROVEMENT_DATA_DIR,
    IMPROVEMENT_LOGS_DIR,
    IMPROVEMENT_RESULTS_DIR,
    OLLAMA_URL,
    SEED,
    ensure_improvement_dirs,
)
from run_improvement_classical import make_pipeline


BOOTSTRAP_ITERATIONS = 5000
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_MARGINS = tuple(round(value, 2) for value in np.arange(0.01, 0.301, 0.01))
DIRECT_CACHE_VERSION = "direct_no_think_v2"


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input data: {path}; run prepare_improvement_data.py first")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "live_id", "content", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row_id in {path}")
    frame["row_id"] = frame["row_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def classify_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return classification_metrics(y_true, prediction)


def ollama_available() -> bool:
    try:
        response = requests.get(OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags", timeout=3)
        return bool(response.ok)
    except requests.RequestException:
        return False


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if "row_id" in record:
                records[str(record["row_id"])] = record
        except json.JSONDecodeError:
            # A truncated final line must not invalidate prior checkpoints.
            continue
    return records


def load_baseline_test_cache(row_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Read the original direct-Qwen cache without mutating baseline artifacts."""

    path = Path(__file__).resolve().parents[1] / "logs" / "qwen_classification.jsonl"
    records = load_jsonl(path)
    if not records:
        prediction_path = Path(__file__).resolve().parents[1] / "results" / "predictions_qwen_direct.csv"
        if prediction_path.exists():
            frame = pd.read_csv(prediction_path, encoding="utf-8-sig")
            for row in frame.to_dict(orient="records"):
                if str(row.get("row_id")) in row_ids and row.get("prediction") in (0, 1):
                    records[str(row["row_id"])] = {
                        "row_id": str(row["row_id"]),
                        "prediction": int(row["prediction"]),
                        "input_hash": "",
                        "raw_response": str(row.get("raw_response", "")),
                        "error": "",
                        "cache_version": "baseline_classify_qwen_v1",
                    }
    return {key: value for key, value in records.items() if key in row_ids}


def direct_prediction(
    session: requests.Session,
    row: pd.Series,
) -> dict[str, Any]:
    """Reuse classify_qwen's prompt/parser while using an explicit no-think call.

    Qwen3 can spend tens of seconds in an unnecessary reasoning phase for a
    one-token label. ``think=False`` is an inference option, not a change to
    the prompt or label semantics, and keeps this gate practical on an 8 GB
    card. The hash remains the original ``classify_qwen.input_hash`` so old
    test caches remain reusable.
    """

    # Import lazily so dry-run and cache-only operation do not require Ollama.
    from classify_qwen import input_hash, parse_label, prompt_for

    prompt = prompt_for(row)
    signature = input_hash(row)
    payload = {
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 8,
            "num_ctx": 512,
            "seed": 20260718,
        },
        "keep_alive": "30m",
    }
    last_error = ""
    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            response = session.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            raw = str(data.get("response", ""))
            prediction = parse_label(raw)
            if prediction not in (0, 1):
                raise ValueError(f"Could not parse label from {raw!r}")
            record = {
                "row_id": str(row["row_id"]),
                "input_hash": signature,
                "prediction": int(prediction),
                "raw_response": raw,
                "inference_seconds": time.perf_counter() - started,
                "generated_tokens": int(data.get("eval_count", 0)),
                "attempts": attempt,
                "error": "",
            }
            break
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = f"{type(exc).__name__}: {exc}"
            record = {
                "row_id": str(row["row_id"]),
                "input_hash": signature,
                "prediction": None,
                "raw_response": "",
                "inference_seconds": 0.0,
                "generated_tokens": 0,
                "attempts": attempt,
                "error": last_error,
            }
            if attempt < 3:
                time.sleep(attempt)
    record["row_id"] = str(row["row_id"])
    record["input_hash"] = signature
    record["cache_prompt"] = "classify_qwen.prompt_for"
    record["cache_version"] = DIRECT_CACHE_VERSION
    record["inference_options"] = {
        "think": False,
        "temperature": 0,
        "num_predict": 8,
        "num_ctx": 512,
        "seed": 20260718,
    }
    return record


def ensure_qwen_cache(
    frame: pd.DataFrame,
    cache_tag: str,
    reuse_baseline_cache: bool,
    allow_network: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return frame with ``qwen_prediction`` and checkpoint generation stats."""

    ensure_improvement_dirs()
    cache_path = IMPROVEMENT_LOGS_DIR / f"qwen_direct_{cache_tag}.jsonl"
    completed = load_jsonl(cache_path)
    row_ids = set(frame["row_id"].astype(str))
    reused_baseline = 0
    if reuse_baseline_cache and cache_tag == "test400":
        baseline = load_baseline_test_cache(row_ids)
        # Baseline records use the same classify_qwen input hash.  Keep only
        # parseable labels and let current records win if already checkpointed.
        for row_id, record in baseline.items():
            if row_id not in completed and record.get("prediction") in (0, 1):
                completed[row_id] = {**record, "cache_source": "baseline"}
                reused_baseline += 1

    from classify_qwen import input_hash

    pending = []
    for _, row in frame.iterrows():
        row_id = str(row["row_id"])
        expected_hash = input_hash(row)
        record = completed.get(row_id, {})
        cache_is_current = record.get("cache_version") == DIRECT_CACHE_VERSION
        baseline_is_explicit = reuse_baseline_cache and record.get("cache_source") == "baseline"
        if record.get("prediction") in (0, 1) and (cache_is_current or baseline_is_explicit) and (
            not record.get("input_hash") or record.get("input_hash") == expected_hash
        ):
            continue
        pending.append((row_id, row, expected_hash))

    if pending and not allow_network:
        raise RuntimeError(
            f"{len(pending)} Qwen predictions are not cached for {cache_tag}. "
            "Start Ollama or omit --offline."
        )

    started = time.perf_counter()
    if pending:
        session = requests.Session()
        with cache_path.open("a", encoding="utf-8") as handle:
            for number, (row_id, row, expected_hash) in enumerate(pending, start=1):
                record = direct_prediction(session, row)
                record["input_hash"] = expected_hash
                record["cache_source"] = "ollama"
                completed[row_id] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[qwen {cache_tag} {number}/{len(pending)}] {row_id}: "
                    f"{record.get('prediction')} ({record.get('inference_seconds', 0):.2f}s)",
                    flush=True,
                )

    # Keep one latest checkpoint per requested row. This makes the cache
    # self-contained after protocol upgrades instead of retaining stale lines.
    with cache_path.open("w", encoding="utf-8") as handle:
        for row_id in frame["row_id"].astype(str):
            record = completed.get(row_id)
            if record:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = completed.get(str(row["row_id"]), {})
        rows.append(
            {
                "row_id": str(row["row_id"]),
                "qwen_prediction": record.get("prediction"),
                "qwen_input_hash": record.get("input_hash", ""),
                "qwen_raw_response": record.get("raw_response", ""),
                "qwen_inference_seconds": record.get("inference_seconds", 0.0),
                "qwen_error": record.get("error", ""),
                "qwen_cache_source": record.get("cache_source", ""),
            }
        )
    qwen = pd.DataFrame(rows)
    merged = frame.merge(qwen, on="row_id", how="left", validate="one_to_one")
    valid = merged["qwen_prediction"].isin([0, 1])
    merged["qwen_prediction"] = merged["qwen_prediction"].where(valid, np.nan)
    output_path = IMPROVEMENT_RESULTS_DIR / f"predictions_qwen_direct_{cache_tag}.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    stats = {
        "cache_path": str(cache_path),
        "prediction_path": str(output_path),
        "rows": int(len(merged)),
        "valid_predictions": int(valid.sum()),
        "coverage": float(valid.mean()),
        "pending_network_calls": int(len(pending)),
        "reused_baseline_records": int(reused_baseline),
        "network_seconds": float(time.perf_counter() - started),
        "sum_recorded_inference_seconds": float(
            sum(float(record.get("inference_seconds", 0.0) or 0.0) for record in completed.values() if str(record.get("row_id", "")) in row_ids)
        ),
        "mean_recorded_inference_seconds": float(
            np.mean(
                [
                    float(record.get("inference_seconds", 0.0) or 0.0)
                    for record in completed.values()
                    if str(record.get("row_id", "")) in row_ids
                ]
            )
            if row_ids
            else 0.0
        ),
        "cache_version": DIRECT_CACHE_VERSION,
        "cache_sources": {
            "ollama": int(
                sum(
                    1
                    for record in completed.values()
                    if record.get("cache_source") == "ollama"
                    and str(record.get("row_id", "")) in row_ids
                )
            ),
            "baseline": int(
                sum(
                    1
                    for record in completed.values()
                    if record.get("cache_source") == "baseline"
                    and str(record.get("row_id", "")) in row_ids
                )
            ),
        },
    }
    return merged, stats


def fit_base_predictions(
    fit_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen-style content-only TF-IDF baseline and return pred/prob."""

    model = make_pipeline()
    content_fit = fit_frame["content"].fillna("").astype(str).tolist()
    content_target = target_frame["content"].fillna("").astype(str).tolist()
    model.fit(content_fit, fit_frame["label"].astype(int))
    probability = model.predict_proba(content_target)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    return prediction, probability


def choose_margin(
    y_true: np.ndarray,
    base_probability: np.ndarray,
    qwen_prediction: np.ndarray,
    margins: tuple[float, ...],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    base_prediction = (base_probability >= 0.5).astype(int)
    for margin in margins:
        low, high = 0.5 - margin, 0.5 + margin
        uncertain = (base_probability >= low) & (base_probability <= high)
        valid_qwen = np.isin(qwen_prediction, [0, 1])
        replace = uncertain & valid_qwen
        gate_prediction = base_prediction.copy()
        gate_prediction[replace] = qwen_prediction[replace].astype(int)
        score = float(classify_metrics(y_true, gate_prediction)["macro_f1"])
        candidates.append(
            {
                "margin": float(margin),
                "lower": float(low),
                "upper": float(high),
                "macro_f1": score,
                "replacement_count": int(replace.sum()),
                "replacement_fraction": float(replace.mean()),
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["macro_f1"],
            item["replacement_fraction"],
            item["margin"],
        )
    )
    if not candidates:
        raise ValueError("No uncertainty margins supplied")
    best = dict(candidates[0])
    best["selection_metric"] = "validation_macro_f1"
    best["candidate_count"] = len(candidates)
    best["candidates"] = candidates
    return best


def apply_gate(
    base_probability: np.ndarray,
    qwen_prediction: np.ndarray,
    selection: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_prediction = (base_probability >= 0.5).astype(int)
    uncertain = (base_probability >= float(selection["lower"])) & (
        base_probability <= float(selection["upper"])
    )
    valid_qwen = np.isin(qwen_prediction, [0, 1])
    replaced = uncertain & valid_qwen
    gate_prediction = base_prediction.copy()
    gate_prediction[replaced] = qwen_prediction[replaced].astype(int)
    return gate_prediction, uncertain, replaced


def bootstrap_difference(
    y_true: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    point = float(classify_metrics(y_true, right)["macro_f1"] - classify_metrics(y_true, left)["macro_f1"])
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        values[index] = classify_metrics(y_true[sample], right[sample])["macro_f1"] - classify_metrics(
            y_true[sample], left[sample]
        )["macro_f1"]
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "macro_f1_difference": point,
        "macro_f1_ci95_low": float(low),
        "macro_f1_ci95_high": float(high),
        "bootstrap_iterations": iterations,
        "fraction_above_zero": float(np.mean(values > 0)),
    }


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    if not train_path.is_absolute():
        train_path = IMPROVEMENT_DATA_DIR / train_path
    if not test_path.is_absolute():
        test_path = IMPROVEMENT_DATA_DIR / test_path
    train = load_frame(train_path)
    test = load_frame(test_path)
    if len(train) != 1000 or len(test) != 400:
        raise ValueError(f"Gate requires fixed 1000/400 data, got {len(train)}/{len(test)}")
    if set(train["row_id"]) & set(test["row_id"]):
        raise ValueError("Train/test row_id overlap")

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.validation_fraction,
        random_state=SEED,
    )
    fit_indices, validation_indices = next(splitter.split(train, train["label"]))
    fit_frame = train.iloc[fit_indices].reset_index(drop=True)
    validation_frame = train.iloc[validation_indices].reset_index(drop=True)

    if args.dry_run:
        return {
            "dry_run": True,
            "train_rows": len(train),
            "validation_rows": len(validation_frame),
            "test_rows": len(test),
            "validation_positive_rate": float(validation_frame["label"].mean()),
            "ollama_available": ollama_available(),
            "margins": list(args.margins),
        }

    validation, validation_cache = ensure_qwen_cache(
        validation_frame,
        "validation200",
        reuse_baseline_cache=False,
        allow_network=args.allow_network,
    )
    test_with_qwen, test_cache = ensure_qwen_cache(
        test,
        "test400",
        reuse_baseline_cache=args.reuse_baseline_test_cache,
        allow_network=args.allow_network,
    )

    validation_base_prediction, validation_base_probability = fit_base_predictions(fit_frame, validation)
    test_base_prediction, test_base_probability = fit_base_predictions(train, test_with_qwen)
    validation_qwen = validation["qwen_prediction"].fillna(-1).to_numpy(dtype=int)
    test_qwen = test_with_qwen["qwen_prediction"].fillna(-1).to_numpy(dtype=int)
    selection = choose_margin(
        validation["label"].to_numpy(dtype=int),
        validation_base_probability,
        validation_qwen,
        tuple(args.margins),
    )
    validation_gate_prediction, validation_uncertain, validation_replaced = apply_gate(
        validation_base_probability, validation_qwen, selection
    )
    test_gate_prediction, test_uncertain, test_replaced = apply_gate(
        test_base_probability, test_qwen, selection
    )

    def split_summary(
        frame: pd.DataFrame,
        base_prediction: np.ndarray,
        base_probability: np.ndarray,
        qwen_prediction: np.ndarray,
        gate_prediction: np.ndarray,
        uncertain: np.ndarray,
        replaced: np.ndarray,
    ) -> dict[str, Any]:
        labels = frame["label"].to_numpy(dtype=int)
        qwen_valid = np.isin(qwen_prediction, [0, 1])
        summary = {
            "rows": int(len(frame)),
            "qwen_valid_predictions": int(qwen_valid.sum()),
            "qwen_coverage": float(qwen_valid.mean()),
            "base": classify_metrics(labels, base_prediction),
            "qwen": classify_metrics(labels[qwen_valid], qwen_prediction[qwen_valid])
            if qwen_valid.any()
            else {},
            "gate": classify_metrics(labels, gate_prediction),
            "uncertain_count": int(uncertain.sum()),
            "uncertain_fraction": float(uncertain.mean()),
            "replaced_count": int(replaced.sum()),
            "replaced_fraction": float(replaced.mean()),
        }
        if np.array_equal(base_prediction, gate_prediction):
            summary["gate_vs_base"] = {
                "macro_f1_difference": 0.0,
                "macro_f1_ci95_low": 0.0,
                "macro_f1_ci95_high": 0.0,
            }
        else:
            summary["gate_vs_base"] = bootstrap_difference(labels, base_prediction, gate_prediction)
        return summary

    validation_summary = split_summary(
        validation,
        validation_base_prediction,
        validation_base_probability,
        validation_qwen,
        validation_gate_prediction,
        validation_uncertain,
        validation_replaced,
    )
    test_summary = split_summary(
        test_with_qwen,
        test_base_prediction,
        test_base_probability,
        test_qwen,
        test_gate_prediction,
        test_uncertain,
        test_replaced,
    )

    def prediction_frame(
        frame: pd.DataFrame,
        base_prediction: np.ndarray,
        base_probability: np.ndarray,
        qwen_prediction: np.ndarray,
        gate_prediction: np.ndarray,
        uncertain: np.ndarray,
        replaced: np.ndarray,
        split: str,
    ) -> pd.DataFrame:
        output = frame[["row_id", "live_id", "label", "content"]].copy()
        output.insert(1, "split", split)
        output["base_probability"] = base_probability
        output["base_prediction"] = base_prediction
        output["qwen_prediction"] = qwen_prediction
        output["qwen_valid"] = np.isin(qwen_prediction, [0, 1])
        output["uncertain"] = uncertain
        output["replaced"] = replaced
        output["gate_prediction"] = gate_prediction
        output["selected_margin"] = float(selection["margin"])
        output["selected_lower"] = float(selection["lower"])
        output["selected_upper"] = float(selection["upper"])
        return output

    validation_predictions = prediction_frame(
        validation,
        validation_base_prediction,
        validation_base_probability,
        validation_qwen,
        validation_gate_prediction,
        validation_uncertain,
        validation_replaced,
        "validation",
    )
    test_predictions = prediction_frame(
        test_with_qwen,
        test_base_prediction,
        test_base_probability,
        test_qwen,
        test_gate_prediction,
        test_uncertain,
        test_replaced,
        "test",
    )
    validation_path = IMPROVEMENT_RESULTS_DIR / "uncertainty_gate_validation_predictions.csv"
    test_prediction_path = IMPROVEMENT_RESULTS_DIR / "uncertainty_gate_test_predictions.csv"
    validation_predictions.to_csv(validation_path, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(test_prediction_path, index=False, encoding="utf-8-sig")

    result: dict[str, Any] = {
        "experiment": "qwen_direct_uncertainty_gate_v1",
        "base_model": "character_tfidf_logistic_regression_content_only",
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "validation_fraction": float(args.validation_fraction),
        "selection_seed": SEED,
        "selection_policy": "validation_only_macro_f1; test_evaluated_once",
        "margins": list(args.margins),
        "selection": selection,
        "validation": validation_summary,
        "test": test_summary,
        "qwen_cache": {"validation": validation_cache, "test": test_cache},
        "prediction_files": {"validation": str(validation_path), "test": str(test_prediction_path)},
        "test_tuning_evaluations": 0,
        "ollama_url": OLLAMA_URL,
    }
    result_path = IMPROVEMENT_RESULTS_DIR / "uncertainty_gate.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        default="douyin_train_improvement1000.csv",
        help="Absolute path or filename under data/processed/improvement",
    )
    parser.add_argument(
        "--test-file",
        default="douyin_test_improvement400.csv",
        help="Absolute path or filename under data/processed/improvement",
    )
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARGINS),
        help="Symmetric uncertainty margins around probability 0.5; chosen on validation only",
    )
    parser.add_argument("--offline", action="store_true", help="Require all Qwen rows to be cached")
    parser.add_argument(
        "--reuse-baseline-test-cache",
        dest="reuse_baseline_test_cache",
        action="store_true",
        help="Opt in to the older classify_qwen test cache (not used by the primary result)",
    )
    parser.set_defaults(reuse_baseline_test_cache=False)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/split and report Ollama status")
    args = parser.parse_args()
    if any(value <= 0 or value >= 0.5 for value in args.margins):
        raise ValueError("Each margin must be in (0, 0.5)")
    args.margins = tuple(sorted(set(round(float(value), 4) for value in args.margins)))
    args.allow_network = not args.offline
    result = run_gate(args)
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
