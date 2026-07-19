from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiment"
DATA = EXPERIMENT / "data" / "processed"
CONFIG = EXPERIMENT / "config" / "same_budget_protocol_v1.json"
RESULTS = EXPERIMENT / "results" / "same_budget"
LOGS = EXPERIMENT / "logs" / "same_budget_qwen_gate"
FIGURES = EXPERIMENT / "figures"
REPORT_ASSETS = EXPERIMENT / "report"
REPORT = REPORT_ASSETS / "improvement_comparison"

PRESETS = (
    "strict70_base_ce_5seed_v1",
    "strict70_dapt_ce_5seed_v1",
    "strict70_base_rdrop_5seed_v1",
    "strict70_dapt_rdrop_5seed_v1",
)

QWEN_BASE_TAG = "strict70_dapt_rdrop_5seed_v1"
QWEN_MODEL = "qwen3:8b"
QWEN_MODEL_DIGEST = "e4b5fd7f8af048d3c02e0357274238a9e93da51936665599ccb957aa42bfe173"
QWEN_PROMPT_VERSION = "rag_verifier_3view_v1"
QWEN_PARSER_VERSION = "json_evidence_parser_v1"
QWEN_VARIANTS = ("explicit_behavior", "latent_intent", "negative_filter")
QWEN_VOTE_REDUCER = "adaptive_first_two_then_third_v1"
QWEN_GENERATION_OPTIONS = {"temperature": 0, "num_predict": 64, "num_ctx": 2048}
QWEN_MIN_CONFIDENCE = 80.0
PAPER_SPT_RII_70 = 0.7923
PAPER_QWEN3_8B_PEFT = 0.7524
BOOTSTRAP_ITERATIONS = 5000

SIGNATURE_COLUMNS = ["row_id", "content", "official_background", "context_json", "label"]
LABEL_FREE_COLUMNS = ["row_id", "live_id", "content", "all_sequence_index"]
QWEN_INPUT_COLUMNS = ["row_id", "live_id", "content", "context_json"]

PROTOCOL_HASH_KEYS = (
    "model",
    "model_digest",
    "prompt_version",
    "parser_version",
    "generation_options",
    "variants",
    "context_window",
    "retrieval_per_class",
    "min_confidence",
    "adaptive_test_voting",
    "vote_reducer",
    "label_free_train_signature",
    "label_free_test_signature",
    "context20_corpus_signature",
    "qwen_test_input_signature_locked",
    "base_tag",
    "human_train_signature",
    "human_validation_signature",
    "base_validation_file_sha256",
    "base_test_file_sha256",
    "selection",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frame_signature(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame[columns].fillna("").astype(str).to_csv(index=False)
    return sha256_text(payload)


def assert_close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def assert_metric_dict(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("macro_f1", "macro_precision", "macro_recall", "accuracy"):
        assert_close(actual[key], expected[key], f"{label}.{key}")


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def bool_array(values: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = normalized.map(mapping)
    if converted.isna().any():
        raise AssertionError(f"Invalid boolean value in {label}")
    return converted.to_numpy(dtype=bool)


def sample_per_label(frame: pd.DataFrame, shot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _, group in frame.groupby("label", sort=True):
        chosen.extend(rng.choice(group.index.to_numpy(), size=shot, replace=False).tolist())
    return frame.loc[sorted(chosen)].reset_index(drop=True)


def make_split(frame: pd.DataFrame, shot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = sample_per_label(frame, shot, seed)
    remaining = frame.loc[~frame["row_id"].isin(train["row_id"])].copy()
    validation = sample_per_label(remaining, shot, seed + 100_003)
    if set(train["row_id"]) & set(validation["row_id"]):
        raise AssertionError("Few-shot train/validation overlap")
    return train, validation


def load_training_source() -> pd.DataFrame:
    frame = pd.read_csv(
        DATA / "douyin_train_full.csv",
        encoding="utf-8-sig",
        usecols=SIGNATURE_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if frame["row_id"].duplicated().any():
        raise AssertionError("Duplicate row_id in training source")
    return frame.reset_index(drop=True)


def label_free_source_signature(path: Path) -> str:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=LABEL_FREE_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    return sha256_text(frame[LABEL_FREE_COLUMNS].to_csv(index=False))


def rebuild_context20() -> dict[str, list[str]]:
    frames = [
        pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=LABEL_FREE_COLUMNS,
            dtype=str,
            keep_default_na=False,
        )
        for path in (DATA / "douyin_train_full.csv", DATA / "douyin_test_full.csv")
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["all_sequence_index"] = pd.to_numeric(combined["all_sequence_index"], errors="raise")
    if combined["row_id"].duplicated().any():
        raise AssertionError("Duplicate row_id in context corpus")
    if combined.duplicated(["live_id", "all_sequence_index"]).any():
        raise AssertionError("Ambiguous live_id/all_sequence_index context order")
    windows: dict[str, list[str]] = {}
    ordered = combined.sort_values(["live_id", "all_sequence_index"])
    for _, group in ordered.groupby("live_id", sort=False):
        history: list[str] = []
        for row in group.itertuples(index=False):
            windows[str(row.row_id)] = history[-20:]
            history.append(str(row.content))
    return windows


def protocol_hash(payload: dict[str, Any]) -> str:
    return sha256_text(canonical({key: payload[key] for key in PROTOCOL_HASH_KEYS}))


def verify_live_model_digest(model: str, expected_digest: str) -> None:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
    except Exception as exc:
        raise AssertionError(f"Cannot verify the live Ollama model digest: {exc}") from exc
    match = next((item for item in models if item.get("name") == model), None)
    if match is None or match.get("digest") != expected_digest:
        raise AssertionError(f"Ollama model digest mismatch for {model}: {match}")


def load_jsonl_cache(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    line_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"Invalid cache JSON at line {line_number}") from exc
            key = str(record.get("cache_key", ""))
            expected_key = f"{record.get('row_id')}::{record.get('variant')}"
            if key != expected_key or record.get("variant") not in QWEN_VARIANTS:
                raise AssertionError(f"Malformed Qwen cache identity at line {line_number}")
            for hash_key in ("input_hash", "prompt_hash"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(hash_key, ""))):
                    raise AssertionError(f"Malformed {hash_key} at cache line {line_number}")
            if record.get("valid"):
                if int(record["label"]) not in (0, 1) or not 0 <= float(record["confidence"]) <= 100:
                    raise AssertionError(f"Invalid valid-vote payload at cache line {line_number}")
                if not record.get("evidence_ok") or record.get("abstain"):
                    raise AssertionError(f"Valid cache vote violates evidence/abstain rules at line {line_number}")
            records[key] = record
    return records, line_count


def reduce_qwen_votes(row_id: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    votes = [records.get(f"{row_id}::{variant}", {}) for variant in QWEN_VARIANTS]
    first_two = votes[:2]
    early_stop = bool(
        all(record.get("valid") for record in first_two)
        and int(first_two[0]["label"]) == int(first_two[1]["label"])
        and np.mean([record["confidence"] for record in first_two]) >= QWEN_MIN_CONFIDENCE
    )
    active_votes = first_two if early_stop else votes
    valid = [record for record in active_votes if record.get("valid")]
    labels = [int(record["label"]) for record in valid]
    counts = Counter(labels)
    consensus_label = -1
    if counts and max(counts.values()) >= 2:
        consensus_label = int(counts.most_common(1)[0][0])
    supporting = [record for record in valid if int(record["label"]) == consensus_label]
    confidence = float(np.mean([record["confidence"] for record in supporting])) if supporting else 0.0
    consensus = consensus_label in (0, 1) and counts[consensus_label] >= 2 and confidence >= QWEN_MIN_CONFIDENCE
    return {
        "consensus": bool(consensus),
        "consensus_label": consensus_label,
        "active_vote_count": len(active_votes),
        "valid_vote_count": len(valid),
    }


def macro_f1_fast(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool, copy=False)
    predictions = predictions.astype(bool, copy=False)
    tp = int(np.count_nonzero(labels & predictions))
    tn = int(np.count_nonzero(~labels & ~predictions))
    fp = int(np.count_nonzero(~labels & predictions))
    fn = int(np.count_nonzero(labels & ~predictions))
    pos_denominator = 2 * tp + fp + fn
    neg_denominator = 2 * tn + fp + fn
    pos_f1 = 2 * tp / pos_denominator if pos_denominator else 0.0
    neg_f1 = 2 * tn / neg_denominator if neg_denominator else 0.0
    return 0.5 * (pos_f1 + neg_f1)


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


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
    for iteration in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled = labels[indices]
        baseline_scores[iteration] = macro_f1_fast(sampled, baseline[indices])
        improved_scores[iteration] = macro_f1_fast(sampled, improved[indices])
    deltas = improved_scores - baseline_scores
    return {
        "unit": "row",
        "iterations": iterations,
        "seed": seed,
        "baseline_ci95": percentile_interval(baseline_scores),
        "improved_ci95": percentile_interval(improved_scores),
        "delta_ci95": percentile_interval(deltas),
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
    unique = np.unique(clusters)
    indices_by_cluster = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    baseline_scores = np.empty(iterations, dtype=float)
    improved_scores = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled_clusters = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([indices_by_cluster[cluster] for cluster in sampled_clusters])
        sampled = labels[indices]
        baseline_scores[iteration] = macro_f1_fast(sampled, baseline[indices])
        improved_scores[iteration] = macro_f1_fast(sampled, improved[indices])
    deltas = improved_scores - baseline_scores
    return {
        "unit": "live_id_cluster",
        "clusters": int(len(unique)),
        "iterations": iterations,
        "seed": seed,
        "baseline_ci95": percentile_interval(baseline_scores),
        "improved_ci95": percentile_interval(improved_scores),
        "delta_ci95": percentile_interval(deltas),
        "probability_delta_le_zero": float(np.mean(deltas <= 0)),
    }


def assert_bootstrap(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key in ("unit", "iterations", "seed"):
        if actual[key] != expected[key]:
            raise AssertionError(f"{label}.{key}: {actual[key]!r} != {expected[key]!r}")
    if "clusters" in expected and actual.get("clusters") != expected["clusters"]:
        raise AssertionError(f"{label}.clusters mismatch")
    for key in ("baseline_ci95", "improved_ci95", "delta_ci95"):
        if not np.allclose(actual[key], expected[key], rtol=0.0, atol=1e-12):
            raise AssertionError(f"{label}.{key} mismatch")
    assert_close(
        actual["probability_delta_le_zero"],
        expected["probability_delta_le_zero"],
        f"{label}.probability_delta_le_zero",
    )


def audit_qwen_gate() -> dict[str, Any]:
    selection_path = RESULTS / "strict70_qwen_gate_selection.json"
    result_path = RESULTS / "strict70_qwen_gate_result.json"
    analysis_path = RESULTS / "strict70_qwen_gate_analysis.json"
    prediction_path = RESULTS / "strict70_qwen_gate_predictions.csv"
    cache_path = LOGS / "strict70_test.jsonl"
    train_path = DATA / "douyin_train_full.csv"
    test_path = DATA / "douyin_test_full.csv"
    base_path = RESULTS / f"{QWEN_BASE_TAG}_predictions.csv"
    base_validation_path = RESULTS / f"{QWEN_BASE_TAG}_validation_predictions.csv"
    validation_votes_path = RESULTS / "strict70_qwen_gate_validation_votes.csv"

    for path in (
        selection_path,
        result_path,
        analysis_path,
        prediction_path,
        cache_path,
        train_path,
        test_path,
        base_path,
        base_validation_path,
        validation_votes_path,
    ):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty Qwen audit artifact: {path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    if selection["status"] != "validation_locked" or result["status"] != "completed":
        raise AssertionError("Qwen selection/result status is not locked/completed")
    if selection["selection_hash"] != sha256_text(canonical(selection["selection"])):
        raise AssertionError("Qwen selection hash mismatch")
    if selection["protocol_hash"] != protocol_hash(selection):
        raise AssertionError("Qwen protocol hash mismatch")

    expected_protocol = {
        "model": QWEN_MODEL,
        "model_digest": QWEN_MODEL_DIGEST,
        "prompt_version": QWEN_PROMPT_VERSION,
        "parser_version": QWEN_PARSER_VERSION,
        "generation_options": QWEN_GENERATION_OPTIONS,
        "variants": list(QWEN_VARIANTS),
        "retrieval_per_class": 3,
        "min_confidence": QWEN_MIN_CONFIDENCE,
        "adaptive_test_voting": True,
        "vote_reducer": QWEN_VOTE_REDUCER,
        "context_window": 20,
        "base_tag": QWEN_BASE_TAG,
    }
    for key, expected in expected_protocol.items():
        if selection.get(key) != expected:
            raise AssertionError(f"Unexpected locked Qwen protocol value: {key}")
        if key != "base_tag" and result.get(key) != expected:
            raise AssertionError(f"Qwen result protocol differs: {key}")

    for key in (
        "selection_hash",
        "protocol_hash",
        "selection",
        "human_train_signature",
        "human_validation_signature",
        "context20_train_signature",
        "context20_validation_signature",
        "label_free_train_signature",
        "label_free_test_signature",
        "context20_corpus_signature",
        "qwen_test_input_signature_locked",
        "base_test_file_sha256",
    ):
        if result.get(key) != selection.get(key):
            raise AssertionError(f"Selection/result protocol binding mismatch: {key}")

    if result["selection_file_sha256"] != sha256(selection_path):
        raise AssertionError("Selection file hash is not bound to the test result")
    if result["prediction_file_sha256"] != sha256(prediction_path):
        raise AssertionError("Prediction file hash is not bound to the test result")
    if result["qwen_cache_file_sha256"] != sha256(cache_path):
        raise AssertionError("Qwen cache file hash is not bound to the test result")
    if result["test_source_file_sha256"] != sha256(test_path):
        raise AssertionError("Test source file hash is not bound to the test result")
    if result["base_test_file_sha256"] != sha256(base_path):
        raise AssertionError("Base prediction file hash is not bound to the protocol")
    if selection["base_validation_file_sha256"] != sha256(base_validation_path):
        raise AssertionError("Base validation prediction hash mismatch")

    full_train = load_training_source()
    human_train, human_validation = make_split(full_train, 70, 20260718)
    if frame_signature(human_train, SIGNATURE_COLUMNS) != selection["human_train_signature"]:
        raise AssertionError("Recomputed human train signature mismatch")
    if frame_signature(human_validation, SIGNATURE_COLUMNS) != selection["human_validation_signature"]:
        raise AssertionError("Recomputed human validation signature mismatch")
    if label_free_source_signature(train_path) != selection["label_free_train_signature"]:
        raise AssertionError("Label-free train source signature mismatch")
    if label_free_source_signature(test_path) != selection["label_free_test_signature"]:
        raise AssertionError("Label-free test source signature mismatch")

    windows = rebuild_context20()
    if sha256_text(canonical(windows)) != selection["context20_corpus_signature"]:
        raise AssertionError("Recomputed context20 corpus signature mismatch")
    human_train["context_json"] = human_train["row_id"].map(
        lambda value: json.dumps(windows[str(value)], ensure_ascii=False)
    )
    human_validation["context_json"] = human_validation["row_id"].map(
        lambda value: json.dumps(windows[str(value)], ensure_ascii=False)
    )
    if frame_signature(human_train, ["row_id", "content", "context_json", "label"]) != selection["context20_train_signature"]:
        raise AssertionError("Recomputed context20 train signature mismatch")
    if frame_signature(human_validation, ["row_id", "content", "context_json", "label"]) != selection["context20_validation_signature"]:
        raise AssertionError("Recomputed context20 validation signature mismatch")

    test_inputs = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        usecols=["row_id", "live_id", "content", "official_background", "context_json"],
        dtype=str,
        keep_default_na=False,
    )
    test_inputs["context_json"] = test_inputs["row_id"].map(
        lambda value: json.dumps(windows[str(value)], ensure_ascii=False)
    )
    test_input_signature = frame_signature(test_inputs, QWEN_INPUT_COLUMNS)
    if test_input_signature != selection["qwen_test_input_signature_locked"]:
        raise AssertionError("Recomputed locked Qwen test input signature mismatch")
    if result["qwen_input_signature"] != test_input_signature:
        raise AssertionError("Result Qwen test input signature mismatch")

    base_validation = pd.read_csv(
        base_validation_path,
        encoding="utf-8-sig",
        dtype={"row_id": str},
    )
    validation_votes = pd.read_csv(
        validation_votes_path,
        encoding="utf-8-sig",
        dtype={"row_id": str},
    )
    if base_validation["row_id"].tolist() != human_validation["row_id"].astype(str).tolist():
        raise AssertionError("Base validation rows differ from the locked split")
    if validation_votes["row_id"].tolist() != base_validation["row_id"].tolist():
        raise AssertionError("Qwen validation votes differ from the locked split")
    if not np.array_equal(base_validation["label"].to_numpy(dtype=int), human_validation["label"].to_numpy(dtype=int)):
        raise AssertionError("Base validation labels differ from the locked split")

    validation_labels = base_validation["label"].to_numpy(dtype=int)
    validation_probability = base_validation["probability"].to_numpy(dtype=float)
    validation_qwen_label = validation_votes["consensus_label"].to_numpy(dtype=int)
    validation_consensus = bool_array(validation_votes["consensus"], "validation consensus")
    recomputed_candidates: list[dict[str, Any]] = []
    baseline_validation_prediction = (validation_probability >= 0.5).astype(int)
    recomputed_candidates.append(
        {
            "mode": "no_op",
            "margin": 0.0,
            "macro_f1": metrics(validation_labels, baseline_validation_prediction)["macro_f1"],
            "replacement_count": 0,
        }
    )
    for margin in (round(value, 2) for value in np.arange(0.05, 0.401, 0.01)):
        selected_rows = (np.abs(validation_probability - 0.5) <= margin) & validation_consensus
        candidate_prediction = baseline_validation_prediction.copy()
        candidate_prediction[selected_rows] = validation_qwen_label[selected_rows]
        candidate_metrics = metrics(validation_labels, candidate_prediction)
        recomputed_candidates.append(
            {
                "mode": "replace_consensus",
                "margin": margin,
                **candidate_metrics,
                "replacement_count": int(selected_rows.sum()),
                "replacement_fraction": float(selected_rows.mean()),
                "qwen_valid_count": int(validation_consensus.sum()),
            }
        )
    if len(recomputed_candidates) != len(selection["candidates"]):
        raise AssertionError("Validation candidate grid length mismatch")
    for index, (actual, expected) in enumerate(zip(selection["candidates"], recomputed_candidates)):
        for key, value in expected.items():
            if isinstance(value, float):
                assert_close(actual[key], value, f"selection.candidates[{index}].{key}")
            elif actual[key] != value:
                raise AssertionError(f"selection.candidates[{index}].{key} mismatch")
    recomputed_selection = sorted(
        recomputed_candidates,
        key=lambda item: (
            -float(item["macro_f1"]),
            int(item.get("replacement_count", 0)) if item["mode"] != "no_op" else 0,
            float(item.get("margin", 0.0)),
        ),
    )[0]
    for key, value in recomputed_selection.items():
        if isinstance(value, float):
            assert_close(selection["selection"][key], value, f"selection.{key}")
        elif selection["selection"][key] != value:
            raise AssertionError(f"Selected validation candidate mismatch: {key}")
    assert_metric_dict(
        metrics(validation_labels, baseline_validation_prediction),
        selection["validation_base_metrics"],
        "selection.validation_base_metrics",
    )
    assert_close(
        validation_consensus.mean(),
        selection["validation_qwen_consensus_rate"],
        "selection.validation_qwen_consensus_rate",
    )

    cache, cache_line_count = load_jsonl_cache(cache_path)
    base = pd.read_csv(base_path, encoding="utf-8-sig", dtype={"row_id": str, "live_id": str})
    gate = pd.read_csv(prediction_path, encoding="utf-8-sig", dtype={"row_id": str, "live_id": str})
    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        usecols=["row_id", "live_id", "content", "label"],
        dtype={"row_id": str, "live_id": str},
    )
    if base["row_id"].tolist() != gate["row_id"].tolist() or base["row_id"].tolist() != test["row_id"].tolist():
        raise AssertionError("Base, gate, and test row orders differ")
    if base["live_id"].astype(str).tolist() != gate["live_id"].astype(str).tolist():
        raise AssertionError("Base and gate live_id orders differ")
    if "label" in base.columns or "label" in gate.columns:
        raise AssertionError("A pre-scoring prediction artifact contains test labels")
    if not np.allclose(gate["base_probability"], base["probability"], rtol=0.0, atol=0.0):
        raise AssertionError("Gate artifact does not preserve the locked base probabilities")

    base_probability = base["probability"].to_numpy(dtype=float)
    margin = float(selection["selection"]["margin"])
    candidate_mask = np.abs(base_probability - 0.5) <= margin
    candidate_ids = test.loc[candidate_mask, "row_id"].astype(str).tolist()
    candidate_id_set = set(candidate_ids)
    if int(candidate_mask.sum()) != result["candidate_rows"]:
        raise AssertionError("Recomputed Qwen candidate count mismatch")

    relevant_cache = [record for record in cache.values() if str(record.get("row_id")) in candidate_id_set]
    if len(relevant_cache) != result["qwen_stats"]["available_records"]:
        raise AssertionError("Available cache record count mismatch")
    if sum(bool(record.get("valid")) for record in relevant_cache) != result["qwen_stats"]["valid_votes"]:
        raise AssertionError("Valid cache vote count mismatch")
    if result["qwen_stats"]["maximum_records"] != len(candidate_ids) * len(QWEN_VARIANTS):
        raise AssertionError("Maximum cache vote count mismatch")
    if result["qwen_stats"]["http_attempts"] < result["qwen_stats"]["new_records"]:
        raise AssertionError("HTTP attempt accounting is impossible")

    expected_consensus = np.zeros(len(test), dtype=bool)
    expected_qwen_label = np.full(len(test), -1, dtype=int)
    expected_prediction = (base_probability >= 0.5).astype(int)
    row_index = {str(row_id): index for index, row_id in enumerate(test["row_id"])}
    active_vote_total = 0
    for row_id in candidate_ids:
        reduced = reduce_qwen_votes(row_id, cache)
        active_vote_total += int(reduced["active_vote_count"])
        index = row_index[row_id]
        expected_consensus[index] = reduced["consensus"]
        expected_qwen_label[index] = int(reduced["consensus_label"])
        if reduced["consensus"]:
            expected_prediction[index] = int(reduced["consensus_label"])

    observed_consensus = bool_array(gate["qwen_consensus"], "test qwen_consensus")
    if not np.array_equal(observed_consensus, expected_consensus):
        raise AssertionError("Gate consensus column cannot be reconstructed from the bound cache")
    if not np.array_equal(gate["qwen_label"].to_numpy(dtype=int), expected_qwen_label):
        raise AssertionError("Gate Qwen labels cannot be reconstructed from the bound cache")
    if not np.array_equal(gate["prediction"].to_numpy(dtype=int), expected_prediction):
        raise AssertionError("Gate predictions cannot be reconstructed from base plus cache")
    if int(expected_consensus.sum()) != result["consensus_rows"]:
        raise AssertionError("Recomputed consensus route count mismatch")
    assert_close(
        expected_consensus.mean(),
        result["consensus_fraction_test"],
        "result.consensus_fraction_test",
    )

    labels = test["label"].astype(int).to_numpy()
    base_prediction = (base_probability >= 0.5).astype(int)
    base_metrics = metrics(labels, base_prediction)
    gate_metrics = metrics(labels, expected_prediction)
    assert_metric_dict(base_metrics, result["base_metrics"], "result.base_metrics")
    assert_metric_dict(gate_metrics, result["qwen_gate_metrics"], "result.qwen_gate_metrics")
    assert_close(
        100 * (gate_metrics["macro_f1"] - base_metrics["macro_f1"]),
        result["delta_macro_f1_pp"],
        "result.delta_macro_f1_pp",
    )

    if analysis["protocol_verification"]["protocol_hash"] != selection["protocol_hash"]:
        raise AssertionError("Analysis is not bound to the selected protocol")
    if analysis["protocol_verification"]["selection_hash_matches"] is not True:
        raise AssertionError("Analysis did not verify the selection hash")
    if analysis["selection"] != selection["selection"]:
        raise AssertionError("Analysis selection differs from the locked selection")
    if result["test_labels_loaded_after_selection"] is not True:
        raise AssertionError("Result does not attest post-selection test label loading")
    if analysis["protocol_verification"]["test_labels_loaded_after_selection"] is not True:
        raise AssertionError("Analysis does not preserve the post-selection label-loading audit")
    for key in ("human_train_signature", "human_validation_signature"):
        if analysis["protocol_verification"][key] != selection[key]:
            raise AssertionError(f"Analysis protocol signature mismatch: {key}")
    assert_metric_dict(base_metrics, analysis["base_metrics"], "analysis.base_metrics")
    assert_metric_dict(gate_metrics, analysis["qwen_gate_metrics"], "analysis.qwen_gate_metrics")
    assert_close(
        100 * (gate_metrics["macro_f1"] - base_metrics["macro_f1"]),
        analysis["delta_vs_local_base_pp"],
        "analysis.delta_vs_local_base_pp",
    )
    assert_close(
        100 * (gate_metrics["macro_f1"] - PAPER_SPT_RII_70),
        analysis["delta_vs_paper_spt_rii_pp"],
        "analysis.delta_vs_paper_spt_rii_pp",
    )
    assert_close(
        100 * (gate_metrics["macro_f1"] - PAPER_QWEN3_8B_PEFT),
        analysis["delta_vs_paper_qwen_peft_pp"],
        "analysis.delta_vs_paper_qwen_peft_pp",
    )
    if analysis["paper_references"]["spt_rii_table5_70_shot_f1"] != PAPER_SPT_RII_70:
        raise AssertionError("Paper SPT-RII reference mismatch")
    if analysis["paper_references"]["qwen3_8b_table6_peft_f1"] != PAPER_QWEN3_8B_PEFT:
        raise AssertionError("Paper Qwen PEFT reference mismatch")

    base_correct = base_prediction == labels
    gate_correct = expected_prediction == labels
    changed = expected_prediction != base_prediction
    routing = {
        "candidate_rows": int(candidate_mask.sum()),
        "consensus_rows": int(expected_consensus.sum()),
        "changed_rows": int(changed.sum()),
        "corrected_rows": int(np.count_nonzero((~base_correct) & gate_correct)),
        "harmed_rows": int(np.count_nonzero(base_correct & (~gate_correct))),
    }
    routing["net_corrected_rows"] = routing["corrected_rows"] - routing["harmed_rows"]
    for key, value in routing.items():
        if analysis["routing"][key] != value:
            raise AssertionError(f"Analysis routing statistic mismatch: {key}")
    if analysis["compute"] != result["qwen_stats"]:
        raise AssertionError("Analysis compute record differs from the bound result")

    row_ci = row_bootstrap(labels, base_prediction, expected_prediction, BOOTSTRAP_ITERATIONS, 20260722)
    cluster_ci = cluster_bootstrap(
        labels,
        base_prediction,
        expected_prediction,
        test["live_id"].astype(str).to_numpy(),
        BOOTSTRAP_ITERATIONS,
        20260822,
    )
    assert_bootstrap(analysis["paired_bootstrap"]["row"], row_ci, "analysis.row_bootstrap")
    assert_bootstrap(
        analysis["paired_bootstrap"]["live_id_cluster"],
        cluster_ci,
        "analysis.cluster_bootstrap",
    )
    if not (row_ci["delta_ci95"][0] <= 0 <= row_ci["delta_ci95"][1]):
        raise AssertionError("Expected row bootstrap uncertainty warning is missing")
    if not (cluster_ci["delta_ci95"][0] <= 0 <= cluster_ci["delta_ci95"][1]):
        raise AssertionError("Expected cluster bootstrap uncertainty warning is missing")

    source = (EXPERIMENT / "src" / "run_same_budget_qwen_gate.py").read_text(encoding="utf-8")
    prediction_write = source.index("atomic_to_csv(prediction_frame, prediction_path)")
    label_read = source.index('labels = pd.read_csv(TEST_FILE, encoding="utf-8-sig", usecols=["row_id", "label"]')
    if prediction_write >= label_read:
        raise AssertionError("Current implementation reads test labels before locking gate predictions")
    if selection_path.stat().st_mtime_ns > prediction_path.stat().st_mtime_ns:
        raise AssertionError("Selection file is newer than the bound test predictions")
    if prediction_path.stat().st_mtime_ns > result_path.stat().st_mtime_ns:
        raise AssertionError("Result file predates the bound prediction file")

    verify_live_model_digest(QWEN_MODEL, QWEN_MODEL_DIGEST)

    strict_base_result = json.loads((RESULTS / f"{QWEN_BASE_TAG}_result.json").read_text(encoding="utf-8"))
    assert_metric_dict(base_metrics, strict_base_result["ensemble_test_metrics"], "strict_base.ensemble_test_metrics")
    if strict_base_result["train_signature"] != selection["human_train_signature"]:
        raise AssertionError("Qwen verifier and strict base use different human train rows")
    if strict_base_result["validation_signature"] != selection["human_validation_signature"]:
        raise AssertionError("Qwen verifier and strict base use different validation rows")

    return {
        "selection_sha256": sha256(selection_path),
        "result_sha256": sha256(result_path),
        "analysis_sha256": sha256(analysis_path),
        "prediction_sha256": sha256(prediction_path),
        "cache_sha256": sha256(cache_path),
        "test_source_sha256": sha256(test_path),
        "cache_lines": cache_line_count,
        "active_vote_total": active_vote_total,
        "base_macro_f1": base_metrics["macro_f1"],
        "gate_macro_f1": gate_metrics["macro_f1"],
        "delta_local_pp": analysis["delta_vs_local_base_pp"],
        "delta_paper_pp": analysis["delta_vs_paper_spt_rii_pp"],
        "routing": routing,
        "row_ci": row_ci,
        "cluster_ci": cluster_ci,
    }


def numeric_values(text: str) -> list[float]:
    normalized = text.replace("{,}", "").replace(",", "").replace("−", "-")
    return [float(value) for value in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", normalized)]


def assert_number_present(text: str, expected: float, label: str, atol: float = 0.011) -> None:
    values = numeric_values(text)
    if not any(np.isclose(value, expected, rtol=0.0, atol=atol) for value in values):
        raise AssertionError(f"{label} does not contain expected value {expected}")


def assert_frame_number(frame: pd.DataFrame, expected: float, label: str, atol: float = 0.011) -> None:
    values: list[float] = []
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce").dropna()
        values.extend(converted.astype(float).tolist())
    alternatives = (expected, expected / 100.0) if abs(expected) <= 100 else (expected,)
    if not any(
        np.isclose(value, target, rtol=0.0, atol=atol if target == expected else atol / 100)
        for value in values
        for target in alternatives
    ):
        raise AssertionError(f"{label} does not contain expected value {expected}")


def audit_qwen_assets(audit: dict[str, Any]) -> None:
    image_paths = {
        "svg": FIGURES / "fig9_qwen_gate_comparison.svg",
        "pdf": FIGURES / "fig9_qwen_gate_comparison.pdf",
        "png": FIGURES / "fig9_qwen_gate_comparison.png",
        "tiff": FIGURES / "fig9_qwen_gate_comparison.tiff",
    }
    source_path = FIGURES / "source_data_fig9.csv"
    qa_path = FIGURES / "fig9_qa.txt"
    macros_path = REPORT_ASSETS / "qwen_gate_macros.tex"
    comparison_table_path = REPORT_ASSETS / "table_qwen_gate_comparison.tex"
    routing_table_path = REPORT_ASSETS / "table_qwen_gate_routing.tex"
    for path in (*image_paths.values(), source_path, qa_path, macros_path, comparison_table_path, routing_table_path):
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty Qwen report asset: {path}")

    if image_paths["pdf"].read_bytes()[:4] != b"%PDF":
        raise AssertionError("fig9 PDF signature is invalid")
    if image_paths["png"].read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("fig9 PNG signature is invalid")
    if image_paths["tiff"].read_bytes()[:4] not in (b"II*\x00", b"MM\x00*"):
        raise AssertionError("fig9 TIFF signature is invalid")
    svg = image_paths["svg"].read_text(encoding="utf-8")
    if "<svg" not in svg or "<text" not in svg:
        raise AssertionError("fig9 SVG is not an editable text-bearing vector figure")
    for expected, label in ((80.94, "gate Macro-F1"), (80.60, "base Macro-F1"), (79.23, "paper SPT-RII")):
        assert_number_present(svg, expected, f"fig9 SVG {label}")

    source = pd.read_csv(source_path, encoding="utf-8-sig")
    if source.empty:
        raise AssertionError("fig9 source data is empty")
    source_text = source.to_csv(index=False).lower()
    for token in ("qwen", "dapt", "spt"):
        if token not in source_text:
            raise AssertionError(f"fig9 source data is missing method token: {token}")
    for expected, label in (
        (100 * audit["base_macro_f1"], "base Macro-F1"),
        (100 * audit["gate_macro_f1"], "gate Macro-F1"),
        (100 * PAPER_SPT_RII_70, "paper SPT-RII"),
        (100 * PAPER_QWEN3_8B_PEFT, "paper Qwen PEFT"),
        (audit["routing"]["candidate_rows"], "candidate rows"),
        (audit["routing"]["consensus_rows"], "consensus rows"),
        (audit["routing"]["changed_rows"], "changed rows"),
        (audit["routing"]["corrected_rows"], "corrected rows"),
        (audit["routing"]["harmed_rows"], "harmed rows"),
        (audit["routing"]["net_corrected_rows"], "net corrected rows"),
    ):
        assert_frame_number(source, float(expected), f"fig9 source {label}")

    qa = qa_path.read_text(encoding="utf-8")
    for marker in (
        "Figure: fig9_qwen_gate_comparison",
        "Backend: Python/matplotlib",
        "Source data: source_data_fig9.csv",
        "PNG dimensions:",
        "PNG non-white fraction:",
        "Editable SVG text nodes:",
        "Image integrity:",
    ):
        if marker not in qa:
            raise AssertionError(f"fig9 QA is missing marker: {marker}")
    dimensions = re.search(r"PNG dimensions:\s*(\d+)\s*x\s*(\d+)", qa)
    nonwhite = re.search(r"PNG non-white fraction:\s*([0-9.]+)", qa)
    text_nodes = re.search(r"Editable SVG text nodes:\s*(\d+)", qa)
    if not dimensions or min(int(dimensions.group(1)), int(dimensions.group(2))) < 800:
        raise AssertionError("fig9 QA reports inadequate PNG dimensions")
    if not nonwhite or not 0.01 < float(nonwhite.group(1)) < 0.95:
        raise AssertionError("fig9 QA reports an implausible non-white fraction")
    if not text_nodes or int(text_nodes.group(1)) < 10:
        raise AssertionError("fig9 QA reports too few editable text nodes")
    for expected, label in ((80.94, "gate score"), (0.34, "local delta"), (1.71, "paper delta")):
        assert_number_present(qa, expected, f"fig9 QA {label}")

    macros = macros_path.read_text(encoding="utf-8")
    comparison_table = comparison_table_path.read_text(encoding="utf-8")
    routing_table = routing_table_path.read_text(encoding="utf-8")
    if "\\newcommand" not in macros:
        raise AssertionError("Qwen LaTeX macro file contains no commands")
    if not any(token in comparison_table for token in ("\\begin{tabular}", "\\begin{tabularx}")) or not any(
        token in routing_table for token in ("\\begin{tabular}", "\\begin{tabularx}")
    ):
        raise AssertionError("Qwen LaTeX table asset is not a tabular environment")
    for expected, label in (
        (80.60, "base score"),
        (80.94, "gate score"),
        (0.34, "local delta"),
        (1.71, "paper delta"),
        (-0.26, "row CI low"),
        (0.92, "row CI high"),
        (-0.37, "cluster CI low"),
        (1.54, "cluster CI high"),
        (1594, "candidate rows"),
        (1567, "consensus rows"),
        (648, "changed rows"),
        (349, "corrected rows"),
        (299, "harmed rows"),
        (50, "net corrected rows"),
    ):
        assert_number_present(macros, expected, f"Qwen macros {label}")
    for expected, label in (
        (75.24, "paper Qwen PEFT"),
        (79.23, "paper SPT-RII"),
        (80.60, "local base"),
        (80.94, "local gate"),
        (0.34, "local delta"),
        (1.71, "paper delta"),
    ):
        assert_number_present(comparison_table, expected, f"Qwen comparison table {label}")
    for expected, label in (
        (1594, "candidate rows"),
        (1567, "consensus rows"),
        (648, "changed rows"),
        (349, "corrected rows"),
        (299, "harmed rows"),
        (50, "net corrected rows"),
    ):
        assert_number_present(routing_table, expected, f"Qwen routing table {label}")


def main() -> None:
    registry = json.loads(CONFIG.read_text(encoding="utf-8"))
    registry_hash = sha256(CONFIG)
    assert registry["protocol_id"] == "recdy_strict70_factorial_v1"
    shared = registry["shared"]
    assert shared["split_seed"] == 20260718
    assert shared["member_seeds"] == [100, 101, 102, 103, 104]
    assert shared["shot"] == 70

    records = {}
    for preset in PRESETS:
        path = RESULTS / f"{preset}_result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["status"] == "completed"
        assert record["protocol_registry_sha256"] == registry_hash
        assert record["member_seeds"] == [100, 101, 102, 103, 104]
        assert record["train_rows"] == 140
        assert record["validation_rows"] == 140
        assert record["test_rows"] == 9069
        assert record["threshold"] == 0.5
        assert record["test_tuning_evaluations"] == 0
        assert record["test_labels_loaded_after_selection"] is True
        assert len(record["member_results"]) == 5
        assert len(record["member_fixed_threshold_test_metrics"]) == 5
        assert all("macro_f1" in member for member in record["member_fixed_threshold_test_metrics"])
        records[preset] = record

    signatures = {
        key: (value["train_signature"], value["validation_signature"], value["test_input_signature"])
        for key, value in records.items()
    }
    assert len(set(signatures.values())) == 1

    summary = pd.read_csv(RESULTS / "strict70_summary.csv")
    assert set(summary["preset"]) == {"base_ce", "dapt_ce", "base_rdrop", "dapt_rdrop"}
    best = summary.loc[summary["preset"] == "dapt_rdrop"].iloc[0]
    base = summary.loc[summary["preset"] == "base_ce"].iloc[0]
    assert float(best["ensemble_macro_f1"]) > float(base["ensemble_macro_f1"])
    assert float(best["delta_ensemble_vs_paper_pp"]) > 0

    analysis = json.loads((RESULTS / "strict70_analysis.json").read_text(encoding="utf-8"))
    assert analysis["exact_content_overlap_audit"]["test_rows_with_exact_train_content"] == 1310
    assert analysis["paired_initialization_statistics"]["dapt_rdrop"]["all_seed_deltas_positive"] is True
    bootstrap = analysis["ensemble_bootstrap"]["dapt_rdrop"]
    assert bootstrap["row_bootstrap"]["delta_ci95"][0] > 0
    assert bootstrap["cluster_bootstrap"]["delta_ci95"][0] > 0

    dapt = json.loads((RESULTS / "dapt_result.json").read_text(encoding="utf-8"))
    assert dapt["tied_word_embeddings"] is True
    assert dapt["input_content_sha256"] == "39f6ab8b93c28baffdb5618e0a27e4a163cd9a5127334b605c81d2ee04b743ab"

    qwen = audit_qwen_gate()
    audit_qwen_assets(qwen)

    qa = (FIGURES / "fig8_qa.txt").read_text(encoding="utf-8")
    assert "Image integrity:" in qa
    pdf = REPORT / "improvement_comparison.pdf"
    assert pdf.exists() and pdf.stat().st_size > 1_000_000

    print(
        json.dumps(
            {
                "status": "passed",
                "registry_sha256": registry_hash,
                "presets": list(PRESETS),
                "dapt_rdrop_member_mean": round(float(best["member_mean_macro_f1"]) * 100, 2),
                "dapt_rdrop_ensemble": round(float(best["ensemble_macro_f1"]) * 100, 2),
                "ensemble_delta_vs_paper_pp": round(float(best["delta_ensemble_vs_paper_pp"]), 2),
                "qwen_gate_macro_f1": round(float(qwen["gate_macro_f1"]) * 100, 2),
                "qwen_delta_vs_local_pp": round(float(qwen["delta_local_pp"]), 2),
                "qwen_delta_vs_paper_pp": round(float(qwen["delta_paper_pp"]), 2),
                "qwen_artifact_hashes": {
                    key: qwen[key]
                    for key in (
                        "selection_sha256",
                        "result_sha256",
                        "analysis_sha256",
                        "prediction_sha256",
                        "cache_sha256",
                        "test_source_sha256",
                    )
                },
                "pdf_bytes": pdf.stat().st_size,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
