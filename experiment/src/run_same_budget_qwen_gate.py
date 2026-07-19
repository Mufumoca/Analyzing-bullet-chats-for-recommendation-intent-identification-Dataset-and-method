"""Selective Qwen verifier for the strict 70-shot experiment.

The RoBERTa ensemble remains the primary classifier.  Qwen is queried only for
low-confidence rows, with three short, retrieval-grounded prompts.  The gate
rule is selected on the locked 140-row validation split and the test labels are
loaded only after all test-side calls have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import linear_kernel

from run_paper_matched import make_split
from run_same_budget import load_frame, signature

try:
    from generate_qwen import GpuMonitor
except ImportError:  # pragma: no cover - only used on machines without NVML
    GpuMonitor = None  # type: ignore[assignment,misc]


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiment"
DATA = EXPERIMENT / "data" / "processed"
RESULTS = EXPERIMENT / "results" / "same_budget"
LOGS = EXPERIMENT / "logs" / "same_budget_qwen_gate"
TRAIN_FILE = DATA / "douyin_train_full.csv"
TEST_FILE = DATA / "douyin_test_full.csv"
BASE_TAG = "strict70_dapt_rdrop_5seed_v1"
MODEL = "qwen3:8b"
MODEL_DIGEST = "e4b5fd7f8af048d3c02e0357274238a9e93da51936665599ccb957aa42bfe173"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
SPLIT_SEED = 20260718
MEMBER_SEEDS = [100, 101, 102, 103, 104]
PROMPT_VERSION = "rag_verifier_3view_v1"
PARSER_VERSION = "json_evidence_parser_v1"
VARIANTS = ("explicit_behavior", "latent_intent", "negative_filter")
MIN_CONFIDENCE = 80.0
ADAPTIVE_TEST_VOTING = True
VOTE_REDUCER = "adaptive_first_two_then_third_v1"
GENERATION_OPTIONS = {
    "temperature": 0,
    "num_predict": 64,
    "num_ctx": 2048,
}
TEST_COLUMNS = ["row_id", "live_id", "content", "official_background", "context_json"]
QWEN_INPUT_COLUMNS = ["row_id", "live_id", "content", "context_json"]
MARGINS = tuple(round(value, 2) for value in np.arange(0.05, 0.401, 0.01))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verify_model_digest() -> None:
    response = requests.get(OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags", timeout=10)
    response.raise_for_status()
    models = response.json().get("models", [])
    match = next((item for item in models if item.get("name") == MODEL), None)
    if match is None or match.get("digest") != MODEL_DIGEST:
        raise AssertionError(f"Ollama model digest mismatch for {MODEL}: {match}")


def normalize(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).strip(" \t\r\n。，！？；;,.!?：:")


def context_items(raw: str) -> list[str]:
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def context20_map() -> dict[str, list[str]]:
    """Rebuild the paper-style 20-comment window from text-only inputs."""

    columns = ["row_id", "live_id", "content", "all_sequence_index"]
    train = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig", usecols=columns, dtype=str, keep_default_na=False)
    test = pd.read_csv(TEST_FILE, encoding="utf-8-sig", usecols=columns, dtype=str, keep_default_na=False)
    combined = pd.concat([train, test], ignore_index=True)
    combined["all_sequence_index"] = pd.to_numeric(combined["all_sequence_index"], errors="raise")
    if combined["row_id"].duplicated().any():
        raise AssertionError("Duplicate row_id while rebuilding context window")
    result: dict[str, list[str]] = {}
    for _, group in combined.sort_values(["live_id", "all_sequence_index"]).groupby("live_id", sort=False):
        history: list[str] = []
        for row in group.itertuples(index=False):
            result[str(row.row_id)] = history[-20:]
            history.append(str(row.content))
    return result


def label_free_source_signature(path: Path) -> str:
    columns = ["row_id", "live_id", "content", "all_sequence_index"]
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns, dtype=str, keep_default_na=False)
    return sha256_text(frame[columns].to_csv(index=False))


class Retriever:
    def __init__(self, train: pd.DataFrame) -> None:
        self.train = train.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        self.matrix = self.vectorizer.fit_transform(self.train["content"].fillna("").astype(str))

    def examples(self, content: str, per_class: int = 3) -> list[dict[str, Any]]:
        query = self.vectorizer.transform([str(content)])
        scores = linear_kernel(query, self.matrix).ravel()
        chosen: list[tuple[float, int]] = []
        for label in (0, 1):
            indices = np.flatnonzero(self.train["label"].to_numpy(dtype=int) == label)
            ranked = sorted(indices.tolist(), key=lambda index: (-float(scores[index]), index))
            chosen.extend((float(scores[index]), index) for index in ranked[:per_class])
        chosen.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "row_id": str(self.train.iloc[index]["row_id"]),
                "label": int(self.train.iloc[index]["label"]),
                "content": str(self.train.iloc[index]["content"]),
                "similarity": round(score, 6),
            }
            for score, index in chosen
        ]


def build_prompt(row: pd.Series, examples: list[dict[str, Any]], variant: str) -> str:
    history = context_items(str(row.get("context_json", "[]")))
    history_text = "；".join(history[-20:]) if history else "（无前文）"
    example_text = "\n".join(
        f"示例{i}: 标签={item['label']}；弹幕={item['content']}"
        for i, item in enumerate(examples, start=1)
    )
    if variant == "explicit_behavior":
        focus = "优先依据数据集中的相似标注示例，判断是否表达对商品、价格、库存、规格、优惠或购买的推荐/咨询需求。"
        order = "先看标签定义，再看示例。"
    elif variant == "latent_intent":
        focus = "必须寻找当前弹幕中的直接文字证据；没有足够证据时输出 abstain=true，不要凭空补全意图。"
        order = "先找当前弹幕证据，再用示例校准标签。"
    else:
        focus = "把当前弹幕与正负示例对比；不能因为出现商品词就自动判为正类。"
        order = "先做正负对比，再给出标签。"
    return (
        "你是 RecDY 电商直播弹幕的独立复核员。只判断当前弹幕，不输出思维过程。\n"
        "标签0：闲聊、称赞、主播互动或不表达商品推荐/购买需求。\n"
        "标签1：询问或表达商品、价格、库存、规格、优惠、使用方式、购买兴趣或推荐需求。\n"
        f"{focus}{order}\n"
        "前文只能用于指代消歧，不可代替当前弹幕证据。\n"
        "检索到的训练示例（标签来自人工标注）：\n"
        f"{example_text}\n"
        f"前文：{history_text}\n"
        f"当前弹幕：{row['content']}\n"
        "只输出一行 JSON，不要 Markdown。label 必须是整数 0 或 1，confidence 必须是 0 到 100 的整数，"
        "abstain 必须是布尔值；格式示例："
        '{"label":1,"confidence":85,"evidence":"原文片段","abstain":false}'
        "。evidence 必须逐字来自当前弹幕；不确定就 abstain=true。/no_think"
    )


def parse_response(raw: str, content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL | re.IGNORECASE).strip()
    candidate = cleaned
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        label_match = re.search(r"(?:label|标签)\s*[=:：]\s*([01])", cleaned, re.IGNORECASE)
        value = {"label": int(label_match.group(1))} if label_match else {}
    try:
        label = int(value.get("label"))
    except (TypeError, ValueError, AttributeError):
        label = -1
    try:
        confidence = int(value.get("confidence", 0))
    except (TypeError, ValueError, AttributeError):
        confidence = 0
    evidence = str(value.get("evidence", "")) if isinstance(value, dict) else ""
    raw_abstain = value.get("abstain", False) if isinstance(value, dict) else False
    if isinstance(raw_abstain, str):
        abstain = raw_abstain.strip().lower() in {"1", "true", "yes", "y"}
    else:
        abstain = bool(raw_abstain)
    evidence_ok = bool(normalize(evidence)) and normalize(evidence) in normalize(content)
    valid = label in (0, 1) and 0 <= confidence <= 100 and not abstain and evidence_ok
    return {
        "cleaned_response": cleaned,
        "label": label,
        "confidence": confidence,
        "evidence": evidence,
        "evidence_ok": evidence_ok,
        "abstain": abstain,
        "valid": valid,
    }


def request_one(
    session: requests.Session,
    row: pd.Series,
    examples: list[dict[str, Any]],
    variant: str,
    retries: int = 2,
) -> dict[str, Any]:
    prompt = build_prompt(row, examples, variant)
    input_payload = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "variant": variant,
        "row_id": str(row["row_id"]),
        "content": str(row["content"]),
        "context": context_items(str(row.get("context_json", "[]"))),
        "examples": examples,
    }
    input_hash = sha256_text(canonical(input_payload))
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "think": False,
        "format": "json",
        "stream": False,
        "options": {
            **GENERATION_OPTIONS,
            "seed": SPLIT_SEED + VARIANTS.index(variant),
        },
        "keep_alive": "30m",
    }
    started_all = time.perf_counter()
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            started = time.perf_counter()
            response = session.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            raw = str(data.get("response", ""))
            parsed = parse_response(raw, str(row["content"]))
            record = {
                "cache_key": f"{row['row_id']}::{variant}",
                "row_id": str(row["row_id"]),
                "variant": variant,
                "input_hash": input_hash,
                "prompt_hash": sha256_text(prompt),
                "retrieval_row_ids": [item["row_id"] for item in examples],
                "raw_response": raw,
                **parsed,
                "attempts": attempt,
                "inference_seconds": time.perf_counter() - started,
                "total_seconds": time.perf_counter() - started_all,
                "generated_tokens": int(data.get("eval_count", 0)),
                "error": "" if parsed["valid"] else "invalid_response",
            }
            if parsed["valid"] or attempt == retries:
                return record
            last_error = "invalid_response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(attempt)
    return {
        "cache_key": f"{row['row_id']}::{variant}",
        "row_id": str(row["row_id"]),
        "variant": variant,
        "input_hash": input_hash,
        "prompt_hash": sha256_text(prompt),
        "retrieval_row_ids": [item["row_id"] for item in examples],
        "raw_response": "",
        "label": -1,
        "confidence": 0,
        "evidence": "",
        "evidence_ok": False,
        "abstain": True,
        "valid": False,
        "attempts": retries,
        "inference_seconds": 0.0,
        "total_seconds": time.perf_counter() - started_all,
        "generated_tokens": 0,
        "error": last_error,
    }


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["cache_key"])] = record
    return records


def query_rows(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    cache_path: Path,
    adaptive: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    LOGS.mkdir(parents=True, exist_ok=True)
    retriever = Retriever(train)
    records = load_cache(cache_path)
    session = requests.Session()
    started = time.perf_counter()
    monitor = GpuMonitor() if GpuMonitor is not None else None
    if monitor is not None:
        monitor.start()
    pending = 0
    cache_hits = 0
    http_attempts = 0
    new_records: list[dict[str, Any]] = []

    def early_consensus(row_id: str) -> bool:
        first = records.get(f"{row_id}::{VARIANTS[0]}", {})
        second = records.get(f"{row_id}::{VARIANTS[1]}", {})
        confidence = np.mean([first.get("confidence", 0), second.get("confidence", 0)])
        return bool(
            first.get("valid")
            and second.get("valid")
            and int(first["label"]) == int(second["label"])
            and confidence >= MIN_CONFIDENCE
        )

    with cache_path.open("a", encoding="utf-8") as handle:
        for number, (_, row) in enumerate(frame.iterrows(), start=1):
            examples = retriever.examples(str(row["content"]))
            for variant in VARIANTS:
                prompt = build_prompt(row, examples, variant)
                payload = {
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "variant": variant,
                    "row_id": str(row["row_id"]),
                    "content": str(row["content"]),
                    "context": context_items(str(row.get("context_json", "[]"))),
                    "examples": examples,
                }
                key = f"{row['row_id']}::{variant}"
                expected_hash = sha256_text(canonical(payload))
                expected_prompt_hash = sha256_text(prompt)
                cached = records.get(key)
                if (
                    cached
                    and cached.get("input_hash") == expected_hash
                    and cached.get("prompt_hash") == expected_prompt_hash
                    and cached.get("valid")
                ):
                    cache_hits += 1
                    if adaptive and variant == VARIANTS[1] and early_consensus(str(row["row_id"])):
                        break
                    continue
                record = request_one(session, row, examples, variant)
                records[key] = record
                new_records.append(record)
                http_attempts += int(record.get("attempts", 0))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                pending += 1
                if adaptive and variant == VARIANTS[1] and early_consensus(str(row["row_id"])):
                    break
            if number == 1 or number % 10 == 0 or number == len(frame):
                print(
                    f"[qwen gate] {number}/{len(frame)} rows, new_calls={pending}, elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    if monitor is not None:
        monitor.stop()
    samples = monitor.samples if monitor is not None else []
    row_ids = set(frame["row_id"].astype(str))
    relevant_records = [record for record in records.values() if record.get("row_id") in row_ids]
    stats = {
        "new_records": int(pending),
        "http_attempts": int(http_attempts),
        "cache_hits": int(cache_hits),
        "maximum_records": int(len(frame) * len(VARIANTS)),
        "available_records": int(len(relevant_records)),
        "valid_votes": int(sum(bool(record.get("valid")) for record in relevant_records)),
        "this_run_wall_seconds": float(time.perf_counter() - started),
        "this_run_new_inference_seconds": float(sum(float(record.get("inference_seconds", 0.0)) for record in new_records)),
        "historical_recorded_inference_seconds": float(sum(float(record.get("inference_seconds", 0.0)) for record in relevant_records)),
        "peak_gpu_memory_mb": float(max((sample["memory_used_mb"] for sample in samples), default=0.0)),
    }
    return records, stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def protocol_hash(payload: dict[str, Any]) -> str:
    keys = (
        "model", "model_digest", "prompt_version", "parser_version", "generation_options",
        "variants", "context_window",
        "retrieval_per_class", "min_confidence", "adaptive_test_voting", "vote_reducer",
        "label_free_train_signature", "label_free_test_signature", "context20_corpus_signature",
        "qwen_test_input_signature_locked", "base_tag", "human_train_signature",
        "human_validation_signature", "base_validation_file_sha256",
        "base_test_file_sha256",
        "selection",
    )
    return sha256_text(canonical({key: payload[key] for key in keys}))


def aggregate(row_id: str, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    votes = [records.get(f"{row_id}::{variant}", {}) for variant in VARIANTS]
    first_two = votes[:2]
    early_stop = bool(
        all(record.get("valid") for record in first_two)
        and int(first_two[0]["label"]) == int(first_two[1]["label"])
        and np.mean([record["confidence"] for record in first_two]) >= MIN_CONFIDENCE
    )
    active_votes = first_two if early_stop else votes
    valid = [record for record in active_votes if record.get("valid")]
    labels = [int(record["label"]) for record in valid]
    counts = Counter(labels)
    consensus_label = -1
    if valid and max(counts.values()) >= 2:
        consensus_label = int(counts.most_common(1)[0][0])
    supporting_votes = [record for record in valid if int(record["label"]) == consensus_label]
    mean_confidence = (
        float(np.mean([record["confidence"] for record in supporting_votes]))
        if supporting_votes
        else 0.0
    )
    raw_consensus = consensus_label in (0, 1) and counts[consensus_label] >= 2
    return {
        "row_id": row_id,
        "valid_votes": len(valid),
        "labels": labels,
        "consensus_label": consensus_label,
        "raw_consensus": raw_consensus,
        "consensus": raw_consensus and mean_confidence >= MIN_CONFIDENCE,
        "mean_confidence": mean_confidence,
        "supporting_vote_count": len(supporting_votes),
        "vote_records": votes,
        "active_vote_count": len(active_votes),
    }


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def candidate_table(base: pd.DataFrame, qwen: pd.DataFrame, labels: np.ndarray) -> list[dict[str, Any]]:
    base_prob = base["probability"].to_numpy(dtype=float)
    qwen_label = qwen["consensus_label"].to_numpy(dtype=int)
    qwen_valid = qwen["consensus"].to_numpy(dtype=bool)
    candidates = [{"mode": "no_op", "margin": 0.0, "macro_f1": float(metrics(labels, (base_prob >= 0.5).astype(int))["macro_f1"]), "replacement_count": 0}]
    for margin in MARGINS:
        selected = (np.abs(base_prob - 0.5) <= margin) & qwen_valid
        prediction = (base_prob >= 0.5).astype(int)
        prediction[selected] = qwen_label[selected]
        result = metrics(labels, prediction)
        candidates.append(
            {
                "mode": "replace_consensus",
                "margin": float(margin),
                "macro_f1": result["macro_f1"],
                "macro_precision": result["macro_precision"],
                "macro_recall": result["macro_recall"],
                "accuracy": result["accuracy"],
                "replacement_count": int(selected.sum()),
                "replacement_fraction": float(selected.mean()),
                "qwen_valid_count": int(qwen_valid.sum()),
            }
        )
    return candidates


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["macro_f1"]),
            int(item.get("replacement_count", 0)) if item["mode"] != "no_op" else 0,
            float(item.get("margin", 0.0)),
        ),
    )
    return ranked[0]


def validation_stage() -> None:
    verify_model_digest()
    full = load_frame(TRAIN_FILE, with_label=True)
    train, validation = make_split(full, 70, SPLIT_SEED)
    human_train_signature = signature(train, ["row_id", "content", "official_background", "context_json", "label"])
    human_validation_signature = signature(validation, ["row_id", "content", "official_background", "context_json", "label"])
    windows = context20_map()
    train["context_json"] = train["row_id"].map(lambda value: json.dumps(windows[str(value)], ensure_ascii=False))
    validation["context_json"] = validation["row_id"].map(lambda value: json.dumps(windows[str(value)], ensure_ascii=False))
    locked_test = load_frame(TEST_FILE, with_label=False)
    locked_test["context_json"] = locked_test["row_id"].map(
        lambda value: json.dumps(windows[str(value)], ensure_ascii=False)
    )
    base_path = RESULTS / f"{BASE_TAG}_validation_predictions.csv"
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    if not base["row_id"].equals(validation["row_id"]):
        raise AssertionError("Strict validation row order does not match the locked split")
    cache_path = LOGS / "strict70_validation.jsonl"
    records, qwen_stats = query_rows(validation, train, cache_path)
    aggregated = pd.DataFrame([aggregate(str(row_id), records) for row_id in validation["row_id"].astype(str)])
    qwen_path = RESULTS / "strict70_qwen_gate_validation_votes.csv"
    flat = aggregated.drop(columns=["vote_records"])
    atomic_to_csv(flat, qwen_path)
    candidates = candidate_table(base, aggregated, validation["label"].to_numpy(dtype=int))
    selected = select_candidate(candidates)
    output = {
        "experiment": "recdy_strict70_qwen_selective_verifier_v1",
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "generation_options": GENERATION_OPTIONS,
        "base_tag": BASE_TAG,
        "split_seed": SPLIT_SEED,
        "member_seeds": MEMBER_SEEDS,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "human_train_signature": human_train_signature,
        "human_validation_signature": human_validation_signature,
        "context20_train_signature": signature(train, ["row_id", "content", "context_json", "label"]),
        "context20_validation_signature": signature(validation, ["row_id", "content", "context_json", "label"]),
        "label_free_train_signature": label_free_source_signature(TRAIN_FILE),
        "label_free_test_signature": label_free_source_signature(TEST_FILE),
        "context20_corpus_signature": sha256_text(canonical(windows)),
        "qwen_test_input_signature_locked": signature(locked_test, QWEN_INPUT_COLUMNS),
        "selection_policy": "validation_only_macro_f1; test evaluated once after lock",
        "variants": list(VARIANTS),
        "retrieval_per_class": 3,
        "min_confidence": MIN_CONFIDENCE,
        "adaptive_test_voting": ADAPTIVE_TEST_VOTING,
        "vote_reducer": VOTE_REDUCER,
        "context_window": 20,
        "base_validation_file_sha256": sha256_file(base_path),
        "base_test_file_sha256": sha256_file(RESULTS / f"{BASE_TAG}_predictions.csv"),
        "candidates": candidates,
        "selection": selected,
        "validation_base_metrics": metrics(validation["label"].to_numpy(dtype=int), (base["probability"] >= 0.5).astype(int)),
        "validation_qwen_consensus_rate": float(aggregated["consensus"].mean()),
        "validation_votes": str(qwen_path),
        "qwen_stats": qwen_stats,
        "exact_train_content_rows": int(validation["content"].isin(set(train["content"])).sum()),
        "status": "validation_locked",
    }
    output["selection_hash"] = sha256_text(canonical(output["selection"]))
    output["protocol_hash"] = protocol_hash(output)
    atomic_write_text(
        RESULTS / "strict70_qwen_gate_selection.json",
        json.dumps(output, ensure_ascii=False, indent=2),
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


def test_stage() -> None:
    verify_model_digest()
    selection_path = RESULTS / "strict70_qwen_gate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "validation_locked":
        raise AssertionError("Validation selection is not locked")
    if selection.get("protocol_hash") != protocol_hash(selection):
        raise AssertionError("Selection protocol hash mismatch")
    if selection.get("model") != MODEL or selection.get("model_digest") != MODEL_DIGEST:
        raise AssertionError("Qwen model identity mismatch")
    if selection.get("prompt_version") != PROMPT_VERSION or selection.get("variants") != list(VARIANTS):
        raise AssertionError("Qwen prompt protocol mismatch")
    if selection.get("parser_version") != PARSER_VERSION or selection.get("generation_options") != GENERATION_OPTIONS:
        raise AssertionError("Qwen generation/parser protocol mismatch")
    full = load_frame(TRAIN_FILE, with_label=True)
    train, validation = make_split(full, 70, SPLIT_SEED)
    test = load_frame(TEST_FILE, with_label=False)
    current_human_train_signature = signature(train, ["row_id", "content", "official_background", "context_json", "label"])
    current_human_validation_signature = signature(validation, ["row_id", "content", "official_background", "context_json", "label"])
    if current_human_train_signature != selection["human_train_signature"] or current_human_validation_signature != selection["human_validation_signature"]:
        raise AssertionError("Human-label split signature differs from locked validation selection")
    windows = context20_map()
    train["context_json"] = train["row_id"].map(lambda value: json.dumps(windows[str(value)], ensure_ascii=False))
    validation["context_json"] = validation["row_id"].map(lambda value: json.dumps(windows[str(value)], ensure_ascii=False))
    test["context_json"] = test["row_id"].map(lambda value: json.dumps(windows[str(value)], ensure_ascii=False))
    if label_free_source_signature(TRAIN_FILE) != selection["label_free_train_signature"]:
        raise AssertionError("Label-free train source differs from locked protocol")
    if label_free_source_signature(TEST_FILE) != selection["label_free_test_signature"]:
        raise AssertionError("Label-free test source differs from locked protocol")
    if sha256_text(canonical(windows)) != selection["context20_corpus_signature"]:
        raise AssertionError("Reconstructed 20-comment context differs from locked protocol")
    if signature(test, QWEN_INPUT_COLUMNS) != selection["qwen_test_input_signature_locked"]:
        raise AssertionError("Qwen test input differs from validation-locked protocol")
    base_path = RESULTS / f"{BASE_TAG}_predictions.csv"
    if sha256_file(base_path) != selection["base_test_file_sha256"]:
        raise AssertionError("Base test prediction file differs from validation-locked protocol")
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    if not base["row_id"].equals(test["row_id"]):
        raise AssertionError("Strict test row order does not match base predictions")
    gate_enabled = selection["selection"].get("mode") != "no_op"
    margin = float(selection["selection"].get("margin", 0.0))
    candidates = (
        test.loc[np.abs(base["probability"].to_numpy(dtype=float) - 0.5) <= margin].copy()
        if gate_enabled
        else test.iloc[0:0].copy()
    )
    cache_path = LOGS / "strict70_test.jsonl"
    records, qwen_stats = (
        query_rows(candidates, train, cache_path, adaptive=ADAPTIVE_TEST_VOTING)
        if len(candidates)
        else ({}, {})
    )
    aggregated = pd.DataFrame([aggregate(str(row_id), records) for row_id in candidates["row_id"].astype(str)])
    aggregate_map = {str(row["row_id"]): row for row in aggregated.to_dict(orient="records")}
    prediction = (base["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
    replaced = np.zeros(len(test), dtype=bool)
    for index, row_id in enumerate(test["row_id"].astype(str)):
        item = aggregate_map.get(row_id)
        if item and item["consensus"]:
            prediction[index] = int(item["consensus_label"])
            replaced[index] = True
    prediction_frame = test[["row_id", "live_id"]].copy()
    prediction_frame["base_probability"] = base["probability"].to_numpy(dtype=float)
    prediction_frame["qwen_consensus"] = [aggregate_map.get(str(row_id), {}).get("consensus", False) for row_id in test["row_id"]]
    prediction_frame["qwen_label"] = [aggregate_map.get(str(row_id), {}).get("consensus_label", -1) for row_id in test["row_id"]]
    prediction_frame["prediction"] = prediction
    prediction_path = RESULTS / "strict70_qwen_gate_predictions.csv"
    atomic_to_csv(prediction_frame, prediction_path)

    # Test labels are intentionally joined only after Qwen calls and prediction files are locked.
    labels = pd.read_csv(TEST_FILE, encoding="utf-8-sig", usecols=["row_id", "label"], dtype={"row_id": str})
    if not labels["row_id"].equals(test["row_id"]):
        raise AssertionError("Test label order differs from label-free inputs")
    y = labels["label"].astype(int).to_numpy()
    base_prediction = (base["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
    result = {
        "experiment": "recdy_strict70_qwen_selective_verifier_v1",
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "generation_options": GENERATION_OPTIONS,
        "variants": list(VARIANTS),
        "retrieval_per_class": 3,
        "selection_hash": selection["selection_hash"],
        "protocol_hash": selection["protocol_hash"],
        "selection": selection["selection"],
        "context_window": 20,
        "min_confidence": MIN_CONFIDENCE,
        "adaptive_test_voting": ADAPTIVE_TEST_VOTING,
        "vote_reducer": VOTE_REDUCER,
        "base_test_file_sha256": selection["base_test_file_sha256"],
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "human_train_signature": selection["human_train_signature"],
        "human_validation_signature": selection["human_validation_signature"],
        "context20_train_signature": selection["context20_train_signature"],
        "context20_validation_signature": selection["context20_validation_signature"],
        "label_free_train_signature": selection["label_free_train_signature"],
        "label_free_test_signature": selection["label_free_test_signature"],
        "context20_corpus_signature": selection["context20_corpus_signature"],
        "qwen_test_input_signature_locked": selection["qwen_test_input_signature_locked"],
        "qwen_input_signature": signature(test, QWEN_INPUT_COLUMNS),
        "candidate_rows": int(len(candidates)),
        "consensus_rows": int(replaced.sum()),
        "consensus_fraction_test": float(replaced.mean()),
        "qwen_stats": qwen_stats,
        "selection_file_sha256": sha256_file(selection_path),
        "qwen_cache_file_sha256": sha256_file(cache_path),
        "prediction_file_sha256": sha256_file(prediction_path),
        "test_source_file_sha256": sha256_file(TEST_FILE),
        "exact_train_content_rows": int(test["content"].isin(set(train["content"])).sum()),
        "base_metrics": metrics(y, base_prediction),
        "qwen_gate_metrics": metrics(y, prediction),
        "delta_macro_f1_pp": float((f1_score(y, prediction, average="macro") - f1_score(y, base_prediction, average="macro")) * 100),
        "prediction_file": str(prediction_path),
        "test_labels_loaded_after_selection": True,
        "status": "completed",
    }
    atomic_write_text(
        RESULTS / "strict70_qwen_gate_result.json",
        json.dumps(result, ensure_ascii=False, indent=2),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    if args.stage == "validation":
        validation_stage()
    else:
        test_stage()


if __name__ == "__main__":
    main()
