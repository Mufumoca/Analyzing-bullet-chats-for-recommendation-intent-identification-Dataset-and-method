"""Shared constants and prompt utilities for the improvement experiment.

This module deliberately does not import the baseline experiment modules.  The
improvement run therefore has an explicit, versioned prompt and its own output
namespace while still reading the frozen baseline data as input.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "experiment"
BASE_DATA_DIR = EXPERIMENT_ROOT / "data" / "processed"
IMPROVEMENT_DATA_DIR = BASE_DATA_DIR / "improvement"
BASE_RESULTS_DIR = EXPERIMENT_ROOT / "results"
IMPROVEMENT_RESULTS_DIR = BASE_RESULTS_DIR / "improvement"
BASE_LOGS_DIR = EXPERIMENT_ROOT / "logs"
IMPROVEMENT_LOGS_DIR = BASE_LOGS_DIR / "improvement"

MODEL = os.environ.get("BC4RII_QWEN_MODEL", "qwen3:8b")
PROMPT_VERSION = "structured_v2"
SEED = 20260718
CONTEXT_WINDOW = 5
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")


def ensure_improvement_dirs() -> None:
    for path in (
        IMPROVEMENT_DATA_DIR,
        IMPROVEMENT_RESULTS_DIR,
        IMPROVEMENT_LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _context_items(row: Any) -> list[str]:
    value = row.get("context_json", "[]") if hasattr(row, "get") else "[]"
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        parsed = []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def structured_prompt(row: Any) -> str:
    """Build the only prompt used by the improvement generation run.

    The label and the official explanation are intentionally never read here.
    The current comment is delimited separately and the evidence field is
    required to be copied from it, which makes unsupported inferences easier to
    audit after generation.
    """

    history = _context_items(row)
    if history:
        history_text = "\n".join(
            f"历史弹幕{i + 1}：{text}" for i, text in enumerate(history[-CONTEXT_WINDOW:])
        )
    else:
        history_text = "（无可用前文）"
    content = str(row.get("content", ""))
    return (
        "你是电商直播弹幕语义分析助手。请只分析【当前弹幕】，前文仅用于解析当前弹幕中的指代；"
        "如果前文无关，请忽略它。不要判断购买意图，不要输出二分类标签，不要猜测用户没有说出的信息，"
        "不要展示推理过程。\n"
        "必须严格只输出一行，格式为：对象：<对象>；言语行为：<言语行为>；原句证据：<当前弹幕中的连续原文片段>。\n"
        "规则：对象和言语行为使用简短中文短语；原句证据必须逐字来自当前弹幕；"
        "若当前弹幕没有明确对象，填写‘未指明’，但仍须引用当前弹幕；不要添加其它字段、序号或 Markdown。\n"
        "【前文，仅供消歧】\n"
        f"{history_text}\n"
        "【当前弹幕】\n"
        f"{content}\n"
        "现在输出结构化释义。/no_think"
    )


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def input_hash(row: Any, prompt: str | None = None) -> str:
    """Hash model-relevant input, including prompt version and generation options."""

    actual_prompt = prompt if prompt is not None else structured_prompt(row)
    payload = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "seed": SEED,
        "temperature": 0.0,
        "num_predict": 128,
        "num_ctx": 2048,
        "row_id": str(row.get("row_id", "")),
        "content": str(row.get("content", "")),
        "context": _context_items(row),
        "prompt_hash": prompt_hash(actual_prompt),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def clean_response(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```text", "").replace("```", "").strip().strip('"').strip()
    return " ".join(cleaned.split())


_STRUCTURED_RE = re.compile(
    r"对象\s*[：:]\s*(?P<object>.*?)\s*[；;]\s*"
    r"言语行为\s*[：:]\s*(?P<act>.*?)\s*[；;]\s*"
    r"原句证据\s*[：:]\s*(?P<evidence>.*)$",
    flags=re.IGNORECASE,
)


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).strip(" \t\r\n。，！？；;,.!?：:")


def parse_structured_response(raw: str, current_content: str) -> dict[str, Any]:
    """Parse and validate one structured response without using its label."""

    cleaned = clean_response(raw)
    match = _STRUCTURED_RE.search(cleaned)
    if not match:
        return {
            "cleaned_response": cleaned,
            "valid_structure": False,
            "structured_object": "",
            "structured_act": "",
            "structured_evidence": "",
            "validation_error": "missing_required_fields",
        }
    obj = match.group("object").strip()
    act = match.group("act").strip()
    evidence = match.group("evidence").strip()
    norm_content = _normalise_text(current_content)
    norm_evidence = _normalise_text(evidence)
    evidence_ok = bool(norm_evidence) and norm_evidence in norm_content
    valid = bool(obj and act and evidence_ok)
    error = "" if valid else ("evidence_not_in_current_comment" if not evidence_ok else "empty_field")
    return {
        "cleaned_response": cleaned,
        "valid_structure": valid,
        "structured_object": obj,
        "structured_act": act,
        "structured_evidence": evidence,
        "validation_error": error,
    }


def formatted_explanation(parsed: dict[str, Any]) -> str:
    return (
        f"对象：{parsed.get('structured_object', '')}；"
        f"言语行为：{parsed.get('structured_act', '')}；"
        f"原句证据：{parsed.get('structured_evidence', '')}"
    )


def generation_paths(split: str, suffix: str = "") -> tuple[Path, Path, Path]:
    """Return input, checkpoint and generated-output paths for one split."""

    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    size = 1000 if split == "train" else 400
    stem = f"douyin_{split}_improvement{size}_{PROMPT_VERSION}{suffix}"
    input_path = IMPROVEMENT_DATA_DIR / f"douyin_{split}_improvement{size}.csv"
    checkpoint = IMPROVEMENT_LOGS_DIR / f"qwen_generation_{split}_{PROMPT_VERSION}{suffix}.jsonl"
    output = IMPROVEMENT_DATA_DIR / f"{stem}.csv"
    return input_path, checkpoint, output


def metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
