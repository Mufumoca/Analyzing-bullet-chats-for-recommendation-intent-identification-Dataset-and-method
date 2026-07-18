from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import DATA_DIR, RESULTS_DIR, dump_json, ensure_dirs, set_seed
from train_roberta import EncodedTextDataset, evaluate, MODEL_NAME


IMPROVEMENT_RESULTS_DIR = RESULTS_DIR / "improvement"
SEED = 20260718


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DATA_DIR / path


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "content", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row_id in {path}")
    return frame.reset_index(drop=True)


def texts(frame: pd.DataFrame, condition: str, qwen_column: str) -> list[str]:
    content = frame["content"].fillna("").astype(str)
    if condition == "content":
        return content.tolist()
    if qwen_column not in frame:
        raise ValueError(f"Missing {qwen_column}")
    explanation = frame[qwen_column].fillna("").astype(str)
    return ("弹幕：" + content + " [SEP] 语境释义：" + explanation).tolist()


def signature(frame: pd.DataFrame, condition: str, qwen_column: str) -> str:
    columns = ["row_id", "content", "label"]
    if qwen_column in frame:
        columns.append(qwen_column)
    payload = frame[columns].astype(str).to_csv(index=False)
    return hashlib.sha256(
        json.dumps({"condition": condition, "qwen_column": qwen_column, "payload": payload}, sort_keys=True).encode()
    ).hexdigest()


def fit_branch(
    train: pd.DataFrame,
    target: pd.DataFrame,
    condition: str,
    qwen_column: str,
    seed: int,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    max_length: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_encoding = tokenizer(
        texts(train, condition, qwen_column),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    target_encoding = tokenizer(
        texts(target, condition, qwen_column),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    train_dataset = EncodedTextDataset(train_encoding, train["label"].to_numpy(dtype=int))
    target_dataset = EncodedTextDataset(target_encoding, target["label"].to_numpy(dtype=int))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    update_steps = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = update_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**batch)
                loss = output.loss / accumulation_steps
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * accumulation_steps)
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
    metrics, predictions, probabilities = evaluate(model, target_loader, device)
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    result = {
        "condition": condition,
        "seed": seed,
        "train_rows": len(train),
        "target_rows": len(target),
        "elapsed_seconds": elapsed,
        "peak_torch_memory_mb": peak_memory,
        "history": history,
        **metrics,
    }
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result, predictions, probabilities


def choose_fusion(y: np.ndarray, content_probability: np.ndarray, explanation_probability: np.ndarray) -> dict[str, object]:
    candidates: list[tuple[float, float, float]] = []
    from sklearn.metrics import f1_score

    for alpha in np.arange(0.0, 1.0001, 0.05):
        fused = alpha * content_probability + (1.0 - alpha) * explanation_probability
        for threshold in np.arange(0.40, 0.6001, 0.01):
            prediction = (fused >= threshold).astype(int)
            score = float(f1_score(y, prediction, average="macro", zero_division=0))
            candidates.append((score, float(alpha), float(threshold)))
    candidates.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), abs(item[2] - 0.5)))
    score, alpha, threshold = candidates[0]
    return {
        "alpha_content": alpha,
        "alpha_explanation": 1.0 - alpha,
        "threshold": threshold,
        "validation_macro_f1": score,
        "grid_alpha": [round(float(value), 2) for value in np.arange(0.0, 1.0001, 0.05)],
        "grid_threshold": [round(float(value), 2) for value in np.arange(0.40, 0.6001, 0.01)],
    }


def run(
    train_path: Path,
    test_path: Path,
    tag: str,
    qwen_column: str,
    seed: int,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    max_length: int,
    validation_fraction: float,
) -> dict[str, object]:
    train = load_frame(train_path)
    test = load_frame(test_path)
    if set(train["row_id"]) & set(test["row_id"]):
        raise ValueError("Train/test overlap")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=SEED)
    train_indices, validation_indices = next(splitter.split(train, train["label"]))
    fit_frame = train.iloc[train_indices].reset_index(drop=True)
    validation_frame = train.iloc[validation_indices].reset_index(drop=True)
    validation_runs: dict[str, object] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for condition in ("content", "qwen"):
        result, _, probabilities = fit_branch(
            fit_frame,
            validation_frame,
            condition,
            qwen_column,
            seed,
            epochs,
            batch_size,
            accumulation_steps,
            max_length,
        )
        validation_runs[condition] = result
        validation_probabilities[condition] = probabilities
    selection = choose_fusion(
        validation_frame["label"].to_numpy(dtype=int),
        validation_probabilities["content"],
        validation_probabilities["qwen"],
    )

    test_runs: dict[str, object] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for condition in ("content", "qwen"):
        result, _, probabilities = fit_branch(
            train,
            test,
            condition,
            qwen_column,
            seed,
            epochs,
            batch_size,
            accumulation_steps,
            max_length,
        )
        test_runs[condition] = result
        test_probabilities[condition] = probabilities
    fused_probability = (
        selection["alpha_content"] * test_probabilities["content"]
        + selection["alpha_explanation"] * test_probabilities["qwen"]
    )
    fused_prediction = (fused_probability >= selection["threshold"]).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_test = test["label"].to_numpy(dtype=int)
    fused_metrics = {
        "accuracy": float(accuracy_score(y_test, fused_prediction)),
        "precision_pos": float(precision_score(y_test, fused_prediction, zero_division=0)),
        "recall_pos": float(recall_score(y_test, fused_prediction, zero_division=0)),
        "f1_pos": float(f1_score(y_test, fused_prediction, zero_division=0)),
        "macro_precision": float(precision_score(y_test, fused_prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, fused_prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, fused_prediction, average="macro", zero_division=0)),
    }
    IMPROVEMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = test[["row_id", "live_id", "label", "content"]].copy()
    predictions[qwen_column] = test[qwen_column]
    predictions["content_probability"] = test_probabilities["content"]
    predictions["explanation_probability"] = test_probabilities["qwen"]
    predictions["positive_probability"] = fused_probability
    predictions["prediction"] = fused_prediction
    prediction_path = IMPROVEMENT_RESULTS_DIR / f"predictions_roberta_{tag}_seed{seed}.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    result = {
        "model": MODEL_NAME,
        "tag": tag,
        "seed": seed,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_rows": len(train),
        "validation_rows": len(validation_frame),
        "test_rows": len(test),
        "validation_fraction": validation_fraction,
        "selection_seed": SEED,
        "train_signature": signature(train, "qwen", qwen_column),
        "test_signature": signature(test, "qwen", qwen_column),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "max_length": max_length,
        "validation_runs": validation_runs,
        "selection": selection,
        "test_branch_runs": test_runs,
        "test_fused_metrics": fused_metrics,
        "test_evaluations_per_branch": 1,
        "test_tuning_evaluations": 0,
        "prediction_file": str(prediction_path),
    }
    dump_json(IMPROVEMENT_RESULTS_DIR / f"roberta_{tag}_seed{seed}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--qwen-column", default="qwen_background_compact")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    ensure_dirs()
    train_path = resolve_path(args.train_file)
    test_path = resolve_path(args.test_file)
    results = [
        run(
            train_path,
            test_path,
            args.tag,
            args.qwen_column,
            seed,
            args.epochs,
            args.batch_size,
            args.accumulation_steps,
            args.max_length,
            args.validation_fraction,
        )
        for seed in args.seeds
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
