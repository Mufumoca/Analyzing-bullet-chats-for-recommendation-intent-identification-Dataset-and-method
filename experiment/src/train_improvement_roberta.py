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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import DATA_DIR, RESULTS_DIR, dump_json, ensure_dirs, set_seed
from train_roberta import EncodedTextDataset, evaluate, MODEL_NAME


IMPROVEMENT_RESULTS_DIR = RESULTS_DIR / "improvement"


def resolve_data_path(value: str) -> Path:
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
    return frame


def build_text(frame: pd.DataFrame, condition: str, qwen_column: str) -> list[str]:
    content = frame["content"].fillna("").astype(str)
    if condition == "content":
        return content.tolist()
    if condition != "qwen":
        raise ValueError(f"Unknown condition: {condition}")
    if qwen_column not in frame:
        raise ValueError(f"Missing {qwen_column} for qwen condition")
    background = frame[qwen_column].fillna("").astype(str)
    return ("弹幕：" + content + " [SEP] 语境释义：" + background).tolist()


def frame_signature(frame: pd.DataFrame, condition: str, qwen_column: str) -> str:
    columns = ["row_id", "content", "label"]
    if qwen_column in frame:
        columns.append(qwen_column)
    payload = frame[columns].astype(str).to_csv(index=False)
    return hashlib.sha256(
        json.dumps(
            {"condition": condition, "qwen_column": qwen_column, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def run(
    train_path: Path,
    test_path: Path,
    tag: str,
    condition: str,
    qwen_column: str,
    seed: int,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    max_length: int,
) -> dict[str, object]:
    set_seed(seed)
    train = load_frame(train_path)
    test = load_frame(test_path)
    train_text = build_text(train, condition, qwen_column)
    test_text = build_text(test, condition, qwen_column)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_encoding = tokenizer(
        train_text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    test_encoding = tokenizer(
        test_text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    train_dataset = EncodedTextDataset(train_encoding, train["label"].to_numpy(dtype=int))
    test_dataset = EncodedTextDataset(test_encoding, test["label"].to_numpy(dtype=int))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this experiment")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    update_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = update_steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history: list[dict[str, object]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**batch)
                loss = output.loss / accumulation_steps
            scaler.scale(loss).backward()
            epoch_losses.append(float(loss.detach().cpu()) * accumulation_steps)
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        mean_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": mean_loss})
        print(
            f"tag={tag} condition={condition} seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={mean_loss:.4f}",
            flush=True,
        )

    final_metrics, predictions, probabilities = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    IMPROVEMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_frame = test[["row_id", "live_id", "label", "content"]].copy()
    if "official_background" in test:
        prediction_frame["official_background"] = test["official_background"]
    if qwen_column in test:
        prediction_frame[qwen_column] = test[qwen_column]
    prediction_frame["prediction"] = predictions
    prediction_frame["positive_probability"] = probabilities
    prediction_path = IMPROVEMENT_RESULTS_DIR / f"predictions_roberta_{tag}_seed{seed}.csv"
    prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    result: dict[str, object] = {
        "model": MODEL_NAME,
        "tag": tag,
        "condition": condition,
        "seed": seed,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_signature": frame_signature(train, condition, qwen_column),
        "test_signature": frame_signature(test, condition, qwen_column),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "max_length": max_length,
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": elapsed,
        "peak_torch_memory_mb": peak_memory_mb,
        "test_evaluations": 1,
        "prediction_file": str(prediction_path),
        "history": history,
        **final_metrics,
    }
    dump_json(IMPROVEMENT_RESULTS_DIR / f"roberta_{tag}_seed{seed}.json", result)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--condition", choices=("content", "qwen"), required=True)
    parser.add_argument("--qwen-column", default="qwen_background_structured")
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    ensure_dirs()
    train_path = resolve_data_path(args.train_file)
    test_path = resolve_data_path(args.test_file)
    results = [
        run(
            train_path=train_path,
            test_path=test_path,
            tag=args.tag,
            condition=args.condition,
            qwen_column=args.qwen_column,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            max_length=args.max_length,
        )
        for seed in args.seeds
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
