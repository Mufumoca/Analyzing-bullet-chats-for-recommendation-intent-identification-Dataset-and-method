from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import DATA_DIR, RESULTS_DIR, dump_json, ensure_dirs, set_seed, text_for_condition


MODEL_NAME = "hfl/chinese-roberta-wwm-ext"


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def load_frames(suite: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = "qwen_generated" if suite == "qwen" else suite
    train = pd.read_csv(DATA_DIR / f"douyin_train_{suffix}.csv", encoding="utf-8-sig")
    test = pd.read_csv(DATA_DIR / f"douyin_test_{suffix}.csv", encoding="utf-8-sig")
    return train, test


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[int] = []
    probabilities: list[float] = []
    labels: list[int] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(**batch)
            losses.append(float(output.loss.detach().cpu()))
            probs = torch.softmax(output.logits, dim=-1)[:, 1]
            predictions.extend(output.logits.argmax(dim=-1).detach().cpu().tolist())
            probabilities.extend(probs.detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    metrics: dict[str, object] = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics, y_pred, np.asarray(probabilities, dtype=float)


def run(
    suite: str,
    condition: str,
    seed: int,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    max_length: int,
) -> dict[str, object]:
    set_seed(seed)
    train, test = load_frames(suite)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_encoding = tokenizer(
        text_for_condition(train, condition),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    test_encoding = tokenizer(
        text_for_condition(test, condition),
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=True)

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
        mean_train_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch, "train_loss": mean_train_loss})
        print(
            f"suite={suite} condition={condition} seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={mean_train_loss:.4f}",
            flush=True,
        )

    final_metrics, predictions, probabilities = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - started
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    prediction_frame = test[["row_id", "label", "content", "official_background"]].copy()
    if "qwen_background" in test:
        prediction_frame["qwen_background"] = test["qwen_background"]
    prediction_frame["prediction"] = predictions
    prediction_frame["positive_probability"] = probabilities
    prediction_frame.to_csv(
        RESULTS_DIR / f"predictions_roberta_{suite}_{condition}_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result: dict[str, object] = {
        "model": MODEL_NAME,
        "suite": suite,
        "condition": condition,
        "seed": seed,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
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
        "history": history,
        **final_metrics,
    }
    dump_json(RESULTS_DIR / f"roberta_{suite}_{condition}_seed{seed}.json", result)
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("main", "qwen"), required=True)
    parser.add_argument("--condition", choices=("content", "context", "official", "qwen"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    if args.condition == "qwen" and args.suite != "qwen":
        raise ValueError("The qwen condition requires --suite qwen")
    ensure_dirs()
    results = [
        run(
            suite=args.suite,
            condition=args.condition,
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
