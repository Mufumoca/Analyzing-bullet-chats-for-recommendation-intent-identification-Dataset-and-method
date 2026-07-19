"""Run a label-budget-locked RecDY comparison.

Every member in one run receives exactly the same 70-per-class train and
validation rows.  Members differ only in initialization/dropout randomness.
The test file is loaded without labels while checkpoints are selected; labels
are joined only by the final scoring block.

The script supports both the original checkpoint and a domain-adapted
checkpoint.  It also supports an explanation view represented as a tokenizer
text pair, avoiding literal prompt-prefix artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as transformers_logging
from transformers.utils.hub import cached_file

from common import DATA_DIR, RESULTS_DIR, ROOT, dump_json, set_seed
from run_paper_matched import make_split
from train_roberta import MODEL_NAME


transformers_logging.set_verbosity_error()

RESULTS = RESULTS_DIR / "same_budget"
DEFAULT_PROTOCOL = ROOT / "experiment" / "config" / "same_budget_protocol_v1.json"
TEST_COLUMNS = ["row_id", "live_id", "content", "official_background", "context_json"]


class EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: np.ndarray | None = None) -> None:
        self.encodings = encodings
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.encodings["input_ids"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def signature(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = frame[columns].fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frame(path: Path, with_label: bool) -> pd.DataFrame:
    columns = ["row_id", "live_id", "content", "official_background", "context_json"]
    if with_label:
        columns.append("label")
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns, dtype=str, keep_default_na=False)
    if frame["row_id"].duplicated().any():
        raise ValueError(f"Duplicate row_id in {path}")
    if with_label:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    return frame.reset_index(drop=True)


def view_texts(frame: pd.DataFrame, view: str, explanation_column: str) -> tuple[list[str], list[str] | None]:
    content = frame["content"].fillna("").astype(str).tolist()
    if view == "content":
        return content, None
    if view == "official_pair":
        if explanation_column not in frame:
            raise ValueError(f"Missing explanation column {explanation_column}")
        explanation = frame[explanation_column].fillna("").astype(str).tolist()
        if any(not value.strip() for value in explanation):
            raise ValueError(f"Empty explanation in {explanation_column}")
        return content, explanation
    if view == "context_pair":
        contexts: list[str] = []
        for raw in frame["context_json"].fillna("[]").astype(str):
            try:
                items = json.loads(raw)
            except json.JSONDecodeError:
                items = []
            contexts.append(" [SEP] ".join(str(item) for item in items))
        return content, contexts
    raise ValueError(f"Unknown view: {view}")


def tokenizer_for(model_path: str) -> Any:
    return AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)


def encode(tokenizer: Any, frame: pd.DataFrame, view: str, explanation_column: str, max_length: int) -> dict[str, torch.Tensor]:
    first, second = view_texts(frame, view, explanation_column)
    encoded = tokenizer(
        first,
        text_pair=second,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value for key, value in encoded.items()}


def metrics_from_probs(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_positive": float(f1_score(labels, predictions, zero_division=0)),
        "precision_positive": float(precision_score(labels, predictions, zero_division=0)),
        "recall_positive": float(recall_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "positive_rate": float(predictions.mean()),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_signature(model_path: str) -> dict[str, str]:
    local_path = Path(model_path)
    if local_path.exists():
        candidates = [local_path / "model.safetensors", local_path / "pytorch_model.bin"]
    else:
        candidates = []
        for filename in ("model.safetensors", "pytorch_model.bin"):
            resolved = cached_file(model_path, filename, local_files_only=True, _raise_exceptions_for_missing_entries=False)
            if resolved:
                candidates.append(Path(resolved))
    for candidate in candidates:
        if candidate.exists():
            return {"file": str(candidate), "sha256": file_sha256(candidate)}
    raise FileNotFoundError(f"No cached model weights found for {model_path}")


def load_classifier(model_path: str, rdrop: bool) -> tuple[torch.nn.Module, dict[str, Any]]:
    local = Path(model_path).exists()
    config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
    # Keep the baseline architecture unchanged; the improved branch uses a
    # modestly larger classifier dropout as a pre-registered regularizer.
    if rdrop:
        config.classifier_dropout = 0.2
        config.hidden_dropout_prob = 0.2
        config.attention_probs_dropout_prob = 0.1
    config.num_labels = 2
    config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    config.label2id = {"LABEL_0": 0, "LABEL_1": 1}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        use_safetensors=False,
    )
    if not local:
        return model, {
            "initialization": "base_sequence_classifier",
            "dapt_encoder_keys_loaded": 0,
        }

    checkpoint_path = Path(model_path) / "model.safetensors"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing DAPT checkpoint: {checkpoint_path}")
    dapt_state = load_safetensors(str(checkpoint_path), device="cpu")
    encoder_prefixes = ("bert.embeddings.", "bert.encoder.")
    encoder_state: dict[str, torch.Tensor] = {}
    renamed_layernorm_keys = 0
    for name, value in dapt_state.items():
        if not name.startswith(encoder_prefixes):
            continue
        normalized_name = name
        if name.endswith(".gamma"):
            normalized_name = name[: -len(".gamma")] + ".weight"
            renamed_layernorm_keys += 1
        elif name.endswith(".beta"):
            normalized_name = name[: -len(".beta")] + ".bias"
            renamed_layernorm_keys += 1
        if normalized_name in encoder_state:
            raise ValueError(f"Duplicate normalized DAPT key: {normalized_name}")
        encoder_state[normalized_name] = value
    if not encoder_state:
        raise ValueError("DAPT checkpoint contains no BERT encoder weights")
    incompatible = model.load_state_dict(encoder_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected DAPT encoder keys: {incompatible.unexpected_keys[:5]}")
    allowed_missing_prefixes = ("bert.pooler.", "classifier.")
    invalid_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(allowed_missing_prefixes)
    ]
    if invalid_missing:
        raise ValueError(f"Unexpected missing keys after DAPT encoder load: {invalid_missing[:5]}")
    return model, {
        "initialization": "base_sequence_classifier_plus_dapt_encoder",
        "dapt_checkpoint": str(checkpoint_path),
        "dapt_encoder_keys_loaded": len(encoder_state),
        "renamed_layernorm_keys": renamed_layernorm_keys,
        "preserved_base_pooler": True,
        "expected_missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = torch.log_softmax(logits_a, dim=-1)
    log_b = torch.log_softmax(logits_b, dim=-1)
    a = log_a.exp()
    b = log_b.exp()
    return 0.5 * (
        torch.nn.functional.kl_div(log_a, b, reduction="batchmean")
        + torch.nn.functional.kl_div(log_b, a, reduction="batchmean")
    )


def evaluate_probabilities(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch).logits
            probabilities.extend(torch.softmax(logits, dim=-1)[:, 1].float().cpu().tolist())
    return np.asarray(probabilities, dtype=float)


def fit_member(
    model_path: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    view: str,
    explanation_column: str,
    seed: int,
    max_length: int,
    batch_size: int,
    accumulation_steps: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    rdrop_alpha: float,
    label_smoothing: float,
) -> dict[str, Any]:
    set_seed(seed)
    tokenizer = tokenizer_for(model_path)
    train_encoding = encode(tokenizer, train, view, explanation_column, max_length)
    validation_encoding = encode(tokenizer, validation, view, explanation_column, max_length)
    test_encoding = encode(tokenizer, test, view, explanation_column, max_length)
    train_dataset = EncodedDataset(train_encoding, train["label"].to_numpy(dtype=int))
    validation_dataset = EncodedDataset(validation_encoding, validation["label"].to_numpy(dtype=int))
    test_dataset = EncodedDataset(test_encoding)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
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
    model, model_load_audit = load_classifier(model_path, rdrop_alpha > 0)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    update_steps = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = update_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_score = -1.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits_a = model(**batch).logits
                ce = criterion(logits_a, labels)
                if rdrop_alpha > 0:
                    logits_b = model(**batch).logits
                    ce = 0.5 * (ce + criterion(logits_b, labels))
                    loss = ce + rdrop_alpha * symmetric_kl(logits_a, logits_b)
                else:
                    loss = ce
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            losses.append(float(loss.detach().cpu()))
            if step % accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        validation_probability = evaluate_probabilities(model, validation_loader, device)
        validation_score = metrics_from_probs(validation["label"].to_numpy(dtype=int), validation_probability)["macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "validation_macro_f1": validation_score,
            }
        )
        print(
            f"view={view} seed={seed} epoch={epoch}/{epochs} "
            f"loss={np.mean(losses):.4f} val_macro_f1={validation_score:.4f}",
            flush=True,
        )
        if validation_score > best_score + 1e-9:
            best_score = validation_score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        raise AssertionError("No validation checkpoint captured")
    model.load_state_dict(best_state)
    final_validation_probability = evaluate_probabilities(model, validation_loader, device)
    test_probability = evaluate_probabilities(model, test_loader, device)
    elapsed = time.perf_counter() - started
    result = {
        "model_path": model_path,
        "view": view,
        "seed": seed,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_macro_f1": best_score,
        "validation_metrics": metrics_from_probs(validation["label"].to_numpy(dtype=int), final_validation_probability),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "rdrop_alpha": rdrop_alpha,
        "label_smoothing": label_smoothing,
        "model_load_audit": model_load_audit,
        "elapsed_seconds": elapsed,
        "peak_torch_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "history": history,
    }
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return {
        "result": result,
        "validation_probability": final_validation_probability,
        "test_probability": test_probability,
    }


def summarize_members(member_records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([record[key] for record in member_records], dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(len(values)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the same-budget experiment")
    full_train = load_frame(args.train_file, with_label=True)
    test_inputs = load_frame(args.test_file, with_label=False)
    train, validation = make_split(full_train, args.shot, args.split_seed)
    if set(train.row_id) & set(validation.row_id):
        raise AssertionError("Train/validation overlap")
    if set(train.row_id) & set(test_inputs.row_id):
        raise AssertionError("Train/test overlap")
    if set(validation.row_id) & set(test_inputs.row_id):
        raise AssertionError("Validation/test overlap")

    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = RESULTS / f"{args.tag}_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Locked manifest already exists and will not be overwritten: {manifest_path}"
        )
    protocol_registry_bytes = args.protocol_file.read_bytes()
    protocol = {
        "experiment": "recdy_same_budget_probability_ensemble_v1",
        "protocol_preset": args.preset,
        "protocol_registry": str(args.protocol_file.resolve()),
        "protocol_registry_sha256": hashlib.sha256(protocol_registry_bytes).hexdigest(),
        "model_path": args.model_path,
        "model_checkpoint": checkpoint_signature(args.model_path),
        "train_file": str(args.train_file.resolve()),
        "test_file": str(args.test_file.resolve()),
        "shot_per_class_train": args.shot,
        "shot_per_class_validation": args.shot,
        "split_seed": args.split_seed,
        "member_seeds": [int(value) for value in args.member_seeds],
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test_inputs)),
        "train_signature": signature(train, ["row_id", "content", "official_background", "context_json", "label"]),
        "validation_signature": signature(validation, ["row_id", "content", "official_background", "context_json", "label"]),
        "test_input_signature": signature(test_inputs, TEST_COLUMNS),
        "view": args.view,
        "explanation_column": args.explanation_column,
        "input_role": "oracle_only" if args.view == "official_pair" else "main_label_free_input",
        "threshold": 0.5,
        "test_tuning_evaluations": 0,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "rdrop_alpha": args.rdrop_alpha,
        "label_smoothing": args.label_smoothing,
        "test_labels_loaded_after_selection": True,
        "train_row_ids": train.row_id.astype(str).tolist(),
        "validation_row_ids": validation.row_id.astype(str).tolist(),
    }
    canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    protocol["protocol_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    dump_json(manifest_path, {**protocol, "status": "locked_before_test_scoring"})

    member_records: list[dict[str, Any]] = []
    validation_probabilities: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []
    for member_seed in args.member_seeds:
        output = fit_member(
            model_path=args.model_path,
            train=train,
            validation=validation,
            test=test_inputs,
            view=args.view,
            explanation_column=args.explanation_column,
            seed=int(member_seed),
            max_length=args.max_length,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            epochs=args.epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            rdrop_alpha=args.rdrop_alpha,
            label_smoothing=args.label_smoothing,
        )
        record = output["result"]
        record["fixed_threshold_validation"] = metrics_from_probs(
            validation["label"].to_numpy(dtype=int), output["validation_probability"]
        )
        member_records.append(record)
        validation_probabilities.append(output["validation_probability"])
        test_probabilities.append(output["test_probability"])

    validation_ensemble = np.mean(np.stack(validation_probabilities), axis=0)
    test_ensemble = np.mean(np.stack(test_probabilities), axis=0)
    validation_labels = validation["label"].to_numpy(dtype=int)

    # This is the only point at which the test label column is read.  All
    # checkpoints and the fixed 0.5 decision rule are already locked above.
    label_frame = pd.read_csv(
        args.test_file,
        encoding="utf-8-sig",
        usecols=["row_id", "label"],
        dtype={"row_id": str},
    )
    label_frame["label"] = pd.to_numeric(label_frame["label"], errors="raise").astype(int)
    if not test_inputs["row_id"].equals(label_frame["row_id"]):
        raise AssertionError("Test input and label row order differ")
    test_labels = label_frame["label"].to_numpy(dtype=int)
    ensemble_validation_metrics = metrics_from_probs(validation_labels, validation_ensemble)
    ensemble_test_metrics = metrics_from_probs(test_labels, test_ensemble)
    prediction_frame = test_inputs[["row_id", "live_id"]].copy()
    for member_seed, probabilities in zip(args.member_seeds, test_probabilities):
        prediction_frame[f"probability_seed{member_seed}"] = probabilities
    prediction_frame["probability"] = test_ensemble
    prediction_frame["prediction"] = (test_ensemble >= 0.5).astype(int)
    prediction_path = RESULTS / f"{args.tag}_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    validation_prediction_frame = validation[["row_id", "label"]].copy()
    for member_seed, probabilities in zip(args.member_seeds, validation_probabilities):
        validation_prediction_frame[f"probability_seed{member_seed}"] = probabilities
    validation_prediction_frame["probability"] = validation_ensemble
    validation_prediction_frame["prediction"] = (validation_ensemble >= 0.5).astype(int)
    validation_prediction_path = RESULTS / f"{args.tag}_validation_predictions.csv"
    validation_prediction_frame.to_csv(validation_prediction_path, index=False, encoding="utf-8-sig")
    result = {
        **protocol,
        "status": "completed",
        "manifest": str(manifest_path),
        "member_results": member_records,
        "member_fixed_threshold_test_metrics": [
            {"seed": int(seed), **metrics_from_probs(test_labels, probabilities)}
            for seed, probabilities in zip(args.member_seeds, test_probabilities)
        ],
        "member_summary_test_macro_f1": summarize_members(
            [metrics_from_probs(test_labels, probabilities) for probabilities in test_probabilities],
            "macro_f1",
        ),
        "ensemble_validation_metrics": ensemble_validation_metrics,
        "ensemble_test_metrics": ensemble_test_metrics,
        "prediction_file": str(prediction_path),
        "validation_prediction_file": str(validation_prediction_path),
        "test_scoring_note": "Test labels were joined only after all validation checkpoints and the fixed 0.5 decision rule were locked.",
    }
    dump_json(RESULTS / f"{args.tag}_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("base_ce", "dapt_ce", "base_rdrop", "dapt_rdrop"), required=True)
    parser.add_argument("--protocol-file", type=Path, default=DEFAULT_PROTOCOL)
    cli = parser.parse_args()
    registry = json.loads(cli.protocol_file.read_text(encoding="utf-8"))
    shared = dict(registry["shared"])
    preset = dict(registry["presets"][cli.preset])
    values = {**shared, **preset}
    for key in ("train_file", "test_file"):
        values[key] = (ROOT / values[key]).resolve()
    model_path = str(values["model_path"])
    if model_path.startswith("experiment/"):
        values["model_path"] = str((ROOT / model_path).resolve())
    return argparse.Namespace(
        **values,
        preset=cli.preset,
        protocol_file=cli.protocol_file.resolve(),
    )


if __name__ == "__main__":
    run(parse_args())
