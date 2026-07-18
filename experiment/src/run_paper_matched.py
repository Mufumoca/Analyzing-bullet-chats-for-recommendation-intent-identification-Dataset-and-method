"""Paper-matched few-shot comparison and leakage-safe explanation fusion.

The target paper samples ``shot`` examples per class for both training and
validation.  This script reproduces that sampling convention on the published
RecDY split, evaluates the unchanged full test file, and keeps the proposed
calibrated multi-view fusion separate from the paper-aligned baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import DATA_DIR, RESULTS_DIR, classification_metrics, dump_json, set_seed
from train_roberta import EncodedTextDataset, MODEL_NAME, evaluate


RESULTS = RESULTS_DIR / "paper_comparison"
DATA = DATA_DIR / "paper_comparison"
SEEDS = (100, 101, 102)


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"row_id", "content", "official_background", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    return frame.reset_index(drop=True)


def sample_per_label(frame: pd.DataFrame, shot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label, group in frame.groupby("label", sort=True):
        if len(group) < shot:
            raise ValueError(f"Class {label} has only {len(group)} rows; need {shot}")
        chosen.extend(rng.choice(group.index.to_numpy(), size=shot, replace=False).tolist())
    return frame.loc[sorted(chosen)].reset_index(drop=True)


def make_split(train: pd.DataFrame, shot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part = sample_per_label(train, shot, seed)
    remaining = train.loc[~train["row_id"].isin(train_part["row_id"])].copy()
    # Use a separate stream for validation so the two sets cannot share rows.
    validation_part = sample_per_label(remaining, shot, seed + 100_003)
    if set(train_part["row_id"]) & set(validation_part["row_id"]):
        raise AssertionError("Few-shot train/validation overlap")
    return train_part.reset_index(drop=True), validation_part.reset_index(drop=True)


def text_for(frame: pd.DataFrame, condition: str) -> list[str]:
    content = frame["content"].fillna("").astype(str)
    if condition == "content":
        return content.tolist()
    if condition == "official":
        explanation = frame["official_background"].fillna("").astype(str)
        return ("弹幕：" + content + " [SEP] 语境释义：" + explanation).tolist()
    raise ValueError(condition)


def frame_signature(frame: pd.DataFrame) -> str:
    payload = frame[["row_id", "content", "official_background", "label"]].astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def loaders(
    train: pd.DataFrame,
    target: pd.DataFrame,
    condition: str,
    tokenizer: Any,
    batch_size: int,
    seed: int,
    max_length: int,
) -> tuple[DataLoader, DataLoader]:
    train_encoding = tokenizer(
        text_for(train, condition),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    target_encoding = tokenizer(
        text_for(target, condition),
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
    return train_loader, target_loader


def train_until_best(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    condition: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Train one branch and return validation probabilities plus best state."""
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_loader, validation_loader = loaders(
        train, validation, condition, tokenizer, batch_size, seed, max_length
    )
    _, test_loader = loaders(train, test, condition, tokenizer, batch_size, seed, max_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the paper-matched experiment")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    update_steps = math.ceil(len(train_loader))
    total_steps = update_steps * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    best_score = -1.0
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**batch)
            scaler.scale(output.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(output.loss.detach().cpu()))
        validation_metrics, _, validation_probabilities = evaluate(model, validation_loader, device)
        score = float(validation_metrics["macro_f1"])
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": score})
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is None:
        raise AssertionError("No checkpoint captured")
    model.load_state_dict(best_state)
    final_validation_metrics, _, final_validation_probabilities = evaluate(model, validation_loader, device)
    test_metrics, _, test_probabilities = evaluate(model, test_loader, device)
    result = {
        "condition": condition,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_macro_f1": best_score,
        "validation_metrics": final_validation_metrics,
        "test_metrics": test_metrics,
        "test_evaluations": 1,
        "history": history,
    }
    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()
    return result, final_validation_probabilities, test_probabilities


def choose_fusion(
    labels: np.ndarray, content_probability: np.ndarray, explanation_probability: np.ndarray
) -> dict[str, float]:
    candidates: list[tuple[float, float, float]] = []
    for alpha in np.arange(0.0, 1.0001, 0.05):
        fused = alpha * content_probability + (1.0 - alpha) * explanation_probability
        for threshold in np.arange(0.35, 0.651, 0.01):
            score = float(f1_score(labels, (fused >= threshold).astype(int), average="macro", zero_division=0))
            candidates.append((score, float(alpha), float(threshold)))
    candidates.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), abs(item[2] - 0.5)))
    score, alpha, threshold = candidates[0]
    return {
        "validation_macro_f1": score,
        "alpha_content": alpha,
        "alpha_explanation": 1.0 - alpha,
        "threshold": threshold,
    }


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    candidates: list[tuple[float, float]] = []
    for threshold in np.arange(0.30, 0.701, 0.01):
        score = float(
            f1_score(
                labels,
                (probability >= threshold).astype(int),
                average="macro",
                zero_division=0,
            )
        )
        candidates.append((score, float(threshold)))
    candidates.sort(key=lambda item: (-item[0], abs(item[1] - 0.5)))
    score, threshold = candidates[0]
    return {"validation_macro_f1": score, "threshold": threshold}


def fit_tfidf_branch(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Tune a sparse short-text branch on validation data only."""
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        min_df=1,
        max_features=60_000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train["content"].fillna("").astype(str))
    validation_matrix = vectorizer.transform(validation["content"].fillna("").astype(str))
    test_matrix = vectorizer.transform(test["content"].fillna("").astype(str))
    y_train = train["label"].to_numpy(dtype=int)
    y_validation = validation["label"].to_numpy(dtype=int)
    candidates: list[tuple[float, float, float, LogisticRegression, np.ndarray]] = []
    for c_value in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        model = LogisticRegression(C=c_value, max_iter=3000, solver="liblinear", random_state=seed)
        model.fit(train_matrix, y_train)
        probability = model.predict_proba(validation_matrix)[:, 1]
        for threshold in np.arange(0.35, 0.651, 0.01):
            score = float(
                f1_score(
                    y_validation,
                    (probability >= threshold).astype(int),
                    average="macro",
                    zero_division=0,
                )
            )
            candidates.append((score, c_value, float(threshold), model, probability))
    candidates.sort(key=lambda item: (-item[0], abs(math.log2(item[1])), abs(item[2] - 0.5)))
    score, c_value, threshold, model, validation_probability = candidates[0]
    test_probability = model.predict_proba(test_matrix)[:, 1]
    validation_prediction = (validation_probability >= threshold).astype(int)
    test_prediction = (test_probability >= threshold).astype(int)
    result = {
        "model": "character TF-IDF (1,5) + logistic regression",
        "seed": seed,
        "selected_c": c_value,
        "selected_threshold": threshold,
        "validation_macro_f1": score,
        "validation_metrics": classification_metrics(y_validation, validation_prediction),
        "test_metrics": classification_metrics(test["label"], test_prediction),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "test_evaluations": 1,
    }
    return result, validation_probability, test_probability


def run_one(
    full_train: pd.DataFrame,
    full_test: pd.DataFrame,
    shot: int,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    train, validation = make_split(full_train, shot, seed)
    if set(train["row_id"]) & set(full_test["row_id"]):
        raise AssertionError("Train/test overlap")
    branch_results: dict[str, Any] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for condition in ("content", "official"):
        result, validation_probability, test_probability = train_until_best(
            train, validation, full_test, condition, seed, epochs, patience, batch_size, max_length
        )
        branch_results[condition] = result
        validation_probabilities[condition] = validation_probability
        test_probabilities[condition] = test_probability
    labels = validation["label"].to_numpy(dtype=int)
    content_selection = choose_threshold(labels, validation_probabilities["content"])
    calibrated_content_prediction = (
        test_probabilities["content"] >= content_selection["threshold"]
    ).astype(int)
    calibrated_content_metrics = classification_metrics(
        full_test["label"], calibrated_content_prediction
    )
    tfidf_result, tfidf_validation_probability, tfidf_test_probability = fit_tfidf_branch(
        train, validation, full_test, seed
    )
    clean_selection = choose_fusion(
        labels, validation_probabilities["content"], tfidf_validation_probability
    )
    clean_fused_probability = (
        clean_selection["alpha_content"] * test_probabilities["content"]
        + clean_selection["alpha_explanation"] * tfidf_test_probability
    )
    clean_fused_prediction = (clean_fused_probability >= clean_selection["threshold"]).astype(int)
    clean_fused_metrics = classification_metrics(full_test["label"], clean_fused_prediction)
    oracle_selection = choose_fusion(
        labels, validation_probabilities["content"], validation_probabilities["official"]
    )
    oracle_fused_probability = (
        oracle_selection["alpha_content"] * test_probabilities["content"]
        + oracle_selection["alpha_explanation"] * test_probabilities["official"]
    )
    oracle_fused_prediction = (oracle_fused_probability >= oracle_selection["threshold"]).astype(int)
    oracle_fused_metrics = classification_metrics(full_test["label"], oracle_fused_prediction)
    tag = f"shot{shot}_seed{seed}"
    prediction_frame = full_test[["row_id", "live_id", "label", "content"]].copy()
    prediction_frame["official_background"] = full_test["official_background"]
    prediction_frame["content_probability"] = test_probabilities["content"]
    prediction_frame["calibrated_content_prediction"] = calibrated_content_prediction
    prediction_frame["tfidf_probability"] = tfidf_test_probability
    prediction_frame["official_probability"] = test_probabilities["official"]
    prediction_frame["clean_fused_probability"] = clean_fused_probability
    prediction_frame["clean_fused_prediction"] = clean_fused_prediction
    prediction_frame["oracle_fused_probability"] = oracle_fused_probability
    prediction_frame["oracle_fused_prediction"] = oracle_fused_prediction
    prediction_path = RESULTS / f"predictions_{tag}.csv"
    prediction_frame.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    result = {
        "tag": tag,
        "shot_per_class_train": shot,
        "shot_per_class_validation": shot,
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(full_test),
        "train_signature": frame_signature(train),
        "validation_signature": frame_signature(validation),
        "test_signature": frame_signature(full_test),
        "clean_selection": clean_selection,
        "content_selection": content_selection,
        "oracle_selection": oracle_selection,
        "validation_runs": branch_results,
        "test_branch_runs": branch_results,
        "tfidf_run": tfidf_result,
        "test_fused_metrics": clean_fused_metrics,
        "test_calibrated_content_metrics": calibrated_content_metrics,
        "test_oracle_fused_metrics": oracle_fused_metrics,
        "official_background_role": "oracle_only_not_main_result",
        "test_tuning_evaluations": 0,
        "prediction_file": str(prediction_path),
    }
    dump_json(RESULTS / f"{tag}.json", result)
    # Return validation/test probabilities for the predeclared seed ensemble.
    result["_validation_labels"] = labels
    result["_validation_content_probability"] = validation_probabilities["content"]
    result["_validation_official_probability"] = validation_probabilities["official"]
    result["_validation_tfidf_probability"] = tfidf_validation_probability
    result["_test_content_probability"] = test_probabilities["content"]
    result["_test_official_probability"] = test_probabilities["official"]
    result["_test_tfidf_probability"] = tfidf_test_probability
    return result


def run_ensemble(shot: int, runs: list[dict[str, Any]], test: pd.DataFrame) -> dict[str, Any]:
    labels = np.concatenate([run["_validation_labels"] for run in runs])
    content_val = np.concatenate([run["_validation_content_probability"] for run in runs])
    tfidf_val = np.concatenate([run["_validation_tfidf_probability"] for run in runs])
    official_val = np.concatenate([run["_validation_official_probability"] for run in runs])
    content_selection = choose_threshold(labels, content_val)
    clean_selection = choose_fusion(labels, content_val, tfidf_val)
    oracle_selection = choose_fusion(labels, content_val, official_val)
    content_test = np.mean(np.stack([run["_test_content_probability"] for run in runs]), axis=0)
    tfidf_test = np.mean(np.stack([run["_test_tfidf_probability"] for run in runs]), axis=0)
    official_test = np.mean(np.stack([run["_test_official_probability"] for run in runs]), axis=0)
    content_prediction = (content_test >= content_selection["threshold"]).astype(int)
    content_metrics = classification_metrics(test["label"], content_prediction)
    clean_fused = (
        clean_selection["alpha_content"] * content_test
        + clean_selection["alpha_explanation"] * tfidf_test
    )
    clean_prediction = (clean_fused >= clean_selection["threshold"]).astype(int)
    clean_metrics = classification_metrics(test["label"], clean_prediction)
    oracle_fused = (
        oracle_selection["alpha_content"] * content_test
        + oracle_selection["alpha_explanation"] * official_test
    )
    oracle_prediction = (oracle_fused >= oracle_selection["threshold"]).astype(int)
    oracle_metrics = classification_metrics(test["label"], oracle_prediction)
    payload = {
        "shot_per_class": shot,
        "seeds": [int(run["seed"]) for run in runs],
        "selection": clean_selection,
        "test_metrics": clean_metrics,
        "content_selection": content_selection,
        "test_content_metrics": content_metrics,
        "oracle_selection": oracle_selection,
        "test_oracle_metrics": oracle_metrics,
        "official_background_role": "oracle_only_not_main_result",
        "test_tuning_evaluations": 0,
        "test_rows": len(test),
    }
    dump_json(RESULTS / f"shot{shot}_seed_ensemble.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, nargs="+", default=[50, 60, 70])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    full_train = load_frame(DATA_DIR / "douyin_train_full.csv")
    full_test = load_frame(DATA_DIR / "douyin_test_full.csv")
    if set(full_train["row_id"]) & set(full_test["row_id"]):
        raise AssertionError("Published train/test overlap")
    summary: list[dict[str, Any]] = []
    for shot in args.shots:
        runs: list[dict[str, Any]] = []
        for seed in args.seeds:
            run = run_one(
                full_train,
                full_test,
                shot,
                seed,
                args.epochs,
                args.patience,
                args.batch_size,
                args.max_length,
            )
            summary.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "macro_f1": run["test_fused_metrics"]["macro_f1"],
                    "accuracy": run["test_fused_metrics"]["accuracy"],
                    "content_macro_f1": run["test_branch_runs"]["content"]["test_metrics"]["macro_f1"],
                    "calibrated_content_macro_f1": run["test_calibrated_content_metrics"]["macro_f1"],
                    "tfidf_macro_f1": run["tfidf_run"]["test_metrics"]["macro_f1"],
                    "oracle_macro_f1": run["test_oracle_fused_metrics"]["macro_f1"],
                    "validation_macro_f1": run["clean_selection"]["validation_macro_f1"],
                    "alpha_content": run["clean_selection"]["alpha_content"],
                    "threshold": run["clean_selection"]["threshold"],
                }
            )
            runs.append(run)
        ensemble = run_ensemble(shot, runs, full_test)
        summary.append(
            {
                "shot": shot,
                "seed": "ensemble",
                "macro_f1": ensemble["test_metrics"]["macro_f1"],
                "accuracy": ensemble["test_metrics"]["accuracy"],
                "content_macro_f1": ensemble["test_content_metrics"]["macro_f1"],
                "calibrated_content_macro_f1": ensemble["test_content_metrics"]["macro_f1"],
                "tfidf_macro_f1": float("nan"),
                "oracle_macro_f1": ensemble["test_oracle_metrics"]["macro_f1"],
                "validation_macro_f1": ensemble["selection"]["validation_macro_f1"],
                "alpha_content": ensemble["selection"]["alpha_content"],
                "threshold": ensemble["selection"]["threshold"],
            }
        )
    pd.DataFrame(summary).to_csv(RESULTS / "paper_matched_summary.csv", index=False, encoding="utf-8-sig")
    dump_json(
        RESULTS / "paper_matched_manifest.json",
        {
            "model": MODEL_NAME,
            "train_file": str(DATA_DIR / "douyin_train_full.csv"),
            "test_file": str(DATA_DIR / "douyin_test_full.csv"),
            "shots": args.shots,
            "seeds": args.seeds,
            "sampling": "shot examples per class for train and validation, disjoint; fixed published test",
            "conditions": [
                "content RoBERTa",
                "character TF-IDF",
                "validation-calibrated clean fusion",
                "three-seed ensemble",
                "official explanation oracle (not a main result)",
            ],
            "test_tuning_evaluations": 0,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
