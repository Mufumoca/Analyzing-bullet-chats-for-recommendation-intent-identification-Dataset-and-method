"""Domain-adapt Chinese RoBERTa on unlabeled RecDY training comments.

The script deliberately reads only ``row_id`` and ``content`` from the
published training partition.  Labels, validation data, and test data are not
used.  The resulting checkpoint can therefore be used in a same-human-label-
budget comparison without adding supervised examples.
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
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as transformers_logging

from common import DATA_DIR, EXPERIMENT_ROOT, RESULTS_DIR, dump_json, set_seed
from train_roberta import MODEL_NAME


transformers_logging.set_verbosity_error()

DEFAULT_OUTPUT = EXPERIMENT_ROOT / "models" / "roberta_recdy_mlm_v1"
RESULTS = RESULTS_DIR / "same_budget"
EXPECTED_INPUT = (DATA_DIR / "douyin_train_full.csv").resolve()
EXPECTED_CONTENT_SIGNATURE = "39f6ab8b93c28baffdb5618e0a27e4a163cd9a5127334b605c81d2ee04b743ab"


class TokenDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]]) -> None:
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.tensor(values[index], dtype=torch.long)
            for key, values in self.encodings.items()
        }


def content_signature(frame: pd.DataFrame) -> str:
    payload = frame[["row_id", "content"]].fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_preregistration(path: Path, payload: dict[str, Any]) -> None:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    record = {
        **payload,
        "protocol_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "status": "pre_registered_before_training",
    }
    dump_json(path, record)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DATA_DIR / "douyin_train_full.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for domain-adaptive pretraining")
    if args.epochs < 1 or args.batch_size < 1 or args.accumulation_steps < 1:
        raise ValueError("epochs, batch-size, and accumulation-steps must be positive")
    if args.input_file.resolve() != EXPECTED_INPUT:
        raise ValueError(f"DAPT input must be the published training partition: {EXPECTED_INPUT}")

    frame = pd.read_csv(
        args.input_file,
        encoding="utf-8-sig",
        usecols=["row_id", "content"],
        dtype={"row_id": str, "content": str},
        keep_default_na=False,
    )
    if frame["row_id"].duplicated().any():
        raise ValueError("Duplicate row_id in the unlabeled training partition")
    if frame["content"].str.len().eq(0).any():
        raise ValueError("Empty content in the unlabeled training partition")
    actual_signature = content_signature(frame)
    if actual_signature != EXPECTED_CONTENT_SIGNATURE:
        raise ValueError(
            "The unlabeled training partition differs from the frozen audit signature: "
            f"{actual_signature}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "experiment": "recdy_content_domain_adaptive_mlm_v1",
        "base_model": MODEL_NAME,
        "input_file": str(args.input_file.resolve()),
        "input_columns": ["row_id", "content"],
        "input_rows": int(len(frame)),
        "input_content_sha256": actual_signature,
        "test_data_used": False,
        "labels_used": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "mlm_probability": args.mlm_probability,
        "seed": args.seed,
        "output_dir": str(args.output_dir.resolve()),
    }
    preregistration = RESULTS / "dapt_preregistration.json"
    write_preregistration(preregistration, protocol)

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        local_files_only=True,
    )
    encodings = tokenizer(
        frame["content"].tolist(),
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
    )
    dataset = TokenDataset(encodings)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
        use_safetensors=False,
    )
    model.config.tie_word_embeddings = True
    model.to(device)
    # Transformers 5 may materialize the two tensors separately during the
    # device transfer, so tie only after moving the model to CUDA.
    model.tie_weights()
    embedding_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if embedding_weight.data_ptr() != output_weight.data_ptr():
        raise AssertionError("MLM input/output embeddings are not tied")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    total_steps = update_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(**batch)
                loss = output.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.accumulation_steps)
            if step % args.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0 or step == len(loader):
                print(
                    f"epoch={epoch}/{args.epochs} batch={step}/{len(loader)} "
                    f"mean_mlm_loss={np.mean(losses):.4f}",
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch,
                "mean_mlm_loss": float(np.mean(losses)),
                "batches": len(loader),
                "optimizer_steps": update_steps_per_epoch,
            }
        )

    elapsed = time.perf_counter() - started
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    checkpoint_hashes = {
        path.name: file_sha256(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    result = {
        **protocol,
        "protocol_sha256": json.loads(preregistration.read_text(encoding="utf-8"))[
            "protocol_sha256"
        ],
        "status": "completed",
        "history": history,
        "elapsed_seconds": elapsed,
        "peak_torch_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "cuda_device": torch.cuda.get_device_name(0),
        "tied_word_embeddings": True,
        "checkpoint_sha256": checkpoint_hashes,
    }
    dump_json(RESULTS / "dapt_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
