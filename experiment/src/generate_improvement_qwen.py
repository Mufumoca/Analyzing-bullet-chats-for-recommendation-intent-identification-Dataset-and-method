from __future__ import annotations

import argparse
import json
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests

from generate_qwen import GpuMonitor
from improvement_common import (
    IMPROVEMENT_LOGS_DIR,
    IMPROVEMENT_RESULTS_DIR,
    MODEL,
    OLLAMA_URL,
    PROMPT_VERSION,
    IMPROVEMENT_DATA_DIR,
    clean_response,
    ensure_improvement_dirs,
    formatted_explanation,
    input_hash,
    parse_structured_response,
    prompt_hash,
    structured_prompt,
)


def load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["row_id"])] = record
    return records


def label_like_output(text: str) -> bool:
    return bool(
        re.search(r"标签\s*[：:]?\s*[01]", text)
        or re.search(r"购买意图|推荐意图|二分类标签|类别判断", text)
    )


def request_one(session: requests.Session, row: pd.Series, retries: int = 2) -> dict[str, object]:
    prompt = structured_prompt(row)
    signature = input_hash(row, prompt)
    signature_prompt = prompt_hash(prompt)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 128,
            "num_ctx": 2048,
            "seed": 20260718,
        },
        "keep_alive": "30m",
    }
    last_error = ""
    last_parsed: dict[str, object] = {}
    started_all = time.perf_counter()
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            response = session.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            raw = str(data.get("response", ""))
            parsed = parse_structured_response(raw, str(row["content"]))
            last_parsed = parsed
            if not parsed["valid_structure"]:
                raise ValueError(str(parsed["validation_error"]))
            explanation = formatted_explanation(parsed)
            return {
                "row_id": str(row["row_id"]),
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": signature_prompt,
                "input_hash": signature,
                "qwen_background_structured": explanation,
                "raw_response": clean_response(raw),
                "structured_object": parsed["structured_object"],
                "structured_act": parsed["structured_act"],
                "structured_evidence": parsed["structured_evidence"],
                "valid_structure": True,
                "validation_error": "",
                "label_like_output": label_like_output(explanation),
                "generation_seconds": time.perf_counter() - started,
                "total_generation_seconds": time.perf_counter() - started_all,
                "prompt_tokens": int(data.get("prompt_eval_count", 0)),
                "generated_tokens": int(data.get("eval_count", 0)),
                "total_duration_ns": int(data.get("total_duration", 0)),
                "load_duration_ns": int(data.get("load_duration", 0)),
                "attempts": attempt,
                "error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(attempt)
    return {
        "row_id": str(row["row_id"]),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": signature_prompt,
        "input_hash": signature,
        "qwen_background_structured": "",
        "raw_response": str(last_parsed.get("cleaned_response", "")),
        "structured_object": str(last_parsed.get("structured_object", "")),
        "structured_act": str(last_parsed.get("structured_act", "")),
        "structured_evidence": str(last_parsed.get("structured_evidence", "")),
        "valid_structure": False,
        "validation_error": str(last_parsed.get("validation_error", "")),
        "label_like_output": label_like_output(str(last_parsed.get("cleaned_response", ""))),
        "generation_seconds": 0.0,
        "total_generation_seconds": time.perf_counter() - started_all,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
        "attempts": retries,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--size", type=int, choices=(400, 1000), required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.split == "test" and args.size != 400:
        raise ValueError("The fixed test set has size 400")
    ensure_improvement_dirs()
    input_path = IMPROVEMENT_DATA_DIR / f"douyin_{args.split}_improvement{args.size}.csv"
    checkpoint_path = IMPROVEMENT_LOGS_DIR / (
        f"qwen_generation_{args.split}_improvement{args.size}_{PROMPT_VERSION}.jsonl"
    )
    output_path = IMPROVEMENT_DATA_DIR / (
        f"douyin_{args.split}_improvement{args.size}_{PROMPT_VERSION}.csv"
    )
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    if args.limit > 0:
        frame = frame.head(args.limit).copy()
    completed = load_checkpoint(checkpoint_path)
    monitor = GpuMonitor()
    monitor.start()
    started_all = time.perf_counter()
    session = requests.Session()
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for number, (_, row) in enumerate(frame.iterrows(), start=1):
            row_id = str(row["row_id"])
            signature = input_hash(row, structured_prompt(row))
            cached = completed.get(row_id)
            if cached and cached.get("input_hash") == signature and cached.get("valid_structure"):
                continue
            record = request_one(session, row)
            completed[row_id] = record
            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            print(
                f"[{args.split} {number}/{len(frame)}] {row_id}: "
                f"valid={record['valid_structure']} ({record['total_generation_seconds']:.2f}s)",
                flush=True,
            )
    monitor.stop()
    records = pd.DataFrame(
        [
            completed[str(row["row_id"])]
            for _, row in frame.iterrows()
            if str(row["row_id"]) in completed
            and completed[str(row["row_id"])].get("input_hash") == input_hash(row, structured_prompt(row))
        ]
    )
    merged = frame.merge(records, on="row_id", how="left", validate="one_to_one")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    valid = merged["valid_structure"].fillna(False).astype(bool)
    summary = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "split": args.split,
        "size": args.size,
        "requested_rows": int(len(frame)),
        "successful_rows": int(valid.sum()),
        "failed_rows": int((~valid).sum()),
        "coverage": float(valid.mean()),
        "label_like_outputs": int(merged["label_like_output"].fillna(False).astype(bool).sum()),
        "wall_seconds": time.perf_counter() - started_all,
        "mean_generation_seconds": float(merged.loc[valid, "generation_seconds"].mean()) if valid.any() else 0.0,
        "median_generation_seconds": float(merged.loc[valid, "generation_seconds"].median()) if valid.any() else 0.0,
        "peak_gpu_memory_mb": float(max((sample["memory_used_mb"] for sample in monitor.samples), default=0.0)),
        "mean_gpu_utilization": float(
            sum(sample["gpu_utilization"] for sample in monitor.samples) / max(len(monitor.samples), 1)
        ),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "checkpoint_file": str(checkpoint_path),
    }
    (IMPROVEMENT_RESULTS_DIR / f"qwen_generation_{args.split}_improvement{args.size}_{PROMPT_VERSION}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
