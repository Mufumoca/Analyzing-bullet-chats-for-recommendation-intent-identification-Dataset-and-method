from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests

from common import DATA_DIR, LOGS_DIR, RESULTS_DIR, dump_json, ensure_dirs


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:8b"


def input_hash(row: pd.Series) -> str:
    payload = json.dumps(
        {"model": MODEL, "prompt": make_prompt(row)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GpuMonitor:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.pynvml = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.samples.append(
                {
                    "time": time.time(),
                    "memory_used_mb": memory.used / 1024**2,
                    "gpu_utilization": float(utilization.gpu),
                }
            )
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if getattr(self, "pynvml", None) is not None:
            self.pynvml.nvmlShutdown()


def clean_response(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().strip('"').strip()
    for prefix in ("解释：", "语境释义：", "该用户想表达的含义是：", "该用户想表达的是："):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return " ".join(text.split())


def make_prompt(row: pd.Series) -> str:
    context = json.loads(row["context_json"])
    context_text = "；".join(context) if context else "（无前文）"
    return f"""你是电商直播弹幕语义分析助手。请结合前文，用一句自然的现代汉语解释当前弹幕的真实意图。
要求：不要判断或输出分类标签，不要重复原句，不要展示推理过程，不要添加解释以外的内容。
前文弹幕：{context_text}
当前弹幕：{row['content']}
只输出一句语境释义。 /no_think"""


def request_explanation(session: requests.Session, row: pd.Series, retries: int = 3) -> dict[str, object]:
    signature = input_hash(row)
    payload = {
        "model": MODEL,
        "prompt": make_prompt(row),
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 96,
            "num_ctx": 1024,
            "seed": 20260718,
        },
        "keep_alive": "30m",
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            response = session.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            cleaned = clean_response(str(data.get("response", "")))
            if not cleaned:
                raise ValueError("empty response")
            return {
                "row_id": row["row_id"],
                "input_hash": signature,
                "qwen_background": cleaned,
                "generation_seconds": time.perf_counter() - started,
                "prompt_tokens": int(data.get("prompt_eval_count", 0)),
                "generated_tokens": int(data.get("eval_count", 0)),
                "total_duration_ns": int(data.get("total_duration", 0)),
                "load_duration_ns": int(data.get("load_duration", 0)),
                "attempts": attempt,
                "error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(attempt)
    return {
        "row_id": row["row_id"],
        "input_hash": signature,
        "qwen_background": "",
        "generation_seconds": 0.0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
        "attempts": retries,
        "error": last_error,
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    completed: dict[str, dict[str, object]] = {}
    if not path.exists():
        return completed
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            completed[str(record["row_id"])] = record
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 processes the complete fixed sample")
    args = parser.parse_args()
    ensure_dirs()

    input_path = DATA_DIR / f"douyin_{args.split}_qwen.csv"
    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    if args.limit > 0:
        frame = frame.head(args.limit).copy()

    checkpoint_path = LOGS_DIR / f"qwen_generation_{args.split}.jsonl"
    completed = load_checkpoint(checkpoint_path)
    monitor = GpuMonitor()
    monitor.start()
    started = time.perf_counter()
    session = requests.Session()
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for number, (_, row) in enumerate(frame.iterrows(), start=1):
            row_id = str(row["row_id"])
            signature = input_hash(row)
            if (
                row_id in completed
                and completed[row_id].get("qwen_background")
                and completed[row_id].get("input_hash") == signature
            ):
                continue
            record = request_explanation(session, row)
            completed[row_id] = record
            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            print(
                f"[{args.split} {number}/{len(frame)}] {row_id}: "
                f"{record['qwen_background']} ({record['generation_seconds']:.2f}s)",
                flush=True,
            )
    total_seconds = time.perf_counter() - started
    monitor.stop()

    records = pd.DataFrame(
        [
            completed[row_id]
            for _, row in frame.iterrows()
            if (row_id := str(row["row_id"])) in completed
            and completed[row_id].get("input_hash") == input_hash(row)
        ]
    )
    merged = frame.merge(records, on="row_id", how="left", validate="one_to_one")
    output_path = DATA_DIR / f"douyin_{args.split}_qwen_generated.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    monitor_path = RESULTS_DIR / f"qwen_gpu_trace_{args.split}.csv"
    pd.DataFrame(monitor.samples).to_csv(monitor_path, index=False)
    summary = {
        "model": MODEL,
        "split": args.split,
        "requested_rows": int(len(frame)),
        "successful_rows": int(records["qwen_background"].fillna("").ne("").sum()),
        "failed_rows": int(records["qwen_background"].fillna("").eq("").sum()),
        "wall_seconds": total_seconds,
        "mean_generation_seconds": float(records["generation_seconds"].mean()),
        "median_generation_seconds": float(records["generation_seconds"].median()),
        "aggregate_generation_seconds": float(records["generation_seconds"].sum()),
        "total_generated_tokens": int(records["generated_tokens"].sum()),
        "tokens_per_second": float(
            records["generated_tokens"].sum() / max(records["generation_seconds"].sum(), 1e-9)
        ),
        "peak_gpu_memory_mb": float(
            max((sample["memory_used_mb"] for sample in monitor.samples), default=0.0)
        ),
        "mean_gpu_utilization": float(
            sum(sample["gpu_utilization"] for sample in monitor.samples) / max(len(monitor.samples), 1)
        ),
    }
    dump_json(RESULTS_DIR / f"qwen_generation_{args.split}.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
