from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from common import DATA_DIR, LOGS_DIR, RESULTS_DIR, classification_metrics, dump_json, ensure_dirs
from generate_qwen import GpuMonitor


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3:8b"


def input_hash(row: pd.Series) -> str:
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt_for(row)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_for(row: pd.Series) -> str:
    context = json.loads(row["context_json"])
    context_text = "；".join(context) if context else "（无前文）"
    return f"""判断当前电商直播弹幕是否表达购买或推荐需求。
标签0：闲聊、称赞主播、与商品购买无关的表达。
标签1：询问商品、价格、尺寸、库存、优惠、使用方式，要求展示或表达购买兴趣。
前文：{context_text}
当前弹幕：{row['content']}
只输出数字0或1，不要解释。 /no_think"""


def parse_label(text: str) -> int | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    matches = re.findall(r"(?<!\d)[01](?!\d)", text)
    return int(matches[-1]) if matches else None


def classify(session: requests.Session, row: pd.Series, retries: int = 3) -> dict[str, object]:
    signature = input_hash(row)
    payload = {
        "model": MODEL,
        "prompt": prompt_for(row),
        "stream": False,
        "options": {"temperature": 0, "num_predict": 24, "num_ctx": 1024, "seed": 20260718},
        "keep_alive": "30m",
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            data = session.post(OLLAMA_URL, json=payload, timeout=120).json()
            raw = str(data.get("response", ""))
            prediction = parse_label(raw)
            if prediction is None:
                raise ValueError(f"Could not parse label from {raw!r}")
            return {
                "row_id": row["row_id"],
                "input_hash": signature,
                "prediction": prediction,
                "raw_response": raw,
                "inference_seconds": time.perf_counter() - started,
                "generated_tokens": int(data.get("eval_count", 0)),
                "attempts": attempt,
                "error": "",
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(attempt)
    return {
        "row_id": row["row_id"],
        "input_hash": signature,
        "prediction": None,
        "raw_response": "",
        "inference_seconds": 0.0,
        "generated_tokens": 0,
        "attempts": retries,
        "error": last_error,
    }


def main() -> None:
    ensure_dirs()
    frame = pd.read_csv(DATA_DIR / "douyin_test_qwen.csv", encoding="utf-8-sig")
    checkpoint_path = LOGS_DIR / "qwen_classification.jsonl"
    completed: dict[str, dict[str, object]] = {}
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[str(record["row_id"])] = record

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
                and completed[row_id].get("prediction") in (0, 1)
                and completed[row_id].get("input_hash") == signature
            ):
                continue
            record = classify(session, row)
            completed[row_id] = record
            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            print(
                f"[{number}/{len(frame)}] {row_id}: true={row['label']} "
                f"pred={record['prediction']} ({record['inference_seconds']:.2f}s)",
                flush=True,
            )
    elapsed = time.perf_counter() - started
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
    valid = merged[merged["prediction"].isin([0, 1])].copy()
    valid["prediction"] = valid["prediction"].astype(int)
    metrics = classification_metrics(valid["label"].astype(int), valid["prediction"])
    merged.to_csv(RESULTS_DIR / "predictions_qwen_direct.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monitor.samples).to_csv(RESULTS_DIR / "qwen_direct_gpu_trace.csv", index=False)
    summary = {
        "model": MODEL,
        "test_rows": int(len(frame)),
        "valid_predictions": int(len(valid)),
        "failed_predictions": int(len(frame) - len(valid)),
        "coverage": float(len(valid) / len(frame)),
        "wall_seconds": elapsed,
        "mean_inference_seconds": float(valid["inference_seconds"].mean()),
        "total_generated_tokens": int(valid["generated_tokens"].sum()),
        "peak_gpu_memory_mb": float(max((x["memory_used_mb"] for x in monitor.samples), default=0.0)),
        "mean_gpu_utilization": float(
            sum(x["gpu_utilization"] for x in monitor.samples) / max(len(monitor.samples), 1)
        ),
        **metrics,
    }
    dump_json(RESULTS_DIR / "qwen_direct.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
