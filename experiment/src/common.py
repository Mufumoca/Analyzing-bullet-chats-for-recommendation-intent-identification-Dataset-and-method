from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "bc4rii_source" / "SPT-RII"
EXPERIMENT_ROOT = ROOT / "experiment"
DATA_DIR = EXPERIMENT_ROOT / "data" / "processed"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
FIGURES_DIR = EXPERIMENT_ROOT / "figures"
LOGS_DIR = EXPERIMENT_ROOT / "logs"

CSV_COLUMNS = ["live_id", "species", "content", "label", "official_background"]


def ensure_dirs() -> None:
    for path in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_generated_split(platform: str, split: str) -> pd.DataFrame:
    path = SOURCE_ROOT / "BC4RII_with_generation" / platform / f"{split}.csv"
    frame = pd.read_csv(
        path,
        header=None,
        names=CSV_COLUMNS,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected labels in {path}")
    frame.insert(0, "source_index", np.arange(len(frame), dtype=int))
    frame.insert(0, "split", split)
    frame.insert(0, "platform", platform)
    frame.insert(0, "row_id", [f"{platform}-{split}-{i:06d}" for i in range(len(frame))])
    return frame


def attach_context(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    result = frame.copy()
    contexts: list[str] = ["[]"] * len(result)
    history: dict[str, list[str]] = {}
    for position, row in enumerate(result.itertuples(index=False)):
        key = str(row.live_id)
        prior = history.setdefault(key, [])
        contexts[position] = json.dumps(prior[-window:], ensure_ascii=False)
        prior.append(str(row.content))
    result["context_json"] = contexts
    return result


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _, group in frame.groupby("label", sort=True):
        count = int(round(n * len(group) / len(frame)))
        count = min(count, len(group))
        chosen.extend(rng.choice(group.index.to_numpy(), size=count, replace=False).tolist())
    if len(chosen) < n:
        remaining = frame.index.difference(chosen).to_numpy()
        chosen.extend(rng.choice(remaining, size=n - len(chosen), replace=False).tolist())
    elif len(chosen) > n:
        chosen = rng.choice(np.asarray(chosen), size=n, replace=False).tolist()
    return frame.loc[sorted(chosen)].reset_index(drop=True)


def text_for_condition(frame: pd.DataFrame, condition: str) -> list[str]:
    content = frame["content"].fillna("").astype(str)
    if condition == "content":
        return content.tolist()
    if condition == "context":
        context = frame["context_json"].map(json.loads).map(lambda items: "；".join(items))
        return ("前文弹幕：" + context + " [SEP] 当前弹幕：" + content).tolist()
    if condition == "official":
        background = frame["official_background"].fillna("").astype(str)
    elif condition == "qwen":
        if "qwen_background" not in frame:
            raise ValueError("qwen_background is missing")
        background = frame["qwen_background"].fillna("").astype(str)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return ("弹幕：" + content + " [SEP] 语境释义：" + background).tolist()


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_pred_array = np.asarray(list(y_pred), dtype=int)
    precision_pos = float(precision_score(y_true_array, y_pred_array, zero_division=0))
    recall_pos = float(recall_score(y_true_array, y_pred_array, zero_division=0))
    f1_pos = float(f1_score(y_true_array, y_pred_array, zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision": precision_pos,
        "recall": recall_pos,
        "f1": f1_pos,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "macro_precision": float(
            precision_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
    }


def bootstrap_ci(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    metric: str = "f1",
    iterations: int = 1000,
    seed: int = 20260718,
) -> tuple[float, float]:
    y_true_array = np.asarray(list(y_true), dtype=int)
    y_pred_array = np.asarray(list(y_pred), dtype=int)
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true_array), len(y_true_array))
        sample_true = y_true_array[indices]
        sample_pred = y_pred_array[indices]
        if metric == "f1":
            score = f1_score(sample_true, sample_pred, zero_division=0)
        elif metric == "macro_f1":
            score = f1_score(sample_true, sample_pred, average="macro", zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(sample_true, sample_pred)
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        scores.append(float(score))
    return tuple(float(x) for x in np.percentile(scores, [2.5, 97.5]))


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class Timer:
    def __enter__(self) -> "Timer":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.seconds = time.perf_counter() - self.started
