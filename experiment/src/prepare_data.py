from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
import pandas as pd

from common import (
    CSV_COLUMNS,
    DATA_DIR,
    SOURCE_ROOT,
    attach_context,
    dump_json,
    ensure_dirs,
    load_generated_split,
    stratified_sample,
)


PLATFORMS = ("douyin", "kuaishou", "xiaohongshu", "tiktok")
MAIN_SAMPLE_SIZE = 2000
QWEN_SAMPLE_SIZE = 400
SEED = 20260718
CONTEXT_WINDOW = 5
MATCH_COLUMNS = ["live_id", "species", "content", "label", "official_background"]


def describe(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "positive": int((frame["label"] == 1).sum()),
        "negative": int((frame["label"] == 0).sum()),
        "positive_rate": float(frame["label"].mean()),
        "unique_live_ids": int(frame["live_id"].nunique()),
        "median_content_chars": float(frame["content"].str.len().median()),
        "median_background_chars": float(frame["official_background"].str.len().median()),
    }
    if "context_json" in frame:
        context_sizes = frame["context_json"].map(lambda value: len(json.loads(value)))
        summary["mean_context_items"] = float(context_sizes.mean())
        summary["rows_with_full_context_window"] = int((context_sizes == CONTEXT_WINDOW).sum())
    return summary


def row_keys(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    normalized = frame[MATCH_COLUMNS].copy()
    for column in MATCH_COLUMNS:
        normalized[column] = normalized[column].astype(str)
    return list(normalized.itertuples(index=False, name=None))


def map_split_to_all(split_frame: pd.DataFrame, all_frame: pd.DataFrame) -> np.ndarray:
    all_keys = row_keys(all_frame)
    split_keys = row_keys(split_frame)
    positions_by_key: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for position, key in enumerate(all_keys):
        positions_by_key[key].append(position)

    mapped = np.full(len(split_frame), -1, dtype=int)
    for position, key in enumerate(split_keys):
        candidates = positions_by_key.get(key, [])
        if not candidates:
            raise ValueError(f"Split row {position} is absent from dy_bg_all.csv")
        if len(candidates) == 1:
            mapped[position] = candidates[0]

    previous_anchor = np.full(len(mapped), -1, dtype=int)
    last = -1
    for position, value in enumerate(mapped):
        previous_anchor[position] = last
        if value >= 0:
            last = value
    next_anchor = np.full(len(mapped), len(all_frame), dtype=int)
    upcoming = len(all_frame)
    for position in range(len(mapped) - 1, -1, -1):
        next_anchor[position] = upcoming
        if mapped[position] >= 0:
            upcoming = mapped[position]

    used = set(int(value) for value in mapped if value >= 0)
    for position, key in enumerate(split_keys):
        if mapped[position] >= 0:
            continue
        candidates = [
            value
            for value in positions_by_key[key]
            if previous_anchor[position] < value < next_anchor[position] and value not in used
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely align duplicate split row {position}; candidates={candidates}"
            )
        mapped[position] = candidates[0]
        used.add(candidates[0])

    if not np.all(np.diff(mapped) > 0):
        raise ValueError("Released split is not a strict subsequence of dy_bg_all.csv")
    if any(all_keys[all_position] != split_key for all_position, split_key in zip(mapped, split_keys)):
        raise ValueError("Aligned row content does not match dy_bg_all.csv")
    return mapped


def load_douyin_splits_with_true_context() -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    all_path = SOURCE_ROOT / "BC4RII_with_generation" / "douyin" / "dy_bg_all.csv"
    all_frame = pd.read_csv(all_path, encoding="gbk", dtype=str, keep_default_na=False)
    all_frame = all_frame.rename(columns={"background": "official_background"})
    if list(all_frame.columns) != CSV_COLUMNS:
        raise ValueError(f"Unexpected columns in {all_path}: {list(all_frame.columns)}")
    all_frame["label"] = pd.to_numeric(all_frame["label"], errors="raise").astype(int)
    all_frame = attach_context(all_frame, window=CONTEXT_WINDOW)

    splits: dict[str, pd.DataFrame] = {}
    mapped_positions: list[int] = []
    for split in ("train", "test"):
        frame = load_generated_split("douyin", split)
        positions = map_split_to_all(frame, all_frame)
        frame["all_sequence_index"] = positions
        frame["context_json"] = all_frame.iloc[positions]["context_json"].to_numpy()
        splits[split] = frame
        mapped_positions.extend(positions.tolist())

    if sorted(mapped_positions) != list(range(len(all_frame))):
        raise ValueError("Train/test splits do not form an exact partition of dy_bg_all.csv")
    duplicate_rows = int(all_frame.duplicated(MATCH_COLUMNS, keep=False).sum())
    audit = {
        "source": str(all_path),
        "source_encoding": "gbk",
        "window": CONTEXT_WINDOW,
        "all_rows": int(len(all_frame)),
        "globally_unique_rows": int(len(all_frame) - duplicate_rows),
        "duplicate_row_instances_aligned_by_anchors": duplicate_rows,
        "exact_partition_verified": True,
        "strict_subsequence_verified": True,
    }
    return splits, audit


def main() -> None:
    ensure_dirs()
    stats: dict[str, object] = {
        "source": str(SOURCE_ROOT / "BC4RII_with_generation"),
        "seed": SEED,
        "platforms": {},
    }

    douyin_splits, context_audit = load_douyin_splits_with_true_context()
    stats["context_construction"] = context_audit
    for platform in PLATFORMS:
        platform_stats: dict[str, object] = {}
        for split in ("train", "test"):
            frame = load_generated_split(platform, split)
            platform_stats[split] = describe(frame)
        stats["platforms"][platform] = platform_stats

    for split, frame in douyin_splits.items():
        frame.to_csv(DATA_DIR / f"douyin_{split}_full.csv", index=False, encoding="utf-8-sig")
        main_sample = stratified_sample(frame, MAIN_SAMPLE_SIZE, SEED + (0 if split == "train" else 1))
        main_sample.to_csv(DATA_DIR / f"douyin_{split}_main.csv", index=False, encoding="utf-8-sig")
        qwen_sample = stratified_sample(main_sample, QWEN_SAMPLE_SIZE, SEED + (10 if split == "train" else 11))
        qwen_sample.to_csv(DATA_DIR / f"douyin_{split}_qwen.csv", index=False, encoding="utf-8-sig")
        stats[f"douyin_{split}_main"] = describe(main_sample)
        stats[f"douyin_{split}_qwen"] = describe(qwen_sample)

    dump_json(DATA_DIR / "dataset_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
