from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA_DIR, dump_json
from improvement_common import IMPROVEMENT_DATA_DIR, ensure_improvement_dirs


SEED = 20260718
ADDITIONAL_TRAIN_ROWS = 600


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ensure_improvement_dirs()
    baseline_train = pd.read_csv(DATA_DIR / "douyin_train_qwen.csv", encoding="utf-8-sig")
    main_train = pd.read_csv(DATA_DIR / "douyin_train_main.csv", encoding="utf-8-sig")
    baseline_test = pd.read_csv(DATA_DIR / "douyin_test_qwen.csv", encoding="utf-8-sig")

    if len(baseline_train) != 400 or len(baseline_test) != 400:
        raise ValueError("The frozen Qwen baseline must contain exactly 400 train and 400 test rows")
    if not set(baseline_train["row_id"]).issubset(set(main_train["row_id"])):
        raise ValueError("Frozen Qwen train rows are not a subset of the main train split")
    if set(baseline_train["row_id"]) & set(baseline_test["row_id"]):
        raise ValueError("Train/test row_id overlap detected")

    frozen_ids = set(baseline_train["row_id"])
    candidates = main_train.loc[~main_train["row_id"].isin(frozen_ids)].copy()
    if len(candidates) < ADDITIONAL_TRAIN_ROWS:
        raise ValueError("Not enough non-overlapping rows for the 1,000-row training set")
    # Use a deterministic stratified sample without depending on pandas' group
    # sampling allocation, which can change across pandas versions.
    chosen_indices: list[int] = []
    rng = np.random.default_rng(SEED + 20)
    for _, group in candidates.groupby("label", sort=True):
        count = int(round(ADDITIONAL_TRAIN_ROWS * len(group) / len(candidates)))
        count = min(count, len(group))
        chosen_indices.extend(rng.choice(group.index.to_numpy(), size=count, replace=False).tolist())
    if len(chosen_indices) < ADDITIONAL_TRAIN_ROWS:
        remaining = candidates.index.difference(chosen_indices).to_numpy()
        chosen_indices.extend(
            rng.choice(remaining, size=ADDITIONAL_TRAIN_ROWS - len(chosen_indices), replace=False).tolist()
        )
    elif len(chosen_indices) > ADDITIONAL_TRAIN_ROWS:
        chosen_indices = rng.choice(
            np.asarray(chosen_indices), size=ADDITIONAL_TRAIN_ROWS, replace=False
        ).tolist()
    additional_ids = set(candidates.loc[chosen_indices, "row_id"])
    train_1000 = main_train.loc[main_train["row_id"].isin(frozen_ids | additional_ids)].copy()
    train_1000 = train_1000.sort_values("source_index").reset_index(drop=True)
    train_400 = main_train.loc[main_train["row_id"].isin(frozen_ids)].copy()
    train_400 = train_400.sort_values("source_index").reset_index(drop=True)
    test_400 = baseline_test.sort_values("source_index").reset_index(drop=True)

    if len(train_400) != 400 or len(train_1000) != 1000 or len(test_400) != 400:
        raise AssertionError("Unexpected improvement split sizes")
    if not set(train_400["row_id"]).issubset(set(train_1000["row_id"])):
        raise AssertionError("The 400-row training set is not nested in the 1,000-row set")
    if set(train_1000["row_id"]) & set(test_400["row_id"]):
        raise AssertionError("Improvement train/test overlap detected")

    outputs = {
        "douyin_train_improvement400.csv": train_400,
        "douyin_train_improvement1000.csv": train_1000,
        "douyin_test_improvement400.csv": test_400,
    }
    manifest: dict[str, object] = {
        "seed": SEED,
        "additional_train_rows": ADDITIONAL_TRAIN_ROWS,
        "nested_train_sets": True,
        "source_train_qwen": str(DATA_DIR / "douyin_train_qwen.csv"),
        "source_train_main": str(DATA_DIR / "douyin_train_main.csv"),
        "source_test_qwen": str(DATA_DIR / "douyin_test_qwen.csv"),
        "splits": {},
    }
    for filename, frame in outputs.items():
        path = IMPROVEMENT_DATA_DIR / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        manifest["splits"][filename] = {
            "rows": int(len(frame)),
            "positive": int(frame["label"].sum()),
            "negative": int((frame["label"] == 0).sum()),
            "row_id_sha256": hashlib.sha256(
                "\n".join(frame["row_id"].astype(str)).encode("utf-8")
            ).hexdigest(),
            "file_sha256": file_sha256(path),
        }
    manifest_path = IMPROVEMENT_DATA_DIR / "improvement_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
