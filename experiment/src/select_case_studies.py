from __future__ import annotations

import pandas as pd

from common import DATA_DIR, RESULTS_DIR


def load_prediction(condition: str) -> pd.DataFrame:
    frame = pd.read_csv(
        RESULTS_DIR / f"predictions_classical_qwen_{condition}.csv",
        encoding="utf-8-sig",
    )
    return frame[["row_id", "prediction", "positive_probability"]].rename(
        columns={
            "prediction": f"prediction_{condition}",
            "positive_probability": f"probability_{condition}",
        }
    )


def main() -> None:
    data = pd.read_csv(DATA_DIR / "douyin_test_qwen_generated.csv", encoding="utf-8-sig")
    conditions = ["content", "content_context", "content_qwen", "content_official"]
    merged = data.copy()
    for condition in conditions:
        merged = merged.merge(load_prediction(condition), on="row_id", validate="one_to_one")

    masks = {
        "qwen_corrects_content": (
            (merged["prediction_content"] != merged["label"])
            & (merged["prediction_content_qwen"] == merged["label"])
        ),
        "qwen_harms_content": (
            (merged["prediction_content"] == merged["label"])
            & (merged["prediction_content_qwen"] != merged["label"])
        ),
        "raw_context_harms": (
            (merged["prediction_content"] == merged["label"])
            & (merged["prediction_content_context"] != merged["label"])
        ),
        "official_correct_qwen_wrong": (
            (merged["prediction_content_official"] == merged["label"])
            & (merged["prediction_content_qwen"] != merged["label"])
        ),
    }
    outputs = []
    for case_type, mask in masks.items():
        candidates = merged[mask].copy()
        candidates["confidence_sum"] = (
            (candidates["probability_content"] - 0.5).abs()
            + (candidates["probability_content_qwen"] - 0.5).abs()
        )
        candidates = candidates.sort_values("confidence_sum", ascending=False).head(12)
        candidates.insert(0, "case_type", case_type)
        outputs.append(candidates)
    output = pd.concat(outputs, ignore_index=True)
    columns = [
        "case_type",
        "row_id",
        "label",
        "content",
        "context_json",
        "qwen_background",
        "official_background",
        "prediction_content",
        "prediction_content_context",
        "prediction_content_qwen",
        "prediction_content_official",
        "probability_content",
        "probability_content_context",
        "probability_content_qwen",
        "probability_content_official",
    ]
    output[columns].to_csv(
        RESULTS_DIR / "case_study_candidates.csv", index=False, encoding="utf-8-sig"
    )
    print(output.groupby("case_type").size().to_string())


if __name__ == "__main__":
    main()
