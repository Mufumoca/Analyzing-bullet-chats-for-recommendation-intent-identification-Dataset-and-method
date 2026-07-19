"""Analyze the strict Qwen verifier against the local baseline and paper values."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from analyze_same_budget import BOOTSTRAP_ITERATIONS, cluster_bootstrap, row_bootstrap
from common import DATA_DIR, RESULTS_DIR


RESULTS = RESULTS_DIR / "same_budget"
BASE_TAG = "strict70_dapt_rdrop_5seed_v1"
PAPER_SPT_RII_70 = 0.7923
PAPER_QWEN3_8B_PEFT = 0.7524


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }


def main() -> None:
    result_path = RESULTS / "strict70_qwen_gate_result.json"
    selection_path = RESULTS / "strict70_qwen_gate_selection.json"
    prediction_path = RESULTS / "strict70_qwen_gate_predictions.csv"
    base_path = RESULTS / f"{BASE_TAG}_predictions.csv"
    test_path = DATA_DIR / "douyin_test_full.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise AssertionError("Qwen gate result is incomplete")
    if result["selection_hash"] != selection["selection_hash"]:
        raise AssertionError("Test result does not use the validation-locked selection")
    expected_hashes = {
        selection_path: result["selection_file_sha256"],
        prediction_path: result["prediction_file_sha256"],
        base_path: result["base_test_file_sha256"],
        test_path: result["test_source_file_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise AssertionError(f"Artifact hash mismatch: {path}")

    base = pd.read_csv(
        base_path,
        encoding="utf-8-sig",
        dtype={"row_id": str, "live_id": str},
    )
    gate = pd.read_csv(
        prediction_path,
        encoding="utf-8-sig",
        dtype={"row_id": str, "live_id": str},
    )
    test = pd.read_csv(
        test_path,
        encoding="utf-8-sig",
        usecols=["row_id", "live_id", "content", "label"],
        dtype={"row_id": str, "live_id": str},
    )
    if base["row_id"].tolist() != gate["row_id"].tolist() or base["row_id"].tolist() != test["row_id"].tolist():
        raise AssertionError("Prediction and label row orders differ")

    labels = test["label"].astype(int).to_numpy()
    base_prediction = (base["probability"].to_numpy(dtype=float) >= 0.5).astype(int)
    gate_prediction = gate["prediction"].to_numpy(dtype=int)
    base_metrics = metrics(labels, base_prediction)
    gate_metrics = metrics(labels, gate_prediction)
    for key, value in base_metrics.items():
        if not np.isclose(value, result["base_metrics"][key], atol=1e-12):
            raise AssertionError(f"Recomputed base metric differs: {key}")
    for key, value in gate_metrics.items():
        if not np.isclose(value, result["qwen_gate_metrics"][key], atol=1e-12):
            raise AssertionError(f"Recomputed gate metric differs: {key}")
    base_correct = base_prediction == labels
    gate_correct = gate_prediction == labels
    routed = gate["qwen_consensus"].astype(bool).to_numpy()
    changed = gate_prediction != base_prediction

    row_stats = row_bootstrap(
        labels,
        base_prediction,
        gate_prediction,
        BOOTSTRAP_ITERATIONS,
        20260722,
    )
    cluster_stats = cluster_bootstrap(
        labels,
        base_prediction,
        gate_prediction,
        test["live_id"].astype(str).to_numpy(),
        BOOTSTRAP_ITERATIONS,
        20260822,
    )

    comparison = pd.DataFrame(
        [
            {
                "source": "Paper Table 6",
                "method": "Qwen3:8B PEFT",
                "human_label_setting": "1,000 PEFT samples",
                "llm_role": "standalone classifier",
                "f1_percent": 100 * PAPER_QWEN3_8B_PEFT,
                "metric_note": "Paper F1 averaging unspecified",
            },
            {
                "source": "Paper Table 5",
                "method": "SPT-RII 70-shot",
                "human_label_setting": "70-shot; public code samples per class",
                "llm_role": "full explanation generator",
                "f1_percent": 100 * PAPER_SPT_RII_70,
                "metric_note": "Paper F1 averaging unspecified",
            },
            {
                "source": "Local strict70",
                "method": "DAPT+R-Drop ensemble",
                "human_label_setting": "same 70/class train and validation rows",
                "llm_role": "none at inference",
                "f1_percent": 100 * base_metrics["macro_f1"],
                "metric_note": "Macro-F1",
            },
            {
                "source": "Local strict70",
                "method": "DAPT+R-Drop + selective Qwen verifier",
                "human_label_setting": "same 70/class train and validation rows",
                "llm_role": "retrieval-grounded low-confidence verifier",
                "f1_percent": 100 * gate_metrics["macro_f1"],
                "metric_note": "Macro-F1",
            },
        ]
    )
    comparison.to_csv(RESULTS / "strict70_qwen_gate_comparison.csv", index=False, encoding="utf-8-sig")

    cases = test.copy()
    cases["base_probability"] = base["probability"].to_numpy(dtype=float)
    cases["base_prediction"] = base_prediction
    cases["qwen_routed"] = routed
    cases["qwen_label"] = gate["qwen_label"].to_numpy(dtype=int)
    cases["gate_prediction"] = gate_prediction
    cases["prediction_changed"] = changed
    cases["corrected"] = (~base_correct) & gate_correct
    cases["harmed"] = base_correct & (~gate_correct)
    cases.loc[routed | changed].to_csv(
        RESULTS / "strict70_qwen_gate_changed_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    analysis: dict[str, Any] = {
        "protocol_verification": {
            "selection_hash_matches": True,
            "protocol_hash": result["protocol_hash"],
            "human_train_signature": result["human_train_signature"],
            "human_validation_signature": result["human_validation_signature"],
            "test_labels_loaded_after_selection": result["test_labels_loaded_after_selection"],
        },
        "paper_references": {
            "spt_rii_table5_70_shot_f1": PAPER_SPT_RII_70,
            "qwen3_8b_table6_peft_f1": PAPER_QWEN3_8B_PEFT,
            "metric_note": "Paper F1 averaging is unspecified; local scoring is Macro-F1, so cross-paper deltas are descriptive.",
        },
        "selection": selection["selection"],
        "base_metrics": base_metrics,
        "qwen_gate_metrics": gate_metrics,
        "delta_vs_local_base_pp": 100 * (gate_metrics["macro_f1"] - base_metrics["macro_f1"]),
        "delta_vs_paper_spt_rii_pp": 100 * (gate_metrics["macro_f1"] - PAPER_SPT_RII_70),
        "delta_vs_paper_qwen_peft_pp": 100 * (gate_metrics["macro_f1"] - PAPER_QWEN3_8B_PEFT),
        "routing": {
            "candidate_rows": result["candidate_rows"],
            "consensus_rows": int(routed.sum()),
            "changed_rows": int(changed.sum()),
            "corrected_rows": int(np.count_nonzero((~base_correct) & gate_correct)),
            "harmed_rows": int(np.count_nonzero(base_correct & (~gate_correct))),
            "net_corrected_rows": int(np.count_nonzero((~base_correct) & gate_correct) - np.count_nonzero(base_correct & (~gate_correct))),
        },
        "paired_bootstrap": {
            "row": row_stats,
            "live_id_cluster": cluster_stats,
        },
        "compute": result["qwen_stats"],
        "interpretation_limits": [
            "The public test set had been observed in earlier project experiments.",
            "The paper does not specify its F1 averaging convention.",
            "The verifier uses 20-comment text context and six labeled retrieval demonstrations, while the local student uses content only.",
            "Cross-paper performance differences are descriptive rather than a formal significance test against SPT-RII.",
        ],
    }
    (RESULTS / "strict70_qwen_gate_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
