from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from common import DATA_DIR, EXPERIMENT_ROOT, FIGURES_DIR, LOGS_DIR, RESULTS_DIR, dump_json


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_id_set(frame: pd.DataFrame) -> set[str]:
    assert "row_id" in frame.columns
    row_ids = frame["row_id"].astype(str).tolist()
    assert len(row_ids) == len(set(row_ids))
    return set(row_ids)


def row_id_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        "\n".join(frame["row_id"].astype(str)).encode("utf-8")
    ).hexdigest()


def resolve_artifact_path(raw_path: str | Path) -> Path:
    """Resolve recorded E:\\Lab paths while keeping verification portable."""

    path = Path(raw_path)
    if path.exists():
        return path
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "experiment" in lowered:
        index = lowered.index("experiment")
        candidate = EXPERIMENT_ROOT.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    return EXPERIMENT_ROOT / path.name


def main() -> None:
    checks: dict[str, object] = {}

    for split in ("train", "test"):
        sample = pd.read_csv(DATA_DIR / f"douyin_{split}_qwen.csv", encoding="utf-8-sig")
        generated = pd.read_csv(
            DATA_DIR / f"douyin_{split}_qwen_generated.csv", encoding="utf-8-sig"
        )
        records = read_jsonl(LOGS_DIR / f"qwen_generation_{split}.jsonl")
        latest = {str(record["row_id"]): record for record in records}
        assert len(sample) == 400
        assert len(generated) == 400
        assert generated["qwen_background"].fillna("").str.len().gt(0).all()
        assert set(sample["row_id"].astype(str)) == set(latest)
        assert all(latest[row_id].get("input_hash") for row_id in latest)
        assert all(not latest[row_id].get("error") for row_id in latest)
        checks[f"qwen_generation_{split}"] = {
            "sample_rows": len(sample),
            "generated_rows": len(generated),
            "successful_latest_records": len(latest),
        }

    direct = json.loads((RESULTS_DIR / "qwen_direct.json").read_text(encoding="utf-8"))
    direct_predictions = pd.read_csv(
        RESULTS_DIR / "predictions_qwen_direct.csv", encoding="utf-8-sig"
    )
    assert direct["valid_predictions"] == 400
    assert direct["failed_predictions"] == 0
    assert direct["coverage"] == 1.0
    assert len(direct_predictions) == 400
    checks["qwen_direct"] = {
        "rows": len(direct_predictions),
        "coverage": direct["coverage"],
        "macro_f1": direct["macro_f1"],
    }

    roberta_paths = sorted(RESULTS_DIR.glob("roberta_*_seed*.json"))
    assert len(roberta_paths) == 18
    roberta_records = [json.loads(path.read_text(encoding="utf-8")) for path in roberta_paths]
    assert all(record["test_evaluations"] == 1 for record in roberta_records)
    assert set(record["seed"] for record in roberta_records) == {100, 101, 102}
    checks["roberta"] = {
        "run_files": len(roberta_paths),
        "seeds": [100, 101, 102],
        "test_evaluations_per_run": 1,
    }

    for suite in ("full", "main", "qwen"):
        result = json.loads((RESULTS_DIR / f"classical_{suite}.json").read_text(encoding="utf-8"))
        assert result
        assert all("macro_f1" in row for row in result)
        checks[f"classical_{suite}"] = {"conditions": len(result)}

    improvement_data_dir = DATA_DIR / "improvement"
    improvement_logs_dir = LOGS_DIR / "improvement"
    improvement_results_dir = RESULTS_DIR / "improvement"

    improvement_frames = {
        "train400": pd.read_csv(
            improvement_data_dir / "douyin_train_improvement400.csv",
            encoding="utf-8-sig",
        ),
        "train1000": pd.read_csv(
            improvement_data_dir / "douyin_train_improvement1000.csv",
            encoding="utf-8-sig",
        ),
        "test400": pd.read_csv(
            improvement_data_dir / "douyin_test_improvement400.csv",
            encoding="utf-8-sig",
        ),
    }
    expected_rows = {"train400": 400, "train1000": 1000, "test400": 400}
    improvement_ids = {}
    for name, frame in improvement_frames.items():
        assert len(frame) == expected_rows[name]
        improvement_ids[name] = row_id_set(frame)

    original_train_ids = row_id_set(
        pd.read_csv(DATA_DIR / "douyin_train_qwen.csv", encoding="utf-8-sig")
    )
    original_test_ids = row_id_set(
        pd.read_csv(DATA_DIR / "douyin_test_qwen.csv", encoding="utf-8-sig")
    )
    assert improvement_ids["train400"] == original_train_ids
    assert improvement_ids["train400"] < improvement_ids["train1000"]
    assert improvement_ids["test400"] == original_test_ids
    assert improvement_ids["train1000"].isdisjoint(improvement_ids["test400"])

    split_manifest_path = improvement_data_dir / "improvement_data_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    assert split_manifest["nested_train_sets"] is True
    manifest_names = {
        "train400": "douyin_train_improvement400.csv",
        "train1000": "douyin_train_improvement1000.csv",
        "test400": "douyin_test_improvement400.csv",
    }
    for name, filename in manifest_names.items():
        entry = split_manifest["splits"][filename]
        frame = improvement_frames[name]
        path = improvement_data_dir / filename
        assert entry["rows"] == len(frame)
        assert entry["row_id_sha256"] == row_id_sha256(frame)
        assert entry["file_sha256"] == file_sha256(path)
    checks["improvement_data_splits"] = {
        "train_rows": [400, 1000],
        "test_rows": 400,
        "train400_nested_in_train1000": True,
        "train_test_overlap": 0,
        "fixed_split_hashes_verified": True,
    }

    qwen_v2_checks = {}
    qwen_configs = (("train", 400), ("train", 1000), ("test", 400))
    for split, size in qwen_configs:
        stem = f"{split}_improvement{size}_structured_v2"
        input_path = improvement_data_dir / f"douyin_{split}_improvement{size}.csv"
        generated_path = improvement_data_dir / f"douyin_{stem}.csv"
        summary_path = improvement_results_dir / f"qwen_generation_{stem}.json"
        log_path = improvement_logs_dir / f"qwen_generation_{stem}.jsonl"

        input_frame = pd.read_csv(input_path, encoding="utf-8-sig")
        generated_frame = pd.read_csv(generated_path, encoding="utf-8-sig")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        latest = {
            str(record["row_id"]): record for record in read_jsonl(log_path)
        }

        assert len(input_frame) == size
        assert len(generated_frame) == size
        assert row_id_set(input_frame) == row_id_set(generated_frame) == set(latest)
        assert summary["prompt_version"] == "structured_v2"
        assert summary["requested_rows"] == size
        assert summary["successful_rows"] + summary["failed_rows"] == size
        assert summary["coverage"] >= 0.99
        assert set(generated_frame["prompt_version"].dropna().astype(str)) == {
            "structured_v2"
        }
        valid_mask = (
            generated_frame["valid_structure"].astype(str).str.lower() == "true"
        )
        assert int(valid_mask.sum()) == summary["successful_rows"]
        assert (
            generated_frame.loc[valid_mask, "qwen_background_structured"]
            .fillna("")
            .str.strip()
            .str.len()
            .gt(0)
            .all()
        )
        assert all(record["prompt_version"] == "structured_v2" for record in latest.values())
        assert all(record.get("prompt_hash") for record in latest.values())
        assert all(record.get("input_hash") for record in latest.values())
        qwen_v2_checks[f"{split}{size}"] = {
            "rows": size,
            "successful_rows": summary["successful_rows"],
            "coverage": summary["coverage"],
        }
    checks["improvement_qwen_structured_v2"] = qwen_v2_checks

    classical_oof_checks = {}
    expected_late_fusion = {
        400: {"old_late_fusion", "structured_late_fusion"},
        1000: {"structured_late_fusion"},
    }
    for size, expected_conditions in expected_late_fusion.items():
        rows = json.loads(
            (improvement_results_dir / f"classical_improvement_{size}.json").read_text(
                encoding="utf-8"
            )
        )
        by_condition = {row["condition"]: row for row in rows}
        assert expected_conditions <= set(by_condition)
        for condition in expected_conditions:
            row = by_condition[condition]
            selection = row["selection"]
            assert selection["folds"] == 5
            assert selection["selection_seed"] == split_manifest["seed"]
            assert 0.0 <= selection["alpha_content"] <= 1.0
            assert abs(
                selection["alpha_content"] + selection["alpha_explanation"] - 1.0
            ) < 1e-12
            assert any(
                abs(selection["alpha_content"] - alpha) < 1e-12
                for alpha in selection["grid_alpha"]
            )
            assert any(
                abs(selection["threshold"] - threshold) < 1e-12
                for threshold in selection["grid_threshold"]
            )
            assert row["alpha_content"] == selection["alpha_content"]
            assert row["threshold"] == selection["threshold"]
            assert "oof_macro_f1" in selection
        classical_oof_checks[str(size)] = sorted(expected_conditions)
    checks["improvement_classical_oof_fusion"] = {
        "folds": 5,
        "selection_seed": split_manifest["seed"],
        "conditions": classical_oof_checks,
    }

    roberta_improvement_checks = {}
    expected_seeds = {100, 101, 102}
    for size in (400, 1000):
        for method in ("compact_concat", "compact_late_fusion"):
            tag = f"qwen{size}_{method}"
            paths = sorted(improvement_results_dir.glob(f"roberta_{tag}_seed*.json"))
            assert len(paths) == 3
            records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            assert {record["seed"] for record in records} == expected_seeds
            assert all(record["tag"] == tag for record in records)
            assert all(record["train_rows"] == size for record in records)
            assert all(record["test_rows"] == 400 for record in records)
            assert all(
                Path(record["train_file"]).name
                == f"douyin_train_improvement{size}_compact_object_act_v1.csv"
                for record in records
            )
            assert all(
                Path(record["test_file"]).name
                == "douyin_test_improvement400_compact_object_act_v1.csv"
                for record in records
            )
            assert len({record["test_signature"] for record in records}) == 1

            for record in records:
                prediction_path = Path(record["prediction_file"])
                assert prediction_path.exists()
                predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
                assert len(predictions) == 400
                assert row_id_set(predictions) == improvement_ids["test400"]

                if method == "compact_concat":
                    assert record["test_evaluations"] == 1
                else:
                    assert record["validation_fraction"] == 0.2
                    assert record["validation_rows"] == size // 5
                    assert record["selection_seed"] == split_manifest["seed"]
                    assert set(record["validation_runs"]) == {"content", "qwen"}
                    assert set(record["test_branch_runs"]) == {"content", "qwen"}
                    assert record["test_tuning_evaluations"] == 0
                    assert record["test_evaluations_per_branch"] == 1
                    assert all(
                        branch["target_rows"] == 400
                        for branch in record["test_branch_runs"].values()
                    )
                    selection = record["selection"]
                    assert 0.0 <= selection["alpha_content"] <= 1.0
                    assert abs(
                        selection["alpha_content"]
                        + selection["alpha_explanation"]
                        - 1.0
                    ) < 1e-12
                    assert any(
                        abs(selection["alpha_content"] - alpha) < 1e-12
                        for alpha in selection["grid_alpha"]
                    )
                    assert any(
                        abs(selection["threshold"] - threshold) < 1e-12
                        for threshold in selection["grid_threshold"]
                    )
            roberta_improvement_checks[tag] = {
                "runs": 3,
                "seeds": sorted(expected_seeds),
                "test_rows_per_run": 400,
                "test_tuning_evaluations": 0 if method.endswith("late_fusion") else None,
                "test_evaluations_per_branch": 1
                if method.endswith("late_fusion")
                else None,
            }
    checks["improvement_roberta"] = roberta_improvement_checks

    # Paper-matched full-test comparison: every shot has three independent
    # seed runs plus a predeclared probability ensemble, with no test tuning.
    paper_results_dir = RESULTS_DIR / "paper_comparison"
    paper_summary_path = paper_results_dir / "paper_matched_summary.csv"
    paper_summary = pd.read_csv(paper_summary_path, encoding="utf-8-sig")
    assert set(paper_summary["shot"].astype(int)) == {50, 60, 70}
    assert set(paper_summary["seed"].astype(str)) == {"100", "101", "102", "ensemble"}
    assert len(paper_summary) == 12
    for shot in (50, 60, 70):
        ensemble_path = paper_results_dir / f"shot{shot}_seed_ensemble.json"
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        assert ensemble["shot_per_class"] == shot
        assert ensemble["seeds"] == [100, 101, 102]
        assert ensemble["test_rows"] == 9069
        assert ensemble["test_tuning_evaluations"] == 0
        assert ensemble["official_background_role"] == "oracle_only_not_main_result"
        for seed in (100, 101, 102):
            run = json.loads(
                (paper_results_dir / f"shot{shot}_seed{seed}.json").read_text(encoding="utf-8")
            )
            assert run["shot_per_class_train"] == shot
            assert run["shot_per_class_validation"] == shot
            assert run["test_rows"] == 9069
            assert run["test_tuning_evaluations"] == 0
            prediction_path = resolve_artifact_path(run["prediction_file"])
            assert prediction_path.exists()
            assert len(pd.read_csv(prediction_path, encoding="utf-8-sig")) == 9069
    comparison_path = paper_results_dir / "paper_vs_local_comparison.csv"
    comparison = pd.read_csv(comparison_path, encoding="utf-8-sig")
    assert len(comparison) == 18
    assert set(comparison["role"]) == {
        "reported_reference",
        "clean_primary_mean",
        "clean_primary_ensemble",
        "clean_secondary_mean",
        "clean_secondary_ensemble",
        "oracle_only",
    }
    for role in {
        "reported_reference",
        "clean_primary_mean",
        "clean_primary_ensemble",
        "clean_secondary_mean",
        "clean_secondary_ensemble",
        "oracle_only",
    }:
        assert len(comparison[comparison["role"] == role]) == 3
    assert comparison[comparison["role"].str.endswith("_ensemble")]["sd"].isna().all()
    assert comparison[comparison["role"].str.endswith("_mean")]["sd"].notna().all()
    paper_reference = json.loads(
        (paper_results_dir / "paper_reference_table5.json").read_text(encoding="utf-8")
    )
    assert paper_reference["table"] == "Table 5"
    assert paper_reference["pdf_page"] == 14
    assert paper_reference["values"]["70"]["macro_f1"] == 0.7923
    checks["paper_matched_comparison"] = {
        "test_rows": 9069,
        "shots": [50, 60, 70],
        "seeds": [100, 101, 102],
        "summary_rows": len(paper_summary),
        "oracle_excluded_from_main": True,
    }

    # Validation-only uncertainty gate and its fixed test predictions.
    gate_path = improvement_results_dir / "uncertainty_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["train_rows"] == 1000
    assert gate["validation_rows"] == 200
    assert gate["test_rows"] == 400
    assert gate["selection_policy"] == "validation_only_macro_f1; test_evaluated_once"
    assert gate["test_tuning_evaluations"] == 0
    assert gate["qwen_cache"]["validation"]["valid_predictions"] == 200
    assert gate["qwen_cache"]["test"]["valid_predictions"] == 400
    for split, rows in (("validation", 200), ("test", 400)):
        cache = gate["qwen_cache"][split]
        cache_path = resolve_artifact_path(cache["cache_path"])
        assert cache_path.exists()
        records = read_jsonl(cache_path)
        latest = {str(record["row_id"]): record for record in records}
        assert len(records) == rows
        assert len(latest) == rows
        assert all(record.get("cache_version") == "direct_no_think_v2" for record in latest.values())
        assert all(record.get("cache_source") == "ollama" for record in latest.values())
        assert all(record.get("inference_options", {}).get("think") is False for record in latest.values())
    assert gate["selection"]["margin"] == 0.22
    assert gate["test"]["gate"]["macro_f1"] > gate["test"]["base"]["macro_f1"]
    assert gate["test"]["gate_vs_base"]["macro_f1_ci95_low"] < 0 < gate["test"]["gate_vs_base"]["macro_f1_ci95_high"]
    for filename, rows in (
        ("uncertainty_gate_validation_predictions.csv", 200),
        ("uncertainty_gate_test_predictions.csv", 400),
    ):
        path = improvement_results_dir / filename
        frame = pd.read_csv(path, encoding="utf-8-sig")
        assert len(frame) == rows
        assert {"base_probability", "qwen_prediction", "replaced", "gate_prediction"} <= set(frame.columns)
    checks["uncertainty_gate"] = {
        "validation_rows": 200,
        "test_rows": 400,
        "selected_margin": gate["selection"]["margin"],
        "test_macro_f1": gate["test"]["gate"]["macro_f1"],
        "ci_crosses_zero": True,
        "cache_protocol": "direct_no_think_v2; all primary rows from Ollama",
    }

    # Full-paper reader bundle: source-grounded anchors and existing assets.
    reader_dir = EXPERIMENT_ROOT / "paper_reader"
    reader_markdown = (reader_dir / "paper.md").read_text(encoding="utf-8")
    reader_map = json.loads((reader_dir / "source_map.json").read_text(encoding="utf-8"))
    assert "**Original:**" in reader_markdown and "**中文:**" in reader_markdown
    assert reader_markdown.count("**Original:**") == reader_markdown.count("**中文:**")
    assert reader_map["paper"]["draft_mode"] is True
    assert len(reader_map["blocks"]) >= 200
    for figure in reader_map["figures"]:
        assert (reader_dir / figure["image_path"]).exists()
        assert figure["id"] in reader_markdown
    reader_notes = (reader_dir / "translation_notes.md").read_text(encoding="utf-8").lower()
    assert "draft-mode" in reader_notes or "draft mode" in reader_notes
    checks["paper_reader"] = {
        "pages": len(reader_map["pages"]),
        "source_blocks": len(reader_map["blocks"]),
        "bilingual_pairs": reader_markdown.count("**Original:**"),
        "draft_mode": True,
    }

    figure_files = []
    for stem in (
        "fig1_workflow",
        "fig2_classical_results",
        "fig3_roberta_results",
        "fig4_efficiency",
        "fig5_improvement_results",
        "fig6_paper_comparison",
        "fig7_uncertainty_gate",
    ):
        for extension in ("svg", "pdf", "png", "tiff"):
            path = FIGURES_DIR / f"{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 10_000
            figure_files.append(str(path.relative_to(EXPERIMENT_ROOT)))
        svg_text = (FIGURES_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg_text
    for index in range(1, 8):
        source = FIGURES_DIR / f"source_data_fig{index}.csv"
        assert source.exists() and source.stat().st_size > 0
    fig5_source = pd.read_csv(FIGURES_DIR / "source_data_fig5.csv", encoding="utf-8-sig")
    assert set(fig5_source.columns) == {
        "panel",
        "scale",
        "condition",
        "macro_f1",
        "sd_or_error",
    }
    assert set(fig5_source["panel"]) == {"tfidf", "roberta", "cost"}
    assert len(fig5_source) == 19
    checks["figures"] = {
        "exports": len(figure_files),
        "source_data_files": 7,
        "editable_svg_text": True,
    }

    report_pdf = EXPERIMENT_ROOT / "report" / "report.pdf"
    report_tex = EXPERIMENT_ROOT / "report" / "report.tex"
    contact_sheet = EXPERIMENT_ROOT / "report" / "qa" / "report_contact_sheet.png"
    rendered_pages = sorted((EXPERIMENT_ROOT / "report" / "qa" / "pages_final").glob("page-*.png"))
    assert report_pdf.exists() and report_pdf.stat().st_size > 100_000
    assert report_tex.exists() and report_tex.stat().st_size > 20_000
    assert contact_sheet.exists() and contact_sheet.stat().st_size > 100_000
    assert len(rendered_pages) == 30
    assert rendered_pages[-1].name == "page-30.png"
    assert contact_sheet.stat().st_mtime >= report_pdf.stat().st_mtime
    report_text = report_tex.read_text(encoding="utf-8")
    required_improvement_report_markers = [
        "\\section{\u6539\u8fdb\u65b9\u6cd5\u4e0e\u6539\u8fdb\u524d\u540e\u5bf9\u6bd4}",
        r"\label{tab:improvement-classical}",
        r"\label{tab:improvement-roberta}",
        r"\label{fig:improvement}",
        "fig5_improvement_results",
        r"\label{tab:paper-comparison}",
        r"\label{fig:paper-comparison}",
        r"\label{tab:gate}",
        r"\label{fig:gate}",
        "fig6_paper_comparison",
        "fig7_uncertainty_gate",
    ]
    assert all(marker in report_text for marker in required_improvement_report_markers)
    log_text = (EXPERIMENT_ROOT / "report" / "report.log").read_text(
        encoding="utf-8", errors="replace"
    )
    critical_markers = [
        "LaTeX Warning",
        "Overfull \\hbox",
        "Float too large",
        "Citation `",
        "Reference `",
    ]
    assert not any(marker in log_text for marker in critical_markers)
    checks["report"] = {
        "pdf_bytes": report_pdf.stat().st_size,
        "latex_bytes": report_tex.stat().st_size,
        "contact_sheet_bytes": contact_sheet.stat().st_size,
        "rendered_pages": len(rendered_pages),
        "improvement_section_markers": len(required_improvement_report_markers),
        "critical_latex_warnings": 0,
    }

    required_analysis = [
        "classical_summary.csv",
        "paired_bootstrap.csv",
        "roberta_runs.csv",
        "roberta_summary.csv",
        "roberta_seed_differences.csv",
        "data_audit.json",
        "environment.json",
        "case_study_candidates.csv",
    ]
    for filename in required_analysis:
        path = RESULTS_DIR / filename
        assert path.exists() and path.stat().st_size > 0
    checks["analysis_artifacts"] = required_analysis

    dump_json(RESULTS_DIR / "verification.json", {"status": "passed", "checks": checks})
    print(json.dumps({"status": "passed", "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
