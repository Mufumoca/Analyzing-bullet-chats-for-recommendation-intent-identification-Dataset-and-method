from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common import DATA_DIR, EXPERIMENT_ROOT, RESULTS_DIR


REPORT_DIR = EXPERIMENT_ROOT / "report"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * float(value):.{digits}f}"


def num(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def macro(name: str, value: object) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def select_row(frame: pd.DataFrame, suite: str, condition: str) -> pd.Series:
    return frame[(frame["suite"] == suite) & (frame["condition"] == condition)].iloc[0]


def write_macros() -> None:
    classical = pd.read_csv(RESULTS_DIR / "classical_summary.csv", encoding="utf-8-sig")
    paired = pd.read_csv(RESULTS_DIR / "paired_bootstrap.csv", encoding="utf-8-sig")
    roberta = pd.read_csv(RESULTS_DIR / "roberta_summary.csv", encoding="utf-8-sig")
    audit = load_json(RESULTS_DIR / "data_audit.json")
    qwen_train = load_json(RESULTS_DIR / "qwen_generation_train.json")
    qwen_test = load_json(RESULTS_DIR / "qwen_generation_test.json")
    qwen_direct = load_json(RESULTS_DIR / "qwen_direct.json")

    full_content = select_row(classical, "full", "content")
    full_official = select_row(classical, "full", "content_official")
    small_content = select_row(classical, "qwen", "content")
    small_raw = select_row(classical, "qwen", "content_context")
    small_qwen = select_row(classical, "qwen", "content_qwen")
    small_official = select_row(classical, "qwen", "content_official")

    def paired_row(suite: str, comparison: str) -> pd.Series:
        return paired[
            (paired["suite"] == suite)
            & (paired["comparison"] == comparison)
            & (paired["metric"] == "macro_f1")
        ].iloc[0]

    full_delta = paired_row("full", "content_official")
    qwen_delta = paired_row("qwen", "content_qwen")
    raw_delta = paired_row("qwen", "content_context")
    main_content = select_row(roberta, "main", "content")
    main_official = select_row(roberta, "main", "official")
    small_roberta_content = select_row(roberta, "qwen", "content")
    small_roberta_qwen = select_row(roberta, "qwen", "qwen")

    values = [
        macro("FullContentMacroF", pct(full_content["macro_f1"])),
        macro("FullOfficialMacroF", pct(full_official["macro_f1"])),
        macro("FullOfficialDeltaPP", pct(full_delta["difference"])),
        macro("FullOfficialDeltaLowPP", pct(full_delta["ci95_low"])),
        macro("FullOfficialDeltaHighPP", pct(full_delta["ci95_high"])),
        macro("SmallContentMacroF", pct(small_content["macro_f1"])),
        macro("SmallRawMacroF", pct(small_raw["macro_f1"])),
        macro("SmallQwenMacroF", pct(small_qwen["macro_f1"])),
        macro("SmallOfficialMacroF", pct(small_official["macro_f1"])),
        macro("SmallDirectMacroF", pct(qwen_direct["macro_f1"])),
        macro("SmallQwenDeltaPP", pct(qwen_delta["difference"])),
        macro("SmallQwenDeltaLowPP", pct(qwen_delta["ci95_low"])),
        macro("SmallQwenDeltaHighPP", pct(qwen_delta["ci95_high"])),
        macro("SmallRawDeltaPP", pct(raw_delta["difference"])),
        macro("RobertaMainContentMean", pct(main_content["macro_f1_mean"])),
        macro("RobertaMainContentSD", pct(main_content["macro_f1_sd"])),
        macro("RobertaMainOfficialMean", pct(main_official["macro_f1_mean"])),
        macro("RobertaMainOfficialSD", pct(main_official["macro_f1_sd"])),
        macro(
            "RobertaMainDeltaPP",
            pct(main_official["macro_f1_mean"] - main_content["macro_f1_mean"]),
        ),
        macro("RobertaSmallContentMean", pct(small_roberta_content["macro_f1_mean"])),
        macro("RobertaSmallQwenMean", pct(small_roberta_qwen["macro_f1_mean"])),
        macro(
            "RobertaSmallQwenDeltaPP",
            pct(small_roberta_qwen["macro_f1_mean"] - small_roberta_content["macro_f1_mean"]),
        ),
        macro(
            "QwenGenerationMeanSeconds",
            num((qwen_train["mean_generation_seconds"] + qwen_test["mean_generation_seconds"]) / 2),
        ),
        macro(
            "QwenGenerationWallSeconds",
            num(qwen_train["wall_seconds"] + qwen_test["wall_seconds"], 1),
        ),
        macro(
            "QwenGenerationPeakGiB",
            num(max(qwen_train["peak_gpu_memory_mb"], qwen_test["peak_gpu_memory_mb"]) / 1024, 2),
        ),
        macro("QwenDirectMeanSeconds", num(qwen_direct["mean_inference_seconds"])),
        macro("QwenDirectCoverage", pct(qwen_direct["coverage"], 1)),
        macro("ContentOverlapPct", pct(audit["test_content_seen_in_train_fraction"])),
        macro("ExactOverlapRows", audit["test_exact_five_field_seen_in_train_rows"]),
        macro("QwenTrainCrossContextPct", pct(audit["qwen_train_context"]["fraction_with_cross_split_context"])),
        macro("QwenTestCrossContextPct", pct(audit["qwen_test_context"]["fraction_with_cross_split_context"])),
    ]
    (REPORT_DIR / "results_macros.tex").write_text("\n".join(values) + "\n", encoding="utf-8")


def table_dataset() -> str:
    stats = load_json(DATA_DIR / "dataset_stats.json")
    rows = []
    for platform, label in [
        ("douyin", "Douyin/RecDY"),
        ("kuaishou", "Kuaishou/RecKS"),
        ("xiaohongshu", "Xiaohongshu/RecXHS"),
        ("tiktok", "TikTok/RecTikTok"),
    ]:
        train = stats["platforms"][platform]["train"]
        test = stats["platforms"][platform]["test"]
        rows.append(
            f"{label} & {train['rows']:,} & {test['rows']:,} & {train['rows'] + test['rows']:,} & "
            f"{pct((train['positive'] + test['positive']) / (train['rows'] + test['rows']))}\\% \\\\"
        )
    return "\n".join(rows)


def table_classical() -> str:
    classical = pd.read_csv(RESULTS_DIR / "classical_summary.csv", encoding="utf-8-sig")
    specs = [
        ("full", "content", "Full", "仅当前弹幕"),
        ("full", "official_only", "Full", "仅官方释义"),
        ("full", "content_official", "Full", "当前弹幕+官方释义"),
        ("qwen", "content", "400", "仅当前弹幕"),
        ("qwen", "content_context", "400", "当前弹幕+原始前5条"),
        ("qwen", "content_qwen", "400", "当前弹幕+Qwen释义"),
        ("qwen", "content_official", "400", "当前弹幕+官方释义"),
    ]
    rows = []
    for suite, condition, scale, name in specs:
        row = select_row(classical, suite, condition)
        rows.append(
            f"{scale} & {name} & {pct(row['accuracy'])} & {pct(row['macro_precision'])} & "
            f"{pct(row['macro_recall'])} & {pct(row['macro_f1'])} \\\\"
        )
    direct = load_json(RESULTS_DIR / "qwen_direct.json")
    rows.append(
        f"400 & Qwen3-8B直接分类 & {pct(direct['accuracy'])} & {pct(direct['macro_precision'])} & "
        f"{pct(direct['macro_recall'])} & {pct(direct['macro_f1'])} \\\\"
    )
    return "\n".join(rows)


def table_bootstrap() -> str:
    paired = pd.read_csv(RESULTS_DIR / "paired_bootstrap.csv", encoding="utf-8-sig")
    specs = [
        ("full", "content_official", "Full: +官方释义"),
        ("main", "content_official", "2,000: +官方释义"),
        ("qwen", "content_context", "400: +原始前5条"),
        ("qwen", "content_qwen", "400: +Qwen释义"),
        ("qwen", "content_official", "400: +官方释义"),
    ]
    rows = []
    for suite, comparison, name in specs:
        row = paired[
            (paired["suite"] == suite)
            & (paired["comparison"] == comparison)
            & (paired["metric"] == "macro_f1")
        ].iloc[0]
        rows.append(
            f"{name} & {pct(row['difference'])} & [{pct(row['ci95_low'])}, {pct(row['ci95_high'])}] & "
            f"{int(row['rows']):,} \\\\"
        )
    return "\n".join(rows)


def table_roberta() -> str:
    summary = pd.read_csv(RESULTS_DIR / "roberta_summary.csv", encoding="utf-8-sig")
    specs = [
        ("main", "content", "2,000", "仅当前弹幕"),
        ("main", "official", "2,000", "当前弹幕+官方释义"),
        ("qwen", "content", "400", "仅当前弹幕"),
        ("qwen", "context", "400", "当前弹幕+原始前5条"),
        ("qwen", "qwen", "400", "当前弹幕+Qwen释义"),
        ("qwen", "official", "400", "当前弹幕+官方释义"),
    ]
    rows = []
    for suite, condition, scale, name in specs:
        row = select_row(summary, suite, condition)
        rows.append(
            f"{scale} & {name} & {pct(row['accuracy_mean'])}$\\pm${pct(row['accuracy_sd'])} & "
            f"{pct(row['macro_precision_mean'])}$\\pm${pct(row['macro_precision_sd'])} & "
            f"{pct(row['macro_recall_mean'])}$\\pm${pct(row['macro_recall_sd'])} & "
            f"{pct(row['macro_f1_mean'])}$\\pm${pct(row['macro_f1_sd'])} \\\\"
        )
    return "\n".join(rows)


def table_efficiency() -> str:
    train = load_json(RESULTS_DIR / "qwen_generation_train.json")
    test = load_json(RESULTS_DIR / "qwen_generation_test.json")
    direct = load_json(RESULTS_DIR / "qwen_direct.json")
    rows = [
        f"训练集释义 & 400 & {train['successful_rows']} & {train['mean_generation_seconds']:.3f} & "
        f"{train['wall_seconds']:.1f} & {train['peak_gpu_memory_mb'] / 1024:.2f} \\\\ ",
        f"测试集释义 & 400 & {test['successful_rows']} & {test['mean_generation_seconds']:.3f} & "
        f"{test['wall_seconds']:.1f} & {test['peak_gpu_memory_mb'] / 1024:.2f} \\\\ ",
        f"Qwen直接分类 & 400 & {direct['valid_predictions']} & {direct['mean_inference_seconds']:.3f} & "
        f"{direct['wall_seconds']:.1f} & {direct['peak_gpu_memory_mb'] / 1024:.2f} \\\\ ",
    ]
    return "\n".join(rows)


def wrap_tabular(column_spec: str, header: str, rows: str) -> str:
    return (
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_macros()
    tables = {
        "table_dataset.tex": wrap_tabular(
            "lrrrr", "平台 & 训练集 & 测试集 & 合计 & 正类比例", table_dataset()
        ),
        "table_classical.tex": wrap_tabular(
            "llrrrr",
            "规模 & 输入条件 & Accuracy & Macro-P & Macro-R & Macro-F1",
            table_classical(),
        ),
        "table_bootstrap.tex": wrap_tabular(
            "lrrr", "比较条件 & 差值/pp & 95\\% CI/pp & 测试行数", table_bootstrap()
        ),
        "table_roberta.tex": wrap_tabular(
            "llrrrr",
            "训练规模 & 输入条件 & Accuracy & Macro-P & Macro-R & Macro-F1",
            table_roberta(),
        ),
        "table_efficiency.tex": wrap_tabular(
            "lrrrrr",
            "阶段 & 请求数 & 成功数 & 平均 s/条 & 墙钟时间/s & 峰值/GiB",
            table_efficiency(),
        ),
    }
    for filename, content in tables.items():
        (REPORT_DIR / filename).write_text(content + "\n", encoding="utf-8")
    print(f"Report assets written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
