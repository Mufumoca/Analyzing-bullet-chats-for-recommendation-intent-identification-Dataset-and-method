"""Build a grounded, page-preserving bilingual reader from the extracted PDF text.

This bundle is intentionally marked draft mode: the full English source is
retained with stable page/block anchors, while high-value experimental passages
have checked Chinese translations and the remaining blocks carry an explicit
translation-review note instead of an invented translation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "paper.txt"
OUT = ROOT


HEADING_TRANSLATIONS = {
    "1. Introduction": "1. 引言",
    "2. Related work": "2. 相关工作",
    "2.1. Live-streaming sales": "2.1 直播销售",
    "2.2. Bullet chatting": "2.2 弹幕交流",
    "2.3. Recommendation intent identiﬁcation": "2.3 推荐意图识别",
    "3. Creating BC4RII": "3. BC4RII 数据集构建",
    "3.1. Data preparation": "3.1 数据准备",
    "3.2. Machine-generated pseudo labels": "3.2 机器生成伪标签",
    "4. Data analysis": "4. 数据分析",
    "5. Methodology": "5. 方法",
    "5.1. Motivation": "5.1 动机",
    "5.2. Formalization and overall architecture": "5.2 形式化定义与总体架构",
    "5.3. Explanations generation for bullet chats": "5.3 弹幕解释生成",
    "5.4. Soft prompt-tuning model": "5.4 软提示调优模型",
    "5.5. Verbalizer construction": "5.5 Verbalizer 构建",
    "6. Experiments": "6. 实验",
    "7. Conclusion": "7. 结论",
}


EXPERIMENT_TRANSLATIONS = {
    "Live-streaming sales (LS) have emerged": (
        "直播销售（LS）已经成为重要的电子商务形态，弹幕是观众与主播互动的主要渠道。论文将推荐意图识别（RII）定义为判断弹幕是否表达寻求商品推荐的意图，并构建了 BC4RII 数据集与 SPT-RII 方法。",
        "high",
    ),
    "RQ: Recommendation Intent Identiﬁcation": (
        "研究问题是：判断一条直播弹幕表达的是推荐意图，还是普通闲聊。",
        "high",
    ),
    "The dataset and code are available": (
        "数据集和代码发布在 BC4RII 官方仓库中。",
        "high",
    ),
    "The overall framework of our method": (
        "方法以滑动窗口取得近期弹幕，由大语言模型生成语义解释，再将原弹幕和解释输入软提示模型；同时通过语义距离和时间衰减动态更新扩展词表，以适应主题漂移。",
        "high",
    ),
    "where k is the window size": (
        "实验将历史窗口大小设为 k=20，即使用当前时刻之前的 20 条弹幕生成解释。",
        "high",
    ),
    "The recommendation intent identification task can be formalized": (
        "推荐意图识别被形式化为二分类：给定当前弹幕及其大模型解释，预测 1（购买/推荐意图）或 0（闲聊）。",
        "high",
    ),
    "Table 5": (
        "表 5 报告了四个平台、不同数据规模和多种基线的实验结果。RecDY 的 SPT-RII 在每类 50、60、70 条训练/验证样本时，Macro-F1 分别为 77.84±0.32%、78.92±0.56% 和 79.23±0.31%。",
        "high",
    ),
    "Table 6": (
        "表 6 汇总了与大语言模型的比较；需要注意，表 6 中 RecDY 的 50-shot SPT-RII 数值与表 5 存在不一致。",
        "high",
    ),
}


def normalise(text: str) -> str:
    replacements = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "­": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify(block: str) -> str:
    if re.match(r"^(Fig\.|Table\s+\d+)", block):
        return "caption"
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[^.]{1,90}$", block) and len(block) < 130:
        return "heading"
    if block.startswith("a r t i c l e") or block in {"Contents lists available at ScienceDirect"}:
        return "note"
    return "paragraph"


def translation_for(block: str, block_type: str) -> tuple[str, str]:
    if block_type == "heading":
        key = block.replace("ﬁ", "fi")
        for source, translation in HEADING_TRANSLATIONS.items():
            if key.startswith(source.replace("ﬁ", "fi")):
                return translation, "high"
        return "【标题译文待核对】" + block, "low"
    if block_type == "caption":
        if block.startswith("Table 5"):
            return "表 5：不同数据条件和基线下四个平台的实验结果；数值为均值±标准差（%）。", "high"
        if block.startswith("Table"):
            return "表格标题及说明已保留；中文译文待核对。", "medium"
        if block.startswith("Fig."):
            return "图注已保留；中文译文待核对。", "medium"
    for marker, (translation, confidence) in EXPERIMENT_TRANSLATIONS.items():
        if marker in block:
            return translation, confidence
    return "【中文译文待核对】本块原文已完整保留；本次实验报告优先核对了数据协议、Table 5 数值和方法边界。", "low"


def main() -> None:
    pages = RAW.read_text(encoding="utf-8", errors="replace").split("\f")
    pages = [page for page in pages if page.strip()]
    blocks: list[dict[str, object]] = []
    page_index: list[dict[str, object]] = []
    order = 0
    for page_number, page in enumerate(pages, start=1):
        page_block_ids: list[str] = []
        # Blank lines are reliable paragraph separators in pdftotext output.
        raw_blocks = re.split(r"\n\s*\n", page)
        for raw_block in raw_blocks:
            text = normalise(raw_block)
            if not text:
                continue
            # Drop repeated running headers and isolated page numbers, but keep
            # all substantive paper text.
            if text in {"Arti cial Intelligence 355 (2026) 104528", "Y. Zhu, Q. Han, Y. Yuan et al."}:
                continue
            if re.fullmatch(r"\d+", text):
                continue
            order += 1
            block_id = f"S{order:04d}"
            block_type = classify(text)
            translation, confidence = translation_for(text, block_type)
            item = {
                "id": block_id,
                "page": page_number,
                "type": block_type,
                "order": order,
                "original_text": text,
                "translation": translation,
                "bbox": [0, 0, 0, 0],
                "confidence": confidence,
                "refs": [],
            }
            blocks.append(item)
            page_block_ids.append(block_id)
        page_index.append({"page": page_number, "block_ids": page_block_ids})

    # The extracted Table 5 crop is an approximate visual crop of PDF page 14.
    table_block = {
        "id": "T001",
        "page": 14,
        "type": "table",
        "order": len(blocks) + 1,
        "original_text": "Table 5, RecDY rows and all reported conditions.",
        "translation": "表 5：RecDY 的 50/60/70-shot 结果；本项目主报告使用其 SPT-RII Macro-F1 行。",
        "bbox": [0, 0, 0, 0],
        "confidence": "high",
        "refs": [],
    }

    md_lines = [
        "# BC4RII Paper Reader (Draft)",
        "",
        "> Source: Zhu et al., *Artificial Intelligence* 355 (2026) 104528, DOI: 10.1016/j.artint.2026.104528.",
        "> Mode: full extracted text with stable page/block anchors. Chinese translations marked as review-needed are intentionally not fabricated.",
        "",
        "## Experimental takeaways",
        "",
        "- RecDY Table 5 reports SPT-RII Macro-F1 of 77.84±0.32, 78.92±0.56 and 79.23±0.31 for 50/60/70 examples per class (source: p.14, Table 5).",
        "- The paper uses a 20-comment historical window and combines LLM explanation generation, soft prompts and dynamic verbalizer updates (source: p.9–10, S0030–S0045; anchors depend on extracted layout).",
        "- This workspace's comparison is explicitly a lower-cost substitute, not an exact SPT-RII reproduction.",
        "",
        "## Page-by-page bilingual text",
        "",
    ]
    for page in page_index:
        page_number = int(page["page"])
        md_lines.extend([f"## Page {page_number}", ""])
        for block_id in page["block_ids"]:
            block = next(item for item in blocks if item["id"] == block_id)
            md_lines.extend(
                [
                    f'<a id="{block_id}"></a>',
                    f"**Source:** p.{page_number} {block_id}",
                    "",
                    f"**Original:** {block['original_text']}",
                    "",
                    f"**中文:** {block['translation']}",
                    "",
                ]
            )
            if page_number == 14 and block_id == page["block_ids"][0]:
                md_lines.extend(
                    [
                        '<a id="T001"></a>',
                        "### Table 5. RecDY experimental results",
                        "",
                        "**Placed near:** p.14",
                        "",
                        "**Source:** Table 5 crop (approximate crop from the provided PDF)",
                        "",
                        "![Table 5](assets/table5_recdy.png)",
                        "",
                        "**Original caption:** Experimental results across four platforms and all baselines under varying data conditions.",
                        "",
                        "**中文图注:** 不同数据条件和基线下四个平台的实验结果。",
                        "",
                        "**Reading note:** RecDY 的 SPT-RII 行是本项目 full-test shot 对比的论文参考值。",
                        "",
                    ]
                )

    (OUT / "paper.md").write_text("\n".join(md_lines), encoding="utf-8")
    source_map = {
        "paper": {
            "title": "Analyzing bullet chats for recommendation intent identification: Dataset and method",
            "venue": "Artificial Intelligence 355 (2026) 104528",
            "source_type": "pdf",
            "language": "en",
            "source_path": "provided PDF; extracted to raw/paper.txt",
            "draft_mode": True,
        },
        "blocks": blocks + [table_block],
        "pages": page_index,
        "figures": [
            {
                "id": "T001",
                "page": 14,
                "caption_id": "T001",
                "image_path": "assets/table5_recdy.png",
                "bbox": [0, 0, 0, 0],
                "placement_hint": "near Table 5 discussion",
                "placed_after": page_index[13]["block_ids"][0] if len(page_index) > 13 and page_index[13]["block_ids"] else "",
                "alt_text": "Table 5 crop showing RecDY SPT-RII results for 50, 60 and 70 examples per class.",
            }
        ],
        "glossary": [
            {"term": "bullet chat", "translation": "弹幕", "note": "保留为直播销售场景中的实时滚动评论。"},
            {"term": "recommendation intent identification (RII)", "translation": "推荐意图识别", "note": "二分类任务。"},
            {"term": "soft prompt-tuning", "translation": "软提示调优", "note": "论文方法核心模块。"},
            {"term": "verbalizer", "translation": "verbalizer/标签词表", "note": "本报告保留英文术语以避免与普通词表混淆。"},
            {"term": "Macro-F1", "translation": "宏平均 F1", "note": "比较时使用论文 Table 5 的 F1 列。"},
        ],
    }
    (OUT / "source_map.json").write_text(json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = """# Translation and Grounding Notes

## Mode

This is a draft-mode full-paper reader generated from selectable PDF text. The
English source is retained page by page and every substantive block has a stable
`S####` anchor. The extracted PDF contains ligature and two-column reading-order
artifacts; those are preserved in the source text and should be checked against
the page image when a quotation matters.

## Translation policy

The abstract, task definition, method overview, explanation-window statement and
Table 5/6 experimental facts have checked Chinese translations. Other blocks are
marked `中文译文待核对` rather than silently inventing a translation. This keeps
the reader source-grounded and makes the missing translation visible.

## Table/figure extraction

`assets/table5_recdy.png` is an approximate tight crop of PDF page 14. It is
linked from the Table 5 card in `paper.md` and its source pointer is `T001` in
`source_map.json`. No other image asset is used by the reader bundle.

## Protocol note for this project

The local experiment matches the paper's RecDY test size and per-class shot
count in a separate substitute-model run, but does not implement the paper's
soft prompt, BiLSTM prompt interaction, or dynamic verbalizer. Official
explanations are therefore kept as an oracle-only reference in the report.
"""
    (OUT / "translation_notes.md").write_text(notes, encoding="utf-8")
    print(f"Wrote {len(blocks)} source blocks across {len(pages)} pages")


if __name__ == "__main__":
    main()
