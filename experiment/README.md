# RTX 4060 上的 BC4RII 解释增强缩小复现实验

## 目标

本项目使用官方仓库发布的 RecDY 带解释子集，检验以下问题：

1. RTX 4060 Laptop GPU 8 GB 能否稳定运行本地 Qwen3-8B Q4 模型；
2. 本地一句话释义是否优于仅当前弹幕和直接拼接原始前 5 条；
3. 解释增强在字符 TF-IDF 与 Chinese-RoBERTa、400 与 2,000 条训练规模下是否一致；
4. Qwen3-8B 更适合直接分类还是作为解释生成器。

完整结论、方法、表格、案例和有效性威胁见 `report/report.pdf`。

## 已完成实验

- Full TF-IDF：21,159 train / 9,069 test；
- Main RoBERTa：2,000 / 2,000，3 个随机种子；
- Qwen 子集：400 / 400；
- Qwen3-8B Q4_K_M 解释生成：800/800 成功；
- Qwen3-8B 零样本直接分类：400/400 可解析；
- 输入消融：仅弹幕、原始前 5 条、本地 Qwen 释义、仓库官方释义；
- 5,000 次配对行 bootstrap、数据重合审计、跨划分上下文审计；
- 5 张论文级主图，每张均含 SVG/PDF/PNG/TIFF 和 source-data CSV；
- 完整中文 LaTeX 报告，经逐页渲染检查。

## 已完成的改进实验

- 固定原 400 条测试集，新增嵌套的 1,000 条训练集；
- 结构化、证据约束的 Qwen3-8B 提示词，训练/测试覆盖率分别为 99.75%/99.50%；
- 400/400 与 1,000/400 的 TF-IDF 2×2 对照和训练集 OOF 晚期概率融合；
- RoBERTa 对象--行为紧凑表示、三随机种子验证集融合和 400/1,000 训练规模对比；
- 新增 `fig5_improvement_results` 及 `results/improvement/` 全部原始记录，旧基线结果保持不变。

改进后的代表性结果：1,000 条训练时 TF-IDF 由当前弹幕的 82.29% 提高到结构化晚期融合的 83.82%；RoBERTa 由 87.19% 提高到验证集晚期融合的 87.63%。400 条设置下 RoBERTa 晚期融合为 84.88%，与当前弹幕 84.90% 基本持平，因此报告不把小样本结果写成普遍增益。

## 核心结果

- Qwen 解释生成平均约 0.562 s/条，峰值整卡显存约 6.40 GiB；
- 400 条 TF-IDF Macro-F1：仅弹幕 78.88%，原始前 5 条 61.69%，本地 Qwen 释义 80.75%，官方释义 81.91%；
- 本地 Qwen 释义相对仅弹幕 +1.87 pp，但 95% CI 为 [-2.80, 6.69]，只能称正向趋势；
- Full TF-IDF 中官方释义 +1.26 pp，95% CI [0.58, 1.94]；
- Main RoBERTa 中官方释义从 87.16+/-0.45% 提高到 88.30+/-0.08%；
- 400 条 RoBERTa 中本地 Qwen 释义反而下降 2.34 pp；
- Qwen 直接分类 Macro-F1 75.77%，低于同子集监督基线。

## 推荐重跑顺序

在 `E:\Lab` 目录执行：

```powershell
D:\Anaconda3\python.exe experiment\src\prepare_data.py

D:\Anaconda3\envs\lora_env\python.exe experiment\src\run_classical.py --suite full
D:\Anaconda3\envs\lora_env\python.exe experiment\src\run_classical.py --suite main

D:\Anaconda3\envs\lora_env\python.exe experiment\src\generate_qwen.py --split train
D:\Anaconda3\envs\lora_env\python.exe experiment\src\generate_qwen.py --split test
D:\Anaconda3\envs\lora_env\python.exe experiment\src\classify_qwen.py
D:\Anaconda3\envs\lora_env\python.exe experiment\src\run_classical.py --suite qwen
```

RoBERTa 示例：

```powershell
$env:HF_HOME='D:\hf'
$env:HF_HUB_OFFLINE='1'
D:\Anaconda3\python.exe experiment\src\train_roberta.py `
  --suite main --condition official --seeds 100 101 102 `
  --epochs 3 --batch-size 8 --accumulation-steps 2 --max-length 128
```

汇总与交付物：

```powershell
D:\Anaconda3\python.exe experiment\src\analyze_results.py
D:\Anaconda3\python.exe experiment\src\select_case_studies.py
D:\Anaconda3\python.exe experiment\src\collect_environment.py
D:\Anaconda3\python.exe experiment\src\make_figures.py
D:\Anaconda3\python.exe experiment\src\make_report_assets.py

cd experiment\report
latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
```

改进实验的独立运行命令见报告附录“关键复现实验命令”；所有改进产物位于 `data/processed/improvement/`、`logs/improvement/` 和 `results/improvement/`，不会覆盖原始 `results/`。

## 目录

- `src/`: 所有实验、统计、绘图和报告资产脚本；
- `data/processed/`: 固定抽样、真实上下文和本地释义；
- `logs/`: Qwen 逐条 JSONL 断点日志，记录输入哈希、延迟和错误；
- `results/`: 指标 JSON、逐行预测、bootstrap、环境和数据审计；
- `figures/`: 四种格式主图及源数据；
- `report/`: LaTeX、BibTeX、PDF 和渲染 QA；
- `archive/invalid_sparse_context_20260718/`: 首轮错误稀疏上下文结果，仅作审计留档，不得用于结论。

## 重要边界

- 使用的是仓库带解释 RecDY 子集，不是 143,957 条全量数据；
- 只复现解释增强组件，不包含 SPT-RII soft prompt 和动态 verbalizer；
- 12 个直播间 ID 均同时出现在 train/test；
- 真实在线上下文含另一划分的未标注文本，属于在线/传导式设置，不是金标泄漏；
- 行级 bootstrap 未对直播间聚类；
- 本地 Qwen 使用窗口 5、Q4 量化和通用释义提示，不能与论文窗口 20 的官方解释源作纯模型比较。
