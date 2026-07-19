# 提升数据与原论文逐项对比

这是 RecDY 本地提升实验的独立附件。LaTeX 是主生成链：

- `improvement_comparison.tex`：LaTeX 源文件
- `提升数据与原论文逐项对比.pdf`：由 XeLaTeX 编译的提交/阅读版
- `提升数据与原论文逐项对比.docx`：早期可编辑快照，不是当前权威版本
- `提升数据与原论文逐项对比.md`：早期纯文本快照，不是当前权威版本

当前严格同人工标签预算完整结果：DAPT+R-Drop 五初始化均值为
`80.01 +/- 0.62%`，同标签五模型集成为 `80.60%`；加入证据约束的
Qwen3-8B 选择性复核后为 `80.94%`。相对本地学生的点估计提高
`+0.34 pp`，但逐行 CI `[-0.26,+0.92] pp`、直播间聚类 CI
`[-0.37,+1.54] pp` 均跨 0。论文 Table 5 的 70-shot 报告点为
`79.23%`，完整系统的跨论文差值 `+1.71 pp` 仅作描述性比较。

编译 LaTeX：

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error .\improvement_comparison.tex
Copy-Item .\improvement_comparison.pdf .\提升数据与原论文逐项对比.pdf -Force
```

重新生成 Word 和 Markdown 备份：

```powershell
& D:\Anaconda3\python.exe .\build_document.py
```

`build_document.py` 读取 `experiment/results` 下的 CSV/JSON 证据，并生成 Word、Markdown 和 LaTeX 使用的 Table 5 裁剪图。原论文 PDF 位于 `E:\zotero\storage\7DATFQRR`。

通俗讲解三联图由 Python 生成：

```powershell
& D:\Anaconda3\python.exe ..\..\src\analyze_same_budget.py
& D:\Anaconda3\python.exe ..\..\src\make_same_budget_report_assets.py
& D:\Anaconda3\python.exe ..\..\src\make_plain_comparison_assets.py
& D:\Anaconda3\python.exe ..\..\src\make_qwen_gate_figure.py
& D:\Anaconda3\python.exe ..\..\src\make_qwen_gate_report_assets.py
& D:\Anaconda3\python.exe ..\..\src\verify_same_budget_artifacts.py
```

基础提升图位于 `experiment/figures/fig8_plain_comparison.*`；大模型完整对照图位于
`experiment/figures/fig9_qwen_gate_comparison.*`，对应源数据为
`source_data_fig8.csv` 和 `source_data_fig9.csv`。
