# 提升数据与原论文逐项对比

这是 RecDY 本地提升实验的独立附件。LaTeX 是主生成链：

- `improvement_comparison.tex`：LaTeX 源文件
- `提升数据与原论文逐项对比.pdf`：由 XeLaTeX 编译的提交/阅读版
- `提升数据与原论文逐项对比.docx`：可编辑备份
- `提升数据与原论文逐项对比.md`：纯文本备份

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
