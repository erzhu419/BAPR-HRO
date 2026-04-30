# OR Review Round 7 for `paper_opre.tex`

日期：2026-04-30  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、`paper/paper_opre.log`、OR guideline/template、Lean/source artifact  
目标期刊：Operations Research

## 总体判断

这版我认为已经可以作为 **OR reviewer-facing manuscript** 进入正式投稿准备。上一轮真正的 P0，也就是 Zenodo DOI 暴露作者身份，已经修掉：正文现在写 anonymous supplementary archive through ScholarOne，并说明 public archival DOI/source URL after acceptance。这个处理是对的。

如果只看论文 PDF，本轮我会给：

**Ready to submit, subject to final package assembly.**

不建议再大改理论、重跑主实验或继续扩展实验。剩下的都是打包前清洁度问题：supplementary source 注释、OR/ScholarOne 最终文件顺序、少量小 overfull。它们不改变论文判断。

## 本轮确认通过

1. **匿名 artifact disclosure 已修好。** `paper_opre.tex:1268-1271` 和 `paper_opre.tex:2273-2283` 都不再放 Zenodo DOI，而是写 ScholarOne anonymous supplementary archive，public DOI/repository after acceptance。
2. **PDF 页数稳定。** 当前 `paper_opre.pdf` 是 59 页；Code/Data Disclosure 和 EC 从 PDF page 39 开始，References 从 page 57 开始。正文仍在 OR Lengthy Manuscript 40 页边界内。
3. **摘要合规。** 当前 abstract 粗略计数 163 words，低于 OR 200-word limit。
4. **LaTeX 关键问题已清。** 当前 log 没有 unresolved references，也没有 `Float too large`。只剩几个小 overfull hbox 和常规 underfull/font warnings。
5. **Lean build 通过。** 12 个相关 targets 重新 `lake build` 成功；总行数 5,322，与论文 `tab:lean` 一致。
6. **formal claim 没有回退。** 精确搜 `axiom/sorry/admit/unsafe` 声明没有命中；论文的 `0 sorry, 0 project-level axiom` 口径可信。
7. **A7 audit 口径稳了。** `In this audit` / `under this audit budget` 已经加上，15-trial audit 和 45-trial main protocol 不会混读。
8. **主文“内部版本感”已经降到可接受。** `R15` 显式文本清掉了；`R16/A7/A1--A10` 还在，但现在主要作为 audit table labels，问题不大。

## 最后必须看一眼

### 1. Supplementary Lean source 还有三处 review-round 痕迹

PDF 里已经干净，但如果 source artifact 随 ScholarOne 上传，Lean 注释里还剩三处：

- `Wasserstein/DROBellman.lean:42`: `as flagged in the round-1 external review`
- `BAPR-HRO/LowerBound.lean:316-318`: `R3 fix #6` / `round-2 reviewer`
- `BAPR-HRO/BAPRHRO.lean:708-710`: `R6` / `round-5b reviewer's`

这不是数学问题，也不暴露作者，但会让 artifact 看起来像内部修订记录。建议上传 zip 前改成中性写法：

- `A previous version...`
- `Trajectory form: same statement...`
- `Final theorem: derives the event from posterior measures...`

这一步很快，值得做。

### 2. OR final package 顺序仍需按 ScholarOne 决定

当前 PDF 结构仍是 EC before References：

- `\begin{APPENDICES}` / `\ECHead{Electronic Companion}`: `paper_opre.tex:2286-2288`
- Acknowledgments: `paper_opre.tex:3459`
- References: `paper_opre.tex:3470`

OR guideline 的字面顺序是 acknowledgments -> references -> electronic companion，并且 tables after references。作为 reviewer PDF，当前 inline tables 和 EC placement 可读性更好；作为最终 submission package，建议确认 ScholarOne 是否要求 EC 单独上传。如果单独上传，就让 main manuscript 到 Code/Data Disclosure 后接 acknowledgments/references，EC 另传。

我不把这个列为论文内容 P0，但它是 submission admin checklist。

## 可忽略或低优先级

1. **小 overfull hbox。** 当前只剩 `8.26pt`、`16.66pt`、几处 `4.06pt`，集中在 Paper-to-Lean longtable。已经不是上一轮那种 300pt 级别问题。
2. **Lean linter warnings。** `lake build` 成功，warnings 是 unused variables / unused simp args / unused section vars。若要 artifact 更漂亮可以清，但不是投稿阻塞。
3. **Endnotes warning。** `No endnotes found` 可以忽略；OR 不用 footnotes，当前没有 endnotes 是正常的。
4. **`R16/A7` 标签。** 主文里仍有不少，但现在有 descriptive framing，保留作为 audit labels 可以接受。

## OR 审稿视角

### 内容判断

这篇现在像一篇可以送审的 OR paper，而不是一个不断补丁的技术报告。主线已经足够清晰：

- disruption 下 hyperpath 不是结构失效，而是排序失效；
- keep structure + online re-rank 是 operational insight；
- per-candidate score 有 Wasserstein-DRO interpretation；
- deployed gain 的大头来自 hyperpath-structural penalties，论文没有再把它强行说成 Lean 证明过的 DRO core；
- Swiss 35-day panel、paired/day-level analysis、LOO、baseline+A7 audit 支撑实证可信度；
- Lean artifact 是数学核心的证据，而不是夸大成 end-to-end simulator verification。

这套定位对 OR 是可辩护的。

### 最大可能审稿质疑

审稿人最可能抓的仍然是：

1. deployed gain 主要来自 A7 structural penalties，而不是 four-term DRO core；
2. A7 retrofit 没有覆盖 PS-SSP/BAMCP/EXP3；
3. audit tables 是 15 trials/cell，不是主实验 45 trials/cell；
4. Lean 证明范围不覆盖 simulator/data pipeline。

但这些现在都已经在文中主动承认和限定了。只要别在 cover letter 或 abstract 里重新把话说过头，这些更像 discussion points，不像 rejection-level flaw。

## 最终建议

**我建议停止内容层面的反复审稿，进入投稿包整理。**

提交前最后做三件事即可：

1. 清掉 supplementary Lean source 里的三处 review-round 注释。
2. 按 ScholarOne 决定 EC 是单独 supplemental file 还是合并 PDF，并据此调整最终文件顺序。
3. 生成最终 reviewer-facing PDF 后，再跑一次 `rg "doi.org|zenodo|author|GPT|reviewer flagged|round-"` 和一次 LaTeX log sweep。

做完这些，这版可以按 OR Lengthy Manuscript 投。
