# OR Review Round 6 for `paper_opre.tex`

日期：2026-04-30  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、`paper/paper_opre.log`、OR guideline/template、Lean source tree、artifact disclosure  
目标期刊：Operations Research

## 总体判断

这一轮是真正进入 **submission package cleanup** 了。正文逻辑、主实验、A7 attribution、Lean scope、ablation 数字这些核心内容已经基本稳住；上一轮两个最刺眼的 `Float too large` 也已经修掉。

我现在会给：

**Content-ready, but not yet anonymity/package-ready.**

最需要马上处理的不是理论，也不是再跑实验，而是：正文现在直接给了一个 Zenodo DOI，我打开 DOI 后能看到作者元数据。这会破坏匿名审稿阶段的“不要在稿件中主动暴露作者身份”的基本目标。OR 是 soft double-anonymous，作者可以有线上版本，但 manuscript 里主动放一个会跳到作者信息的 DOI，风险很高。

## 本轮确认已经修好的点

1. **大版面 warning 修掉了。** 当前 `paper_opre.log` 不再有 `Float too large`。上一轮 Algorithm 1 的 `27.4pt` 和 Paper-to-Lean table 的 `306pt` 都已经消失。
2. **PDF 页数更好。** 当前 `paper_opre.pdf` 是 59 页；Code/Data Disclosure 和 EC 从 PDF page 39 开始，References 从 page 57 开始。正文仍在 OR Lengthy Manuscript 的 40 页边界内。
3. **摘要仍合规。** 当前 `\ABSTRACT` 粗略计数 163 words，低于 OR 的 200-word limit。
4. **Paper-to-Lean 表已经改成 `longtable`。** 这解决了上一轮巨大 float 的主要问题。
5. **A7 audit 口径更稳。** 正文已经加了 `In this audit` / `under this audit budget`，这能避免把 15-trial audit 和 45-trial main protocol 混读。
6. **`R15` 显式文本基本清掉。** 主文已经改成 “earlier simulator version”，这比以前像内部 changelog 的写法好很多。
7. **Lean build 成功。** 我重新跑了 12 个相关 Lean targets，`lake build` 成功；12 个文件总行数正好是 5,322，和 `tab:lean` 对得上。

## P0 / 投稿前必须处理

### 1. Zenodo DOI 会暴露作者身份

`paper_opre.tex:2282-2285` 现在写：

> The same archive is permanently deposited on Zenodo at `https://doi.org/10.5281/zenodo.19854150`; the public source-repository URL is withheld...

我打开这个 DOI 后，Zenodo landing page 的 metadata 显示了作者信息。也就是说，虽然你把 repository URL withheld 了，但 DOI 本身已经足够 deanonymize。

这比 “soft double-anonymous 允许作者有线上版本” 更危险，因为现在是 manuscript 主动把 reviewer 引到一个带作者 metadata 的页面。OR guideline 同时说 authors may post online，也说 manuscript 要 eliminate references to author names。我的建议是：

- 匿名审稿版正文不要放这个 DOI。
- 改成：`The artifact is submitted as an anonymous supplementary archive through ScholarOne; the public DOI and repository URL will be released after acceptance.`
- 如果一定要放 DOI，则必须让 Zenodo record 的 creator/metadata/filenames/README 都匿名化，并确认 landing page 不出现作者、机构、GitHub 用户名、ORCID、email。
- cover letter 可以单独给 editor 说明 artifact DOI，但不要让 reviewer-facing PDF 直接暴露作者。

这是当前唯一真正的 P0。

### 2. OR submission 顺序仍要最终确认

当前结构仍是：

- Code/Data Disclosure：`paper_opre.tex:2274`
- `\begin{APPENDICES}` + `\ECHead{Electronic Companion}`：`paper_opre.tex:2288-2290`
- Acknowledgments：`paper_opre.tex:3461`
- References：`paper_opre.tex:3471` 之后

也就是 EC 仍在 References 前。OR guideline 的字面顺序是 Acknowledgments -> References -> Electronic companions，并且 tables should be after Reference List, not inline。

这个我不再当内容硬伤；当前稿件作为 review PDF 可读性更好。但正式上传前要确认 ScholarOne 接收方式：

- 如果 EC 是单独 supplemental PDF/zip，主稿应在 Code/Data Disclosure 后进 acknowledgments/references，EC 另传。
- 如果 EC 合并在一个 PDF，最好按 guideline 放在 references 后。
- tables inline 是否允许，要按 OR/informs4 当前 submission-mode 说明确认；若编辑严格按 guideline，可能需要生成一个 submission-layout 版本。

## P1 / 强烈建议修

### 3. Supplementary Lean source 还有内部 review 痕迹

PDF 里基本干净，但如果 Lean source tree 随 artifact 上传，源码注释里还有多处内部 review 痕迹：

- `Wasserstein/DROBellman.lean:28`：`GPT55_REVIEW_v3`
- `Wasserstein/DROBellman.lean:347`：`The round-2 reviewer flagged...`
- `Wasserstein/DRO.lean:588`：`The round-3 reviewer flagged...`
- `BAPR-HRO/LowerBound.lean:347`：`The round-3 reviewer flagged...`
- `BAPR-HRO/EXP3Regret.lean:597`：`The round-5 reviewer flagged...`
- `BAPR-HRO/BAPRHRO.lean:348/392/629/849`：多处 `reviewer flagged...`

这些不影响 Lean 正确性，但 reviewer 下载 artifact 看到会很不专业，也会暴露论文多轮内部审稿/LLM修订痕迹。建议统一改成中性注释：

- `A previous version assumed...`
- `This lemma avoids postulating...`
- `We make the scope explicit here...`

不要出现 `GPT`、`reviewer`、`round-5`、`R6 reviewer`。

### 4. 主文里的 `R16/A7/A1--A10` 仍应再降一点密度

这一轮已经把 subsection title 改好了：

`Deployed Layered-Risk Score: DRO Core + Hyperpath-Structural Penalties`

这是对的。但主文仍有很多 `R16`、`A7`，尤其：

- `paper_opre.tex:1363-1409`：formal proof scope 段仍以 `R16 algorithm changes` 和 A1--A10 组织。
- `paper_opre.tex:1558-1647`：baseline fairness/A7 audit 段可接受，但 `A7` 很密。
- `paper_opre.tex:2082-2143`：ablation 段还比较像工程审计日志。

我的建议不是删掉这些标签，而是继续坚持“描述性名称在前，代号在括号里”：

- `R16 algorithm` -> `deployed layered-risk algorithm (R16 in the audit tables)`
- `A7` -> `two hyperpath-structural penalties (A7)`
- `A1 row` -> `prior-calibration-only row (A1)`
- `A2` -> `structural-penalty row (A2)`

这样不会牺牲精确性，但 OR reviewer 会少一点“这是什么实验内部版本号”的感觉。

### 5. Paper-to-Lean longtable 还有小 overfull

当前 log 已经没有大 float，但还有几个小 overfull：

- `8.26074pt` around lines `2538--2539`
- `16.66074pt` around lines `2555--2556`
- `4.06075pt` around lines `2582--2593`

这不是 blocker。若要再干净一点，最有效的是把 file path 列再窄一点没用；应该缩短 theorem names 或把 `Wasserstein/DROBellman.lean` 这类路径拆行。现在已经可以接受，但最终 PDF 质检时可以顺手清。

### 6. Lean linter warnings 仍然很多

`lake build` 成功，所以 correctness claim 没问题。剩下是 unused variables / unused simp args / unused section vars。

这不是投稿障碍；如果时间有限，优先级低于匿名 DOI 和源码 review 痕迹。但 artifact 交给 OR reviewer 时，cleaner build output 会更好看。

## P2 / 小问题

### 7. `\label{sec:method-r15-score}` 和 `\label{eq:r15-score}` 还残留旧内部名

这不进 PDF，严格说不影响审稿。但如果提交 TeX source，建议顺手改成：

- `sec:method-layered-score`
- `eq:deployed-layered-score`

### 8. Endnotes package warning 可以忽略

log 里有：

`Package endnotes Warning: No endnotes found`

OR 不允许 footnotes，你现在没有 endnotes；这条不是问题。若想清 log，可以移除 endnotes block，但不必要。

## OR 针对性评价

### 内容

现在的内容已经能送 OR 审了。文章的 OR value 不只是 “一个 routing heuristic”，而是：

- stochastic hyperpath 的 disruption failure mode；
- keep-structure/re-rank 的 operational insight；
- per-candidate robust score 的 DRO interpretation；
- 真实 35-day GTFS-RT panel；
- formal proof artifact；
- 对 deployed gains 的诚实归因。

尤其最后一点很重要。你现在承认四项 DRO core alone 不直接带来 deployed gain，A7 structural penalties 是收益大头；这会让 reviewer 更难用 “the theorem does not prove the deployed algorithm” 一刀砍掉。

### 实证

主结果、paired analysis、day-level wins、LOO robustness、baseline+A7 audit 都够支撑 submission。不要再为了审稿意见无限补实验了。现在更重要的是保持口径：主表是 45 trials/cell，A7 audit 是 15 trials/cell audit-budget。

### 证明

Lean scope 现在基本正确：证明的是 per-candidate DRO identity、bounds、scope lemmas，不是整套 simulator。`lake build` 通过，`0 sorry / 0 project-level axiom` 口径可信。源码注释清理后，formal artifact 会更像正式补充材料。

### 格式

正文 39 页进入 Code/Data/EC，摘要 163 words，页数已经压得很好。现在格式上的主要问题不是页数，而是 OR final package 的 ordering 和 artifact anonymity。

## 推荐处理顺序

1. 先把 reviewer-facing manuscript 里的 Zenodo DOI 拿掉，或确认 Zenodo record 完全匿名。
2. 清理 Lean/source artifact 里的 `GPT`、`reviewer flagged`、`round-X` 注释。
3. 确认 OR/ScholarOne 要求的 EC/references/tables 最终上传结构。
4. 把主文剩余 `R16/A7/A1--A10` 再做一轮 descriptive-first polish。
5. 可选清小 overfull 和 Lean linter warnings。

## 最终建议

**可以进入正式投稿准备，但当前 reviewer-facing 版本先不要上传。**

理由很简单：内容已经够了，证明也能 build，PDF 大问题修掉了；但 Zenodo DOI 会暴露作者身份。把匿名 artifact 这件事处理干净后，这篇稿子就可以按 OR Lengthy Manuscript 路线提交。
