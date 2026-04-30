# OR Review Round 5 for `paper_opre.tex`

日期：2026-04-30  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、`paper/paper_opre.log`、OR submission guideline/template、Lean source tree  
目标期刊：Operations Research

## 总体判断

这版已经不是“论文本身站不住”的阶段了。主线现在很清楚：静态 stochastic hyperpath 本身保留了可用备选项，真正的问题是 disruption 下排序失效；论文用在线 pessimistic re-ranking 解决这个排序问题，并用 Wasserstein-DRO/Lean 给核心 score 一个可审计的数学解释。

如果我是 OR 预审视角，这版我会给：

**可以准备投稿，但上传前必须做 final production cleanup。**

内容层面的最大风险已经降下来了：baseline fairness、A7 attribution、R16 ablation 数字、Paper-to-Lean mapping、double-anonymous 处理，都比上一轮稳。现在剩下的主要问题是：当前本地 PDF/log 还不干净，OR 包装顺序还要按 guideline 最终整理，以及主文里仍有少量工程版本号/内部代号影响“人话”阅读。

## 本轮已明显改善

1. **A7 attribution 现在可接受。** `tab:baseline_a7_audit` 明确写了 `15 trials per cell`、single seed、audit-budget 口径；正文也承认这不是主表 45-trial protocol。这不再是硬伤。
2. **Baseline fairness 叙述已经自洽。** Static/SW-LCB 做了 A7 retrofit，PS-SSP/BAMCP/EXP3 没 retrofit 的原因也讲清楚了：需要 loss/rollout adapter。
3. **R16 ablation 数字已经同步。** 主表和 Appendix A1--A10 现在是一套数：A0 `+13.0%`，A1 `+3.5%/+6.0%`，A2 `-3.0%/-7.1%`，A3 `-6.2%/-6.8%`。
4. **Paper-to-Lean 口径更稳。** `regret_sandwich` 已经降成 lower/upper-bound scope lemma，保留 legacy theorem name 但不再把它包装成过强结果。
5. **Lean claim 经当前工程验证能 build。** 我跑了 12 个相关 Lean targets，`lake build` 成功；仍有 linter warnings，但没有 build failure。
6. **OR 长文页数基本踩线成功。** 当前 PDF 60 页；`pdftotext` 显示 Code/Data Disclosure 和 EC 从 PDF page 39 开始，References 从 page 54 开始。按 Lengthy Manuscript “正文一般不超过 40 页，不含 references”的口径，现在是能解释的。
7. **摘要合规。** 当前 `\ABSTRACT` 粗略计数 163 words，低于 OR 200-word limit；没有明显不适合网页 metadata 的复杂公式。

## P0 / 上传前必须处理

### 1. 当前本地 LaTeX log 仍不是 clean build

你上一轮说过具体 warning 可能来自旧 PDF；但我检查的是当前目录里 2026-04-30 08:51 生成的 `paper_opre.log`，仍然有明显问题：

- `Float too large for page by 27.41983pt on input line 607`，对应 Algorithm 1，`paper_opre.tex:560-607`。
- `Float too large for page by 306.15945pt on input line 2595`，对应 Paper-to-Lean cross-reference table，`paper_opre.tex:2518-2595`。
- EC Lean table 附近还有多处 overfull hbox，集中在 `paper_opre.tex:2534-2589`。

这不是理论问题，但 OR 上传前最好别留。尤其 306pt 的 float-too-large 太大，属于很容易被 production 或 reviewer 注意到的版面问题。

建议：

- Algorithm 1 可以压缩伪代码，把 posterior update 或 cancel handling 移到正文/EC，只保留 score-and-board 主流程。
- Paper-to-Lean cross-reference table 建议拆成两张：主文/EC 摘要表只列 paper result、Lean file、main theorem name；长 theorem names 和解释放到第二张表或 README。
- 如果你本地另一份编译确实是 0 warnings，就把那份 PDF/log 同步到当前目录；按当前 workspace，不能写“compile clean”。

### 2. OR guideline 的最终顺序还需要整理

本地 guideline 写得很明确：

- Lengthy manuscripts 一般不超过 40 页，不含 references。
- text order 是 title/abstract、main sections、appendices、acknowledgments、references、electronic companions。
- tables should be placed together after the Reference List, not embedded in the manuscript text。

当前稿件的实际结构是：

- `Code and Data Disclosure` 在 `paper_opre.tex:2276`；
- `\begin{APPENDICES}` 和 `\ECHead{Electronic Companion}` 在 `paper_opre.tex:2289-2291`；
- Acknowledgments 在 `paper_opre.tex:3466`；
- References 从 `paper_opre.tex:3468` 之后开始；
- tables 仍大量 inline。

我不建议现在为了版面把全稿表格强行搬家，因为这会降低可读性，也可能和 `informs4` review style 的实际工作流冲突。但最终 ScholarOne 上传前要确认一件事：**EC 是单独 supplemental PDF/zip，还是和 main manuscript 合并在一个 PDF。** 如果按 guideline 的字面顺序，当前 EC 在 references 前面是不规范的。

最低成本修法：

- main PDF 保留正文到 Code/Data Disclosure；
- EC 另成 `paper_opre_ec.tex/pdf` 或在主 PDF references 之后；
- tables 是否 inline 按 ScholarOne/OR latex instruction 最终确认；如果必须 after references，就至少生成一个 submission-mode 版本。

## P1 / 强烈建议修

### 3. 主文里 `R16/R15/A7/A1--A10` 仍像工程内部代号

这版比之前好，但对 OR reader 来说，`R16 deployed score`、`R15 reports`、`A7 layered penalties` 仍然有点像实验室 changelog。问题不是定义缺失，而是读者要不断把代号翻译成含义。

典型位置：

- `paper_opre.tex:617-634`：subsection 直接叫 `R16 Deployed Score`，还提 `earlier R15 reports`。
- `paper_opre.tex:1366-1411`：formal proof scope 用 A1--A10 解释算法变化。
- `paper_opre.tex:2082-2119`：ablation 段落密集使用 A0/A1/A2/A3/A7/R16。
- `paper_opre.tex:3382-3458`：EC 里的 A1--A10 mapping 没问题，适合保留。

建议把主文改成“人话名在前，代号在括号里”：

- `R16 Deployed Score` -> `Deployed Layered-Risk Score (R16 audit label)`。
- `earlier R15 reports` -> `an earlier simulator version`。
- `A7` 首次出现时用 `two hyperpath-structural risk penalties (A7 in the audit table)`。
- 主文尽量少说 A1/A2/A3，改成 `prior calibration only`、`+ structural penalties`、`all deployed components`；EC 里继续保留 A1--A10。

这个不是硬伤，但 OR 非本领域审稿人会更容易读懂。

### 4. Paper-to-Lean table 同时有版面问题和阅读问题

`tab:xref` 的目标是好目标：让 reviewer 能查 paper theorem 到 Lean theorem 的映射。但现在每行 theorem name 太长，导致 overfull/float warning，也让表格阅读成本很高。

建议改成三层：

1. 主文或 EC summary table：paper theorem、Lean file、primary theorem。
2. 详细 theorem-name list：放到 EC 或 README。
3. code archive README：给 `lake build` command 和 exact target list。

这样比把全部 theorem identifiers 塞进一张 TeX 表更符合 OR 读者的阅读习惯。

### 5. A7 audit 的文字要继续守住“audit”口径

当前 `paper_opre.tex:1619-1649` 已经写得比较稳，尤其是最后承认它是 `15`-trial audit budget 而不是 `45`-trial main protocol。

我建议只做一个小收口：凡是说 “A7 penalties alone capture roughly 60%...” 或 “almost all reach-rate improvement” 的句子，最好加上 `in this audit` / `under this audit budget`。这样 reviewer 即使盯 single seed，也不会觉得你把 audit table 当成主实验来卖。

这不是要求重跑 45 trials/cell。当前 caption 已经足够防混读；45-trial rerun 只是锦上添花。

### 6. Lean source build 成功，但 warnings 可以轻量清理

`lake build` 成功，这支持论文里的 “all 12 files compile” claim。剩下的是 Lean linter warnings，主要是 unused variables / unused simp args / unused section vars。

这不影响 correctness，也不需要写进论文。但如果 supplementary code 会交给 reviewer，本轮可以顺手清一部分最显眼的 warnings，尤其：

- `BAPR-HRO/BAPRHRO.lean`
- `BAPR-HRO/BAPRHRO_V2.lean`
- `BAPR-HRO/LowerBound.lean`
- `BAPR-HRO/EXP3Regret.lean`
- `Wasserstein/DRO.lean`
- `Wasserstein/Distance.lean`
- `Wasserstein/DROBellman.lean`

不清也能投；清掉会让 formalization package 看起来更专业。

## P2 / 小问题

### 7. TeX source 还有一段旧 draft 注释

`paper_opre.tex:2935-2941` 有注释：

> The Cross-Domain Validation and Neural Acceleration sections from earlier drafts...

这不进 PDF，不影响匿名审稿。但如果提交 source，建议删掉或改成中性注释。OR reviewer 通常不看 TeX source，但 production 可能会拿到。

### 8. Code/Data disclosure 目前匿名处理是合理的

`paper_opre.tex:2276-2286` 写明 supplementary archive、README、`lake build`、Python runners，并说明 public URL/Zenodo DOI 因 double-anonymous withheld until acceptance。这个处理符合 soft double-anonymous。

注意：如果 ScholarOne 要求 code/data zip 同时上传，cover letter 里也要说明 artifact 和匿名策略，否则编辑可能误以为代码不可用。

## OR 针对性评价

### Problem fit

适合 OR。问题不是纯 ML routing demo，而是 stochastic network routing under disruption；文章把 hyperpath recomputation、ranking fragility、robust scoring、formal proof和真实 GTFS-RT evaluation 放在同一个框架里，有 OR 读者能识别的 optimization/network/stochastic-modeling 价值。

### Contribution clarity

现在的 contribution 比早期清楚很多：

- 保留 hyperpath 结构，不重算整个 hyperpath；
- 在线重排 candidate connections；
- per-candidate score 有 Wasserstein-DRO interpretation；
- Swiss 35-day panel 给出主实证；
- A7 audit 把工程收益和 formal DRO core 的边界讲清楚；
- Lean proof 作为 credibility layer，而不是夸大成全系统验证。

### Empirical credibility

主结果现在可信度够送审。35-day cross-day、paired CI、day-level wins、LOO robustness、baseline+A7 audit 都能支撑“这不是一个 seed 或一个 OD 的偶然结果”。

最大可辩点仍然是：部署收益的大头来自 hyperpath-structural penalties，而不是 Lean 证明过的 DRO core。你现在已经正面承认这一点，所以 reviewer 很难再抓成 fatal flaw。需要继续避免 abstract/conclusion 把 formal DRO core 写成全部 deployed gain 的直接来源。

### Formal verification positioning

Lean 的位置现在比较健康：证明 per-candidate identity、bounds、scope lemmas，而不是声称证明了整套 simulator/GTFS pipeline。当前 `lake build` 成功；论文也承认 simulator/data-loading scripts are not verified。这是正确口径。

### Readability

Abstract 和 introduction 现在基本能读。剩下的人话问题集中在工程代号：`R16`、`R15`、`A7`、`A1--A10`。这些在 EC 里没问题，但主文应尽量用 descriptive names，不要让 OR reviewer 像读 internal experiment log。

## 推荐处理顺序

1. 先处理 LaTeX log：Algorithm 1 和 Paper-to-Lean table 两个 float-too-large。
2. 确认 OR submission package：EC 是否分离、references/EC/tables 的最终顺序。
3. 把主文里的 `R16/R15/A7/A1--A10` 改成 descriptive-first label-second。
4. 在 A7 attribution 段补 `in this audit` 这类限定语。
5. 可选清 Lean linter warnings 和 TeX source 注释。

## 最终建议

**Submission-ready after production cleanup.**

我不建议再大改理论或重跑核心实验。现在最值钱的工作是把稿子包装成 OR reviewer 一眼能读懂、PDF/log 干净、supplementary artifact 可验证的形态。内容上已经够进入正式投稿流程。
