# OPRE-Targeted Review for `paper_opre.tex`

日期：2026-04-29  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、OPRE template 与 general submission guideline  
PDF 状态：`paper_opre.pdf` 61 页；main paper 到 `Code and Data Disclosure / Electronic Companion` 约第 40 页，References 从约第 55 页开始。  
结论：**内容已经接近 OPRE lengthy manuscript 形态，但还不建议直接投。先修 P0：abstract/intro 合规、未定义引用、内部修稿话术、formal-claim 口径和表格/EC 版式。**

## 总体判断

这版重构方向是对的：用了 `\documentclass[opre,dblanonrev]{informs4}`，匿名作者、空 manuscript number、1.5 spacing、11/12pt house style、Code and Data Disclosure、Electronic Companion 都有了。主文被压到 40 页左右，这已经符合 OPRE “Lengthy Manuscript 一般不超过 40 页（不含 references）” 的上限。

但 OPRE 的 guideline 不只是页数。现在最大风险是：**这篇还保留了内部修稿/技术报告的写法，而不是一篇给 broad OR audience 的首轮投稿稿。** 例如 `P0 #2`、`round-3`、`Round-6 reviewer ask`、`GPT-5.5 algorithm review`、`CODE_OPTIMIZATION_REVIEW_GPT.md` 这类话术必须删掉。审稿人不应该看到你的内部迭代历史。

另一个核心问题是“说人话”。`EXP3`、`PS-SSP`、`BAMCP`、`BA-SSP-MDP`、`GTFS-RT`、`CSA`、`PMF`、`OOD`、`R16/A7/V1/V2/V3` 同时堆在 introduction 和 contribution list 里，OR 读者会累。应先说“这是 adversarial-bandit baseline / posterior-sampling baseline / rollout-planning baseline”，再把缩写放括号里。

## OPRE Guideline 对照

| 要求 | 当前状态 | 评价 |
|---|---:|---|
| Double anonymous | `opre,dblanonrev`，作者匿名 | 基本合规 |
| Regular ≤30 页，Lengthy 通常 ≤40 页 | 主文约 40 页，全文含 EC/refs 61 页 | 只能按 **Lengthy Manuscript** 投，建议再留 1-2 页余量 |
| Abstract ≤200 words | 约 229 words | **不合规** |
| Abstract 尽量 text-only，少数学符号 | 有 `\texttt`、百分比、箭头、多个缩写 | 需压缩并降低技术密度 |
| Introduction 面向 broad OR audience，不应含公式/数学符号 | contribution list 有 `sup`、`epsilon`、CI、p-value 等 | **不合规风险** |
| Tables after Reference List, not embedded | 当前 tables 大量嵌入正文 | 与 guideline 明文冲突，至少要确认并准备 end-of-paper table 版本 |
| Figures near relevant text | 当前 figure placement 基本符合 | 可保留 |
| Code/Data disclosure after main body | 有，且匿名处理 URL/DOI | 好 |
| Footnotes not used | 无 `\footnote`，endnotes 空 | 好 |
| Subject classification / area | 有 `\SUBJECTCLASS` 和 `\AREAOFREVIEW` | 可用，但 subject phrase 可再精炼 |
| ScholarOne keywords up to 3 | 当前 manuscript keywords 5 个 | 建议压到 3 个核心词 |

## P0 必修问题

### 1. Abstract 超 200 词，而且术语太密

OPRE guideline 明确 abstract 不超过 200 words，且应 text-only、可被 metadata system 直接复用。当前 abstract 约 229 words，并且把 Lean、DRO、LCB、GTFS-RT、V2-LCB、Wasserstein、Hedge regret 等全塞进去。

建议 abstract 只保留四件事：

1. 问题：实时公交扰动下，静态 hyperpath 的备选路线还在，但排序过时。
2. 方法：用 Bayesian posterior 做一个 pessimistic arrival-time score。
3. 理论：这个 per-candidate score 等价于 Wasserstein DRO worst-case arrival；Lean 验证数学核心。
4. 实证：Swiss 35-day 数据 5--6% 改善。

一版更 OPRE 风格的 abstract 可以是：

```text
Transit routing systems often precompute stochastic hyperpaths: ranked fallback options at each stop. Under real-time disruptions, the fallback options may remain useful, but their ranking can become stale. We study this ranking problem as a Bayesian adaptive stochastic shortest path model and propose a pessimistic online scoring rule that updates each route from real-time delay and cancellation observations. At the candidate-connection level, the score equals the worst-case expected arrival time over a Wasserstein ambiguity set whose radius is supplied by posterior uncertainty; the mathematical core is machine-checked in Lean 4. On a 35-day Swiss SBB real-time-feed benchmark with 28,350 journeys per method, the deployed LCB family with layered hyperpath-risk terms reduces cell-mean expected total travel time by 5--6% and improves reach rate by 2.8 percentage points. The results suggest that stochastic hyperpaths are structurally robust but ranking-fragile: disruption handling should often re-rank existing alternatives rather than recompute the route structure from scratch.
```

这版约 150-170 词，重点也更像 OPRE 摘要。

### 2. Introduction 违反 “no equations or mathematical notation” 的精神

OPRE guideline 写得很直接：introduction should not contain equations or mathematical notation。当前 introduction 没有 display equation，但 contribution list 里有：

- `LCB(c) = sup...`
- `epsilon = beta sigma + gamma p_cancel`
- `95% CI [-2.95, -2.26]`
- `p < 10^-10`
- `O(1/T)`, `log K/eta + eta T`

建议改法：

- Introduction 只用 plain English 写贡献。
- 把精确公式全部移到 Section 4/5。
- Swiss 结果可以保留一两个关键数字，但不要放 CI/p-value 细节。
- `EXP3` 不要首次作为裸缩写出现，应写成 “an adversarial-bandit parameter-tuning wrapper (EXP3-IX)”。

现在 intro 的开头故事很好，建议保留；后面的 seven contribution list 应压成 3 个 primary contributions + 1 个 “additional analyses in EC” 段落。

### 3. 删除所有内部修稿话术

当前有多处会让审稿人觉得这是 revision log，不是投稿稿：

- `paper_opre.tex:1642-1643`：`round-3 typed-cancellation fix` 和 `P0 #2`
- `paper_opre.tex:1939-1940`：`round-6 reviewer ask`
- `paper_opre.tex:2013`：`GPT-5.5 algorithm review`
- `paper_opre.tex:3308-3310`：`CODE_OPTIMIZATION_REVIEW_GPT.md`
- 多处 `earlier drafts`, `reviewer concern`, `reviewer comments`, `flagged`

建议全部改成正常研究叙述：

- “A leave-one-day-out audit checks...” 而不是 “round-6 reviewer ask”
- “A component ablation isolates...” 而不是 “GPT-5.5 algorithm review”
- “The typed-cancellation correction...” 而不是 “round-3 fix”
- 不要在论文正文提 `P0`、`GPT`、内部 review 文件名。

### 4. 编译还有未定义引用

我重新跑了：

```bash
latexmk -pdf -interaction=nonstopmode paper_opre.tex
```

编译能出 PDF，但仍有 5 个 unresolved references：

- `rem:r15-score-scope`，出现 2 次。现在对应 label 似乎是 `rem:lcb_dro_scope`。
- `fig:zurich_network`，正文引用了但图/label 不存在。
- `sec:dro-equiv` 和 `sec:regret`，EC detailed proofs 里引用旧 section label。

这些必须投前清零。OPRE ScholarOne 生成 proof 时如果还有 `??`，第一印象会很差。

### 5. Formal verification 口径还有一处自相矛盾

OPRE 版已经修好了 Theorem 6 和 Table 10：mixture 是 `gamma`-contraction，Wasserstein 是 `gamma(1+epsilon)` Lipschitz。这很好。

但 `paper_opre.tex:1349-1351` 仍写：

```text
the robust Bellman gamma-contraction (both for the discrete-mixture instance ... and for the Wasserstein ball)
```

这又回到了 Round 8 的问题。应改成：

```text
gamma-contraction for the discrete-mixture ambiguity and a gamma(1+epsilon) Lipschitz bound for the Wasserstein-ball ambiguity
```

这个必须修，因为同一份稿子里 Table 10 已经承认 Wasserstein 不是一般 `gamma`-contraction。

### 6. `EXP3`/`PS-SSP`/`BAMCP` 需要“人话版”指代

你说 `EX3` 这种表达会一头雾水，这个判断是对的。OPRE 读者不一定熟悉 bandit/RL 缩写。建议全稿首次出现和表格方法名改成下面这种模式：

- `EXP3-IX adversarial bandit` → “adversarial-bandit baseline (EXP3-IX)”
- `Adaptive-beta` → “online beta tuning wrapper (EXP3-IX over beta values)”
- `PS-SSP` → “posterior-sampling transit planner (PS-SSP)”
- `BAMCP-60` → “Bayes-adaptive Monte Carlo planner with 60 rollouts”
- `SW-LCB` → “sliding-window pessimistic score”
- `LCB` → “pessimistic arrival-time score (LCB)”

尤其是 introduction 和 abstract，先说作用，再给缩写。不要让缩写先行。

## P1 问题

### 7. Main paper 正好卡在 40 页，建议留余量

`paper_opre.pdf` 全文 61 页；Code/Data + EC 从第 40 页开始。OPRE lengthy manuscript 通常不超过 40 页（excluding references），所以你现在是踩线过关。

建议再砍 1--2 页作为缓冲。优先砍：

- Introduction 的 secondary contributions
- Related Work 中过细的 formal verification line-count 描述
- Main paper 里的 A1/A7/R16 实现细节，移到 EC
- Cross-domain 在 main conclusion 只留一句，细节放 EC

如果目标是 Regular Manuscript，则必须砍到 30 页；现在只能按 Lengthy Manuscript 投。

### 8. EC/References 顺序和表格位置要按 OPRE 口径确认

General guideline 写的是：

- Appendices
- Acknowledgments
- References
- Electronic companions

且 tables “should be placed together after the Reference List, not embedded in the manuscript text.”

当前 `paper_opre.tex` 是：

- Main sections
- Code and Data Disclosure
- `\begin{APPENDICES}` + `\ECHead{Electronic Companion}`
- Acknowledgment
- References

也就是说 EC 在 References 前，且 tables 全部嵌在正文/EC 中。模板样例本身对 table/appendix placement 没完全演示清楚，所以这里不一定会被 desk reject，但 guideline 明文在那。建议准备两个版本：

1. 审稿友好版：保留 figures near text，但把 tables 移到 references 后或 EC；正文中引用 table。
2. 工作版：当前 inline tables 便于阅读。

至少在最终提交前确认 OPRE ScholarOne/LaTeX instructions 是否允许 inline tables。不要默认没事。

### 9. Electronic Companion 有严重 float 溢出

log 里还有：

- `Float too large for page by 285.45946pt` at line 2511，基本是 Table 10 paper-to-Lean cross-reference。
- 多个 `Overfull hbox` 20--65pt，集中在 `paper_opre.tex:2450-2505` 的 Lean file/path 表。
- Algorithm 1 也有 `Float too large for page by 27.4pt` at line 664。

建议：

- Paper-to-Lean cross-reference 表拆成两张表，或放 landscape/缩短 theorem names。
- 文件路径列只写 basename，不写 `BAPR-HRO/...`。
- Algorithm 1 从 main paper 移到 EC，主文只放 8--10 行 pseudo-code summary。

### 10. Keywords 太多，subject classification 可更贴 OPRE

ScholarOne guideline 写 “up to 3 keywords”。当前 5 个：

```text
Stochastic transit routing; distributionally robust optimization;
Wasserstein distance; Bayesian online learning; formal verification
```

建议压成 3 个：

```text
Stochastic transit routing; distributionally robust optimization; formal verification
```

`Bayesian online learning` 可放 abstract/intro，不一定进 keywords。

### 11. Reference list 顺序

References 要 alphabetically by author name。当前 `Anonymous(2025)` 放在最后，不是 alphabetical。双盲自引可以匿名，但仍建议按 `Anonymous` 排到 A 段，或者用 OPRE 推荐的 anonymous self-citation 方式统一处理。

### 12. 表格列名太多缩写

OPRE guideline 说 table column headings should be brief and not use abbreviations。当前表格里有：

- `Norm`, `Disr`
- `P95`
- `cell-cond`
- `vs. Static`
- `LOO`
- `V1/V2/V3`

技术读者能懂，但 OPRE 风格更偏清楚。建议至少主文关键表改成：

- `Normal`, `Disrupted`
- `95th pct.`
- `Mean time among completed trips`
- 表注里解释 `V1/V2/V3`，或者用 “LCB-fixed”, “LCB-ensemble”, “LCB-topology”。

## 内容评价

### 强项

1. **OPRE fit 变强了。** 现在论文不是单纯算法 paper，而是“用 analytical methods improve real-time transit decision-making”，符合 OPRE editorial statement。

2. **核心故事清楚。** “Hyperpaths are structurally robust but ranking-fragile” 是一句好话，值得作为 introduction 和 conclusion 的主线。

3. **Round 8 的 proof-claim 问题大部分已修。** Theorem 6、Table 10、Proposition 1 的 V2 witness/prose distinction 都比 `paper.tex` 更诚实。

4. **Swiss benchmark 的 caveat 写得好。** `Paradeplatz-origin disrupted-route OD panel, not city-wide random OD sample` 这句话很重要，OR 审稿人会认可。

5. **Code/Data disclosure 合规意识到位。** 匿名审稿阶段 withheld URL/DOI，同时说明 supplement archive 和 README，会比裸 GitHub 链接更合适。

### 仍弱的地方

1. **论文还像“修稿记录压缩版”。** 要把内部工程史改成研究设计叙述。

2. **A7 是 dominant fix，但不是 DRO theorem 的一部分。** 现在已经承认了，但 main result 仍容易被读成 “DRO/LCB 理论解释了全部 5--6% Swiss gain”。建议 abstract/conclusion 都明确 “with deployed layered hyperpath-risk terms”。

3. **Contribution list 太贪。** Lean、Swiss、cross-domain、neural、Adaptive-beta、BAMCP、A1-A10 全放主线，OPRE editor 会觉得不聚焦。主线应是 transit routing + DRO identity + Swiss validation；cross-domain/neural/A1-A10 全做 supporting/EC。

4. **EXP3/Adaptive-beta 不是主贡献。** 当前实验里 Adaptive-beta 基本追踪 V1，不是新性能来源。把它写成 safety wrapper 即可，不要占太多 introduction real estate。

## 建议修改顺序

1. **P0：Abstract 压到 200 词以内，去掉重数学/重缩写表达。**
2. **P0：Introduction 改成人话，删公式和 CI/p-value，缩写首次出现必须有作用解释。**
3. **P0：删除所有 `P0/round/reviewer/GPT/CODE_OPTIMIZATION` 内部修稿痕迹。**
4. **P0：修 5 个 undefined references。**
5. **P0：修 `paper_opre.tex:1349-1351` 的 Wasserstein Bellman overclaim。**
6. **P1：确认 OPRE tables-after-references 规则；至少准备 tables-end 版本。**
7. **P1：把 main paper 再压 1--2 页，避免卡 40 页边界。**
8. **P1：拆 Table 10 / 缩 Algorithm 1，清掉 float-too-large。**
9. **P1：keywords 压到 3 个，references alphabetize。**
10. **P2：主文表头减少缩写，方法名换成 OR 读者能懂的名字。**

## 最终建议

**当前版本：Revise before submission to OPRE.**  

不是说内容不够，而是投稿形态还差最后一层。理论 claim 比 `paper.tex` 更稳，主文 40 页也已经很接近 OPRE lengthy manuscript；但 abstract 超词、intro 数学符号、未定义引用、内部修稿话术和 EC/table 版式都是会被 editor/referee 直接注意到的问题。

修完 P0 后，这篇可以按 **Lengthy Manuscript** 投 OPRE。若想提高 desk-review 通过率，建议再做一次“人话化”：让 abstract/introduction/conclusion 先回答 OR 读者的三个问题：问题是什么、为什么 OR、结果有多大；把 LCB/DRO/EXP3/Lean 的技术细节放到后面。
