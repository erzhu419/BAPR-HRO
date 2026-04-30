# OR/OPRE Review Round 2 for `paper_opre.tex`

日期：2026-04-30  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、R16 ablation/recompute scripts and JSON results  
目标期刊：Operations Research (`opre`)

## 总体判断

这版比上一轮明显更接近可投状态。几个最危险的问题已经修了：abstract 174 词，低于 200 词；没有 unresolved references；Beta prior 不再前后冲突；hyperpath recomputation failure 加了数字；A7 的归因说得更诚实；V2/Lean 和 lower-bound/sandwich 的口径也降下来了。

我的当前建议是：**还差一轮小到中等 revision，再投 OR lengthy manuscript。** 现在不是“方向不对”，而是有几处会让审稿人觉得作者刚从 rebuttal/内部 review 改稿中复制了内容，或者 ablation 证据还不够干净。修掉后，首轮送审的把握会高很多。

PDF 状态：61 页；Code/Data Disclosure 和 Electronic Companion 从 PDF 第 40 页开始，References 从第 55 页开始。也就是说仍然是卡线的 Lengthy Manuscript。

Lean 验证状态：我用正确 target 重跑了相关 Lean 库：

```bash
lake build BAPRHRO BAPRHRO_V2 LowerBound Irrecoverability IrrecoverabilityBridge EXP3Regret AdaptiveConvergence Architecture WassersteinCoupling WassersteinDistance WassersteinDRO WassersteinDROBellman
```

结果：build 成功，只有 linter warnings，没有 build error。

## 已修好的关键点

1. **Abstract 合规**：约 174 words，符合 OR abstract 不超过 200 words 的要求。
2. **引用状态合格**：当前 log 没有 undefined references / citations。
3. **Beta prior 统一了**：Section 3 不再写 Beta(1,5)，实验值统一到 Appendix parameter table 的 Beta(1,99)。
4. **Hyperpath recomputation failure 有量化证据了**：新增 Static / BOCD+recompute / keep+re-rank 表，方向清楚。
5. **A7 归因更诚实**：introduction 和 conclusion 已经明确 deployed gain 不是 four-term DRO core alone。
6. **Baseline fairness 问题被正式承认**：现在写成 deployable routing stacks comparison，而不是 same-information learning paradigms comparison。
7. **V2 Lean 口径好多了**：现在说两个 Lean building blocks 分别验证，Proposition prose-level 组合，不再声称单独 Lean theorem。
8. **Lower bound 降级正确**：Theorem 改成 adversarial existential minimax，比上一轮严谨。
9. **R16 ablation 重跑了同一 simulator**：不再混 R15/R16 simulator，这是重要进步。

## P0 / 投前必须修

### 1. 正文还有明显的“审稿回复口吻”

`paper_opre.tex:3152-3154`：

> The reviewer correctly noted that prediction MAE / Pearson r are predictive accuracy metrics...

这个不能出现在首轮投稿稿里。审稿人会立刻意识到这份稿子是按某个 review 文件改出来的。建议改成正常论文口吻：

> Prediction MAE and Pearson correlation measure surrogate accuracy, not decision equivalence.

同类问题也在代码补充材料里：

- `run_component_ablation_R16.py:5-6` 写了 reviewer concern / P1 #11。
- `run_recompute_baseline.py:5-6` 写了 reviewer question / P1 #6。

如果这些脚本作为 supplementary archive 提交，建议全部删掉内部 review 编号，改成普通说明，比如 “This script evaluates whether...”

### 2. 新 ablation 表还有一处正文-表格矛盾

`paper_opre.tex:1938-1943` 写：

> disabling the two hyperpath-structural penalties (rows A1 and A2) flips the sign of the V1, V2, V3, and Adaptive-beta improvements...

但 Table `tab:swiss_ablation` 里：

- V1: A1 +3.5% 到 A2 -3.0%，确实翻符号。
- V2: A1 +6.0% 到 A2 -7.1%，确实翻符号。
- V3-Topo: 全部行都是 -1.8%，没有翻符号。
- Adaptive-beta: 全部行都是 -6.2%，没有翻符号。
- DRO: 全部行都是 +11.6%，也没有 toggled。

所以这句话必须改。建议写成：

> Comparing A1 and A2 shows that adding the A7 layered penalties flips V1 and V2 from worse-than-static to better-than-static. V3-Topo, fixed-beta DRO, and Adaptive-beta are listed only as R16 reference columns because they do not consume the row toggles in this ablation.

### 3. Ablation 表的列设计容易误导

Table `tab:swiss_ablation` 现在把 V3-Topo、DRO、Adaptive-beta 放在 A0/A1/A2/A3 四行里，但 caption 又说这些列不 consume toggled components，保留 R16 defaults。这个对审稿人非常不友好：行名写 “A0 pre-fix priors + no A4/A7”，但 Adaptive-beta 那列还是 -6.2%，读者会误以为 “pre-fix Adaptive 也很好”。

建议二选一：

1. **最稳**：主表只放真正被 toggled 的 V1-LCB 和 V2-LCB。V3/DRO/Adaptive 放到 EC reference table。
2. **保留宽表**：对不参与 ablation 的列用 em dash 或灰色 “R16 ref.”，不要在每一行重复数值。

另外 A2 同时打开 A7 layered penalties 和 V2 cold-start fix。对 V1 来说基本是 A7；对 V2 来说 A7 和 cold-start fix 被合并了。若要强说 “A7 是 V2 dominant fix”，最好再加一个 `A2a: cold-start only` 或 `A2b: A7 only`。否则就写成 “A7 plus V2 cold-start correction is the dominant deployed fix for V2”。

### 4. 新增 R16 ablation / recompute 结果统计强度低于主 Swiss 表

主 Swiss 表是 35 days × 18 ODs × 45 trials = 28,350 journeys per method。  
新增 ablation 和 recompute baseline 是 35 days × 18 ODs × 15 trials = 9,450 journeys per method，而且 JSON 里 `seeds: [0]`。

这不是不能用，但要避免让读者误以为这和主结果同一 Monte Carlo 精度。现在 recompute 段落说 “same 35-day Swiss cross-day panel used elsewhere”，后面确实补了 15 trials/cell；还算诚实，但建议再加一句：

> This audit uses the same days and OD panel as the main Swiss benchmark, but a smaller 15-trial-per-cell Monte Carlo budget.

更好是直接用 45 trials/cell 重跑 ablation/recompute，这样 main table、ablation table、recompute table都可直接对齐。

### 5. 脚本默认参数和论文表格参数不一致

两个新增脚本默认都是 `seeds=(0,1,2)`，但当前 JSON 和论文表格用的是 `seeds: [0]`、`n_per_cell=15`。如果 reviewer 按默认命令重跑，会得到 45 trials/cell，不一定复现论文数字。

建议：

- README 中明确表格复现命令必须带 `--seeds 0`。
- 或者把脚本默认改成论文表格默认，并另设 `--full-seeds 0,1,2`。
- 论文 Code/Data Disclosure 里可以说 “Table X was generated with ...”。

## P1 / 强烈建议修

### 6. “regret sandwich” 这个词还没完全收干净

虽然 Corollary 已经改成 “not a tight sandwich”，但 Related Work 仍写：

- `paper_opre.tex:287`: “We answer it with a regret sandwich”
- `paper_opre.tex:366`: “the minimax lower bound and regret sandwich”

这会和 `paper_opre.tex:1276-1278` 的 “not a tight sandwich” 打架。建议统一改成：

> a minimax lower-bound construction and a separate LCB per-journey upper bound

另外 `paper_opre.tex:1290-1297` 说 algorithm uses un-reset posterior，但又说 “per-regime reset cost”。这个还是有点矛盾。建议把 “reset cost” 改成 “adaptation cost”。

### 7. Baseline+A7 仍是最大实证残余风险

现在你已经把这个写成 limitation，这很正确。但如果 OR 审稿人抓住它，还是可能要求一个 controlled experiment。

最低成本补强方案：

- `Static + A7`: 不学习，只用 StopLabel feasibility 和 on-time PMF 重排。
- `SW-LCB + A7`: 直接给 sliding-window pessimistic score 加同样两个 penalties。

这两个就能回答最尖锐的问题：Swiss 5-6% 到底是 “A7 label score 就够了”，还是 “A7 + posterior pessimism 才够”。如果这两个没时间跑，现在的 limitation 可以保住诚实性，但无法完全消除审稿风险。

### 8. Methods 仍然有缩写先行的问题

你已经把 abstract 和 introduction 大幅改善了，但 Methods 里仍然是：

- `PS-SSP`
- `BAMCP-60`
- `EXP3`
- `SW-LCB`
- `V3-Topo`
- `Adaptive-beta`

对 OR broad audience，建议 method list 改成表格，第一列是人话：

| Plain-language role | Implementation label |
|---|---|
| Static hyperpath ranking | Static |
| Pessimistic posterior score | LCB-V1 |
| Ensemble pessimistic score | LCB-V2 |
| Posterior-sampling planner | PS-SSP |
| Monte Carlo belief planner | BAMCP-60 |
| Adversarial-bandit route learner | EXP3-IX |
| Sliding-window pessimistic score | SW-LCB |

不要让缩写当读者入口。

### 9. Introduction 仍含 OR guideline 不喜欢的数学符号

OR guideline 写 introduction should not contain equations or mathematical notation。当前 introduction 没有大公式，但仍有 `stop $A$`、`destination $B$`、`online beta-tuning wrapper` 里的 `\beta` 等符号。这个不一定 desk reject，但能改就改：

- stop A / B 直接写 “a downtown stop” / “the destination”。
- introduction 里的 `\beta` 改成 “pessimism parameter”。
- 公式和希腊字母全部放到 Method。

### 10. Recompute baseline 表最好改成正式编号表

`paper_opre.tex:1462-1473` 用 `center + tabular` 放了一个无编号表。OR guideline 要求 tables should be numbered, have a title, and be referred sequentially。建议改成正式 `table` 环境，加 caption 和 label：

> Table X. Hyperpath recomputation audit on the 35-day Zurich panel.

这是小改，但对 OR 格式合规很有用。

### 11. Title 还在过度强调 DRO

当前标题是：

> Distributionally Robust Stochastic Transit Routing via Wasserstein Online Learning: Theory, Formal Verification, and Experiments

但论文现在最稳的贡献其实是 “re-ranking stochastic hyperpaths under disruptions”，DRO 是 core score 的解释，不是 deployed gain 的全部来源。这个标题可能再次把审稿人引向 “为什么 empirical gain 不是 DRO core alone?” 的质疑。

建议考虑更贴近真实贡献的标题，例如：

> Re-Ranking Stochastic Transit Hyperpaths Under Disruption: A Wasserstein-Robust Scoring Approach

或者：

> Ranking-Fragile Hyperpaths: Wasserstein-Robust Online Scoring for Disrupted Transit Routing

## P2 / 版面和提交细节

1. PDF 仍是 61 页，main paper 卡在 40 页。Lengthy Manuscript 可以，但没有余量。
2. log 仍有 Algorithm 1 float too large 27.4pt，EC table float too large 285.5pt，多处 EC overfull hbox。投前建议处理。
3. Tables 仍然 inline；OR guideline 写 tables should be placed together after Reference List。模板和实际投稿容忍度可能不同，但最终提交前要确认。
4. Current order 是 EC before References；guideline 推荐 References before Electronic Companion。这个也建议确认，不一定是内容问题。
5. Cross-domain EC 现在主 conclusion 只写 exploratory，已经比上一版好。保持这个低调口径。

## 建议的投前优先级

1. 立刻删掉正文和 supplementary scripts 里的 reviewer/P1/internal review 口吻。
2. 修正 Table `tab:swiss_ablation` 的正文矛盾，最好把表拆成 “true ablation columns” 和 “reference methods”。
3. 在 ablation/recompute 结果旁明确写 15 trials/cell 是小预算 audit；若时间允许，重跑 45 trials/cell。
4. 把 “regret sandwich” 全部改成 “lower-bound construction + LCB upper bound”，并把 “reset cost” 改成 “adaptation cost”。
5. 把 recompute baseline 的无编号 tabular 改成正式 table。
6. 对 Methods 的缩写做一次人话化。

## 最终意见

这版已经不是“先别投”的状态，而是“再清一轮就可以投”的状态。论文现在最强的卖点是：

> Hyperpaths are structurally robust but ranking-fragile; a deployable layered score can re-rank existing alternatives under disruption, with a Wasserstein-DRO interpretation for its pessimistic core and Lean-checked mathematical support.

只要别再让标题/叙事把读者带回“DRO core alone explains 5-6%”这个坑，OR 的定位是能讲通的。
