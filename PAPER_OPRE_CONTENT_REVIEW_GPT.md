# OPRE Content Review for `paper_opre.tex`

日期：2026-04-29  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、`OR templet/General_submission_guildline.md`、`OR templet/INFORMS-OPRE-Template.tex`  
定位：这一轮主要审论文内容和 OPRE 适配，不只是版面合规。

## 总体判断

这版比上一版强很多。Abstract 现在约 174 词，低于 OPRE 200 词要求；关键词压到 3 个；编译日志里没有 unresolved references；主文叙事也明显更像一篇投稿稿，而不是内部技术报告。

但我仍然不建议直接投 OPRE。当前最主要风险不是格式，而是审稿人会追问：**论文的理论贡献是 LCB/DRO，但 Swiss 实证主收益主要来自 A7 layered hyperpath-risk penalties，这两个东西到底是什么关系？** 论文已经开始承认 A7 不属于 Wasserstein-DRO 解释，但 abstract、introduction、discussion、conclusion 的主叙事还容易让人读成“DRO/LCB 理论直接解释 5-6% 的部署收益”。这个口径必须再压实。

我会给一个偏审稿人的结论：**Major revision before submission, promising but not yet clean enough for OPRE.** 亮点是真亮，尤其是“keep hyperpath, re-rank alternatives”的问题抽象、Swiss 35-day stress test、Lean-checked theory core。但 OPRE 审稿人会对 empirical identification、baseline parity、理论声明边界和可读性非常敏感。

## 已经明显改善的地方

1. OPRE 基本形式更稳了。`opre,dblanonrev`、匿名作者、subject classification、area of review、Code/Data disclosure、Electronic Companion 都在。
2. Abstract 已经合规，约 174 词，并且不再像以前那样堆满内部缩写。
3. 文章开头的人话叙事更好。8:05 AM passenger example 能让 OR 读者明白问题。
4. 论文已经明确写出 DRO equivalence 是 per-candidate scalar score，不是 global-policy DRO optimality claim。这一点很重要。
5. Swiss benchmark caveat 写得比较诚实：这是 Paradeplatz-origin disrupted-route OD panel，不是 city-wide random OD sample。
6. LOO audit 是加分项。历史 prior 是否偷看 evaluation day 这个问题，现在有了清楚说明和结果。
7. Lean 口径比之前严谨，尤其是 robust Bellman 部分已经承认 Wasserstein-ball 只有 `gamma(1+epsilon)` Lipschitz，不再硬说都是 contraction。

## P0 问题

### 1. 主贡献叙事和实证归因仍然不够一致

当前 abstract 说 deployed pessimistic score augmented by two hyperpath-structural risk terms 带来 5-6% 改善，这是比上一版诚实的。但全文读下来，审稿人还是会发现一个硬事实：

- `paper_opre.tex:485-489` 明确说两个 A7 penalties 不在 DRO ball 内，没有 DRO interpretation。
- `paper_opre.tex:1564-1576` 明确说 A1 对称信息比较里 LCB family 反而比 Static 差 3.7-4.9%，A2/A3 的 sign change 来自 A7。
- `paper_opre.tex:1996-2055` 的 component ablation 进一步说明 A7 是 dominant fix。
- `paper_opre.tex:2153-2159` conclusion 把 5-6% 和 LCB family with R16 layered score 绑定，但理论核心仍主要是 four-term DRO-LCB core。

这不是小问题。OPRE 审稿人会问：到底是“DRO/LCB 理论使 routing 更好”，还是“一个 hyperpath label PMF/feasibility heuristic 使 routing 更好，LCB 是其中一部分解释”？现在最稳的说法应该是：

> The deployable contribution is an R16 layered hyperpath-risk score built around a formally justified LCB/DRO core. The Swiss gains are driven mainly by the two hyperpath-structural risk terms; the DRO core supplies the pessimistic uncertainty/cancellation layer and its theoretical interpretation.

建议修改：

1. Title/abstract/introduction/conclusion 里都避免让读者以为 5-6% 是纯 DRO-LCB 的结果。
2. 把 “LCB family beats baselines” 改成 “R16 layered hyperpath-risk score improves a disrupted-route stress test”。
3. Swiss main result后加一句更直白的话：`The ablation shows that the deployed gain is not caused by the four-term DRO core alone; it requires the hyperpath-structural penalties.`

### 2. Baseline fairness 现在是“披露了”，但还没“解决”

`paper_opre.tex:1540-1576` 已经写了 information-access protocol，这比隐藏问题强。但核心问题还在：A7 features 只给 LCB family 用，SW-LCB / EXP3 / PS-SSP / BAMCP 没有 retrofit。文章解释说 PS-SSP/BAMCP rollouts see PMF tail, EXP3 没有 natural place 加 per-candidate penalty。这个解释对审稿人不够硬。

原因很简单：如果 A7 是 Swiss gain 的 dominant fix，那么 baseline 没吃到 A7，就很难说这是 learning paradigm 的胜利。审稿人可能会要求一个最小对照：

- `Static + A7`: 不更新 posterior，只用 hyperpath label 的 feasibility 和 on-time PMF 重排。
- `SW-LCB + A7`: 原 SW-LCB score 加同样两个 A7 terms。
- `PS-SSP/BAMCP candidate score + A7`: rollout/posterior planner 最终选择 candidate 前加同样 structural penalty，哪怕只是 adapter。
- `EXP3 + A7`: 把 per-candidate structural penalty 转成 loss shaping，或明确说明这样会改变 EXP3 问题定义。

如果时间不够，至少把 limitation 写得更像正式承认，而不是“future work”轻轻带过。现在 `paper_opre.tex:1573-1576` 的说法还偏防御。建议改成：

> Because the A7 features are consumed only by the LCB-family implementations, the Swiss comparison should be interpreted as a comparison of deployable routing stacks, not a clean same-information comparison of learning paradigms.

### 3. Beta prior 前后不一致，必须修

主模型里写：

- `paper_opre.tex:399-402`: cancellation prior 是 Beta(1,5)，prior cancel rate 约 0.17。

参数表和 R16 说明里写：

- `paper_opre.tex:2255-2260`: R16 calibration 是 Beta(1,99)，prior cancel rate 约 0.01。
- `paper_opre.tex:2000-2006`: A1 fix 也说从 Beta(1,9) 改成 Beta(1,99)。

这个会被审稿人抓。取消率 prior 是实验结果的核心参数之一，不能一处 0.17、一处 0.01。建议：

1. Section 3 的 model 不要给默认 Beta(1,5)，只写 generic Beta(alpha_c, beta_c)。
2. 把具体实验默认值统一放在 parameter table。
3. 如果需要一个 pedagogical example，就明确写 “for illustration only, not used in R16 experiments”。

### 4. Ensemble V2 的 Lean/数学口径还有不稳的句子

`paper_opre.tex:571-594` 的 “What Lean verifies vs. what is prose” 整体方向是对的，但里面有两个危险点。

第一，`paper_opre.tex:580-583` 写 argmin identity 不需要 iSup equality，lower-bound witness alone 就支持。这句话数学上很容易被反驳。一个 witness 只能说明某个 feasible distribution 达到某个值；要说 DRO objective 的 argmin 一样，仍然需要上界把 supremum 压住，或者需要明确说“我们排序的是 witness value，不是 full DRO value”。现在这句话会让形式化读者警觉。

第二，`paper_opre.tex:591-593` 写 ensemble radius contracts at ensemble rate `O(1/sqrt(n))`，并称它是 Theorem exp3 的 Lean counterpart `ensemble_std_bound`。EXP3 theorem 和 ensemble std bound 不是一类东西，引用会显得混乱。建议拆开：

- `The EXP3 theorem is only for the adaptive-beta meta-bandit / Hedge regret.`
- `The ensemble_std_bound, if retained, should be cited as a V2 auxiliary lemma, not as Theorem exp3's counterpart.`

更稳的写法：

> The Lean files verify the shifted-empirical witness and the abstract upper bound separately. Proposition 2 combines these ingredients in prose; we do not claim a standalone Lean theorem named ensemble_lcb_equals_empirical_dro.

### 5. “Tight regret sandwich” 目前像过度声明

`paper_opre.tex:1234-1253` 的 lower bound 定理写成 “for any algorithm, in a single-shot BA-SSP-MDP with C changepoints”，但 proof 实际上是构造一个 adversarial two-action/two-regime instance。更严谨的 theorem statement 应该是 existential minimax:

> For any algorithm, there exists an instance/trajectory with C changepoints such that regret is at least C d_min.

现在写法容易被读成“所有这样的 BA-SSP-MDP 都有这个下界”，这不对。

`paper_opre.tex:1255-1275` 的 sandwich 也不稳。下界是按 C changepoints 累积，上界是 per-journey LCB excess cost `2D(1+beta)sigma_max`。除非把上界也明确累积到 C 个 regime/episodes，否则不能直接夹成 tight sandwich。并且 proof 写“after each regime change, posterior resets to prior”，但前文和实际方法强调的是 per-journey posterior/reset protocol，不是 BOCD-style hard reset。这会造成口径冲突。

建议：

1. 把 corollary 标题从 “Tight regret sandwich” 改成 “Lower-bound intuition and LCB upper bound”。
2. 删掉 “LCB matches lower bound rate Theta(C)” 这类强表述。
3. 明确区分 single journey、cross-journey meta-bandit、multi-regime adversarial construction。

### 6. Hyperpath recomputation fails 是核心论点，但证据太少

`paper_opre.tex:1452-1462` 只用一个短段落说 BOCD + recomputation no better than static, sometimes worse。这个论点是文章的主线之一：不是重算 hyperpath，而是保留结构、重排 ranking。现在只有 prose，没有 table、figure、appendix ref。

建议至少加一个小表或 EC 表：

| Strategy | Mean | Reach | Failure mode |
|---|---:|---:|---|
| Static hyperpath | ... | ... | stale ranking |
| BOCD + recompute | ... | ... | drops fallback route |
| Keep + re-rank R16 | ... | ... | best |

如果没有全量结果，也要放 focused OD 的 concrete numbers。否则审稿人会觉得“recomputation fails”是一个 anecdote。

另外 `paper_opre.tex:1485-1486` 有坏句子：

> The geographic layout of this sub-network is described in sub-network on an OpenStreetMap basemap.

这是明显删图后残留，必须修掉。

## P1 问题

### 7. Synthetic large-network 结果不能被讲成 LCB 明显赢

Table main 里 large/disrupted：

- Static: 73.8
- LCB-V2: 73.7
- Oracle: 73.0
- DRO: 95.4
- SW-LCB: 99.2

这说明 V2 没有显著改善 Static，只是没有像 fixed-beta DRO/SW-LCB 那样崩掉。`paper_opre.tex:1617-1634` 已经部分承认这一点，但 figure caption `paper_opre.tex:1640` 写 “LCB/DRO eliminates the long tail” 会和表格矛盾，因为 DRO 在 large/disrupted 明显失败。

建议把 synthetic large 的结论改成：

> On the large synthetic disruption, Static is already robust; the useful result is not a large gain over Static, but that dynamic-beta V2 avoids the over-correction suffered by fixed-beta pessimistic baselines.

### 8. Cross-domain EC 有价值，但 main conclusion 里别讲太重

`paper_opre.tex:2166-2169` 在 conclusion 里说 LCB transfers to unit commitment and SDN with large percentage gains。这个容易分散 OPRE 主线。你的主论文已经够复杂：transit hyperpath、DRO、Lean、Swiss GTFS-RT、A1-A10、EXP3/BAMCP。再把 UC/SDN/VRP 放到 conclusion，会让审稿人怀疑文章想证明太多。

建议 main text 只留一句：

> Additional EC stress tests suggest that the same pessimistic-ranking motif can be useful outside transit, but those results are exploratory.

不要在主结论里高调报 `-61% delay` 这种跨域数字，除非你愿意接受一轮跨域 baseline 审稿。

### 9. OPRE 读者仍会被缩写绊住

你说 “EX3 这种表达会一头雾水”，这个问题还在。实际文中是 `EXP3`，但对非 bandit/RL 读者也一样突兀。建议全稿首次出现和表格 method label 采用“功能名 + 缩写”的写法。

建议替代表：

| 当前写法 | 更像人话的写法 |
|---|---|
| EXP3 / EXP3-IX | adversarial-bandit baseline (EXP3-IX) |
| Adaptive-beta | online beta tuning wrapper (EXP3-IX over beta values) |
| PS-SSP | posterior-sampling transit planner (PS-SSP) |
| BAMCP-60 | Bayes-adaptive Monte Carlo planner with 60 rollouts |
| SW-LCB | sliding-window pessimistic score |
| LCB | pessimistic arrival-time score (LCB) |
| DRO | Wasserstein robust score / fixed-beta robust score |
| A7 | layered hyperpath-risk penalties, not just “A7” |
| R16 | deployed score version, not just “R16” |

尤其是 method table、abstract、intro、discussion 里，不要让缩写先于含义出现。Appendix 可以用 A7/R16，但 main paper 应该少用这些内部版本号。

### 10. Related work 里有几处数字/表述可能不一致

`paper_opre.tex:261-263` 写 synthetic large disrupted PS-SSP 91.1、BAMCP-60 96.6 vs Static 73.8，但 main table `paper_opre.tex:1663-1664` 是 PS-SSP 96.0、BAMCP-60 101.0。可能是不同 run 或旧数值。如果是旧数值，统一；如果是另一个 table，说明清楚。OPRE 审稿人对数字不一致很敏感。

### 11. Component ablation 混用 R15 和 R16 simulator，因果归因会被削弱

`paper_opre.tex:2037-2042` 已经承认 A0/A1/A2 是 R15-era historical code states，A3 是 R16 simulator。这个 disclosure 是好的，但从审稿角度，因果 ablation 最好同一 simulator 语义下重跑。否则 “A2 是 dominant fix” 会被质疑成历史版本比较。

如果重跑成本可接受，建议重跑 A0/A1/A2 under R16 typed-cancellation semantics。若不能重跑，表格标题别叫 component ablation，改成：

> Development-history audit of component fixes

这样就不会装成完全因果分解。

### 12. OPRE guideline 仍有两个提交前检查项

根据 `General_submission_guildline.md`：

- Lengthy manuscript generally no more than 40 pages excluding references。当前 PDF 61 页，主文大约卡在 40 页附近，能投 lengthy，但几乎没有余量。
- Tables guideline 写 tables should be placed together after the Reference List, not embedded in manuscript text。当前大量 table inline。模板本身没有完全强制这个样式，但最终提交前应确认 ScholarOne/OPRE 是否接受 inline tables。

编译层面还有：

- Algorithm 1 float too large by 27.4pt。
- EC proof/code mapping table float too large by 285.5pt。
- EC cross-reference table 有多处 overfull hbox。

这些不是内容硬伤，但会影响第一印象。

## P2 文字和结构建议

1. Conclusion 第一段太技术密集。`paper_opre.tex:2141-2152` 一口气塞 BA-SSP-MDP、Wasserstein-1、posterior/empirical ensemble、attainment witness、Bellman、Chebyshev、Hedge、Lean lines。建议第一句先说人话：

   > Real-time transit disruptions do not usually require recomputing the whole hyperpath. They require re-ranking the alternatives the hyperpath already contains.

   然后再讲 theory 和 Lean。

2. `paper_opre.tex:1802-1805` 还有 “Earlier drafts inadvertently reported ...” 这种内部修稿口吻。建议改成正式口吻：

   > We report the cell-mean expected total time throughout; this avoids mixing cell-level and trial-level summaries.

3. `paper_opre.tex:655-701` R16 subsection很关键，但 `R15/R16/A7/A9/A10` 这些版本号会让读者感觉在看 release note。建议 main text 用 descriptive name，版本号放括号里。

4. `paper_opre.tex:1510-1536` methods list 建议改为 table：left column “Plain-language role”，right column “Implementation label”。OPRE 读者扫表比读缩写 bullet 更顺。

5. Formal verification 部分最好再强调“Lean verifies math lemmas, not implementation correctness”。你已经写了，但可以在 abstract/conclusion 降低一点 Lean 的中心性，避免审稿人以为 Lean 是 deployment guarantee。

## 建议的首轮投稿口径

我建议把整篇文章的主 claim 改成这三个层次：

1. **Decision insight:** In disrupted transit, hyperpaths are structurally robust but ranking-fragile; keep the contingency set and update rankings online.
2. **Method:** Use a deployable layered hyperpath-risk score. Its core is a pessimistic arrival-time score with a Wasserstein-DRO interpretation; its deployed Swiss version adds two bounded hyperpath-structural penalties.
3. **Evidence:** On a 35-day Zurich disrupted-route stress test, this deployed score improves expected total time by 5-6% and reach by 2.8pp; ablation shows the structural penalties are essential.

这样比“LCB/DRO beats everything”更稳，也更像 OPRE 会接受的应用型 OR 论文。

## 最终建议

这版已经有投稿潜力，但我会先做一轮 targeted revision 再投：

1. 统一 cancellation prior，修 Beta(1,5) vs Beta(1,99)。
2. 给 hyperpath recomputation failure 加数字证据。
3. 重写 A7/LCB/DRO 的 claim boundary，让 5-6% 归因不混。
4. 处理 baseline + A7 公平性，至少加 limitation，最好加一两个 adapter baseline。
5. 降级 regret sandwich 和 Ensemble V2 argmin/Lean 过强表述。
6. 把 main text 的 EXP3/PS-SSP/BAMCP/A7/R16 改成更“人话”的指代。

完成这些后，OPRE lengthy manuscript 的内容形态会更稳。现在如果直接投，最可能被打回来的点不是页数，而是“empirical gains are driven by engineering penalties outside the claimed DRO theory”和“baseline comparison is not same-information once A7 dominates”。
