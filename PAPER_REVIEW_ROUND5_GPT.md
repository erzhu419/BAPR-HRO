# BAPR-HRO Paper Review Round 5

日期：2026-04-29  
对象：`paper/paper.pdf`，生成时间 2026-04-29 12:31 CST，56 页  
结论：**Major Revision，接近 Weak Accept，但还不能直接投 OR 主刊**

## 总体评价

新版比上一轮强很多。最关键的几个问题已经被正面处理：摘要和贡献列表明确把 `LCB = Wasserstein DRO` 收窄为 **per-candidate scalar score identity**，不再声称完整 BA-SSP-MDP 策略是全局 DRO 最优；Lean 侧也不再是 theorem-shaped fragments，`dro_iSup_real_equality`、`posterior_product_high_prob_lcb_bound`、`hedge_regret_real_bound`、`exists_adversarial_trajectory_cumulativeRegret` 等声明形状已经接近论文声称；真实数据从单日/单 OD 扩展到 35 天 × 18 ODs × 45 trials，并加入 paired bootstrap、sign test 和组件消融。

这轮可以从之前的 **Weak Reject / Major Revision** 上调到 **Major Revision leaning Weak Accept**。但当前 PDF 还有若干严重的一致性问题和叙事问题，会让审稿人怀疑结果表格是否来自同一套实验。最重要的修复不是继续加 theorem 或实验，而是做一次彻底的 **paper consistency audit**。

## 已明显修好的点

1. **理论 claim 收窄到位。** 摘要、Introduction 和 Remark 3 都明确说明 Corollary 5 是 per-candidate ranking-layer identity，不是 full-policy DRO optimality。这基本解决了上一轮最大的理论过度包装问题。

2. **LCB 命名有解释。** 作者解释了 cost minimization 下 `mu + beta sigma` 与 `-cost` lower confidence bound 的关系。这个解释仍略绕，但已经足以避免“方向写反”的直接质疑。

3. **Lean 形式化可信度提升。** 我抽查了 Lean 目录，没有发现顶层 `axiom`、`sorry`、`admit`。`dro_iSup_real_equality` 现在确实含有 `iSup` over probability measures、W1 ball membership 和 translation witness；`posterior_product_high_prob_lcb_bound` 也把 product posterior bad-event bound 和 deterministic LCB gap 链起来；Hedge bound 已经定义 weights/losses/regret 并给出 cumulative theorem。

4. **真实数据证据大幅增强。** 35-day Swiss benchmark、paired `(day, OD)` cell analysis、per-day sign test、Oct 29 disrupted-day breakdown 都比上一版更像 OR-grade empirical section。

5. **V2 cold-start 不再被回避。** Table 7 明确报告旧 V2 collapse 到 Static 的问题，并给出 R15 修复后 V2 mean 从 98.4 改到 87.7 的结果。

6. **Neural surrogate 过度 claim 已削弱。** Section 10 明确说 224x 是 compute claim，不是 decision-equivalence claim，这一点很重要。

7. **组件消融有价值。** Table 11 把 A0/A1/A2/A3 拆开，显示主要收益来自 layered risk penalties 和 hyperpath PMF 信息。这比只报最终方法更有说服力。

## 主要问题

### 1. 表格、图和文字存在严重残留矛盾

这是当前版本最容易被审稿人抓住的问题。

- **Table 8 与 Table 9 的核心数值对不上。** Table 8 中 Adaptive-β 的 E[total] 从 Static `60.75` 降到 `51.99`，差值应为 `-8.76 min`。但 Table 9 的 paired ∆E[total] 对 Adaptive-β 是 `-2.24`，V1/V2/V3 也同样差很多。如果每个 `(day, OD)` cell 都有相同 45 trials，cell-average paired mean 应该与 aggregate mean difference 接近。除非 Table 8 和 Table 9 使用不同 weighting、不同 config 或不同 seed pool，否则这是重大不一致。

- **Figure 6 明显还是旧图。** 图内仍写 `Swiss multi-OD reach rate (17 viable ODs)`，柱子数值是旧的 `29/34` reach-rate 和 `30.2/33.2` conditional mean；但 caption 已改成 “Swiss 35-day cross-day evaluation, 18 viable ODs, E[total] by method”。这会直接暴露为 stale figure。

- **OD 数量不统一。** 正文多处说 `18 viable ODs`，Figure 4 caption 仍说 Paradeplatz 是 `17-OD multi-OD experiment`，Figure 6 图内也写 `17 viable ODs`。

- **年份不统一。** Section 8.4.2 说 GTFS-RT 数据是 `Oct 1–Nov 4, 2023`；Figure 4 caption 仍写 `Oct 29, 2025`。这会影响数据来源可信度。

- **SDN 结论前后矛盾。** Section 9 前面说 multi-seed 后 SDN 上 V1-LCB `-60.9%`、Hybrid `-61.1%`，并说 earlier single-seed loss 被推翻；但后面 “Why this is a useful negative” 仍写 “LCB wins on UC, breaks even on VRP, fails on SDN”。这段必须重写。

- **V2 cold-start 叙事冲突。** Section 8.4.1 和 Conclusion 说 V2 cold-start failure 已修复；Discussion 的 “V2 cold start” 段仍把 jitter/V1 fallback 写成 future mitigation，像是旧版本遗留。

- **Conclusion 仍有旧 claim。** 结论里说 “Oracle upper bound confirms that LCB-V2 leaves only a 1.5 min gap”，但新版 Table 4 中 large disrupted LCB-V2 是 `72.6`，Oracle 是 `73.0`；small disrupted LCB-V2 是 `69.0`，Oracle 是 `67.6`。应按具体表格重写。

这些不是文字润色问题，而是会直接削弱实验可信度。建议先全部修完再投稿。

### 2. 最终算法与核心 DRO 理论之间又出现新 gap

理论主线证明的是：

```text
score = mean + beta * sigma + gamma * p_cancel
```

但 Table 11 说明最终有效的 A2/A3 score 依赖：

```text
mu_dest + delta + beta*sigma + gamma*p_cxl
  + 60*(1 - feasibility)
  + 60*(1 - P_on_time)
```

并且还包括 typed cancellation counters、historical route priors、adaptive top-k/lookahead、persistent EXP3 meta-state。也就是说，35-day benchmark 的主要收益并不完全来自 Corollary 5 的简单 posterior LCB/DRO score，而来自把 hyperpath PMF 的 feasibility/on-time probability 加入风险 score。

这不是坏事，反而可能是实际贡献。但论文需要更诚实地连接：

- 如果 layered risk penalties 也可以解释为 Wasserstein radius 或 Lipschitz cost augmentation，需要给出形式化或半形式化说明。
- 如果不能，应明确区分 “DRO-theoretic core” 与 “R15 engineering score”，不要让读者以为 Table 8 的 14.4% 改善全部由 Corollary 5 解释。
- Table 11 显示 A2 是 dominant fix，那么 A2 的定义应进入 Method 主文，而不是主要放在实验解释和 Appendix E。

目前最强实验结果和最强理论结果之间还有一层工程桥接没有完全写清楚。

### 3. Theorem 9 的概率界仍然偏松，不能作为实用高概率保证

Theorem 9 的 union-Chebyshev bound 是正确方向，但实用性很弱。正文自己给出 `D=6, K=5, k=6` 时 `P(Good) >= 17%`；如果要 95% confidence，需要 `k >= sqrt(DK/0.05) ≈ 24.5`，而 cost bound 又线性乘以 `k`，会非常松。

建议把 Theorem 9 定位为 “formal Bayesian-to-deterministic bridge / sanity certificate”，不要暗示它给出 tight operational guarantee。若想增强，可补充 sub-Gaussian/Bernstein 或 Normal-Gamma Student-t tail 的 sharper bound。

### 4. Baseline 公平性仍需进一步说明

新版加入 BAMCP rollout-budget sweep 是好改动，但仍有几个问题：

- PS-SSP、BAMCP、EXP3、SW-LCB 是否使用了与 R15 LCB 相同的 typed cancellation signal、blacklist、patience window、layered timeout/on-time penalty？如果不是，Table 5 可能比较的是 “full R15 router” 对 “未工程化 baseline”。
- EXP3 的理论 upper bound 仍不能直接解释具体性能失败；它只能说明 regret guarantee 在 single-shot 预算下不强。
- BAMCP-240 在 small disrupted 反而不错，但 large disrupted 很差。需要解释这种 network-dependence，而不是简单归因为 “structural failure”。
- 如果最终方法使用 hyperpath PMF feasibility/on-time probability，baseline 是否也可以使用这些 risk features？如果不给 baseline，需要说明这是 proposed method 的可用信息优势。

### 5. Cross-domain section 需要重新组织

Table 13 比旧版强：8 seeds、有 CI，UC 结果显著，VRP wash，SDN multi-seed 翻转。但叙事没有完全同步：

- 不能再说 “SDN fails / recoverable per-flow mismatch”，因为新版结果显示 LCB family 在 SDN 大幅降低 mean delay。
- 如果解释是 Static seed-heavy-tail，应该从“irrecoverability boundary”改成“LCB as robust tail controller under volatile congestion”。
- React-UCB 仍显著更好，应保留“LCB transfers but is not domain-best”的定位。
- Table 13 的 `SIG` 标记使用 “95% CI disjoint from Static CI”，这比 paired seed test 弱且不标准。建议使用 paired seed-level differences 或 bootstrap over seeds。

Cross-domain section 现在可以保留，但它支持的是 “scope study”，不是主定理验证。

### 6. 论文过长且贡献层级仍偏散

新版 56 页，包含 Lean、Swiss、cross-domain、neural acceleration、A1-A10 engineering review、BAPR relation、full posterior proof。材料很丰富，但主线变得拥挤。

建议：

- 主文保留三条核心贡献：per-candidate DRO identity、Lean theory artifact、35-day Swiss validation。
- Cross-domain、neural surrogate、A1-A10 mapping 放 appendix 或 shorter secondary sections。
- 把 Table 11 的核心 A2 score 移到 Method，Appendix 只放完整 A1-A10 review。

## 次要问题

1. `LCB` 命名解释已经有了，但仍建议在第一次出现时写 “cost-side pessimistic LCB score / UCB-on-cost” 之类短注，减少读者心理阻力。

2. Related work 新增 risk-aware transit routing 是正确方向，但还可以更具体地区分 SOTA、robust shortest path、CVaR transit assignment 与本文在线 posterior update 的差异。

3. Table 11 在 PDF 文本抽取中 DRO / Adaptive-β 行似乎不完整，需检查排版是否被截断。

4. “zero axiom” 建议表述为 “zero project-level axiom declarations” 或 “no additional axioms beyond Mathlib/kernel”，避免被注释中的 “axiom” 字样误导。

5. 如果投稿双盲，GitHub 用户名、Zenodo DOI、repository commit 可能需要匿名化处理。

6. Figure 5 的 old synthetic distributions 看起来仍来自旧图，需确认是否与 R15 Table 4/6 同一配置。

7. `35/35 days better, sign-test p < 10^-4` 对 35 天不是独立同分布的假设比较强，建议写成 descriptive non-parametric evidence，而不是过度强调显著性。

## 给作者的问题

1. Table 8 的 aggregate E[total] difference 为什么与 Table 9 的 paired ∆E[total] 差这么多？是否使用了不同实验文件？

2. R15 最终 deployed score 是否就是 A3 all-on？如果是，Method section 是否应该以 A3 score 为主，而不是 Algorithm 1 的原始 DRO-LCB score？

3. A2 中的 `60*(1-P_on_time)` 能否被解释为 chance-constrained / CVaR-like penalty？如果可以，这会比把它当工程 trick 更有说服力。

4. Swiss 35-day benchmark 的 OD 筛选是在所有方法结果之前固定的吗？是否存在 selection bias？

5. Baselines 是否能访问 hyperpath StopLabel feasibility 和 destination-arrival PMF？如果不能，为什么这不是信息不对称？

## 建议修改优先级

1. **P0：一致性 audit。** 修 Table 8/9、Figure 6、17/18 ODs、2023/2025、SDN 结论、V2 cold-start 段、Conclusion 旧 claim。

2. **P0：把最终 R15/A3 score 写进 Method。** 明确哪些部分有 DRO identity，哪些是 empirical risk engineering。

3. **P1：重写 cross-domain narrative。** SDN 不再是失败案例，应改成 “LCB transfers but React-UCB remains domain-best”。

4. **P1：补 baseline information parity。** 用表格列出每个方法可访问的信息、取消处理、timeout penalty、compute budget。

5. **P1：收紧篇幅。** 把 secondary contributions 后移，主文集中在 transit + Lean + Swiss。

## 评分建议

- Novelty: 4/5
- Theory: 4/5 after claim narrowing
- Formal verification: 4/5, with theory-layer caveat
- Empirical evidence: 3.5/5, pending consistency fixes
- Clarity: 3/5, mainly hurt by stale figures/text
- Reproducibility: 3.5/5 for Lean, 3/5 for experiments

## 最终建议

**Major Revision, likely salvageable.**  
这版已经从“主张过大、证据不足”变成“核心贡献可信，但稿件需要一次严肃一致性清理”。如果修掉表格/图/年份/SDN/Conclusion 的残留矛盾，并把 R15 layered risk score 与 DRO 理论的关系讲清楚，这篇可以进入 Weak Accept 区间。当前不建议直接投，因为 Table 8 vs Table 9 和 Figure 6 stale 这类问题会让审稿人对实验 pipeline 产生不必要的不信任。
