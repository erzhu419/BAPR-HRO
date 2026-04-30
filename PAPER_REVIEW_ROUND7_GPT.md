# BAPR-HRO Paper Review Round 7

日期：2026-04-29  
对象：`paper/paper.pdf`，生成时间 2026-04-29 17:14 CST，61 页  
结论：**Weak Accept after targeted cleanup / 当前仍建议小修后再投**

## 总体评价

这一轮又有实质进展。Round 6 的几个硬伤基本都被处理了：Conclusion 的 SDN 旧叙事已经改掉；Section 2.3 的旧 `90.5 vs 84.3` 数值已更新；A4 historical prior 的数据泄漏风险新增了 leave-one-day-out audit；A7/R15 的理论边界被写得更克制；baseline information asymmetry 也被正面披露。

现在论文的主要风险不再是“核心主张站不住”，而是 **最终稿一致性、命名口径、表格版式和少量残留 claim**。如果目标是投稿 OR/INFORMS 级别 venue，我不建议带着这些明显可修的问题直接投；但从审稿判断上，这版已经进入 **Weak Accept / Borderline Accept** 区间。

## Round 6 问题修复情况

| Round 6 问题 | 当前状态 | 评价 |
|---|---|---|
| Conclusion 旧 SDN 结论 | **已修** | 现在写成 LCB 在 SDN multi-seed 下也有收益，但 React-UCB 是 domain-specific best。 |
| Section 2.3 旧数值 | **已修** | 已改为 PS-SSP `91.1/96.0` 等 R16/R15 表格附近的新数值。 |
| A4 数据泄漏风险 | **基本修复** | 新增 LOO audit，声称 LOO 与 non-LOO 差异 ≤ 0.02 min。 |
| A7 理论边界 | **明显改善** | 现在明确 A7 是 empirical augmentation，formal theorem 只 cover four-term core。 |
| Baseline + A7 公平性 | **诚实披露但未完全解决** | Table 3 承认 A7 只给 LCB family；文中说明直接 baseline+A7 留作 future work。 |
| 版式溢出 | **仍未修完** | log 仍有 float too large 和极端 overfull；Table 12 文本抽取明显截断。 |

## 主要优点

1. **A4 leave-one-day-out audit 是关键补强。** 之前最危险的数据泄漏质疑被正面处理。文中给出 LOO numbers：V1 `46.75 vs 46.76`、V2 `46.49 vs 46.47`、V3 `48.26 vs 48.27`、DRO `55.70 vs 55.70`、Adaptive-β `46.75 vs 46.76`。如果 artifact 里确有对应 JSON，这个问题基本关闭。

2. **R15/A7 理论 claim 降调到位。** Theorem 8 下方现在明确说 `D*CA7` slack 在单 journey 上可能是 vacuous，不把 R15 layered score 包装成同等强度的 DRO theorem。这是审稿人会认可的诚实写法。

3. **Baseline fairness 口径更合理。** 论文现在明确说 cross-day Swiss claim 是 “LCB family with A7 hyperpath-structural risk score”，不是“同信息预算下纯 LCB/DRO beats all baselines”。这比之前强行比较更稳。

4. **SDN 叙事已自洽。** Section 9 和 Conclusion 都承认 React-UCB 仍是 SDN-specific best，同时说明 LCB family 在 high-volatility SDN 下也能作为 robust mean controller transfer。

5. **实验结果更像最终版。** Swiss 35-day 结果现在是 V2 `49.34 → 46.47`、V1/Adaptive `49.34 → 46.76`，paired CI 和 Figure 6 已同步；Oct 29 day-only 表也更新为 R16。

## 主要问题

### 1. R15/R16 命名混用，必须统一

正文和结论多处说 “R16 algorithm”，例如 Section 8.1、Conclusion；但多个表格 caption 仍写 “R15 algorithm / R15 configuration”：

- Table 5 caption: `under the R15 algorithm`
- Table 6 caption: `large network, R15 algorithm`
- Table 7 caption: `R15 configuration`
- Table 8 caption: `R15 historical-prior configuration`
- Table 13 caption: `R15 configuration`

同时 Section 4.5 仍叫 “R15 Deployed Score”。如果 R16 是 R15 加上 typed-cancellation/deadline correction，那么需要在 Method 或 Experiment Setup 中定义：

```text
R16 = R15 layered score + typed-cancellation accounting + A7 deadline correction
```

并统一所有 table captions。否则读者会怀疑 Table 5/6/9/11 是否来自同一配置。

### 2. 表格版式仍是直接投稿风险

LaTeX log 仍有：

- `Float too large for page by 37pt`
- `Float too large for page by 31pt`
- `Overfull hbox 760pt`
- `Overfull hbox 176pt`
- 多处 80pt/98pt overfull

PDF 文本抽取中 Table 12 的 Adaptive-β 列明显截断：

```text
Adaptive-
56.33 (+14.
51.18 (+3.7
47.60 (-3.5
46.76 (-5.2
```

Table 3 的列结构也不易读，A7/σ/p_cancel 列在抽取文本里混在一起。即使 PDF 视觉上可能勉强可读，审稿人看大表时也会感到工程痕迹重。建议：

- Table 3 改成横向 `sidewaystable` 或拆成两张表。
- Table 12 缩列名、用 `\scriptsize` + `tabularx`，或把 A0–A3 ablation 放 appendix。
- Table 14 cross-domain 也适合横向页。

### 3. A7 feature asymmetry 已披露，但仍是主实验解释的弱点

论文现在诚实地承认 A7 features 只给 LCB family，而不是给 PS-SSP/BAMCP/EXP3/SW-LCB。这个披露很好，但它也意味着主结论必须保持克制：

- 可以说 “LCB family with hyperpath structural risk features improves Swiss cross-day E[total]”。
- 不应强说 “LCB/DRO alone beats exploration baselines under equal information”。

当前文中基本已经这样写，但 Abstract 仍容易被读成 LCB/DRO 本身赢。建议在 Abstract 的 Swiss 结果句中直接加上 “with the R16 layered hyperpath-risk score”。

### 4. A4 LOO audit 不进主表，最好加脚注或 appendix table

文本说 LOO 与 non-LOO 差异 ≤ 0.02 min，这足够强。但主表仍报告 non-LOO numbers。为避免审稿人继续追问，建议：

- Table 9 caption 加一句 “LOO historical-prior audit changes all E[total] values by ≤0.02 min; see §8.4.2.”
- 或在 appendix 加小表列 non-LOO vs LOO。

这样 A4 问题会彻底关闭。

### 5. Adaptive-β “safety net” 的贡献现在变弱了

结果显示 Adaptive-β 基本追踪 V1：Swiss 上 V1 和 Adaptive-β 都是 `46.76`，SDN 上 Adaptive-β `-59%` 与 fixed-β LCB 接近。它不像早期版本那样“救回 fixed-β failure”。现在更准确的说法是：

> Adaptive-β provides a robust β-selection wrapper, but in the reported settings it mostly tracks the best fixed-β LCB rather than producing a distinct gain.

贡献列表第 7 条仍写 “recovers to Static-level performance in regimes where fixed-β LCB over-corrects (e.g. recoverable per-flow SDN)”。这句已经和新版 SDN 结论不完全一致，建议删除 “e.g. SDN”，改成泛化的 safety-net framing。

### 6. 篇幅仍偏长

61 页对主刊投稿压力较大。当前包含 Lean theorem mapping、Swiss 35-day、A1–A10 review、cross-domain、neural acceleration、appendix proof。建议主文压到更干净的结构：

1. Core transit method + per-candidate DRO identity
2. Lean artifact summary
3. Synthetic + Swiss real-data validation
4. Short limitations

Cross-domain、A1–A10、neural surrogate、full high-prob proof可以进 appendix / online supplement。

## 次要问题

1. Abstract 中仍写 “LCB rule substantially helps when decisions are irreversible ... and substantially also helps on SDN”，这已经比较长。可以简化成一两句，避免 abstract 过载。

2. “zero sorry, zero axiom” 已改成 “zero project-level axiom declarations”，很好；Conclusion 里仍有 “zero axiom beyond Mathlib’s standard kernel”，建议统一措辞。

3. Table 5/6 的 R16 typed-cancellation correction 导致 static-β DRO 大幅变差，这是新发现。建议在 method/baseline 部分解释为什么 DRO row 不是与 LCB-V1 等价的 R16 score，否则读者会问 “LCB=DRO 为什么 DRO 反而差”。

4. Theorem 12 的 irrecoverability hypothesis 现在被定位为 sufficient condition，不是 necessary condition；这个改法好，但 contribution list 里仍有一点旧 “recoverable per-flow” 味道。

5. Sign-test `p < 10^-10` 对 35 天结果有些夸张，建议写精确二项检验值或 `p=2^{-35}` 量级，并提醒 days are not necessarily independent。

## 建议修改优先级

1. **P0：统一 R15/R16 命名和所有 table captions。**
2. **P0：修 Table 12 / Table 3 / Table 14 版式溢出。**
3. **P0：删掉 contribution 7 中 “e.g. recoverable per-flow SDN” 的旧 safety-net 例子。**
4. **P1：在 Table 9 caption 或 appendix 加 LOO audit 小表/脚注。**
5. **P1：Abstract 明确 Swiss result 是 R16 layered hyperpath-risk score，不是 naked four-term DRO core。**
6. **P1：压缩篇幅，把 A1–A10 和 cross-domain 细节下沉。**

## 评分建议

- Novelty: 4/5
- Theory: 4/5
- Formal verification: 4/5
- Empirical evidence: 4/5 after LOO audit
- Clarity: 3.7/5，主要扣 R15/R16 和表格版式
- Reproducibility: 3.8/5

## 最终建议

**Weak Accept after cleanup.**  
这版核心已经够强：理论 claim 收窄得当，Lean 形式化可信，Swiss 35-day 证据更扎实，A4 数据泄漏质疑也基本被 LOO audit 处理掉。当前不应继续大改算法或加实验，应该做最后的投稿级整理：统一 R15/R16，修表格溢出，删旧 safety-net 例子，给 LOO audit 一个表注。完成这些后，我会把建议上调到 **Weak Accept / Accept**。
