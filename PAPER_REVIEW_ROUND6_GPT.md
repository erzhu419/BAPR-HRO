# BAPR-HRO Paper Review Round 6

日期：2026-04-29  
对象：`paper/paper.pdf`，生成时间 2026-04-29 12:51 CST，60 页  
结论：**接近可投，但仍需一次 targeted cleanup；当前建议 Major Revision leaning Weak Accept**

## 总体评价

这一轮明显修掉了 Round 5 的核心 P0。Table 8/9 的 cell-mean vs paired bootstrap 不一致已经改正为 Table 9/10 的统一口径；Figure 6 已更新为 35-day cell-mean E[total]；OD 数量和年份基本统一；R15 deployed score 被移入 Method 4.5，并明确区分了 DRO core 与 A7 layered risk penalties；Theorem 9 的 Chebyshev bound 也被诚实解释为 conservative certificate，而不是 tight operational bound。

这版已经不像“还没整理好的实验草稿”，而是一个基本可信的论文版本。剩下的问题主要是 **旧叙事残留、A4 historical prior 的数据泄漏风险、R15 工程层与理论层边界、baseline 信息公平性**。这些都可以修，但当前 PDF 还不建议直接投。

## Round 5 问题修复情况

| Round 5 问题 | 当前状态 | 评价 |
|---|---|---|
| Table 8 vs Table 9 数值不一致 | **已修** | 现在 Table 9 `49.34 → 46.91/47.11` 与 Table 10 paired ∆ `-2.43/-2.24` 对齐。 |
| Figure 6 是旧图 | **已修** | 图内已改成 35-day cell-mean E[total]，18 ODs。 |
| 17/18 ODs、2023/2025 冲突 | **基本修复** | Figure 4 已改为 Oct 29, 2023；主文统一 18 viable ODs。 |
| SDN section 前后矛盾 | **Section 9 已修，Conclusion 未修** | Section 9 已改成“LCB also helps on SDN under multi-seed”；Conclusion 仍保留旧的“hurts/recoverable mismatch”说法。 |
| V2 cold-start 段落冲突 | **已基本修** | Discussion 现在写成“failure mode and R15 fix”，不再像纯 future work。 |
| R15/A3 score 没进 Method | **已修** | Section 4.5 明确写出 `score_R15`，并说明 A7 不属于 DRO formal scope。 |
| Baseline 信息公平性 | **部分修** | Table 3 新增 information-access parity，但也暴露出 A7 features 只给 LCB family。 |

## 主要优点

1. **实证口径更严谨。** 新版明确区分 per-cell reach、cell-cond、cell-mean E[total]，并解释早期混用 `rbar * cbar + (1-rbar)*tmax` 的错误。这一段很加分。

2. **R15 score 终于透明。** Section 4.5 明确写出：

```text
score_R15 = mu_dest + delta + beta*sigma + gamma*p_cxl
          + 60*(1-feasibility) + 60*(1-P_ontime)
```

并承认后两项是 bounded engineering penalties，不是 Wasserstein-DRO identity 的一部分。这是正确写法。

3. **Table 3 信息公平性是好补充。** 审稿人会问 baselines 吃不吃同样信号，现在至少能直接看到 GTFS-RT、blacklist、patience、sigma、cancel posterior、A7 features 的分配。

4. **Theorem 9 的弱点被主动说明。** `k=6` 只有 17% good-event probability、95% Chebyshev 需要 `k≈24.5`，这类诚实说明会降低审稿人的攻击欲望。

5. **Cross-domain 叙事更稳。** Section 9 现在承认 SDN-specific React-UCB 仍然最好，同时说明 LCB 在 multi-seed SDN 上作为 robust-mean controller 也有收益。这比上一版的“SDN fails”更符合 Table 14。

## 主要问题

### 1. Conclusion 仍残留旧 SDN 结论，必须改

正文 Section 9 已经改成：

- SDN multi-seed 下 V1-LCB `-60.9%`
- Hybrid `-61.1%`
- Adaptive-β `-59.0%`
- React-UCB column best `-74.2%`
- 结论是 LCB transfers to SDN but is not domain-best

但 Conclusion 仍写：

> it helps on daily commitment decisions, is neutral on VRP, and hurts when per-flow SDN decisions are recoverable. Crucially, Adaptive-β ... recovers to Static-level performance in this last regime mismatch

这与 Table 14 和 Section 9 直接矛盾。必须改成新版口径：LCB 在 UC 显著好、VRP wash、SDN multi-seed 也显著降低 mean/tail delay，但 React-UCB 仍是 SDN-specific best；irrecoverability 是 sufficient condition，不是 necessary condition。

### 2. Related Work 仍有旧实验数值残留

Section 2.3 还写：

> PS-SSP achieves 90.5 min under disruption vs. Static’s 84.3 min (Table 6)

当前 Table 6 是 R15 large network：PS-SSP `91.1`，Static `73.8`。旧 `84.3` 来自早期版本。这里需要同步，否则审稿人会怀疑表格是拼接的。

类似地，Table 13 caption 里有一句：

> Note on Table 5 discrepancy ... Static mean differs (84.3 vs. 98.3)

这也像旧版本残留，且现在 Table 5/13 的上下文不清。建议删除或重写。

### 3. A4 historical prior 可能有数据泄漏风险

R15/A3 使用 “hierarchical route priors built from the 34 normal days”。如果 35-day benchmark 同时评估这些 normal days，那么对任一 normal-day cell，prior 可能包含该目标日的数据。这会使 Swiss 35-day result 不再是严格 out-of-sample evaluation。

需要明确下面三种之一：

- A4 prior 是用评估窗口之前的历史数据训练的；
- 对每个 evaluated day 使用 leave-one-day-out prior，不包含目标日；
- 这是 transductive/offline calibration，不能被表述为 out-of-sample real-time validation。

建议优先改成 leave-one-day-out 或 temporal split，并更新 Table 9/10/12。如果不重跑，至少在 limitations 中承认 A4 可能使用同一 35-day archive 的信息。

### 4. R15/A7 理论桥接仍然偏弱

Section 4.5 和 Remark 5 说 A7 penalties 不属于 DRO scope，但 “Theorem 8’s O(sigma_max) excess-cost rate continues to apply with additive constant CA7=120 min per stop”。这个说法数学上可以作为 bounded perturbation，但实际意义很弱：

- 每 stop 加 `120 min` slack，在 `tmax=120` 的实验里几乎吞掉了整个 guarantee。
- 这不是 Lean formalized theorem，而是 prose extension。
- 如果 A7 是 dominant empirical fix，那么最强实验结果主要依赖一个没有 DRO interpretation 的 risk penalty。

建议写得更克制：Theorems 8–9 certify the four-term DRO-LCB core; R15/A7 is an empirically validated augmentation. 不要让读者以为 deployed R15 有同等强度的 theoretical guarantee。

### 5. Baseline fairness 现在被 Table 3 自己暴露出来

Table 3 很诚实地显示 A7 features 只被 LCB family 使用，PS-SSP/BAMCP/EXP3/SW-LCB 没有使用 StopLabel feasibility 和 on-time probability。既然 Table 12 显示 A7 是 dominant fix，这意味着主实验的一部分优势来自额外 feature engineering，而不是纯算法类别优势。

需要二选一：

- 给 baselines 也加同样 A7 risk penalty，报告 “baseline + A7”；
- 或者明确主张变成 “LCB family plus hyperpath structural risk features”，而不是 “LCB/DRO beats exploration baselines”。

现在已有 Table 3 披露信息不对称，这是好事，但最好再补一个 ablation：without A7, LCB family vs baselines；with A7, all methods that can consume A7 vs R15。

### 6. 文稿排版还有明显风险

LaTeX log 中仍有很多 overfull/underfull，且至少两个 float too large：

- `Float too large for page by 122pt`
- `Float too large for page by 89pt`
- 多处 overfull hbox，包括 760pt、176pt 等极端值

这通常对应大表格或长公式溢出。Table 3、Table 12、Table 14、Appendix table 都应检查 PDF 视觉效果。当前 PDF 已 60 页，篇幅也偏长；投 OR 主刊前需要压缩主文，把 A1–A10、cross-domain、neural surrogate 的细节下沉。

## 次要问题

1. **Table numbering 已基本稳定，但仍有旧引用。** 主要是 Section 2.3 和 Table 13 caption 的旧 Table 5/6 数值说明。

2. **DRO method 命名可能混淆。** Table 12 说 A7 retrofitted into LCB family but not standalone DRO router；但理论上 DRO and LCB core are equivalent。建议解释 “DRO row” 是 fixed-β four-term router without A7，而不是 R15 layered LCB。

3. **Sign-test 依赖天级样本。** `35/35 days, p<1e-4` 可以保留，但应避免暗示 35 days 完全独立；更稳妥写作是 “non-parametric descriptive evidence over days”。

4. **A7 penalty constant 60 min 来自 ablation。** 可以更明确说明是否调过该常数；如果调过，需要给 validation split 或 sensitivity。

5. **Neural surrogate 仍只是 compute claim。** 现在已经写清楚，这是好事；建议在 abstract 中也避免让人误读为 policy-preserving acceleration。

6. **“zero axiom” 表述。** 建议统一写 “zero project-level axioms / no additional axioms beyond Mathlib/kernel”，避免审稿人 grep 到注释里的 “axiom” 字样产生误解。

## 建议修改优先级

1. **P0：修 Conclusion 的 SDN 旧结论。**
2. **P0：修 Section 2.3 的旧 baseline 数值和 Table 13 caption 旧说明。**
3. **P0：澄清或重跑 A4 historical prior，避免 35-day Swiss benchmark 的数据泄漏质疑。**
4. **P1：降低 R15/A7 理论 guarantee 的语气，把它明确定位为 empirical augmentation。**
5. **P1：补 baseline + A7 或 no-A7 fairness ablation。**
6. **P1：修 LaTeX overflow 和 float-too-large，压缩主文。**

## 评分建议

- Novelty: 4/5
- Theory: 4/5 for core DRO/Lean layer
- Formal verification: 4/5
- Empirical evidence: 3.8/5，若 A4 prior 无泄漏可到 4/5
- Clarity: 3.5/5，主要扣在旧残留和篇幅
- Reproducibility: 3.5/5

## 最终建议

**Major Revision leaning Weak Accept.**  
这轮已经把 Round 5 的大部分硬伤修掉，论文的核心贡献现在可信：per-candidate LCB=DRO identity、Lean theory artifact、35-day Swiss validation 和 R15 layered-risk engineering 都能站住。当前最危险的不是理论或实验本身，而是几个会让审稿人迅速失去信任的残留冲突：Conclusion 的 SDN 旧叙事、Related Work 旧数值、A4 prior 是否泄漏、以及 A7 feature 不对称。修完这些后，这篇可以进入 Weak Accept/Accept 讨论区。
