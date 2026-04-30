# BAPR-HRO Paper + Proof Review Round 8

日期：2026-04-29  
目标期刊：`Operations Research`  
对象：

- 论文：`paper/paper.pdf`，最新生成时间约 2026-04-29 17:25 CST，61 页
- 论文源文件：`paper/paper.tex`
- Lean 证明：`/home/erzhu419/mine_code/proof/BAPR-HRO/*.lean` 与 `/home/erzhu419/mine_code/proof/Wasserstein/*.lean`

结论：**Weak Accept after proof-claim cleanup / 当前不建议直接投 OR，建议先修 P0 问题。**

## 总体评价

这一轮比 Round 7 明显更接近可投稿状态。R16 的定义、A7 的理论边界、Swiss 35-day 主结果、Table 9/10、Oct 29 disrupted-day 结果，以及 A4 leave-one-day-out audit 都已经比前几版稳很多。Lean artifact 也确实能构建通过，且在 BAPR-HRO 与 Wasserstein 目录下没有发现顶层 `sorry`、`axiom`、`admit` 或 `unsafe` 声明。

但如果目标是 OR，当前最大风险已经从“实验结果是否可信”转移到 **论文的强声明是否精确匹配 Lean 中实际证明的定理**。现在有两个 P0 级别问题：Wasserstein robust Bellman contraction 的论文表述强于 Lean 证明；V2 “Ensemble LCB = Empirical DRO” 在 Lean 中是 witness construction，而不是与 V1 一样完整的 `iSup` 等式。它们都可修，但不能带着现在的 cross-reference 直接投。

## 我已核查的内容

- `lake build` 在 `/home/erzhu419/mine_code/proof` 下通过：`Build completed successfully (16056 jobs).`
- 构建中只有无关文件的 unused-variable linter warning：`RESAC-Counterproof.lean`、`ApproxContraction.lean`。
- 定点扫描 `BAPR-HRO` 与 `Wasserstein` 证明目录，没有发现顶层 `sorry`、`axiom`、`admit`、`unsafe` 或 `set_option maxHeartbeats 0`。
- `DRO.lean` 已有真正的 real `iSup` equality：`dro_iSup_real_equality`，并包含 translation witness 与 `w1_translation_le`。
- `BAPRHRO.lean` 的 product posterior high-probability LCB bound 已经不是空壳，确实组合了 joint bad-event probability 与 good-event 条件下的 cost bound。
- `BAPRHRO_V2.lean` 已补上 empirical measure、translation push-forward identity 和 W1 membership，但没有完整的 ensemble-specific `iSup` supremum theorem。
- `DROBellman.lean` 明确区分 mixture 的 `γ`-contraction 与 Wasserstein surrogate 的 `γ(1+ε)` Lipschitz bound。

## 主要优点

1. **核心 Swiss 结果现在自洽。** Table 9/10 已对齐：Static `49.34`，V2-LCB `46.47 (-5.8%)`，V1/Adaptive `46.76 (-5.2%)`，DRO `55.70 (+12.9%)`。paired CI、days better、sign-test 口径也基本同步。

2. **A4 数据泄漏风险基本被处理。** Section 8.4.2 新增 LOO audit，报告 LOO 与 non-LOO 差异均在 `0.02 min` 以内。对审稿人来说，这足以关闭“历史 prior 偷看当天数据”的主要质疑，前提是 artifact 中确有 `swiss_multi_day_loo.json`。

3. **R16 部署分数的理论边界写得更诚实。** Section 4.5 和 Remark 5 已明确 Corollary 5 只 cover `μ_dest + δ + βσ + γp_cxl` 四项，A7 的 `feasibility` 与 `P_ontime` 是 empirical augmentation。

4. **Lean artifact 的工程状态可用。** 证明能整体 build，且没有项目级 axiom/sorry。对 OR 审稿来说，formal artifact 是加分项，但前提是论文不要把未 formalize 的版本写成已 formalize。

5. **R15/R16 命名已比前一轮好很多。** Table 5/6/7/8/13 的主 caption 已基本改到 R16；Table 12 对 A0/A1/A2 的 R15-era 历史语义也有解释。

## P0 问题

### 1. Wasserstein robust Bellman 的论文声明强于 Lean 证明

论文当前写法：

- `paper.tex:1101-1110`：Theorem 6 标题为 `DRO-Bellman contraction`，并声明 Wasserstein robust Bellman operator `T_ε` 是 `γ`-contractive。
- `paper.tex:1464-1467`、Conclusion 附近也说 formalized results include robust Bellman `γ`-contraction for rectangular-Wasserstein and discrete-mixture ambiguity。
- `paper.tex:1745-1749` 把 `wasserstein_robust_bellman_lipBound` 放在 “Robust Bellman γ-contraction” 这一行。

Lean 实际证明：

- `Wasserstein/DROBellman.lean:25-47` 明确说 mixture instance 是 fully `γ`-contractive，但 Wasserstein instance has honest `(1 + εC)` Lipschitz bound。
- `Wasserstein/DROBellman.lean:305-324` 的定理 `wasserstein_robust_bellman_lipBound` 结论是
  `≤ γ * (1 + ε) * supDiff V W`，不是 `≤ γ * supDiff V W`。

这不是小措辞问题。OR 审稿人如果打开 Lean 文件，会发现论文 cross-reference 把一个 Lipschitz bound 写成 contraction theorem。建议二选一：

1. 修改论文：Theorem 6 改成 “generic non-expansive ambiguity gives `γ`-contraction; discrete mixture is an instantiated `γ`-contraction; Wasserstein KR surrogate is `γ(1+ε)`-Lipschitz and is a contraction only when `γ(1+ε)<1`。”
2. 或补 Lean：定义 exact Wasserstein-ball supremum ambiguity functional，并证明它 satisfies `NonExpansiveAmbiguity`，再把该 theorem 接到 Theorem 6。当前 `wassersteinAmbFunc = E_Phat[V] + ε Lip(V)` 不是这个 exact-sup formalization。

另外，`paper.tex:1106-1107` 的 displayed definition 里漏了 `γ`：

```tex
c(s,a) + \sup_{P:W_1\leq\varepsilon} E_P[V(s')]
```

但 proof 的 Step 2 又使用 `c(s,a) + γ sup ...`。即使保留这个 theorem，也要先把公式修一致。

### 2. Proposition 1 的 V2 empirical DRO formalization 仍有 overclaim

论文 Section 4.4 写：

```text
Ensemble LCB is DRO on the empirical measure.
LCB_ens(r) = ... = sup_{Q: W1(Q, P_hat_ens) <= eps_r} E_Q[arr(r)].
```

Lean 当前更准确的状态是：

- `ensemble_lcb_equals_empirical_dro` 只证明 shifted ensemble 的 mean 等于 LCB score，且 std translation-invariant。
- `ensemble_W1_le_eps` 证明 shifted empirical measure 在原 empirical measure 的 W1 ball 内。
- `ensemble_dro_witness_complete` 证明 push-forward identity + mean shift + std shift。
- `ensemble_argmin_equals_empirical_dro_argmin` 最后只是把 LCB score rewrite 成 shifted ensemble mean 的比较，没有出现真正的 `⨆` / `iSup` over W1 ball。

这对数学直觉来说基本够用，因为可以借 V1 的 translation witness 逻辑补上 supremum equality；但对 “machine-verified Proposition 1” 来说还差最后一层。建议：

- 最干净：在 `BAPRHRO_V2.lean` 增加一个 ensemble-specific `iSup` theorem，形式对齐 `DRO.lean` 的 `dro_iSup_real_equality`。
- 最省事：论文把 Proposition 1 的 formalization 改成 “verified witness construction for the empirical-DRO equality; the full supremum equality follows by applying Corollary 5 to the empirical measure”，不要写成 Lean 已直接证明了完整 sup equality。

## P1 问题

### 3. `DRO.lean` 和 `Distance.lean` 使用了两套 W1 定义

`Wasserstein/Distance.lean` 定义：

```lean
def wassersteinDist (μ ν : Measure X) : ℝ≥0∞ := ...
```

`Wasserstein/DRO.lean` 又定义：

```lean
def W₁ (μ ν : Measure X) : ℝ≥0∞ := ...
```

二者结构上相同，但没有 bridge theorem 说明 `W₁ = wassersteinDist`。因此论文说 “W1 pseudometric structure + DRO equality” 是一条统一 formal chain 时，严格说中间有一个 integration gap。建议补一个小 theorem：

```lean
theorem W₁_eq_wassersteinDist ... : W₁ μ ν = wassersteinDist μ ν := ...
```

或者让 `DRO.lean` 直接复用 `Distance.lean` 的定义。这个问题不是 soundness bug，但会被 formal-methods 取向的审稿人抓到。

### 4. R16/A7 的 bound 语言还有一处不够严谨

`paper.tex:1223-1244` 已经诚实说明 Theorem 8 只对应 four-term core，A7 会带来 `D * C_A7` slack，且在当前设置下 slack 可达 `~6 tmax`，所以 operationally vacuous。

但同一段仍写 “the substitution preserves O(σ_max) asymptotically”。如果 A7 slack 是常数项，严格表达应是：

```text
four-term core: O(σ_max)
R16 layered score: O(σ_max) + D*C_A7, hence not a vanishing bound in σ_max unless A7 terms are separately controlled.
```

建议删除 “preserves O(σ_max)” 或改成 “the four-term verified component remains O(σ_max); the deployed R16 score has an additional bounded perturbation.”

### 5. Baseline + A7 公平性已披露，但仍是 OR 审稿弱点

当前 Table 3 和 Section 7.3 已承认 A7 hyperpath-structural features 只给 LCB family，没有 retrofit 到 PS-SSP / BAMCP / EXP3 / SW-LCB。这个披露是正确的，但 OR 审稿人会继续问：

```text
LCB family wins because of Bayesian pessimism, or because it gets destination-arrival PMF / feasibility features?
```

论文现在最稳的 claim 是：

```text
LCB family with the R16 layered hyperpath-risk score improves Swiss cross-day E[total].
```

不应再出现裸的 “DRO-optimal update rule explains the deployed Swiss gains” 这种口径。Abstract 最后一段仍有一点这个味道，建议继续降调。

### 6. Swiss benchmark 代表性需要更克制

Section 8.4 是 35 days × 18 ODs × 45 trials，但 OD 是从 Paradeplatz origin 的 20 candidate ODs 里筛出 18 个，并要求 hyperpath 中有 disrupted route 与 safe alternative。这是合理的 stress-test benchmark，但不是 city-wide random OD benchmark。

建议在 OR 投稿口径中明确：

- “Zurich sub-network stress-test benchmark”
- “Paradeplatz-origin disrupted-route OD panel”
- 不要写成一般性的 “Swiss real-data validation proves city-wide superiority”

如时间允许，最好加一个 appendix robustness check：随机多 origin OD sample 或至少多几个 origin 的 OD panel。没有也能投，但 limitation 要写清。

### 7. EXP3/Adaptive-β formalization 的措辞需要保持精确

论文现在已经写 “standard EXP3-IX reduction from bandit feedback to Hedge via importance-weighted losses is presented in prose, not formalized”，这是好的。需要确保所有 summary/table 都不要写成 “full EXP3-IX theorem is machine-verified”。Lean 里强的是 deterministic/full-information Hedge potential bound；bandit reduction 是 prose layer。

### 8. 排版仍不是投稿级

最新 `paper.log` 仍有严重 overfull：

- `Overfull \hbox (760.46701pt too wide)` at `paper.tex:3160-3203`，对应 hyperparameter table。
- `Overfull \hbox (176.79779pt too wide)` at `paper.tex:3320-3327`，对应 proof appendix 的 long Mathlib theorem name。
- 还有 `165pt`、`62pt`、`49pt` 等 overfull。

`paper.tex:3156-3203` 的 Table 16 四列表格太宽，尤其是 route prior rationale 那一行。建议用 `tabularx`、`\scriptsize`、缩短 rationale，或放入 landscape appendix。OR 投稿 PDF 不能带这种明显溢出。

## 次要问题

1. `lcb_equals_dro_attained` 在 Table 2 中作为 Corollary 5 counterpart 是可以的，但如果强调 real `sup equality`，最好同时列 `dro_iSup_real_equality`。

2. Figure 3 caption 说 “LCB=DRO equivalence” 是对 four-term core 成立。建议 caption 加 “four-term core” 或在图注中避免让读者误以为 A7 两项也在球内。

3. Table 12 的 A0/A1/A2 是 R15-era historical code state，A3 是 R16 simulator。caption 已解释，但建议视觉上分隔 A0-A2 与 A3，避免读者把它当成同一 simulator 下的 strict ablation ladder。

4. Sign-test `p < 10^-10` 对 35 days 可以成立为 `2^-35` 量级，但 days 不是完全独立样本。建议写成 “descriptive sign-test” 或在 caption 加 independence caveat。

5. Conclusion 中 “DRO-optimal update rule is O(1) posterior-based LCB score” 这类泛化句建议改成 “the four-term LCB core has a DRO interpretation; the deployed R16 router adds hyperpath-structural risk terms.”

## 建议修改优先级

1. **P0：修 Theorem 6 和 Table 1/2 的 Wasserstein Bellman contraction claim。**
2. **P0：修 Proposition 1 / `BAPRHRO_V2.lean` 的 formalization 口径，补 `iSup` theorem 或降调论文表述。**
3. **P0：补 `γ` 到 Theorem 6 的 displayed robust Bellman operator。**
4. **P1：补 W1 definition bridge theorem，或让 `DRO.lean` 复用 `Distance.lean` 的 `wassersteinDist`。**
5. **P1：把 R16/A7 bound 写成 `O(σ_max) + D*C_A7`，不要说 deployed score preserves pure `O(σ_max)`。**
6. **P1：修 Table 16 和 proof appendix 的 overfull boxes。**
7. **P1：Abstract/Conclusion 继续压低裸 “DRO-optimal deployed rule” 口径，强调 R16 layered score。**
8. **P2：OR 版本中把 Swiss benchmark 的 OD selection 和 stress-test nature 写得更清楚。**

## 评分建议

- Novelty: 4/5
- OR relevance: 4/5
- Theory: 3.7/5 当前扣在 Bellman/V2 formalization mismatch；修后可到 4.2/5
- Formal verification: 4/5 工程强，但 cross-reference 需精确
- Empirical evidence: 4/5，LOO audit 后可信度明显提高
- Clarity: 3.5/5，主要扣 proof claim mismatch 与版式
- Reproducibility: 4/5

## 最终建议

**不要再大改算法或继续堆实验。** 这版的主体已经够强，Swiss real-data + Lean artifact + R16 ablation 是有 OR 投稿价值的组合。当前最该做的是投稿前的 proof-claim 对齐：

1. 让论文中每个 “machine-verified” claim 都能在 Lean 中找到同强度 theorem。
2. 让每个 “DRO/LCB exact equivalence” 都明确只针对 four-term core，还是包括 empirical ensemble/V2。
3. 清掉明显 overfull 的表格。

完成 P0 后，我会把建议上调到 **Weak Accept / Accept after minor cleanup**。当前版本如果直接投，最可能被抓的问题不是实验，而是 “formal verification table overstates what the Lean files prove”。
