# OR/OPRE Review Round 3 for `paper_opre.tex`

日期：2026-04-30  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、`src/baseline_a7_routers.py`、`experiments/swiss_full/run_baseline_a7_audit.py` 及相关 JSON  
目标期刊：Operations Research (`opre`)

## 总体判断

这版已经明显进入“可投前最后清理”的阶段。上一轮最大的问题 baseline+A7 现在真的补了：`Static+A7` 和 `SW-LCB+A7` 都跑了，结果也很有用。它把故事讲清楚了：A7 structural penalties 贡献了大部分 reach rescue，但 LCB-V1 的 posterior pessimism 在 A7 之上还有额外 travel-time gain。

当前建议：**Minor revision before OR submission**。不再是需要大改的状态。剩下的问题主要是几处口径冲突、audit 结果统计强度标注、source/code 中的内部 review 痕迹和 OPRE 排版细节。

我重新核了当前状态：

- PDF：61 页；Code/Data Disclosure 和 EC 从 PDF 第 40 页开始，References 第 55 页开始。
- Abstract：约 174 words，合规。
- LaTeX：没有 unresolved references/citations；仍有 Algorithm 1 和 EC 大表 float/overfull warnings。
- Lean：相关 12 个 target build 成功，只有 linter warnings。

```bash
lake build BAPRHRO BAPRHRO_V2 LowerBound Irrecoverability IrrecoverabilityBridge EXP3Regret AdaptiveConvergence Architecture WassersteinCoupling WassersteinDistance WassersteinDRO WassersteinDROBellman
```

## 这轮显著进步

1. **标题改对了**：现在是 “Re-Ranking Stochastic Transit Hyperpaths Under Disruption”，不再把读者引到 “DRO core alone explains everything”。
2. **Methods 更像人话**：新增 `tab:methods_overview`，用 plain-language role 对应 implementation label，比之前直接扔 EXP3/BAMCP/PS-SSP 好很多。
3. **Recompute baseline 正式成表**：`tab:recompute_audit` 有 caption/label/sample size，符合 OR table 基本要求。
4. **Ablation 表拆干净了**：现在主表只放 V1/V2 两个真实 toggled columns，V3/DRO/Adaptive-beta 放 reference line，上一轮的误导基本解决。
5. **Baseline+A7 audit 是关键补强**：`Static+A7`、`SW-LCB+A7`、`LCB-V1 (R16)` 同表比较，回答了 “是不是 A7 alone 就够了” 的问题。
6. **内部审稿口吻基本删掉**：正文里 `The reviewer correctly noted...` 这类句子已经没了。

## P0 / 投前必须修

### 1. Baseline-fairness 段和新增 A7 audit 现在互相矛盾

`paper_opre.tex:1608-1610` 仍写：

> SW-LCB/EXP3/PS-SSP/BAMCP could in principle consume them; we did not retrofit them into the baselines...

但 `paper_opre.tex:1629-1649` 已经报告了 `SW-LCB+A7`。这会被审稿人抓成明显自相矛盾。

建议改成：

> We did not retrofit A7 into PS-SSP, BAMCP, or EXP3 because their update rules would require a different loss/rollout adapter. We did retrofit A7 into Static and SW-LCB as a controlled audit, reported in Table X.

同理，`paper_opre.tex:2284-2288` conclusion 的 future work 还写：

> fairness retrofit that exposes the hyperpath-structural penalties to baselines other than the LCB family

这也过时了。应改为：

> a broader retrofit for rollout and adversarial-bandit baselines

### 2. Baseline+A7 结论有点过强，应降半档

`paper_opre.tex:1651` 写 “The retrofit gives a clean attribution”。这个实验确实有价值，但还不能叫完全 clean attribution，因为：

- 只 retrofit 了 Static 和 SW-LCB，没有 retrofit PS-SSP/BAMCP/EXP3。
- 只用 `15 trials/cell, single Monte Carlo seed`，低于主 Swiss 表的 `45 trials/cell`。
- `Static+A7` 和 `SW-LCB+A7` 是 adapter-style baselines，不是对应论文原方法的完整重设计。

建议改成：

> The retrofit gives a useful attribution audit.

或者：

> This audit suggests the following attribution...

同时建议加上 day-level robustness：

- `Static+A7` 优于 Static：30/35 days。
- `SW-LCB+A7` 优于 Static：29/35 days。
- `LCB-V1 (R16)` 优于 Static：35/35 days。

这比只报均值更有说服力，也能说明 posterior core 的稳定性不是均值偶然。

### 3. Source/code archive 里仍有内部 review 痕迹

论文 PDF 正文基本干净了，但提交 supplemental archive 时，代码里还有不少类似：

- `src/bandit_router.py`: `A3 (GPT review)`, `A7 (GPT-5.5 review)`, `P0 #1 R3 review fix`, `reviewer's concern`
- `src/sw_lcb_router.py`: `P1 R3 review`

如果 OR reviewer 打开代码包，这些会很不专业。建议全局清理：

- `GPT review` → `Implementation note`
- `P0/P1/R3` → 删除
- `reviewer concern` → `design concern`

论文 tex 的注释里也有 `earlier drafts`，虽然 PDF 不显示，但 source file 提交时也建议删掉。

### 4. `regret sandwich` 仍在 Lean cross-reference table 里出现

正文 corollary 已经降级为 “not a tight sandwich”，但 `paper_opre.tex:2516` 仍写：

> regret sandwich preserved (`regret_sandwich`)

这会把旧的强叙事又带回来。建议改成：

> lower-bound/upper-bound scope lemma (`regret_sandwich`, despite legacy theorem name)

或者干脆不写 “sandwich”，只写 Lean theorem name in parentheses。

## P1 / 建议投前修

### 5. Baseline+A7 audit 最好跟主 Swiss 表用同一 Monte Carlo budget

现在 `tab:baseline_a7_audit`、`tab:recompute_audit`、`tab:swiss_ablation` 都是 9,450 journeys/method，single seed；主 Swiss 表是 28,350 journeys/method。你已经在 caption 写清楚了，这是好事。

但从审稿角度，baseline+A7 是现在支撑归因的关键表。若算力允许，建议至少把 `baseline_a7_audit` 跑到 `45 trials/cell`，这样可以直接和主 Swiss 表对齐。否则保留 audit-budget 字样，不要把 `-3.7% / -6.2%` 和主文 `-5.2% / -5.8%` 混用。

### 6. `all eleven methods` 这句现在需要核一下口径

`paper_opre.tex:1593-1596` 说 “all eleven methods share the same feed...”。现在 main methods overview 加上 Oracle 是 11 个，但 baseline+A7 又额外引入 `Static+A7` / `SW-LCB+A7` 两个 audit methods。建议写清楚：

> In the main comparison, all eleven methods...

而 baseline+A7 audit 是 additional controlled audit，不属于主 11-method comparison。

### 7. Introduction 仍有一点技术密度，但可接受

现在 intro 已经比之前好多了。不过 OR guideline 说 introduction should not contain equations or mathematical notation。现在 intro 仍有 LCB、DRO、Wasserstein-1、Lean 4、Section refs 等技术词，但没有大公式。这个我不再视为硬伤；如果还想更稳，可以把 “lower confidence bound, abbreviated LCB” 改成 “pessimistic score” 先行，缩写放到 Method。

### 8. OPRE 版式仍有老问题

LaTeX log 还剩：

- Algorithm 1 float too large by 27.4pt。
- EC proof/code mapping table float too large by 285.5pt。
- EC code mapping table多处 overfull hbox。

这些不是内容杀伤点，但 OR 第一眼会看版面。投前建议处理。

此外 OPRE guideline 写 tables should be placed together after Reference List，而当前 tables inline；EC 也在 References 前。这个可能是模板/ScholarOne 容忍问题，但最终上传前最好确认。

## P2 / 小建议

1. `tab:baseline_a7_audit` 可以加一列 “Days better than Static”：30/35、0/35、29/35、35/35。这个很直观。
2. `SW-LCB` 在 baseline+A7 audit 中非常差（+15.2%），但 main synthetic small disrupted里 SW-LCB 很好。可以加一句说明这是 Swiss 35-day panel under audit-budget setting，避免读者误以为 SW-LCB 结论前后冲突。
3. `A2 (+A7 layered + V2 cold-start fix)` 对 V2 来说仍是两个改动合并。你现在 caption 已经承认 “A7 + cold-start correction”，这可以接受；不要再说 “A7 alone flips V2”。
4. Conclusion 里 “fairness retrofit” 需要和已经完成的 Static/SW-LCB retrofit 同步，见 P0 #1。

## 当前审稿意见

如果我是 OR reviewer，这一版我会倾向于：

**Major contribution, minor revision before submission.**

理由是：论文的主叙事已经清楚，关键归因也基本诚实；baseline+A7 audit 让 empirical story 可信很多；Lean 和理论口径也没有明显越界。现在最需要避免的是让审稿人看到几处文本自相矛盾后降低信任。

投前最后 checklist：

1. 修 baseline-fairness 段和 conclusion future work 的 retrofit 矛盾。
2. 把 “clean attribution” 改成 “attribution audit”，并尽量加 days-better counts。
3. 清理 code/source 中 `GPT review`、`P0/P1/R3`、`reviewer concern` 等内部痕迹。
4. 把 `regret sandwich` 残留改掉。
5. 处理 LaTeX float/overfull，或至少确认 PDF 视觉上没有溢出版面。

做完这几项，我认为可以按 OR Lengthy Manuscript 进入投稿准备。
