# OR/OPRE Review Round 4 for `paper_opre.tex`

日期：2026-04-30  
对象：`paper/paper_opre.tex`、`paper/paper_opre.pdf`、baseline+A7 audit、R16 ablation/recompute audit  
目标期刊：Operations Research (`opre`)

## 总体判断

这版已经基本到了 **submission cleanup** 阶段。上一轮最重要的几个点已经处理得很好：baseline+A7 audit 加了 day-level robustness，fairness 段不再和 `SW-LCB+A7` 矛盾，conclusion 的 future work 也改成 rollout/adversarial-bandit baselines，`regret_sandwich` 在 Lean summary 里也降级成 legacy theorem name。

我现在的建议是：**可以准备投 OR Lengthy Manuscript，但投前还要做一次 consistency sweep。** 主要不是理论或实证大问题，而是当前文稿里有几处旧数字、旧说法和本轮新表不一致。

## 对本轮回复的处理

1. **P1 #5 baseline+A7 未重跑 45 trials/cell**：接受你的处理。表 caption 已明确写 `audit-budget`、`15 trials/cell`、`single Monte Carlo seed`，不会和主表 `45 trials/cell` 混读。这不再作为硬伤，只是如果算力足够，45 trials/cell 会更漂亮。
2. **P1 #7 intro 仍有 LCB/DRO 技术词**：同意，不再视为硬伤。现在 intro 的人话程度已经够 OPRE 送审。
3. **P1 #8 float warnings**：这里我不能完全撤回。当前本地 `paper_opre.log` 是 2026-04-30 08:36 生成的，仍有 exact warnings：Algorithm float too large by 27.41983pt，EC table float too large by 285.45946pt，并有多处 overfull hbox。如果你另一套编译环境已经是 0 warnings，需要把对应 PDF/log 同步到当前目录。
4. **P2 #3 A2 caption 合并 V2 cold-start**：接受。caption 已明确写 `+A7 layered + V2 cold-start fix`，这不再是问题。

## 已明显改善

1. `tab:baseline_a7_audit` 现在有 `Days < Static`，这很好。`Static+A7` 是 30/35，`SW-LCB+A7` 是 29/35，`LCB-V1` 是 35/35，说明 posterior core 在 A7 上确实提供更稳定的增益。
2. Baseline-fairness 段现在说清楚了：Static/SW-LCB 已 retrofit，PS-SSP/BAMCP/EXP3 没 retrofit，原因是要做 rollout/loss adapter。
3. `clean attribution` 已降成 `useful attribution audit`，口径稳了。
4. Conclusion 的 future work 已改成 broader retrofit for rollout/adversarial-bandit baselines，不再否认当前已做的 retrofit。
5. `regret sandwich` 在 Lean results 表里已改成 lower-bound/upper-bound scope lemma，并注明 theorem name 是 legacy identifier。

## P0 / 投前必须修

### 1. A1--A10 附录仍残留旧 ablation 数字

主文 `tab:swiss_ablation` 已经更新为 R16 audit-budget 数字：

- A0 V1：57.18，`+13.0%`
- A1 V1：52.34，`+3.5%`
- A2 V1：49.09，`-3.0%`
- A3 V1：47.44，`-6.2%`
- A2 V2：46.99，`-7.1%`

但 Appendix `app:a1a10` 仍有旧数字：

- `paper_opre.tex:3401-3403` 写 original implementation `+9.6%`，但当前表 A0 V1 是 `+13.0%`。
- `paper_opre.tex:3450-3452` 写 A7 moves V1 from `+3.7%` to `-3.5%`，但当前表是 `+3.5%` to `-3.0%`。
- `paper_opre.tex:3467-3469` 写 `+3.7%` to `-3.5%` and `7.2 pp swing`，也应更新。

建议把 Appendix 的 A1--A10 数字全部按当前 `tab:swiss_ablation` 同步。否则审稿人会觉得主表和附录不是同一次实验。

### 2. 主文 baseline-fairness 段仍有旧 ablation 数字

`paper_opre.tex:1598-1603` 仍写：

> A1 row ... LCB family is at +3.7 to +4.9% vs Static ... sign change to -3.5 to -4.9% at A2/A3

当前 R16 ablation 表不是这个数。建议改成：

> In the R16 audit-budget ablation, A1 leaves V1/V2 worse than Static (+3.5% and +6.0%), while A2 flips them to -3.0% and -7.1%; A3 lands at -6.2% and -6.8%.

同时不要把 DRO 的 `+7.0%` 放进 A1 row 叙述；当前 reference DRO 是 `+11.6%`，且不是 row toggle。

### 3. 当前本地 LaTeX log 仍非 clean

按当前 `/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.log`：

- Algorithm 1: `Float too large for page by 27.41983pt`
- EC Lean mapping / proof table: `Float too large for page by 285.45946pt`
- EC table lines 2550--2605 有多处 overfull hbox

如果实际 PDF 视觉上已经没问题，可以把这降为 cosmetic；但从 OR submission hygiene 看，最好在最终上传前清掉，尤其是 285pt 的 EC float warning。

### 4. Code archive 还有一处内部 review 痕迹

全局搜下来，代码里基本干净了，只剩：

- `src/bandit_router.py:63`: `The reviewer flagged this as...`

这句如果随 supplementary code 提交，建议改成普通设计说明：

> This over-pessimism biased V1 toward already-ridden routes.

PDF 里 `earlier drafts` 只在 TeX 注释中，不影响 PDF，但如果提交 source，也建议删掉注释。

## P1 / 建议修

### 5. A7 audit 很强，但和主表的尺度要一直保持区分

当前 caption 已经写清楚 audit-budget，这很好。主文解释也写了 `15-trial-per-cell`，可以保留。建议不要在 abstract/conclusion 里引用 baseline+A7 的 `-3.7% / -6.2%` 数字，避免和主表 `-5.2% / -5.8%` 混在一起。

### 6. `tab:baseline_a7_audit` 可以再精炼一点

现在表很好，但 caption 比较长。OR 读者可接受，不过如果版面紧张，可以把 “Days < Static” 的解释移到正文一句话。

### 7. `Regret sandwich` 仍在 theorem title / table label 层面可见

`Corollary~\ref{cor:sandwich}` 的 label 叫 sandwich 没关系，Lean theorem name 也可以保留。但 Paper-to-Lean cross-reference table 行名仍是 “Regret sandwich (Cor...)”。建议改成：

> Lower/upper bound scope lemma (Cor.~\ref{cor:sandwich})

这样全文口径完全一致。

### 8. EC/table placement 仍需按 OR 最终要求确认

Guideline 写 References before Electronic companions、tables after Reference List。当前稿件是 EC before References，tables inline。这个已经多轮提过，我不再当内容硬伤，但 final ScholarOne proof 前要确认。

## 当前推荐

如果我是 OR reviewer，这版我会给：

**Ready for submission after consistency cleanup.**

核心贡献现在是清楚的：hyperpath 结构保留，在线重排；A7 structural penalties 是部署收益的大头，LCB/DRO core 提供额外稳定收益和理论解释。baseline+A7 audit 把之前最大的一块 empirical fairness 风险补上了。

投前最后做这四件事：

1. 同步主文和 Appendix A1--A10 的所有 ablation 数字。
2. 把本地 LaTeX log 中两个 float warnings 处理掉，或确认并同步 clean compile。
3. 清掉 `src/bandit_router.py` 里最后一处 reviewer 痕迹。
4. 把 Paper-to-Lean table 的 “Regret sandwich” 行名改成 lower/upper-bound scope lemma。

做完这些，我认为可以按 OR Lengthy Manuscript 准备正式投稿。
