GPT：
我直说：**以现在这个稿子的形态投 Operations Research，我觉得 desk reject 风险偏高，大概 60–75%**。不是因为稿子粗糙，而是因为它很容易被 AE 读成：

> 一个有趣的 transit reranking 工程/实证 heuristic，外加一个相对标准的 Wasserstein-Lipschitz 解释；但真正带来 empirical gain 的不是被 formalized 的核心理论。

最可能卡在 **Associate Editor 的 desk screen**；如果一开始就进 **Stochastic Models** area，Area Editor/AE 端的风险会更高。EIC 直接拒的概率我反而觉得不高，除非匿名性或 fit 出问题。

### 我的主观概率

| 结果                                | 我对当前稿的判断 |
| --------------------------------- | -------: |
| EIC 行政/fit reject                 |    5–10% |
| Area Editor 觉得不适合 OPRE 或不适合该 area |   20–30% |
| Associate Editor desk reject      |   35–45% |
| 能送外审                              |   25–40% |
| 送外审后拿到 R&R                        |   10–20% |
| 以当前 framing 最终进 OPRE              |     <10% |

如果被分到 **Transportation / Networks** 口味强的编辑，送审概率会上升；如果被分到 **Stochastic Models**，我会更担心，因为理论贡献会被拿来和 Wasserstein DRO、robust MDP、stochastic transit routing 的主流理论标准比，而这篇的理论核心容易被认为“解释性强、创新性弱”。

### 最危险的三个 desk-reject 理由

第一，**headline 和实证贡献有错位**。题目和摘要强调 “Wasserstein-Robust Scoring”，但你的 ablation 自己显示，四项 posterior-risk core 单独用时反而比 Static 差；真正把结果翻正的是 structural reliability terms，也就是 feasibility 和 on-time probability 两项。Table 10 里，posterior-risk score only 的 V1/V2 都差于 Static，而加 structural reliability 后才变好。这对 AE 来说会非常刺眼：形式化证明的是 A，但 empirical gain 主要来自 B。

第二，**baseline fairness 会被质疑**。Table 14 明确写了，LCB family 使用了 stop-label feasibility 和 destination on-time probability，而 PS-SSP、BAMCP、EXP3 没用这些 reliability features；后面虽然 retrofitted Static 和 SW-LCB，但没有把同等信息接口给 rollout / bandit baselines。这会让 reviewer 很容易说：你证明的是“带有更好 hyperpath label features 的 reranker 优于没有这些 features 的 baselines”，而不是“Wasserstein posterior pessimism 优于强基线”。

第三，**OR 理论贡献可能被认为太轻**。LCB = Wasserstein DRO 这件事在 1-Lipschitz affine cost 下，本质上是 Kantorovich-Rubinstein 上界加 translation witness；它是漂亮、干净、可 formalize 的，但在 OPRE 的 Stochastic Models 口味下，AE 可能会认为这不是足够新的 stochastic model / DRO theorem。更麻烦的是，full policy 的理论 bound 对 structural reliability terms 只是 additive perturbation，而且你文中承认这个 bound 在实际参数下基本 vacuous。

### 但它不是“没戏”的稿子

它有几个很强的点：选题是真问题，hyperpath “structure robust but ranking fragile” 这个 operational insight 很清楚；Zurich GTFS-RT stress-test 做得比普通 toy transit paper 强；paired bootstrap、leave-one-day-out prior audit、Oct 29 disrupted-day breakdown 都是加分项。你还很诚实地写了 benchmark 是 Paradeplatz-origin stress-test，不是 city-wide random OD sample，这比很多 paper 更可信。

Lean artifact 也是差异化卖点。我看了 zip 里的目录和文本，至少静态搜索没有看到真正的 `sorry` 或项目级 `axiom` 声明；但我这里没有 `lake/lean`，所以不能替你确认 `lake build` 是否真的通过。这个卖点在 OPRE 里有新鲜感，但也可能变成双刃剑：如果 AE 觉得 formal verification 只是证明一个相对标准的小 identity，而没有覆盖 simulator、GTFS pipeline、empirical statistics，那么它不会显著降低 desk risk。

### 我觉得 AE 可能会写出的拒稿逻辑

很可能是类似这样的判断：

> The paper studies an interesting transit routing heuristic under disruptions and provides a formal interpretation of a component of the scoring rule. However, the main empirical improvements appear to come from structural reliability features already present in the hyperpath labels rather than from the Wasserstein-DRO core. The validation is limited to a selected Zurich stress-test panel, and several baselines do not receive the same reliability information. The theoretical contribution is not sufficiently substantial for Operations Research.

这类话就是典型 AE desk reject，不一定会送外审。

### 最值得立即改的地方

我会优先做四件事。

**1. 换 framing。** 不要把 “Wasserstein DRO equivalence” 放在主标题/主贡献第一位。更稳的 framing 是：

> Keep-and-rerank disrupted stochastic hyperpaths using risk-augmented label reliability, with a Wasserstein interpretation for the posterior-risk component.

这样 reviewer 就不会拿整个 empirical gain 去反问 “不是 DRO 在起作用吗？”

**2. 把 baseline 信息接口补齐。** 至少主表里要有这些版本：Static、Static+reliability、SW-LCB+reliability、BOCD+recompute+reliability、LCB posterior-only、LCB full。如果 PS-SSP/BAMCP/EXP3 不容易加入 reliability terms，就不要把它们作为主要 defeated baselines；否则 reviewer 会抓住 fairness 不放。

**3. 把 empirical scope 扩大或降调。** 现在 35-day Zurich stress-test panel 有价值，但对 OPRE 来说偏窄。最好加一个 random-origin/random-OD panel；哪怕结果弱一些，也能证明不是 hand-picked stress-test。文中已经承认不是 city-wide superiority，这很诚实，但在 OPRE 里可能被 AE 直接当作“empirical contribution not general enough”。

**4. 修掉所有内部不一致。** 例如 Table 8 说 Adaptive-β 35/35 days better，但后文有一句说 31/35 days net-improved；Table 7 的 45-trial 主实验和 Table 10 的 15-trial audit 数字并列时容易让人困惑；DRO 在 focused OD 上强、cross-day 上很差，这个 narrative 要更早更清楚地解释。

### 我的最终判断

**当前稿直接投 OPRE：我会预期大概率 desk reject，尤其是 Stochastic Models area。**

但这不是“低质量拒稿”，而是“OR 顶刊贡献定位不够干净”的拒稿。
如果你把它改成 **transportation operational insight + fair risk-augmented reranking benchmark**，把 DRO/Lean 从“主贡献”降为“theory certificate for one component”，并补一个更广泛的 OD panel，我觉得送审概率可以从现在的 **25–40%** 提到 **50–60%**。
如果还能证明 full ranking objective 本身有一个非 vacuous robust/chance-constrained interpretation，而不是只证明四项 posterior core，那才有机会真正冲 OPRE。

我：所以有希望么？

GPT：
有希望，**但不是“现在这样直接投 OPRE 就很稳”的那种希望**。

我的判断是：

> **这篇论文作为一个可发表的 OR / transportation / stochastic routing 方向工作，是有希望的；作为 Operations Research 正刊稿件，当前版本属于“有亮点但 desk risk 较高”的状态。**

如果问得更具体一点：

| 问题                                     |            我的判断 |
| -------------------------------------- | --------------: |
| 这是不是一篇有研究价值的论文？                        |           **是** |
| 能不能投一个不错的 OR / transportation journal？ |          **可以** |
| 当前版本 OPRE 送外审概率                        |   **大约 25–40%** |
| 当前版本 OPRE 最终中稿概率                       |     **偏低，<10%** |
| 认真重构后 OPRE 送外审概率                       | **可提高到 50% 左右** |
| 重构后 OPRE 拿 R&R 的概率                     |  **可能到 15–25%** |

我会把它归为：**不是没希望，而是“贡献定位还没完全打通”的稿子。**

---

### 为什么我说它有希望

最强的地方不是 Wasserstein theorem，也不是 Lean，而是这个 operational insight：

> disruption 发生时，hyperpath 的 alternative set 往往还在；真正坏掉的是 ranking。

这个说法很有 OR 味道，因为它不是单纯做一个机器学习 heuristic，而是在重新定义 operational decision object：**不是 recompute route，而是 keep structure + rerank**。你的 paper 里 Table 1 的 recomputation audit 支撑了这个观点：BOCD+recompute 没有改善，甚至在最 disrupted day 表现更差，而 keep-and-rerank 有明显改善。这个是一个真实、有解释力、也有反直觉性的 finding。

实证也不是弱的。35-day Zürich benchmark 上，V2-LCB 把 cell-mean expected total time 从 49.34 降到 46.47，约 **−5.8%**；reach rate 从 78.2% 到 81.2%，约 **+3.0 percentage points**。paired analysis 里 V1/V2/Adaptive-β 都是 35/35 天优于 Static，CI 也很干净。这个结果作为 transportation routing paper 是有说服力的。

而且稿子现在最好的地方之一是**诚实**。你没有硬说 city-wide superiority，而是明确写这是 Paradeplatz-origin disrupted-route stress-test panel，不是 city-wide random OD sample；也承认 posterior-risk core alone 不够，structural reliability terms 是关键。这个诚实会降低 reviewer 的敌意。

---

### 为什么我仍然觉得 OPRE 当前版本危险

最大的风险是：**标题、摘要、理论主线和实证真正起作用的机制之间有错位。**

题目叫 “Wasserstein-Robust Scoring Approach”，摘要和贡献也突出 Wasserstein-DRO + Lean。但 Table 10 自己显示，posterior-risk score only 的 V1/V2 并没有打败 Static；真正把结果翻正的是 structural reliability objective，也就是 feasibility 和 on-time probability 这些 hyperpath label terms。换句话说，实证主贡献像是：

> hyperpath reliability labels + online reranking improve disrupted transit routing。

但理论主贡献像是：

> four-term posterior LCB score has a Wasserstein DRO interpretation。

这两者有关联，但不是同一个东西。OPRE 的 AE 很可能抓住这个点说：你 formalize 的不是 empirical gain 的主要来源。

第二个危险是 baseline fairness。Table 14 已经承认，LCB family 使用了 stop-label feasibility 和 destination on-time probability，而 PS-SSP、BAMCP、EXP3 没有这些 structural reliability features。虽然你做了 Static+reliability 和 SW-LCB+reliability 的 retrofit，这是很好的补救，但 reviewer 还是可能说：主表里被打败的几个强 baseline 没有拿到同等信息。

第三个危险是 OPRE 的理论门槛。Corollary 1 的 LCB = Wasserstein DRO identity 是干净的，也适合 Lean verification，但它是 **per-candidate scalar identity**，不是 full policy DRO optimality。稿子自己也说 full risk-augmented objective 的 theorem bound 加了 reliability perturbation 后在实际参数下是 vacuous。对普通 transportation journal 这不是致命问题；但对 OPRE Stochastic Models 来说，AE 可能觉得 theory contribution 不够“主菜”。

---

### 我觉得它“有希望”的正确打开方式

这篇最应该主打的不是：

> We propose a Wasserstein-DRO transit routing algorithm.

而是：

> Disrupted stochastic hyperpaths are structurally robust but ranking-fragile; a reliability-aware reranking rule can rescue reachability without recomputing the hyperpath.

然后把 Wasserstein / Lean 放成：

> The posterior-risk component has a formally certified Wasserstein interpretation.

这样就顺了。因为 Table 10 已经告诉你，**structural reliability 是主要 empirical driver，posterior pessimism 是 additional layer + theoretical certificate**。你现在其实已经在摘要里写了这句话，但标题、贡献排序和 related work 还是让人以为 DRO 是主角。

我会建议把贡献排序改成：

1. **Operational finding:** recomputation can fail because it drops useful fallbacks.
2. **Modeling contribution:** keep hyperpath structure, rerank candidates online using hyperpath reliability labels and posterior disruption evidence.
3. **Empirical contribution:** 35-day Zürich disrupted-route benchmark with reach-rate rescue.
4. **Theory contribution:** posterior-risk core admits Wasserstein-DRO certificate, machine-checked in Lean.

现在的顺序更像是把 2 和 4 放得太前，导致 reviewer 用理论标准审全文。

---

### 如果我是作者，我会怎么判断投稿策略

如果目标是 **Operations Research**，我会先改，不会直接投当前版。
当前版不是“没希望”，但 desk reject 风险确实不小。我觉得最可能的拒稿逻辑会是：

> Interesting empirical transit-routing heuristic and useful stress-test, but the main empirical improvements come from structural reliability features rather than the Wasserstein-DRO component. The theoretical contribution is per-candidate and does not establish global policy optimality. The empirical scope is a selected Zürich stress-test panel rather than a broad OD/city evaluation.

这个拒稿理由很合理，所以要提前消灭它。

如果你愿意补一个 **random-origin / random-OD panel**，哪怕 gain 从 5–6% 下降到 1–3%，反而会让稿子更强。因为 OPRE 很看重“这个机制是不是一般性 operational phenomenon”。目前最容易被质疑的是：这个结果是不是 Paradeplatz-origin disrupted-route stress-test 特有。你已经诚实承认 scope，但 OPRE reviewer 可能会说，诚实承认不等于解决问题。

---

### 我的最终判断

**有希望。**

但我会把希望分成两层：

**作为一篇研究论文：希望很大。**
核心 insight 清楚，实证有真实数据，negative recomputation audit 有价值，paired results 也漂亮，Lean artifact 是加分项。

**作为当前版本的 OPRE 投稿：希望有限但不是没有。**
我不会说它没戏；我会说它现在卡在“贡献叙事不匹配”。如果不改，容易被 AE 认为是一个 strong empirical heuristic + formalized side theorem，而不是 OPRE 级别的 integrated contribution。

我最真实的建议是：**不要放弃 OPRE，但先重构。** 把 “Wasserstein-Robust” 从主标题降下来，把 “structurally robust but ranking-fragile hyperpaths” 推上去；补 baseline fairness 或至少更强地解释；再加一个 broader OD robustness check。这样它就从“容易 desk 的 clever paper”变成“有可能送审、有可能拿 R&R 的 OR paper”。

