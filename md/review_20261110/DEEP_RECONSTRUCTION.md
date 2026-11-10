**从新版 BAPR 到 HRO 的深度重构（2026-11-10）**

可以更彻底地重构，但应把它当成一项新的研究工作。上一份修改方案主要修复正确性、描述与归因；这些是可信论文的前提，尚不足以回答“即使修好，方法贡献仍不够 OR”。建议保留数据、可用实现和历史证据，重新建立研究问题、强基线和理论对象。

**新版 BAPR 已经做对了什么**

当前 [BAPR 主稿](/home/erzhu419/mine_code/BAPR/paper/paper.tex:36)是 *Belief-Aware Policy Reparameterization for Persistent Actuator Regimes*，已经明确放弃旧 BOCD/LCB 主线。它找到的结构是：已知、可逆且保持动作约束的工况变换允许一个参考策略继续工作；在线任务变成因果辨识与坐标补偿。

| 当前证据 | 对 HRO 的启发 |
|---|---|
| V32 HalfCheetah 十策略种子：3297.1 对等策略交互预算 SAC 2287.1，配对差 1009.9，95% CI [391.7,1628.2]。[主稿](/home/erzhu419/mine_code/BAPR/paper/paper.tex:51) | 总效果与组件增量分开；记录估计器额外成本。 |
| 独立 Reacher 同信息比较：补偿相对 conditioned SAC 为 +7.641，[5.317,9.965]，5/5。[结果](/home/erzhu419/mine_code/BAPR/reproducibility/reacher_confirmation/analysis.json) | 同一信息可以通过不同决策接口产生不同效果；HRO 应固定信息、预算和候选来源来比较。 |
| V47 独立 signed-permutation 扩展：+4.674，[2.920,6.427]，5/5。[报告](/home/erzhu419/mine_code/BAPR/reports/submission_extensions_v47_results_2026-10-05.md:21) | 扩展研究应检验机制适用范围，而非只增加任务数量。 |
| 正确的有限时域回报差由实际策略轨迹上的 Bellman telescoping 得到。[证明](/home/erzhu419/mine_code/BAPR/paper/appendix_submission.tex:124) | HRO 应同样从真实旅程成本与因果信息出发，重新推导性能关系。 |

HRO 不应照搬动作补偿。取消服务、错过换乘和已执行路径前缀通常不可逆；不存在已知双射把它们变回正常网络。两篇可以共享“利用结构减少适应难度”的研究思路，但应各有独立的问题与贡献。旧稿中把 HRO 当作旧 BAPR 悲观性理论的应用，需要重写。

**我推荐的新研究问题**

> 在部分实时信息和明确的计算预算下，怎样维护一个足够好的公交候选策略子图，并判断何时可以继续使用、何时必须扩展？

建议工作题目为 *Adaptive Maintenance of Transit Routing Policies under Partial Disruption Information*。核心对象是带后续换乘选择的策略子图，而非一个连接的加权分数。研究输入包括时刻表、观测与预测的到达时序、扰动模型以及计算预算；输出是可执行的因果路由策略和所保留候选的质量说明。

预算优先用项目实际存在的连接扫描数、label 更新数、候选数量和实测运行时间。当前 [V1](/home/erzhu419/mine_code/BAPR-HRO/src/bandit_router.py:352)已经把 top-k 从 5 放大到 8，同时把等待窗口从 25 放大到 50 分钟，这些因素应拆开。不要人为收取“保留备选费用”，也不要假设亚秒计算必然导致错过车辆。如果完整维护全部候选在目标网络中已足够便宜，这条预算压缩路线就缺少实际必要性。

**已有研究决定了创新必须落在哪里**

| 已有工作 | 新稿不能重复宣称的贡献 |
|---|---|
| [Durner 2024 原文](</home/erzhu419/mine_code/BAPR-HRO/reference/Stochastic Strategies for Public Transport Journeys Based on Realtime Delay Predictions.pdf>)，第 4 章及 Algorithm 5.4 | 已讨论实时自适应策略、完整到达分布和被支配连接的价值；“有用备选可能被忽略”不是新发现。 |
| [Rambha et al. 2016](https://pubsonline.informs.org/doi/10.1287/trsc.2015.0613)、[Geissmann & Gianinazzi 2019](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ATMOS.2019.4) | 随机公交自适应路由、有限时域 MDP、预处理和高效求解已有直接研究。 |
| [K-adaptability，Hanasusanto et al. 2015](https://pubsonline.informs.org/doi/10.1287/opre.2015.1392) | 预存 K 个策略、获得信息后选优本身不新。 |
| [Büsing 2012](https://onlinelibrary.wiley.com/doi/10.1002/net.20487) | 预算限制下的路线恢复已有成熟问题。 |
| [Ren et al. 2024](https://proceedings.mlr.press/v244/ren24a.html)、[Zheng et al. 2026，预印本](https://arxiv.org/html/2608.29465v1) | 决策相关分布变化、基于 optimality gap 触发重新决策已有近邻。 |
| [Smith & Simmons 2004，HSVI](https://publications.ri.cmu.edu/heuristic-search-value-iteration-for-pomdps) | belief 空间上的上下界、局部展开和带误差界的 anytime planning 也已有成熟方法。 |

因此，“Bayesian + hyperpath + 局部修复 + 上下界”这个组合还不能自动成为贡献。值得追求的具体目标是：利用时刻表的时间结构、共享下游换乘和局部扰动，构造比通用 belief-space 搜索更便宜且足够紧的候选质量界，并据此增量修复策略子图。需要证明这种结构何时有效、最坏情况下何时退化为全量规划；随后证明它在同信息、同预算强基线下有实际价值。这是本次选定的研究目标，不是已经完成的定理。

**本轮已做的精确诊断**

新增 [run_probe.py](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_support_v1/run_probe.py)，结果在 [probe_results.json](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_support_v1/probe_results.json)。执行命令：`python3 experiments/reconstruction_support_v1/run_probe.py`。

1. 在现有 Paradeplatz → Goldbrunnenplatz、08:10 查询中处理 1588 个连接；均值近似去重删除 353 个 label，其中 46 次与保留项线路不同、75 次发车时刻不同、73 次 feasibility 不同，三个计数有重叠。例：线路 9 的 08:49 连接被线路 14 的 08:50 连接替代，只因二者预测到达均值相同。
2. 当前 [去重代码](/home/erzhu419/mine_code/BAPR-HRO/src/durner/topocsa.py:208)直接按均值差小于 0.01 删除，传入 connection 却不使用其发车分布。Durner Algorithm 5.4（原文第 55–56 页）使用期望发车时间做支配判断，并对相同发车时刻保留弱支配情形。当前实现不能当作原算法的忠实基线。
3. 两工况三候选成本向量 (0,10)、(10,0)、(4,4)：逐工况最优项的并集，在工况未揭示且各半概率时成本为 5；保留折中项成本为 4；若选择前工况完全揭示，前两项可以达到 0。因此候选保留问题必须包含信息到达时序。
4. 两阶段网络中全动作最优成本 2、受限最优成本 6、实际策略成本 6。正确分解得到结构损失 4、受限集合内决策损失 0。若错误地用全网络 Q 值衡量“集合内排序损失”，会得到 1，把未来候选受限混入排序归因。
5. 一条由两条连续边组成的替代路线，单独添加任一边都无收益，合并才有收益。这阻止我们直接套用单层菜单的次模贪心保证。

这些诊断已排除了几种不成立的重构方式，并表明应先纠正基线语义。候选删除的次数没有被解释为造成乘客损失的次数。

**可以现在写清楚的数学基础**

令 h_t 为决策前可用的完整历史，A_t(h_t) 为物理可行动作。候选规则 H_t(h_t)⊆A_t(h_t) 必须非空、因果，并对相关反事实历史有定义。有限时域、可积成本、初始分布、超时和终端成本相同。full 与 H 内最优策略接收相同信息。

全动作最优策略可以模拟受限策略，因此 J_full* ≤ J_H* ≤ J_π。无额外计算成本或模型误差时，完整最优重规划不会比固定候选策略差；“重规划本身有害”应退出主张。

用 V_H*、Q_H* 表示未来仍按同一 H 规则执行的最优值，则

\[
J_\pi-J_{\mathrm{full}}^*
=\underbrace{J_H^*-J_{\mathrm{full}}^*}_{\text{候选支持损失}}
+\underbrace{\mathbb E_\pi\sum_{t=0}^{T-1}
[Q_{H,t}^*(h_t,a_t)-V_{H,t}^*(h_t)]}_{\text{受限候选内的决策损失}}.
\]

证明：Bellman 方程把第二项改写为实际策略轨迹上的一步成本加下一时刻值减当前值；求和后望远镜抵消，得到 J_π−J_H*，再加减 J_H*。两项非负；受限最优策略的第二项为零。这样允许不同策略经过不同站点，也不会重复累加最终到达时间。

若实际策略对估计的 Q_H 贪心，并在 H 内满足统一误差 |Q̂_H−Q_H*|≤δ_t(h)，则第二项至多 2 E_π Σδ_t(h_t)。这里 δ 是 continuation cost 的误差界，不是直接拿 delay 后验标准差代替。这些属于标准集合包含和 performance-difference 论证，作为正确分析基础，不列为新方法贡献。

必须分开三个参照：同信息 full optimum、知道当前工况但不知道未来的 optimum、知道未来全部实现的事后最优。后两者用于识别信息价值；不能把其优势归因于候选结构。有限 rollout 得到的参考策略也不自动是真正最优值或数学界。

**新方法需要真正完成的部分**

最有希望的实现路线是：维护共享后续换乘的候选 labels，使用同一因果信息更新其值界；只删除能在明确的 belief 范围内证明不会改善目标的候选；需要时成组恢复下游 continuation。实时运行返回当前可执行策略和剩余质量差距，而非只返回一个分数。

设 U_H 是受限子图内可执行因果策略的成本上界，L 是全网络同信息最优成本的有效下界。U_H−L≤ε 足以说明该策略 ε-近优，但这个判据本身是标准的。研究难点是以较低成本构造、更新足够紧的 L 和 U_H。直接使用提前知道未来的松弛值可能在全候选集下仍留下很大信息差距，不能据此误判为结构缺失。估计模型下的界与对真实环境的统计保证也要分开。

论文需要交付的新增结果应围绕：哪些 continuation 能安全共享或删除；如何定位必须成组恢复的换乘结构；更新成本与受影响结构之间的关系；在何种可检验条件下获得非空泛的质量—计算权衡。不能先承诺多项式复杂度、次模性、固定百分比收益或全局局部性。

**执行顺序**

第一步：重建可信参考。修复上一份方案中的事件语义和后验定义；按原文实现并验证 TopoCSA 标签插入及持续重规划；分别控制候选数量、等待窗口和评分规则。用少量已存在 OD 的轨迹比较确认收益是否仍存在，避免先大批重跑。

第二步：建立可精确求解的有限时域信息树基准。只纳入与项目对应的现象：同线路相关延误、真实取消、延迟获得信息、错过换乘和已执行前缀。比较全动作同信息最优、固定候选最优、现有 reranker，直接计算上述两项损失。模型规模小到可以穷举，防止用新的近似 oracle 掩盖解释问题。

第三步：只围绕主导误差开发算法。若支持缺失主导，做候选结构维护；若主要是候选内估值错误，则转向 continuation 估值，停止把保留结构当创新。如果正确完整规划已足够便宜，应放弃预算压缩的必要性主张。

第四步：算法和协议冻结后，在同实时信息、同计算预算下比较原生自适应路由、完整更新、固定 K、预算受限通用 belief planner 与新方法。记录实际成本、到达率、连接扫描和延迟分布，报告质量—计算曲线。先复用修正后的原面板，再用独立起点/日期验证；同一批策略、估计器与重复旅程不扩大独立样本数。

正文由“决策问题 → 结构性障碍 → 算法与对应保证 → 匹配计算实验 → 运营解释”组成。DRO 等式与 Lean 不再主导题目和摘要；旧稿错误修正保留在版本记录。论文应由新方法与证据决定能否达到 OR 的贡献要求。

**局限与当前结论**

已完成的是本地 BAPR 主线/证据核查、近邻文献检索、数学分解和一个现有图查询诊断；未完成新算法、统计校准或新的乘客收益实验，也没有证明候选维护方向的新颖性。诊断没有重跑旧稿实验，当前依赖库不是已确认的历史运行环境。

BAPR 的证据边界继续保留：V32 组合条件仍 FAIL；HalfCheetah 同信息对照区间跨零；Reacher 没有建立过滤或 ensemble 的独立收益。其已知可逆动作变换的保证不能移植到公交取消。详细数值见 [同信息诊断](/home/erzhu419/mine_code/BAPR/reproducibility/reacher_confirmation/halfcheetah_information_diagnostic.json)、[主稿讨论](/home/erzhu419/mine_code/BAPR/paper/paper.tex:436)。

我的判断是：值得沿上述问题做一次有明确止损条件的研究重构；先证明强基线下仍存在结构与计算瓶颈，再投入新理论和大规模验证。修稿可以消除错误，OR 层面的创新性必须由这个新增研究结果来回答。
