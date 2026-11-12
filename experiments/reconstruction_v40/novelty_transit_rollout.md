# V40：交通自适应路径与 rollout 近邻核查

核查日期：2026-11-12。只使用作者原文、作者机构库和出版方记录；未运行新的控制实验。

当前证据支持把 V37–V39 定位为**具有具体消息语义和精确实现缩减的标准 rollout 基线**。上车／等待比较、途中下车换乘、完整基策略续程评估、无关状态消除与共同随机数，各自都有直接先例。下面的区别是对原文与本地实现的比较判断，不能由文献检索直接推出“所有可能贡献均不存在”。

| V39 操作 | 已核查的原始依据与位置 | 实际区别及方法判断 |
| --- | --- | --- |
| 当前登乘与分别等待其他候选比较 | Liu、Blandin、Samaranayake，*Stochastic on-time arrival problem in transit networks*，2018 年作者稿 v2，§4.3 **Heuristic 2**，PDF 第17页：当前线路效用不低于逐条等待任一其他线路的效用时登乘。[原文](https://arxiv.org/pdf/1808.04360v2#page=17) | 本项目具名车次、截断到达损失和消息后验不同；“当前车对单目标等待”的比较结构已有先例。不能将更完整的数值积分本身视作新决策原理。 |
| 逐站决定继续乘坐、下车和换乘；维护尚可等待候选 | 同文 §2.1–2.3，PDF 第4–8页，明确包括 boarding/alighting/transferring；站点、到达、线路节点刻画候选集合与到达后选择。§3.2、§4.1，PDF 第12–13页，还讨论分布量复用及不可行路径删除。[原文](https://arxiv.org/pdf/1808.04360v2#page=4) | 其频率线路、首班候选、准时概率目标与 V39 并非同一模型。V39 保留同一 physical trip 的条件链及完整消息来源是具体扩展；“自适应下车”和“相关子图”这些概念本身不能重新立项为新颖性。 |
| 基于实时车辆信息的状态及不可达状态缩减 | Rambha、Boyles、Waller，*Adaptive Transit Routing in Stochastic Time-Dependent Networks*。作者 **TRB 2013 会议稿** §2.1、§3、§4.3，PDF 第5、8–14页：逐车状态、可达性预处理、分车计算转移及复用查表。[作者全文](https://sboyles.github.io/research/2013rambhaboyles.pdf#page=14)。同题 2016 年期刊文献的 MDP／预处理贡献由[出版方摘要](https://pubsonline.informs.org/doi/10.1287/trsc.2015.0613)确认。 | V39 的带噪预测、非网格有序证据、原始预测重选和固定剩余菜单，不等于该文的已观察车辆状态。现有后缀／无关 trip 删除属于针对这些语义的精确实现结论；尚未显示新的可解结构、复杂度阶改进或独立控制方法。 |
| 多个车次及其下车站组成菜单 | Bérczi、Jüttner、Laumanns、Szabó，*Arrival time dependent routing policies in public transport*，2018，§4、§5.1–5.3，作者机构库存稿 PDF 第6–7页：有限候选策略 DP，以及 `(service, alighting stop)` 的扩展与相关服务过滤。[原文](https://real.mtak.hu/86687/1/1-s2.0-S0166218X18302920-main.pdf#page=7) | 该文的 `(s,w)` 指到站选定服务及下车站，不应说它已经实现 V39 的登乘后逐站重选。它证明“服务＋下车点菜单”已有先例；更强的后续适应仍须通过强同信息基线分离其新增收益。 |
| 枚举当前动作，随后完整执行共同便宜策略，滚动重算 | Bertsekas、Castañon，*Rollout Algorithms for Stochastic Scheduling Problems*，1999，§4、式(4.2)–(4.5)，期刊第99–100页（PDF 第11–12页）：基策略成本、一步改进，以及首动作后按基策略模拟完整轨迹的 Monte Carlo 计算。[作者原文](https://web.mit.edu/dimitrib/www/quiz.pdf#page=11) | V39 在收到的消息快照下比较 ALIGHT／CONTINUE，再执行 combined_rolling，正是这一架构。保留完整尾程、用128个世界或加速求值，都没有改变该控制原理。经典精确期望改进性质不能直接转移给有限样本、未解投影质量或实际模型误差。 |
| 两动作共享物理世界，比较配对差值 | CRN 至少已有 Wright、Ramsay 1979 的原始研究，且并非必然降方差。[出版方摘要](https://pubsonline.informs.org/doi/10.1287/mnsc.25.7.649)。更近的 Yadav 等，*Using Common Random Numbers for Simulation-based Planning with Rollouts*，2026，§1.2 Proposition 1、§2 Theorem 2，PDF 第4–5页，区分全共享与按深度共享的估计器。[原文](https://arxiv.org/pdf/2605.04732#page=4) | V39 的每车一次抽样和两臂共享是保持物理依赖并做 CRN 比较；没有实现该文 XDD 构造，也没有其方差保证。不可把“相同随机数”直接写作新的估计方法或普遍更高样本效率。 |

本地对照落点是 [V39 理论](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v39/THEORY.md)、[当前决策引擎](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v39/state.py)、[完整续程评估器](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v39/evaluator.py)及 [V39 汇总](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v39/summary.json)。fullmemo 与 frontier 的对照正确地分离了普通动作缓存和剩余状态投影；局部非网格及预测重选反例则说明为何不能随意删除活跃车次内部历史。这些成果应保留为可核查的模型契约、实现技术和负面证据。

论文方向的判断是：**这组文献对照尚未找到足以独立支撑算法论文的新增方法机制。** 完整质量—预算结果可以证明优化后的标准 rollout 在该模型和菜单下是否实用；即使表现良好，也还不是对上述先例的独立方法突破。若 V40 的另一部分核查同样只留下等价工程加速，应按本轮要求收束当前算法扩展路线，保留可信执行器、强基线、全部正负案例和数据／信息条件分析。

范围说明：这是面向当前主张的近邻核查，不是整个交通与 POMDP 文献的穷尽检索。Rambha 的详细页码来自公开2013会议稿；2016期刊全文未取得，未将两个版本的页码混用。Yadav 原文通过 arXiv v1 核查，条目自列 *Reinforcement Learning Journal 2026*；不额外推断其审稿状态。本文没有据此判断本项目不可能产生建模或经验洞见，也没有新增实验预算。
