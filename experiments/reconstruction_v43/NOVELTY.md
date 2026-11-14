# V43：误差归因与局部积分的最近先例

**本轮先把评价目标和近似来源说清楚，没有预先登记新算法贡献。** 完整 π 的两动作值仍由 V41 固定生成法则下的 rollout 评价；下一平台的精确积分只评价明确冻结的局部替代策略。

| 拟使用的工具 | 直接原始先例 | 本轮判断 |
|---|---|---|
| 反复决定登乘／等待，逐个目标等待比较及子集 DP | [Liu 等，*Stochastic on-time arrival problem in transit networks*](https://arxiv.org/html/1808.04360v2)，§2、§3、§4.3 Heuristic 2 | 该文已包括登乘、下车、换乘；目标等待与子集递推不是新控制原理。具名物理车、当前消息规则及截断损失是本实验差别，尚须产生额外结构结果 |
| 交通支配与算到足以判定即停止 | 同文 §4.2 Propositions 1–3 | 已有候选集合、子问题及剩余质量界剪枝；“只把排名算准”本身不能作为新颖性 |
| 对可积局部结构求条件期望，前缀继续采样 | [Murray 等，AISTATS 2018](https://proceedings.mlr.press/v84/murray18a/murray18a.pdf)，§1–2 | 属于条件 Monte Carlo／Rao–Blackwell 化工具。若局部策略或终值标记也改变，不能直接套用对同一目标的方差改善结论 |
| 按访问分布累计局部评价误差 | [Kakade–Langford，ICML 2002](https://homes.cs.washington.edu/~sham/papers/rl/aoarl.pdf)；[Scherrer，JMLR 2013](https://www.jmlr.org/papers/volume14/scherrer13a/scherrer13a.pdf)，§4 | 性能差与残差传播已有理论。有限截止的望远镜恒等式是分析工具，不是本项目新贡献；转移法则必须匹配原世界生成法则 |
| 估值误差小于动作间隙即可保序 | [Farahmand，NIPS 2011](https://proceedings.neurips.cc/paper/2011/file/013d407166ec4fa56eb1e1f8cbe183b9-Paper.pdf)，§1–3 | 已有动作间隙理论；本项目真正待争取的是便宜可计算、适用于交通信息更新结构的误差包络 |
| 难分辨状态多分配 rollout、显著优胜后停止 | [Dimitrakakis–Lagoudakis，2008](https://arxiv.org/pdf/0805.2015) 及 [配套算法论文](https://arxiv.org/pdf/0805.2027) | 已有自适应样本分配与优胜动作识别；本轮固定样本诊断不把这些原则重新命名 |

V42 的零净收益没有否证所有近似计算，但排除了继续通过同类会合检查寻找贡献。V43 检验另一个明确问题：完整下一换乘的评价与合法前缀更新能否解释剩余两动作排序差异。若有效，应先保留为更强基线或明确的工程方法；若只得到标准 DP、Rao–Blackwell 化、误差恒等式与间隙判据的组合，仍不足以宣布 G2 通过。

真正未完成的研究目标是：从当前菜单及证据构造小范围计算，并为其遗漏的完整续程差值给出便宜、有效的控制；还需要说明它超出上述一般方法的部分。未得到该构造或其净计算价值时，按本轮证据停止，不靠扩大样本、换日期或剔除负例使其成立。未来消息价值不是本轮替代分支，因为加入未来发布过程会改变固定模型和信息条件。
