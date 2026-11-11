# 基线与最近方法的对应

Durner 原文已核对到模型定义、transfer strategy 和 Algorithms 5.1–5.4（印刷页32–34、44–46、50–59）。新实现为 `src/durner/native.py`，独立实现 strict 支配及最低换乘时间切环/回退。它恢复了一个可检查的原生策略参照，不把已有算法包装为新贡献。

| 方法 | 实际对象 | 本项目需要遵守的边界 |
|---|---|---|
| Durner strict port | 独立到发边际分布、显式取消率、指定 strict transfer strategy；成功条件终点均值 | 同车连续区间可续乘；Algorithm 5.4 同时比较终点均值和发车均值，发车均值相等时保留；不是一般策略最优，也不直接优化120分钟截断损失 |
| 当前 `topocsa.py` | 保留全部 label，但继续用不含 `P_after` 的递推 | 既不能称 strict 原算法，也不能称 fuzzy 原算法；fuzzy 需要原文式(4.7)的额外排序概率 |
| 精确信息树参照 | 在同观察历史下行动相同的有限模型最优策略 | full/H 比较必须冻结全部未来候选规则、共同目标与信息；不能逐场景事后选最优路径 |
| 目标一致的重规划/固定候选 | 同一因果模型、同一旅程损失下改变候选维护 | 才能判断候选限制损失；不能把原生策略的不同目标混进候选归因 |
| K-adaptability、HSVI | 预存策略及上下界驱动局部展开已有先例 | 有限策略库存和 `U−L` 本身不能作为新贡献 |
| 状态压缩、影响图、相关随机路由 | 保留决策信息、因子依赖、信息揭示与相关更新已有先例 | 需要额外可检查结构和计算优势；“共同事故+belief”不足以主张新颖 |

本地 `TransitGraph` 没有单独的实时预测标量；native port 用输入PMF均值作为切环所需的预测时间代理，查询窗口按可能发车支持选取，完全取消连接移出可行动作。无步行边的本地图只验证无步行子类。调用方须提供归一化的条件PMF及单独 `cancel_prob`，旧版缺失质量表示不能直接传入。以上是明确的接口适配，不声称复制作者全部工程实现。

`src/native_router.py` 将该策略接入固定车辆场景执行器：使用与匹配算法相同的先验、在线均值/取消更新、5个候选和25分钟窗口，保留原生条件终点均值排序。缺失质量只转换一次为单独取消概率，条件PMF归一化。该列是不同目标的次要策略参照，不参与“保留/重算候选”的主归因。双trip固定场景执行与取消不重复计数的两项回归已通过。

Durner 已明确承认独立到发模型可能产生不符合真实物理的时序，以及 transfer strategy 本身的简化（第33、45–46页）。在固定车辆场景、相关事故及延迟公告模型中，可以用该策略作为同环境的可执行比较，但不能把其递推值当作同信息精确最优或质量证书。切环可能删除真实备选，需保留 cut 记录。

小样本数据另见 `experiments/reconstruction_data_audit/field_audit.json`。仅凭旧图31个公开线路名，有28个存在多route/agency歧义；随后改用精确 `Connection.trip_id → trips.txt → route_id → agency_id`，189个trip、2000个连接全部匹配，无缺失或歧义（`graph_identity.json`）。31个图线路名中有2个实际包含多个route ID，逐连接身份可分开，不能仍按名称混池。身份问题已解除；采样feed仍是预测/已公布状态，不能据此宣称真实事件回放。

原文：[本地PDF](/home/erzhu419/mine_code/BAPR-HRO/reference/Stochastic%20Strategies%20for%20Public%20Transport%20Journeys%20Based%20on%20Realtime%20Delay%20Predictions.pdf)。原作者实现地址载于论文第50页：https://github.com/traines-source/stochastic-journey-strategies 。
