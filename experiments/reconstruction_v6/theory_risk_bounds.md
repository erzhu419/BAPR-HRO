# 精确首分支、固定救援和换乘时间门

`risk_bounds.py` 对当前完整有限网格概率模型积分，不把四个世界的均值作为总体上界。范围是无actual occurrence feed、固定route参数、独立trip的原有序链；同trip时钟/服务相关、forecast、等车证据均保留。网格条件化仍沿用既有近似。

首动作ready为r、12分钟窗口终点f=min(D,r+P)、catch事件C。因pub{}下所有首失败均在原站f返回，

`L_commit=P(C)E[min(A,D)−t|C]+P(Cᶜ)(f−t) ≤ Q*_full(h,a)`。

这是精确首步成本加非负后续成本；直接到goal且失败停至D给精确可行U。7588的解析失败质量.547801进入下界，不能被四个样本全catch抹掉。

对每trip只出现一次、组内连接事件连续的固定itinerary，使用后验Markov边`K_i e_(i+1) β_(i+1)`逐行归一化，传播未归一的成功arrival pmf。跨trip入口将incoming pmf变为departure可乘质量：对ready clocks排序并用窗口prefix sum，避免arrival×departure×后续clock三维张量。WAIT60相位使ready(t)非单调，不能直接按arrival排序。固定等待阈值来自本次公开history的nominal值，未来实际arrival只决定合法WAIT相位；不使用未来再估计的阈值。

令`G(path)=E[(D−A_goal)1(all legs succeed)]`。全部后续失败停到D的可行策略值是`U=D−t−G(primary)`。若仅首失败切换预编译rescue，且rescue排除失败trip，则独立性给

`U_rescue=D−t−G(primary)−P(Cᶜ)G(rescue at f)`。

根失败之外的后续失败仍停至D；执行器在此时改用任何不超过deadline cap的补救只会降低损失。`L_commit+P(Cᶜ)V(rescue)`只约束固定救援的策略族，不能冒充全因果最优的下界。基本消息运算量是路径trip所需HMM边的`∑M_i M_(i+1)`，加incoming排序/窗口查询；已有future evidence所需前后向工作也计入。

**换乘时间门。** 已在同车r、fresh arrival t，给定连续同车suffix至goal。令`T=min(D,t+沿途最小MCT)`，要求T≤t+P，所有suffix scheduled lookahead当前已合法；另检查沿途同车其他出边不能在[t,T)构成shortcut（依据已知不可用、时间支持和menu opening）。宏立即尝试每段，不采用nominal等待。任意首次离车策略即使先知也不能T前到goal；仅留在当前车的可选前缀受连续slot顺序约束。宏成功到B时最优损失至少min(B,T)−t，取消/超窗口失败时至少T−t。因此成功pmf r(B)、质量p给

`L_gate=Σr(B)(min(B,T)−t)+(1−p)(T−t) ≤ V*_full ≤ U_macro`，
`U_macro−L_gate=E[(B−T)₊;success]+(1−p)(D−T)`。

该证明为同一transfer-buffer合同下的pathwise比较，允许其他trip任意相关、任意未来信息；只需当前车的条件链分布。已支持共享日偏移和每trip独立偏移的混合后验。拓扑/支持检查不成立返回None，不以高概率“假设无shortcut”。图中MCT采用GTFS站内最大声明值或未识别时的120秒假设，没有验证任意实际platform pair的物理minimum；oracle也仅指同一合同，不能外推无约束的真实物理最优。

三个完整city roots的132/125/132动作解析下界各约.38/.25/.38秒；q5的direct-stop完整模型U=83.533秒，可删123/125动作，WAIT的60秒下界保留。53个旧公开状态中45个有同车goal suffix，39个满足结构；4个模型regret≤1秒，5个≤20秒。q5的`L/U=82.981/83.533`，界宽.552秒；其余紧末段约.060/.954/.260秒，两段正例8.420秒。大多数长suffix仍界松，未冒称46/52都可跳算。[最终冷上下文回执](theory_risk_bounds_final_results.json)；早期warming pilot和冷α版本分别保存，不用于最终成本结论。

11项测试通过：完整小链枚举和exact causal Q、多trip固定救援、同车连续段、WAIT相位、未来forecast、取消恢复、unsafe同车shortcut，以及两段gate对worldwise clairvoyant路线仍成立。

近邻：[IS-DESPOT](https://adacomp.org/wp-content/uploads/2020/08/ijrr19isdespot.pdf)已有重要性采样处理关键稀有场景；本实现的分支积分不是新一般抽样定理。[Kong–Indelman2026](https://arxiv.org/abs/2604.01352)已有带界open-loop与安全跳重规划框架；MCT门提供的是本交通合同下无完整future tree/其他trip搜索的具体oracle下界，不能再宣称macro或上下界框架首创。[Geissmann–Gianinazzi2019](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.ATMOS.2019.4)给独立edge失效的近线性策略；这里保留同车相关，gate比较本身也允许跨trip相关。

[CSA–MEAT2018第6节](https://ben-strasser.net/paper/connection_scan_algorithm_acm_j_exp_algorithmics.pdf)已计算包括错过换乘后替代车次的decision graph，故预编译backup本身也不是增量。其§6.3的decision graph要求各leg存在保证到达的safe suffix，§6.4先计算所有connection的最小期望到达再提取图；这里允许取消/尝试失败并以固定deadline截断，新增积分保留同trip有序事件相关和历史条件化，首失败在原站已知f返回时才可用上述独立trip救援分解。该差异限定了可比较的概率/执行合同，不自动证明比MEAT或strict native TopoCSA更强。

限制：这些是当前网格概率模型与固定策略族的结果，真实日期的条件校准仍需独立证据；直接执行后改变网格或重新规划不自动保留整趟regret保证。冷成本包括单车链积分，但不等于图/当前信念构造零成本。整体方法是否优于compiled nominal和低预算UCT由根任务的匹配回放判断，不能由这四个紧界状态决定OR创新门槛。
