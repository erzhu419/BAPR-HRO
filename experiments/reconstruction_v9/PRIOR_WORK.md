# 全程时序割下界与已有工作

把站点与时刻作为展开图节点。给定公开历史 $H$ 和站点阈值 $\ell_v$，令晚侧为 $\{(v,t):t\ge\ell_v\}$，并使当前状态在晚侧。任何在 $\ell_d$ 前到达终点 $d$ 的轨迹必有第一条离开晚侧的边。这条运行边满足发车 $b\ge\ell_u$、到达 $a<\ell_v$。等待和非负停站时间无法越界，零时长边仍适用相同判据。同车继续乘行的边也计入，包括不对新乘客开放的段。步行边同样使用越界判据，也可事先取步行最短路闭包，使 $\ell_v\le\ell_u+w_{uv}$。

按每辆车的有序后验链传播，在每个发车到到达转移中屏蔽该运行段的越界原子，保留其余转移及取消原子。所得剩余总质量是该车没有运行边越界的概率。各车在 $H$ 下独立时，乘积给出所有割边均不可用的概率，也是任意信息策略的 $\Pr(Y\ge \ell_d\mid H)$ 下界。这里 $Y$ 是在截止时刻 $D$ 截断的到达时刻，阈值满足 $\ell_d\le D$。若给定共同潜变量时各车独立，可先求积再积分。只知道条件边际时可用并集界。

多组阈值对时刻 $s$ 的共同生存下界是所有 $\ell_d\ge s$ 对应概率的最大值，再按整数时钟求和。分量递增不足以保证割边集合嵌套，因为发车下限与到达上限同时改变。

## 原文对照

- Sartor–Robledo §4 用所有边均失效的 $d$-cutset 事件限制可达距离，再据此降低模拟方差。时刻展开图上的越界证书沿用这一网络可靠性原理。 [A Simulation Method for Network Performability Estimation](https://arxiv.org/pdf/1303.5302)
- Nakahata–Arizono–Kasahara §2.2–3 在固定时间标签和逐边独立失效模型上，使用 BDD/ZDD 精确计算全路径的时序可达概率。当前时序割只计算一个充分的不可达事件。任意一辆车的随机时刻会共同控制多个展开边，因而使用同车后验链的联合质量。 [Computing Time-varying Network Reliability using Binary Decision Diagrams](https://arxiv.org/html/2504.01339v1)
- Parmentier–Meunier §4 用分布递推构造独立随机弧长下固定路径的随机序下界。Niknami–Samaranayake §3.1 用自适应策略的准点概率上界固定路径的准点概率。时序割给定各车的联合时刻实现，直接约束所有可执行轨迹。 [Stochastic Shortest Paths and Risk Measures](https://arxiv.org/pdf/1408.0272) · [Tractable Pathfinding for SOTA](https://arxiv.org/pdf/1408.4490)
- Liu–Blandin–Samaranayake §4.1 用各边最短可实现时长的静态最短路排除零概率路径。公交重规划中的 envelope 用静态最短路时长和确定性时刻条件筛边。 [Stochastic on-time arrival in transit networks](https://arxiv.org/pdf/1808.04360) · [Dynamic Replanning for Improved Public Transport Routing](https://arxiv.org/html/2505.14193v1)

可检验的增量是对公交有序端点模型的全程割概率及其计算成本。首退出界已经使用同车链掩码。这里增加的是完整旅程上的时序约束，新增证据取决于同预算下的下界收紧、可接受间隙和实际省下的规划时间。
