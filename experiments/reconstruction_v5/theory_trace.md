# 按需因子与观测边界：保持规划对象，减少推断

## 已有实现的轨迹等价结果

冻结同一公开历史、参数、物理动作、尾策略、模型世界数及 transition 预算。按原 `_trips` 插入顺序，为每个 trip 先分配原形状 `rng.random((n,T_trip))[:,::-1]` 的完整 random tape；之后的 FFBS 只读取该 tape，不再消耗规划 RNG。名义事件是同一 conditional law 的确定函数。于是，惰性车辆/名义 mapping 对每次请求返回 eager 实现的同一值，规划程序下一步的控制流、下一次请求、UCT 随机数和 transition 计数也相同。对请求序列归纳，得到同一逻辑规划 trace、动作和估值。未访问的世界变量仍由预留 tape 定义，并未删除候选或替换为假事件。

该论证不要求预先知道相关 trip；公共消息的路径外车辆、first-five 遮挡因子和循环线过去段只要被执行器/查询访问，就必须原样生成。True 的局部公告可能扩大访问集合；False 的空 publication map 则在取车辆事件前直接返回空批。已经观测而出现在 `_revision` 的 trip 也计入预先构建集合，不能只数被采样的 trip。

[完整城市 parity 回执](runtime_bank_parity_results.json) 对四个模型世界的111,580个 VehicleEvent逐项一致，采样后的 RNG state一致，全部连接的名义均值/取消概率误差为零。这是该固定查询上的计算证据；lazy 调度本身若使用原运算可以保持浮点值，新 separator 运算则只证明代数等价并另报数值误差。计入实际计算延迟后，较快实现会改变执行时钟，不能再声称两种实现的整趟现实旅程 trace 相同。

## 工作依赖什么

设 trip r 有 T_r 个事件 slot，每slot网格宽 M_ri。random tape 和静态元数据仍需 `O(n∑T_r)`，不是与全城规模无关。令 A 为本次所有 branch 中实际请求或已触发 revision 的 `(trip, revision, parameter key)` 集合。现有完整 trip 推断的稠密运算量依赖

`∑_(r,v)∈A ∑_i M_r,i−1 M_ri`，

而非每次调用都对所有 trip 求 HMM。跨分支命中、共享不变 kernel 和名义事件缓存另减重复运算；名义最早到达仍要支付实际访问的连接扫描、字典取值与队列操作。route 参数变化须把受影响 route 的全部相关 trip 视为新 key。该式不省略输入、网格、随机数预分配或 branch copy 的代价。

**80不是总推断trip数。** [归档首query profile](profile_lazy_localcache_2023-07-26_q0.json) 的320次FFBS/四世界表明只生成80/2,173个trip的采样事件；但名义尾策略使缓存law达到1,390个trip、名义事件13,606个，包含1,828次marginals计算。该query还有2,055,948次mapping读取和440次nominal tail；名义查询16.42s累计耗时，采样FFBS仅约0.20s。此profile早于后续flat-cache修复，不能当作最新生产耗时，但它足以否定“所有推断只访问80个trip”的说法。

## 可检查的观测边界

固定一条 trip 的全部网格和参数后，状态 `Z_i=(potential clock, service)` 构成链。若某slot k的完整状态被精确观测，则

`P(Z_>k | Z_k, evidence_≤k, evidence_>k)=P(Z_>k | Z_k,evidence_>k)`。

这是直接按链因子分解并约去过去似然的结果。查询 k以后的事件可从该点质量δ重新前向传播；k以前的查询仍需其对应prefix与右侧证据。只有 operating/canceled 状态已知、时钟未知时不是单点separator。未来同车远端forecast是非平凡emission，仍须后向传播。

若一段后缀没有非平凡emission且每个转移行和为1，`β_last=1` 逆推得整个后缀 `β_i=K_i·1=1`，无需后向矩阵运算。不能跳过有全零行的substochastic后缀。当前 ordered Gaussian 核在两种availability状态各有合法后继时按行归一化；在全局相同shift/anchor集合下，非递减scheduled slot时钟保证后继最大网格时钟不小于前驱。July26/27各2,173trip的静态slot序列均无倒序，记录在[边界回执](chain_boundaries_results.json)。

全局evidence必须可行。利用本模型“服务转换许可且后继时钟不早于前驱时，Gaussian潜势严格为正”的结构，可以不建稠密核就检验：每slot、每availability状态只保留最早可达clock；后继clock c可达当且仅当某个许可前态的最早clock≤c且当前emission有限。对slot归纳得这两个最早clock是完整布尔可达性的充分量，时间 `O(∑M_i)`、额外状态O(1)。它维持no-path检查，不能被“真实数据应当自洽”替代。

**adaptive grid仍保留完整历史元数据。** 当前 `_grid` 把同trip全部已知时钟/shift加入每个slot，过去一个新的off-grid观测会改变future网格。因此 separator 消掉的是固定网格下的过去概率运算，不是过去anchor元数据。测试中同一精确当前arrival、仅增加更早1秒的已知departure，就改变future网格及均值；忽略它并复用旧suffix不是等价优化。参数或网格改变需新factor，不能沿用另一网格的缓存。

## 独立reference与实测边界

`chain_boundary_reference.py` 提供固定factor下最近separator/最后emission的单marginal计算和线性可行性检查，不触生产源码。一次查询的稠密传播只覆盖最近left separator到query以及query到最近right separator或最后非平凡emission/非随机行之间；大小为该区间 `∑M_i−1 M_i`。reference尚未共享多query的部分messages，因而不虚报跨query缓存收益。核的row检查也明确计时。

53个归档城市公开状态中有52次连接动作，46次为已知同trip精确arrival之后的续乘。选两个实际trip作核/概率对照：

| 公开状态 | 当前完整trip前/后向步数 | 所选departure / arrival的边界步数 | 全slot概率最大差 |
| --- | --- | --- | --- |
| July26 q0首次选择8341，60slot | 59 / 59 | 前向20 / 21，后向均0 | 1.7e−16 |
| July26 q0第5决策选择11306，66slot；slot55已精确 | 65 / 65 | 前向1 / 2，后向均0 | 7.3e−16 |

此次独立进程中，完整law构造约4.49/11.52ms，完整smoothing2.92/3.89ms；两事件边界查询合计约2.18/0.26ms。reference为验证先构造完整核，row检查另花2.59/3.58ms；线性可行性检查4,200/5,401个网格状态约0.63/0.66ms。这不是生产端到端加速结果，冷态收益也没有已知prefix时明显。选择规则是首动作和已有精确prefix最靠后的实际动作，旨在检验边界，不估计平均收益。[源码与回执](run_chain_boundaries.py)

7项测试覆盖future informative offpath emission、部分service观测、零后继行、adaptive anchor，以及80组随机小链与稠密布尔支撑传播的逐例一致。既有所有动作、信息与已见负结果保持不变。生产接入应先做同公开输入的query parity，再做已获授权的匹配预算和计算延迟实验；不能由矩阵步数直接换算旅客收益。

## 最近方法与具体增量

[Silver–Veness, POMCP](https://papers.nips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html) 已基于黑盒生成器在POMDP中采样规划；[Madsen–Jensen, Lazy Propagation, UAI1998](https://arxiv.org/abs/1301.7398) 已按证据诱导的独立性延迟势函数计算；[Pfeffer等](https://arxiv.org/abs/1509.03564) 给出通用lazy factored inference；[Murphy的DBN论文](https://www.cs.ubc.ca/~murphyk/Thesis/thesis.html)系统覆盖前后向与精确/增量推断。故“按需生成”“Markov separator”“β=1”都不能作为新一般性结论。

这里可核验的具体增量是：实际 committed 选择/消息/UCT合同下保留原random tape的实现等价、clock加service的完整观测何时真实可得、ordered二availability支撑的线性可行性检查，以及不丢adaptive grid元数据的事件查询边界。46/52续乘记录为这种局部计算提供真实工作负载，弱transport closure并非唯一计算入口。是否足以成为独立OR方法贡献，仍取决于与这些先例的具体不可替换差异及匹配的完整计算证据；此处交付能继续验证的实现与边界，不将普通HMM性质改名为新定理。
