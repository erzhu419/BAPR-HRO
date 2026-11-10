**BAPR-HRO：OPRE 审稿核查与修改方案（2026-11-10）**

已读决定邮件、两份完整附件，并对照当前稿件、路由实现、实验脚本、已有结果及 Lean 定义。本次完成的是修改诊断；尚未修改论文、算法或实验结果，也未启动新实验。

**编辑决定与版本**

- 稿号 OPRE-2026-05-3065，题目 *Re-Ranking Stochastic Transit Hyperpaths Under Disruption: A Wasserstein-Robust Scoring Approach*。2026-11-10 的[决定邮件](https://mail.google.com/mail/u/?authuser=erzhu419%40gmail.com#all/1a1c5f39b5af12f8)是三位外审后的正式拒稿，没有返修邀请或返修期限。
- E.1：按拒稿后重构研究处理。E.2：编辑认为核心方法贡献不清楚。E.3：正确性、论述和不实引用共同损害可信度。现有核查支持实质修改，没有形成以审稿人误判为依据的申诉理由。
- 附件未明确标出审稿人编号，以下用 A 表示三页 `referee report.pdf`，B 表示两页 `OR hyperpath DRO.pdf`；邮件正文中的第三位审稿人用 R3。不臆测附件与 R1/R2 的对应。
- 原文留存：[邮件与 R3](/home/erzhu419/mine_code/BAPR-HRO/md/review_20261110/decision_email.txt)、[报告 A](/home/erzhu419/mine_code/BAPR-HRO/md/review_20261110/report_A_referee_report.txt)、[报告 B](/home/erzhu419/mine_code/BAPR-HRO/md/review_20261110/report_B_OR_hyperpath_DRO.txt)。
- 工作区已有大量未提交的论文修改，本次保留。当前 OPRE 源码和本地 PDF 已没有 `chen2024`，但 Git HEAD 的 OPRE 稿及 [paper.tex](/home/erzhu419/mine_code/BAPR-HRO/paper/paper.tex:3776)仍有该条；不能把当前本地 PDF 当作已确认的送审版本。

**对审稿意见的判断与具体处理**

| 意见 | 核查结论 | 具体修改 |
|---|---|---|
| A.1：后验 scale 被当作标准差 | 成立。[正文](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:508)所写 βₙ/(αₙκₙ) 是边际 Student-t 的 scale²；均值后验方差应为 βₙ/((αₙ−1)κₙ)，要求 αₙ>1。 | 统一公式、参数定义、概率界和实现。现有 [ssp_mdp.py](/home/erzhu419/mine_code/BAPR-HRO/src/ssp_mdp.py:94)在 αₙ>1 时已算对，保留；V1 的统计量另见下文。 |
| A.2、R3.4：仅 Lipschitz 不足以推出 DRO 等式 | 成立。[Corollary](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:945)和[V2 Proposition](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:658)都缺少其证明实际使用的仿射与平移可行条件。常数函数即可否定一般 Lipschitz 等式。 | 仅保留仿射代理成本、有限一阶矩、支持允许所需平移时的既有表示；加入 Esfahani–Kuhn 出处，取消新 DRO 定理定位。真实错过换乘与截断旅程成本另行定义。 |
| A.2、B.3：半径不消失、没有统计校准 | 成立。[正文](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:627)把 ε=βσ+γp_cancel 全体称为收缩；真实取消率为正时，第二项不趋零。 | 分开均值估计不确定性与取消成本；删除自动置信覆盖、整个半径消失以及据此 Bayes-optimal 的结论。相关研究中“无需校准”的说法也要删改。 |
| A.3、B.1：Bellman 收缩条件和证明错误 | 成立。[正文](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:997)写的是固定 W₁ 球的最坏期望；固定非空概率集合上的最坏期望在 sup norm 下非扩张，折扣后为 γ-Lipschitz。其所谓必要半径阈值不成立；附录下界被 V≡1 否定。 | 建议删去与实际排序实现无关的 Bellman 定理。若作为背景保留，就引用标准结论，并区分折扣 MDP 与本文未折扣 SSP；不再将 γ(1+ε)<1 写成必要条件。 |
| A.4：全旅程保证没有证明 | 成立。[正文](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:1058)没有建立局部误差假设，也未证明沿不同轨迹的最终到达时间差可以累加为旅程成本差；取消项和比较策略范围不清楚。 | 撤下当前全旅程上界、相应概率保证与 Bayes-optimal 收敛。局部选项比较引理可在明确假设、同一候选集合下保留。 |
| A.5：悲观性、EXP3、Adaptive-β 的保证越界 | 成立。[正文](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:1236)把正风险系数放进假设后当作不可恢复性的推论；宽松 regret 上界不能证明实际性能差；Adaptive 定理只是代数不等式。 | 删除不可恢复性必然要求正悲观系数、上界解释基线失败、Adaptive 收敛保证。已有下界/sandwich 同时存在量词与比较对象不一致，也撤下。比较方法优劣由匹配实验判断。 |
| B.2、R3.2：已知 OT/DRO 结果充当贡献 | 成立。仿射成本下的 nominal expectation + penalty 表示不依赖本文后验，更不单独证明所选 penalty 有用。 | 主贡献转向具体交通决策与增量证据；标准 W₁ 性质改为引用，DRO 表示压缩为一段或补充命题。 |
| B.4、B.7、R3.1：结构碎片化、写作与术语堆叠 | 当前正文在方法和理论中重复 DRO，夹杂 MDP、bandit、Lean、神经代理等多条主线。文风本身不能证明使用了 AI。 | 合并重复介绍，先讲决策问题、完整评分、输入输出，再讲证据。首次解释 SBB、cell-mean；移走与主要实验无关的神经代理及基线内部细节。 |
| B.5：Lean 占比过重 | 应接受，且另有语义对应问题。Lean [wassersteinAmbFunc](/home/erzhu419/mine_code/BAPR-HRO/submission_artifacts/anonymous_bapr_hro_lean_artifact/Wasserstein/DROBellman.lean:257)定义的是 E[V]+ε·lipV，并非正文的 W₁ 球 supremum。 | Lean 放补充材料，只说明实际证明对象。编译通过不能弥补正文、定义与实现间的缺口；删除“零 sorry 因而支撑整套性能保证”的论证方式。 |
| B.6：Bayesian 模型适用性欠缺 | 成立。目前不能用共轭计算方便来代替延误模型的经验适配证据。 | 对留出真实观测检查尾部、偏态和预测覆盖；明确均值后验与预测分布的区别。若只保留工程评分，则按实际统计量描述，不强行声称统计 DRO 覆盖。 |
| B.8：相关研究不足 | 所建议五项来源均可核实，见文末。 | 按其具体贡献补引：已知仿射表示、robust MDP、Bayesian DRO、数据驱动校准、封闭式风险鲁棒化；重写差异，不只扩充书目。 |
| R3.3：疑似不存在的 Chen 文献 | 定点精确标题及 Crossref 检索未找到可靠记录；旧稿确有该条。当前 OPRE 版删除是正确的。 | 在取得真实原文前该引用不可用；清理仍会用于投稿的源文件及依赖它的论断。另修正 [Lin et al. 作者](/home/erzhu419/mine_code/BAPR-HRO/paper/paper_opre.tex:3637)：Ren J 应为 Ren Y。 |

**项目里先改什么**

1. **统一 V1 的统计对象。** [RouteBeliefState.posterior_var](/home/erzhu419/mine_code/BAPR-HRO/src/bandit_router.py:78)是先验与观测方差的加权量，一般不会随样本增加趋零；它不是论文中的 Normal–Gamma 均值后验方差。建议保留旧结果及其方法定义，另做一个明确使用正确均值后验标准差的修正版，与原版及 β=0 的匹配版本比较。不能仅改公式后继承旧实验成绩。V2 也应按[实际组合统计量](/home/erzhu419/mine_code/BAPR-HRO/src/bandit_router_v2.py:85)写清楚，不能只称 ensemble std。
2. **修数据语义和模拟观察接口。** [构建脚本](/home/erzhu419/mine_code/BAPR-HRO/experiments/swiss_full/build_per_day_distributions.py:58)用 |delay|>30 分钟估算 `cancel_rate`；[解析器](/home/erzhu419/mine_code/BAPR-HRO/src/gtfs_rt_parser.py:30)没有读取取消字段；[模拟器](/home/erzhu419/mine_code/BAPR-HRO/src/simulate_bandit.py:105)又将 delay>25 标为 `true` cancellation。应分开显式取消、严重延误、未观测与乘客等待超时，先检查原始 feed 是否提供可用取消字段。GTFS 明确支持 CANCELED；缺少 trip update 表示缺少实时数据，不等于取消。[官方规范](https://gtfs.org/documentation/realtime/reference/)、[Trip Updates 说明](https://gtfs.org/documentation/realtime/feed-entities/trip-updates/)。
3. **如实命名旧实验。** [多日脚本](/home/erzhu419/mine_code/BAPR-HRO/experiments/swiss_full/run_multi_day_loo.py:42)从每日均值/标准差生成截断离散高斯，现有成绩属于“真实数据统计量校准的仿真”。“经验分布直接回放”“47 条线路真实取消”等表述不能由这条数据路径支持。获取真实取消标签后，改变环境分布的比较需重跑所有参评方法。
4. **先把 Adaptive-β 移出主贡献。** [runner](/home/erzhu419/mine_code/BAPR-HRO/experiments/swiss_full/run_multi_day_loo.py:109)每个 trial 新建 router，关闭共享状态；存档主实验与 LOO 结果中，Adaptive 与 V1 的 630 个 cell 完整记录均相同。[权重更新](/home/erzhu419/mine_code/BAPR-HRO/src/adaptive_bandit_router.py:164)也不是正文称谓下的 EXP3-IX。若未来保留，需定义独立序列、学习与评估阶段及重置边界，再评估实际算法。
5. **修正 recompute 比较的名称和解释。** [RecomputeRouter](/home/erzhu419/mine_code/BAPR-HRO/src/recompute_router.py:92)首次收到取消便切换到传入的全日分布，不是实际 BOCD；它与 V1 的评分、先验和筛选同时不同。原表不能证明“重算导致幸存备选被丢弃”。下一次比较应匹配这些条件，并保存具体候选变化与后续决策。

**哪些已有工作可以保留**

弱 Kantorovich–Rubinstein 上界、窄仿射情形的 DRO 表示及显式误差条件下的局部选项比较是有用的正确事实。Normal–Gamma 的正确实现也已部分存在。Lean 中抽象 Chebyshev 结果可以保留，但它假设了方差上界，不能说已验证实际路由后验。

已有 35 日、18 OD 的结果及 LOO 审计继续保存为历史证据。主表 V2 的 46.47 分钟与 Static 的 49.34 分钟来自旧实现和旧环境；不据此否定整个方法，也不当作修正后方法的成绩。已有 reliability retrofit 结果为 Static+A7 48.74、V1 47.44 分钟，说明值得进一步检查完整评分的增量；它还没有单独分离后验悲观项。

**最小补证据顺序与论文重构**

第一步先修上述定义与接口，并检查 Normal–Gamma 更新是否对应解析公式、取消/延误是否按事件区分。这些检查直接决定哪些旧比较受影响。对原始服务器资料先取日志与小样本汇总，确认字段后在服务器端提取需要的统计量，不搬运全部档案。

第二步在修正后的同一环境做一个核心对照：候选集“保留/重算” × 评分“可靠性项/可靠性项加在线后验项”，加原始 Static 参考。其余先验、可行性筛选、调参预算和可用信息一致。环境用同一外生场景实现，避免仅设同一随机种子却因方法耗用随机数不同而得到不同事件。对后验项再比较 β=0 与 β>0，分离实时均值更新、取消成本和不确定性惩罚。

在相同决策状态记录被重算移除但仍可行的候选，以及保留它们是否改变后续成本。若匹配对照没有优势，就删除“保留结构优于重算”的机制结论；若后验项没有增益，就把贡献落在可靠性重排本身。两种结果都应保留。

第三步核心对照有结果后，再评估原始 35 日面板和一个独立选取的多起点 OD 面板。调参/模型检查用训练与校准日，最终评估使用留出日；部署式在线主张应使用按时间划分的历史信息。报告按天配对差异及相应区间，保留正常日损失和失败案例，不把重复旅程当作独立日期。无需先扩充更多城市、神经网络或 bandit 变体。

建议题目：*Reliability-Aware Re-Ranking of Stochastic Transit Hyperpaths under Disruptions*。正文依次为：具体交通问题与待检验机制 → 候选集、事件和旅程成本定义 → 一套完整评分与在线更新 → 匹配实验及机制证据 → 讨论。DRO 作为短解释，Lean 与技术证明放补充材料。摘要中的收益数字待修正实验完成后替换。

下一步应先实现“数据事件语义 + V1 不确定性定义”的小范围修正，再做匹配试验。当前不宜先写宣称已解决问题的英文回复信，也没有证据保证换标题或补引用即可达到 OPRE 的贡献要求。

**已核实的参考来源**

- Mohajerin Esfahani & Kuhn (2018), *Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations*, Mathematical Programming 171:115–166。[出版社全文](https://link.springer.com/article/10.1007/s10107-017-1172-1)：Thm 6.3、Prop 6.5 支持相应仿射特例；统计保证需另外满足校准条件。
- Yang (2021), *Wasserstein Distributionally Robust Stochastic Control: A Data-Driven Approach*, IEEE TAC 66(8):3863–3870。[作者原稿](https://arxiv.org/abs/1812.09808)。
- Wiesemann, Kuhn & Rustem (2013), *Robust Markov Decision Processes*, MOR 38(1):153–183。[出版社](https://pubsonline.informs.org/doi/10.1287/moor.1120.0566)。
- Shapiro, Zhou & Lin (2023), *Bayesian Distributionally Robust Optimization*, SIAM J. Optimization 33(2):1279–1304。[出版社](https://epubs.siam.org/doi/10.1137/21M1465548)。
- Wozabal (2014), *Robustifying Convex Risk Measures for Linear Portfolios: A Nonparametric Approach*, Operations Research 62(6):1302–1315。[出版社](https://pubsonline.informs.org/doi/10.1287/opre.2014.1323)。
- Lin, Ren & Zhou (2022), *Bayesian Risk Markov Decision Processes*。[NeurIPS 原文信息](https://papers.neurips.cc/paper_files/paper/2022/hash/6f7d90b1198fec96defd80b5ebd5bc81-Abstract-Conference.html)：第二作者为 Yuxuan Ren。

**局限**

本次依据邮件、当前本地源码/结果和定点文献核验；未取得并确认投稿系统里的最终送审源包，也未重跑仿真或 Lean 编译。已有结果文件缺少足以唯一关联当前代码版本的运行记录。因此这里区分“当前代码的确定行为”“存档结果数值”和“尚待验证的改进”，未对全部参考文献作穷尽核查。
