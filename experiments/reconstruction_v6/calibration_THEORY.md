# 条件风险、选择与同一联合模型

分组是决策时已知的incoming实测延误≤180秒/>180秒。每个日期d、每组g，取该组预测非合规分数的第`ceil(.9n_dg)`个顺序统计量；日期分数`S_d`是这些组分数的最大值。九个校准日期给`q=max_d S_d`。固定拟合预测器后，若校准日与下一日的完整块可交换，下一日分数在十个日期中的秩给出`P(S_new≤q)≥.9`。因此，概率至少.9时，下一日每个存在的组有至少90%的输入被覆盖。日内观测可以依赖；这不是每个日期、每个incoming值或每个旅客的条件保证。日历和未观测共同冲击仍可能破坏可交换性。

初次实验以已测operating target的pair为单位，覆盖分母排除了无实测时钟的target。它没有保护根据估值选择某个target的优化器。第二次实验先固定两种目标，再开新标签：固定selector取最大冻结catch概率；menu-max则对同一incoming的全部候选取最大非合规分数。若menu-max集合覆盖，则任何选中该菜单元素的selector都被覆盖。对incoming再按上述date/group秩校准，保护的是保留输入状态群中的首选择；它不自动保护自适应旅程访问状态的分布，也不是宏后悔期望上界。

结果是`{CANCELED} ∪ {operating, clock∈interval}`。明确取消是单独服务类别，不能当无限延误；该集合包含它，不意味着识别了取消概率。未知target给分数区间`[-∞,+∞]`。固定selector先由输入选定再评分；menu-max分别最大化各target的分数上下界。顺序统计量和日期最大值都单调，得到校准阈值的可识别上下界。采用上界；若为∞，如实给vacuous集合。测试日期coverage同样给lower/upper，不删除未知动作或把它们当已知失败。

有限可观察分组及conformal predictive distributions已有理论：[Vovk2012](https://proceedings.mlr.press/v25/vovk12.html)、[Vovk等2017](https://proceedings.mlr.press/v60/vovk17a.html)。无分布假设的任意个体条件覆盖存在限制：[Barber等](https://arxiv.org/abs/1903.04684)。本轮具体贡献是纠正本项目的统计对象、选择人口与缺测处理，并完成匹配验证；不把group/max-score秩论证冒充新一般OR定理。

## 同一个当前网格联合律

不能把新失败概率乘旧条件补救均值。给定当前公开H及冻结trip链`P0(Z|H)`，目标departure状态j边际为`p0(j)`。把条件operating CDF按当前clock cells投成质量、限制在`p0>0`支撑并归一；服务/取消边际保持原模型，得到`p*(j)`。乘一个emission `λ(j)=p*(j)/p0(j)`，则

`P*(Z|H)=P0(Z|H) λ(Z_j)`，`P*(Z_j)=p*`，`P*(Z_−j|Z_j,H)=P0(Z_−j|Z_j,H)`。

所以新的名义时钟、概率、FFBS和后续成本可来自同一完整链。每个trip至多替换一个边际；Evolving的独立trip可同时投影，定义一个供全部actions共享的`P_H`。不能逐action选择不同的`P_a`再拼成同一最优证书。

`TripMarginalTilt`固定根时的grids、initial、transitions和λ；未来分支只把新增事实的emission限制到原网格上。于是是对同一有限先验的条件化，virtual world不会暗中进入公开belief。原Evolving会因新观测在所有slot插入anchors，即使观测值来自原slot网格，也会改变其他slot；因此不能直接沿用其adaptive prior并声称固定λ的一致Bayes更新。实际新H出现off-grid事实时另开模型投影；跨H重算λ不保证来自一个全局Bayes先验。

已知departure、取消、已有attempt window和零operating支撑分别记录为skip，不覆盖事实。路由参数在一个投影上下文内冻结。取消质量和已知mask保留，私有copy只更新自身事实。连续CDF、投影后的catch质量及预测区间分别验证；60秒格的边界量化可产生约.22的catch概率差，不能将连续区间coverage移贴为离散`P_H`保证。
