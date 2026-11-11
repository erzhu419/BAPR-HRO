# 首次承诺分层：同一概率模型，显式保留失败分支

`FirstBranchSampler` 对当前固定grid posterior的departure slot取 `C={operating,r≤D≤min(T,r+12min)}`，直接相加raw marginal取得p。仅在私有FFBS的该slot emission乘`1_C`或`1_notC`，不改变grid/参数/公开belief；乘积因子即为原joint law的精确条件分布。整条目标trip的clock、arrival和service关联保留；给定固定参数时其他trip独立，仍按原posterior生成。稀有分支不做拒绝采样，成本不含`1/p`。此前prefix forward和只读transition可复用。

`FirstBranchBank` 一次预留原trip×world×slot uniform tape，原scenes和采样后RNG与Evolving逐项相同。测试一个action时只把目标trip的同一U送入条件FFBS，其余trip逐对象复用原world。默认按`round(n*p)`分配且每正质量分支至少一个world；配额在检查任何rollout loss前确定。零质量分支的全部配额转给另一支。当前支持Evolving与OrderedDuration；共享日模型会被明确拒绝，不能忽略其branch-dependent day posterior。

对任意固定的公开历史尾策略，`E[G]=p μC+(1−p) μF`。分层估计用这两个真实模型质量加权条件样本均值，故无偏；尾策略从原公开belief开始，只在实际模拟反馈后更新。branch mask不能进入该尾策略。若独立分支样本数为nC/nF，方差为`p²σC²/nC+(1−p)²σF²/nF`；IID方差为`[pσC²+(1−p)σF²+p(1−p)(μC−μF)²]/n`。仅精确比例`nC=np`时自动省去between-stratum项，强行2/2没有普遍降方差保证；整数舍入也须实测。[Owen原作者教材§8.4](https://artowen.su.domains/mc/Ch-var-basic.pdf)已给出这些经典结论，本轮不称新一般理论。跨action复用U也不保证任意两行动差值方差下降。

固定July27 q0根（July25训练、全部27,895候选），比较7588/8341及同一公开nominal尾策略。原种子四世界给660/1035秒而选7588；两者解析可追概率分别.45220/.87781。独立分层四世界给1204/1078，复用U的分层给1114/940，均修复该单次选择；实际7588已走而8341可赶。此处只比较两个动作，没有把它称完整网络最优。

另预锁8个模拟seed、n=4/8，分别做IID、独立half分层、复用U half、复用U比例分层，共136个完整估计。每action最多56次transition，两个action合计也低于512；未删慢世界或截断失败rollout。fresh平均成本约2.7–3.8秒/action，完整世界数相同，实际work另报。[原IID/分层68回执](branch_sampling_results.json) · [复用U/比例68回执](branch_coupled_results.json)

| n / action | IID估值方差 | 独立half | 复用U half | 复用U比例 |
| --- | ---: | ---: | ---: | ---: |
| 4 / 7588 | 56234 | 34176 | 53163 | 53163 |
| 4 / 8341 | 39793 | 27353 | 62862 | 41245 |
| 8 / 7588 | 25389 | 19735 | 7650 | 7650 |
| 8 / 8341 | 22836 | 32201 | 37634 | 18236 |

n8独立分层的conditional成本约为7588成功724/失败1663秒、8341成功1168/失败1819秒，加权估值约1238/1248。两者差远小于当前Monte Carlo误差，因此修复p并不自动修复行动排序或实际旅程；成功分支内部仍有较大的后续成本变化。136次只分析同一个已用根的模型内抽样，8seed方差不是独立真实日期收益，也不足以断言任何抽样方式全面更优。

11项测试通过：tiny全路径概率及所有slot条件分布、off-grid7秒/未来forecast/负wait、q0和部分取消、极小operating mass、private history隔离、非目标trip不变、20秒CDF、无拒绝循环、零质量配额、原world/RNG等价和共享日误用拒绝。复现用`python3 -m pytest -q tests/test_first_branch_belief.py tests/test_first_branch_bank.py`。

回执冻结实际运行源码。第二批的relative output路径只影响最终metadata组装，68case已经完整；`branch_summarize.py`核对冻结case身份、权重和值后补生成汇总，未重跑或改变旧源码记录。后续`branch_sampling.py`已修output绝对路径；要重新跑应另选output，不能以当前源码覆盖旧协议。
