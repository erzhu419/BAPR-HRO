# Continuation V2

修复空缓存提前终止、同车续乘被旧排序截掉、尾策略使用不可执行路径和辅助核公开未报告发车时刻。根动作包含一分钟等待。首次上车概率和直达成功条件成本由拟合后验解析计算，有限样本估计其余条件成本。只在首次承诺失败分支施加封顶消融。保留候选仍用同一 rollout 评分，包括缺少新 TopoCSA 标签的连接。见 [V2 理论](THEORY.md#v2-的首次承诺和尾策略)。

## 同一公开状态

- 主面板 002 的 q67 中，旧菜单遗漏的同车 68 可确定在第 46 分钟到达。按生成律固定历史尾策略估值，旧选项 1927 的补救均值为 72.26，拟合律下为 61.41，旧八样本为 76.875。精确局部直达问题中，根候选限制损失在拟合律和生成律下分别为 19.49 和 31.28 分钟。修复后用全图、原公开历史和八个样本重新查询，选中 68。
- 新增面板 001 的 q48 中，68 和 1304 在各模型下分别具有相同的首次成功概率和成功条件成本。生成律下历史尾策略的 1304 减 68 估值为 −0.029 分钟，近似蒙特卡洛区间为 [−0.129, 0.070]。精确局部直达问题中，承诺任一车的最优尾值相同，主动等待节省 1.11 分钟，名义尾增加 0.16 分钟。首轮全图试跑选了上车概率约 4.26e−70 的绕行 1771。最终实现将该动作排除，保留基础场景并在需要时额外补抽，复查给两班车相同的 47.87，等待得 47.63，选择等待。见 [共同状态估值](adverse_states.json) 与 [精确局部参照](local_reference_results.json)。
- 30 趟归档旅程上的同状态拟合对比支持保留在线均值与取消率更新。压力下等外生世界加权的非退化首乘概率 MAE，主面板从冻结先验的 0.320 降到 0.273，新增面板从 0.449 降到 0.255。无取消下平均差很小，主面板两个日期的方向相反。见 [拟合对照](route_fit_results.json)。

## 匹配旅程

旧的 30 个 OD 条件来自 10 个外生世界。V2 的 retain/refresh 与首次失败封顶/补救共 120 趟，全部到达。60 对 retain/refresh 中 59 对损失相同。主面板 020 的补救对照中，retain 比 refresh 多 43 分钟。第一次分歧时两者的当前候选相同，未来冻结菜单不同，导致 531 与 185 的排序互换。下表封顶和补救采用 refresh，单位为分钟。

| 面板 | V1 条件化补救 | V2 封顶 | V2 补救 | V2 名义策略 |
|---|---:|---:|---:|---:|
| 主面板 24 格 | 39.79 | 16.38 | 17.79 | 21.75 |
| 新增生成 6 格 | 25.83 | 14.83 | 14.83 | 15.17 |

V2 补救的两个旧负例分别从 120 降到 19、47 降到 18。相对 V1 补救还保留主面板 019 的退步，12→15。主面板的显式补救比首次失败封顶平均多 1.42 分钟，新增生成面板持平。见 [交叉结果](candidate_cross_results.json)。

直接使用名义策略在同样 30 格全部到达，平均损失为 20.43，最终 rollout 为 17.20。rollout 改善 7 格、变差 2 格、持平 21 格。每格耗时中位数分别为 3.11 秒和 6.94 秒，两批并发数不同。见 [名义基线](nominal_baseline_final_results.json)。

另以 10 月 12 日和 10 月 25 日的参数、两个新 OD 和四个新外生世界检验策略。修正数值候选筛选和补抽后，在同一 8 格重跑，旧评分、名义和 rollout 的平均损失为 103.25、31.00 和 31.63，到达数分别为 2、8 和 8。rollout 对名义为 2 改善、2 变差、4 持平。无取消平均少 0.50 分钟，压力下平均多 1.75 分钟。初次试跑中，名义策略在一格压力场景超时，最终同格两种策略都以 52 分钟到达。见 [重跑结果](fresh_confirmation_final_results.json) 与 [初次试跑](fresh_confirmation_results.json)。

## 范围

各历史参数日期仍使用归档的线路短名统计和生成车辆事件。精确参考的范围是捕获状态后的局部直达子问题。候选对比量化此次留存和刷新规则的差异，整张城市网的因果最优值仍未求解。计时用于描述计算开销。

## 复现

```bash
python3 -m pytest tests -q
python3 experiments/continuation_v1/audit_adverse_states.py
python3 experiments/continuation_v1/run_local_reference.py
python3 experiments/continuation_v1/audit_route_fit.py
python3 experiments/continuation_v1/run_candidate_cross.py --samples 8 --workers 8
python3 experiments/continuation_v1/run_nominal_baseline.py --output /tmp/bapr_nominal_repeat.json
python3 experiments/continuation_v1/run_fresh_confirmation.py
```
