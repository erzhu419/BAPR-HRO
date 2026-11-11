# 2026-11-11：代码、理论与研究筛查

两份诊断 Markdown 和 Word 已完整阅读；按用户要求先处理实验、代码和理论，稿件与既有 Lean 未修改。旧结果、原 pickle 和 Git 中的历史实现保留。

已修取消概率、成功条件 PMF、Normal–Gamma 均值后验、同车续乘与换乘、固定 trip 世界、因果消息、重复学习及等待承诺。旧图实际含6条零时长同站连接，现禁止同一 connection 重乘，保留同车后续区间。全套测试 **61 passed**。

数据只流读两个本地 RT snapshot；明确取消、严重延误、跳站、缺失信息分开。按 trip ID 单次流读静态 trips 表，189/189 trip、2000/2000 connection 均恢复 route/agency 身份；线路短名歧义不再被视为身份恢复阻塞。

## 结果

精确机制面板共36个cell，覆盖相关性、消息时机、候选和截止时间。120分钟上限下，四例 full 最优为12/22/25/25，只保留A时均为25，候选损失13/3/0/0。两处真实拓扑子网各取两班车、四个构造世界，30/60分钟上限下候选损失分别2.5/10和3.25/10.75分钟；受限最优的候选内损失为零。

匹配面板：2个历史参数日期 × 3个不同起点OD × 2个世界种子 × 无取消/人为压力，共24个世界、216次旅程。压力参数固定为延误均值增加8分钟、每trip取消概率20%；先验仅用更早日期，旧的“延误超过30分钟”取消代理未使用。所有方法共享每格的车辆世界；候选上限5、窗口25分钟、旅程上限120分钟，无调参。

| 共同估值变体 | 无取消校准场景：平均截断损失/到达率 | 人为压力：平均截断损失/到达率 |
|---|---:|---:|
| 冻结先验 | 80.000 / 50.0% | 97.667 / 33.3% |
| 在线均值 | 79.917 / 50.0% | 91.083 / 33.3% |
| 在线均值与取消估计 | 74.083 / 58.3% | 91.083 / 33.3% |
| 再加均值后验标准差罚金 | 78.750 / 50.0% | 91.083 / 33.3% |

以上每行的 retain 与 refresh 平均结果相同；**96个逐场景配对的损失全部相同，8组动作序列不同**。记录中确有新增/移除候选，候选变化本身没有转化为本轮观察到的损失差。罚金在无取消面板平均增加4.667分钟（1格改善、1格变差），压力面板平均差为零。原生 strict 策略的次要参照分别为92.583/33.3%与67.667/66.7%；它保留条件均值目标，不属于上述同目标归因。

完整计时含估计、预处理和查询。匹配估值单次查询约0.55–2.00秒；在线各变体整趟平均约7–11秒，冻结先验约0.7秒。完整查询在retain中同样执行，因此此设计不主张保留候选节省计算。

**决定：阶段四暂不启动，转向 continuation 估值诊断。** 当前没有识别出候选保留收益，也没有识别出不确定性罚金增益；不把局部压缩、质量证书或 OR 新颖性写成已经成立的结果。精确 full/H 参照和合法 Bellman 分解已经可用，后续应先在共同信息及动作合同下定位换乘失败概率、成功条件到达与补救价值的估值误差，再决定新算法。

## 产物与复现

- [完整匹配结果](matched_corrected_results.json)：逐世界车辆表、先验、实际公开消息、决策与执行日志、全部开销；每格另存于 `matched_corrected_results_cases/`，同配置重启可复用。
- [精确公交子网](exact_transit_results.json)、[机制面板](mechanism_results.json)。
- [质量与计算](figures/quality_computation.png)、[两类损失](figures/loss_decomposition.png)、[相关性与消息时机](figures/correlation_information_timing.png)；均另有SVG/PDF。
- [理论与证明](../../md/review_20261111/THEORY.md)、[逐项状态](../../md/review_20261111/ACTIONS.md)、[最近方法与基线对应](../../md/review_20261111/BASELINES.md)。

```bash
python3 -m pytest tests -q
python3 experiments/reconstruction_v2/run_mechanisms.py
python3 experiments/reconstruction_v2/run_exact_transit.py
python3 experiments/reconstruction_v2/run_matched_panel.py --replicates 2 --workers 8 --output experiments/reconstruction_v2/matched_corrected_results.json
python3 experiments/reconstruction_v2/plot_diagnostics.py
```

## 限制

这仍是统计量校准的生成仿真与构造有限模型，不是真实事件回放或独立现实确认。两日期的仿真重复不构成新的真实独立样本；不作总体显著性结论。大图估值是边际 continuation 近似，96个启发式配对零差异不证明真实候选最优损失为零；精确子网采用全局已发生公告，匹配面板采用本地公告，不能跨信息制度比较最优差。精确证书先解full，尚无便宜证书或计算压缩保证。并行实测时间受共享机器负载影响，子网计时启用了内存追踪。前两次预备运行在零时长重乘问题上异常/中止，未形成完整统计；本报告仅使用修复后的v3产物。送审系统冻结PDF未取得，稿件、引文清理及新方法独立确认尚未执行。
