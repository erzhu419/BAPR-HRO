# 20秒积分网格的范围

训练zero-dwell elapsed的均值为13.935秒、标准差10.186秒。若冷态相邻事件只使用同一个非负h秒格，`X²≥hX`，所以 `Var(X)≥hE[X]−E[X]²`。h=60要求方差至少约641.9，训练方差仅103.7；无论怎样调drift/sd都不能同时再现。必要条件为 `h≤m+variance/m≈21.38秒`，因此本候选预先固定20秒core，而非按测试日选择步长。20秒仅解除这个必要条件，不证明拟合充分。

`OrderedDurationBelief` 保留原60秒tails和全部精确anchor，在[-300,300]秒增加20秒点。无anchor时每slot90个clock，q=0时只保留operating状态。初始Gaussian按clock Voronoi cell积分，两侧无限tails分别归端点，避免非均匀网格新增点被当作相同权重的密度样本。

下一事件使用 `X~Normal(d+60·drift,(60·sd)²)|X≥0`，其中d是相邻scheduled clock差，drift/sd参数仍为分钟。只允许clock≥前态；第一cell下界是前态clock，内部边界是相邻可选clock的midpoint，最后上界为∞。所有cells构成[0,∞)的分割，故概率和为1；再乘原服务状态链概率，travel保持同一availability，dwell/gap按原q/persistence转移。没有合法后继的行仍为零。logCDF/logSF差用于远tail。

同一 `_ChainLaw` 用于nominal marginal与FFBS采样；所有观测接口、精确秒钟anchor、未知取消后的wait证据及既有Markov边界算法保留。新CDF kernels使用独立只读cache，观察anchor形成的kernel只属于该law，不进入公共固定网格cache。旧Evolving与旧密度kernel不变。

10项独立测试通过：非均匀cells与tails、首cell非中点下界、零/一秒精确事件、远tail、q0与部分取消恢复、brute-force joint、4000次采样与同law均值一致，以及copy/cache隔离。模块明确使用integer second clocks、minute=60。

本模型仍是有限网格近似：exact anchors会改变网格；尾部cell聚合到端点；CDF积分不会自动保证每日期覆盖或真实旅程改善。连续截断MLE与部署cell likelihood也不是同一目标。拟合与匹配回放由独立冻结训练/运行脚本给出，不能由上述表达能力证明替代。
