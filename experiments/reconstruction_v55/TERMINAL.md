# 末平台的完整因果子集求和

`terminal_platform` 仅接收首个保留出口就是目的地、每个候选评分仅读取本车、候选车次互异的平台。相关现场观察必须恰为这些候选的根 departure；存在同车其他现场消息或共享评分依赖时返回 `None`，回到 V53。语义读集由有条件的 goal 屏障给出，同车完整概率链、未来已收到的 emission、原始计划排序均保留。

令 (U) 为尚未观察到离站的候选集合。候选 (i) 在当前时刻 (t) 的条件离站概率为

\[
q_i(t)=\frac{P_0(D_i=t,\mathrm{operating}\mid h)}
 {P_0(\mathrm{canceled}\ \mathrm{or}\ D_i<e\ \mathrm{or}\ D_i\ge t\mid h)}.
\]

这里 (e) 是原始入站时刻，不能替换成换乘准备完成时刻。独立车次使同时离站子集 (A\subseteq U) 的概率成为相应 Bernoulli 项的乘积。记录过的候选先移除，事件网格严格晚于 `after`。准备完成之前观察到的离站仍移除候选，但不触发可选动作、不扣有限决策深度。

三类量分别计算：

- **实际 BOARD 值**：在固定根 (P_0) 下逆推完整同车 posterior transition，再按 departure 原子条件化。仅实际执行的 departure/arrival 检查取消、时序与截止时间；隐藏 slot 传递概率，不额外判服务失败。它与 V53 `terminal_ride_value` 的前向积分相同。
- **WAIT 分数**：复用原 combined 的首机会与单目标公式。尚未离站候选仅新增本车 departure 的否定证据；给定未来 departure (D_i=s>t)，这项证据冗余。否定证据不推进已完成 slot，不改变单消息选择器，因此可复用未观测标记，再按当前存活概率归一化。V41 支持的时钟为整数秒。
- **观测后 BOARD 分数**：克隆历史、观察精确 departure，再调用原 planner。此时 W 可以重选后续 forecast，不能拿 WAIT 标记替代。P0 模式使用 `ConsistentP0History`，与实际根概率链一致。

因此完整因果值只需状态 ((t,U,d))。最优模式对当前全部 BOARD 与下一 WAIT 值取最小；有限深度在每次可行动比较后减一，深度零按原 combined 评分选择，实际损失仍由 (P_0) 积分。没有合法 offer 的时刻保持深度。唯一剩余候选的首次合法 offer 必上车（BOARD 分数不超过 (H=\mathrm{WAIT})）；整个平台只有一个候选时直接积分满足 (s\ge\mathrm{ready},s>\mathrm{after}) 的 departure 质量，其余质量付 (H)。

固定深度的子集求和至多 (O(T3^K)) 次概率分支、(O(T2^K)) 个状态，另计同车积分与原评分成本。它是标准独立竞争事件的有限状态求和，不主张独立方法新颖性。

所有积分、评分调用和 DP 状态计入 host 原 `states`，并检查同一 CPU 限额。预算不足的剩余质量保留物理区间 ([t,H])；概率加权和与动作最小值保持上下界有效。只有所需分支全部完成才标 `complete`。已完成缓存可以在预算耗尽后复用，未完成区间不会伪装成精确值。

验证：`tests/test_v55_terminal.py` 的 71 项检查及组合回归 36 项通过。覆盖五种策略、三车消息重选、同时 offer、原 entry/ready、strict-after、取消与隐藏 slot、未来 emission，以及原状态预算下的真值包含。全部 88 状态的固定模型配对实验由 `run.py` 统一执行，结果见 `RESULTS.md`。
