# 决策相关条件风险：冻结同车片段审计

已实现直接校准决策损失的方案，无需给全部候选同时扩宽时钟区间。固定 Oct1–2 模型、gap≤1秒 selector、180秒分组、3600秒终止时间，Oct3–11 校准，Oct15/19/22 独立确认。只审计已预先发出的 stay 指令，不计作到站后新计算仍能赶上同秒车辆的实验。

令当前时刻为 t，项目换乘 buffer 为 m，T=t+m，D=t+3600。当前站仅有一个可用同 trip 出边，正是静态相邻下一段，终点是其下一站。成功到达时刻 B 给出 R=(B−T)⁺，失败给出 R=D−T。任一其他 trip 按项目执行合同最早在 T 上车；唯一同车边成功时也只能在 B 到达。因此，即使 oracle 知道任意其他车辆、关联性和未来消息，其成本至少 min(B,T)−t；失败时至少 T−t。故该片段相对合同内 clairvoyant 最优的 realized regret≤R。拓扑唯一性避免把模型 hard support 当真实世界支持。

未来缺测或实际 trip 身份不一致保留 R∈[0,D−T]。同实际 trip 任一端点明确取消按失败计。源码修复了 arrival-only cancellation 漏判及未来 arrival 身份漏验，模型、selector 和冻结校准规则未变；旧 cal/dev 与新确认均按同一修复重算。

每个日期、每组先对 selector 选中的有限 context 集合求平均 R 上界，再对该日非空组取最大。九个校准日期取 rank9，得到 **Bcal=0.980769秒**；ordinary 非空9日，late 非空5日。日期块可交换时，这是新日期所有出现组的有限总体平均 R 的90%容忍界，允许10%日期不覆盖；不等于每个 context、任意条件、每个日历日期的硬保证，也不是给定 gate 被调用的条件覆盖。

| 独立日期/组 | 选中数 | 平均 R | 模型/实测片段耗时，秒 |
|---|---:|---:|---:|
| Oct15 ordinary | 67 | 0 | 238.014 / 253.104 |
| Oct19 ordinary | 58 | 0 | 237.045 / 236.655 |
| Oct19 late | 10 | 0 | 239.953 / 217.600 |
| Oct22 ordinary | 16 | 0 | 211.632 / 255.250 |

Oct15/22 的 late 组为空。151个选中片段均可测成功，均在 Bcal 内；late 的独立非空验证实际只有1个日期。主结果严格复现旧校准的1MiB-prefix来源过程；完整日成员仅作同日期 source-completion sensitivity。完整来源只在 Oct19 补了2个 REAL 端点、1个非选中 leg，三个日期的选中人群、分组、结果全部不变。本地仅接收 **79,320B gzip + 9,133B audit**。

保留的负结果是 **Oct27 late：21成功+1已识别取消，平均 R=144.545秒**，超过 Bcal；不再称为缺测，也未事后把27加入校准。S2车18227的 incoming dep/arr 为28708/29071，下一段 dep=PROGNOSE、arr=CANCELED，实际 trip 身份均为85:11:18227:002。该 context 模型 success=1、gap=0.000666秒，但实际 R=3180秒。S2训练40个 departure 全为 REAL，冻结 q=0，模型根本未建立该线路的取消状态；训练可观测同车后续无取消，九个校准日期仅1个 S16 后续取消且未被 selector 选中。该 late 组模型平均 R=0.048300秒、实际144.545秒，片段耗时预测237.789秒、实测379.182秒。回顾数据没有取消发布时间，不能把未来标签提前放入 H。

服务可靠性缺口以**拒绝授权**处理。新增 `service_guard` 先估计“已乘完 incoming 后的下一同车段能否完成”，取消、身份不明、未来缺测、已知错过均保守计失败；不使用线路无条件 departure 取消率。606个训练 contexts 按 route 归入1–2个日期块，以“当日任一 suffix 失败/未决”为 Bernoulli 单位。零事件单侧90%上界 qU=1−.1^(1/n)：两日 qU=.683772，S2只有Oct2一个非空训练日，24个 contexts 对应 qU=.9。假设同来源日期块 iid，这界控制新日出现 suffix 失败的边际概率；不冒称任意 H 或 gate 调用条件下的保证。context-iid 数字仅作敏感性，不能替代日期独立量。

若条件成功的 clock-tail 上界为 g，则总风险可保守上界为 min(D−T, g+(D−T)qU)，还依赖 g 在目标人群有效。现有模型 g 不是独立校准的真实时钟界。新 guard 对已有3日151个 model-selected contexts **全部拒绝1秒真实风险授权**，上界约2174.4–2862.0秒；这是事后敏感性，不是新的独立确认，qU 也不是实际风险下界。只让服务失败项≤epsilon，零失败仍需 n≥ceil(log(.1)/log(1−epsilon/(D−T)))；cap3180、epsilon1 时至少 **7322个独立日期块**，还未留 clock-tail 余量。当前训练日数无法靠调时钟模型取得这种统计依据。

限制：全部独立选中片段都位于 Zürich HB，使用 GTFS stated420秒 buffer；这并非已识别的任意平台对物理最短换乘。相同 contexts 改为0-buffer时平均 R 为217.6–255.25秒。小的模型内证书与真实风险授权已经分开：代码拒绝证据不足的授权，数据仍保留原始正负结果，没有证明城市整体优势或低风险已得到普遍验证。

`python3 -m pytest -q tests/test_decision_risk.py`：7 passed，覆盖合同内 oracle 比较、缺测保留、日块 rank、空组、未来标签不影响 selector、有限样本服务 guard。结果与证明对应 `src/decision_risk.py`、`decision_risk_run.py`、`decision_risk_analysis.json`、`decision_risk_failure_diagnosis.json`、`decision_risk_service_guard.json`；来源修正及实现修复分别记录在 `decision_risk_source_cohort_amendment.json`、`decision_risk_scorer_correction.json`。
