# 线路 ID 面板与实测补证

35 个日期按 33 个精确线路 ID 汇总，得到 1,027,848 条去重预测与 109,335 个已见 trip 状态，其中 237 个声明取消。29 个 ID 有延误预测。策略只使用过去日期构建先验。目标日预测与已见声明用于生成车辆世界，缺失线路回退到过去参数。原图包含多个日历版本，每天在静态 GTFS 中有效的连接为 74–176 条，因此主面板保持原 2,000 连接模板并按生成实验解释。 [身份与日历](../reconstruction_data_audit/graph_identity.json)

| V2，35 日期 × 18 OD，共 2,520 趟 | 平均截断损失（分钟） | 到达率 | 平均整趟计算（秒） |
|---|---:|---:|---:|
| 边际评分 | 52.00 | 70.95% | 4.56 |
| Durner strict | 61.56 | 63.81% | 4.44 |
| 名义策略 | 27.05 | 95.08% | 3.59 |
| Continuation rollout | 25.88 | 95.87% | 16.24 |

该V2 rollout 相对名义改善59格、变差15格、持平556格，按35个生成日期分组重抽样的差值区间为[−2.49，−0.30]分钟。在60、90、120分钟压力对照中，两者每档均到达11/12，rollout各多0.67分钟。 [V2主面板](chronological_results.json) · [截止敏感性](deadline_results.json)

中间空poll信息状态修复后，六个线路ID条件上的full/fixed/较小菜单/UCT按每决策128/512次转移上限完成48趟复跑。512档较小菜单与full逐案同损失，均值28.67分钟，平均整趟计算10.58对17.95秒，实际转移9,012.17对9,058.83次。128档较小菜单多0.17分钟。512档UCT两格损失由17→33、32→31分钟，均值由31.17变为33.67分钟；其余方法终端损失持平。实际工作量与墙钟另列。 [修正128档](budget_128_poll_history_results.json) · [修正512档](budget_512_poll_history_results.json) · [修正预算图](route_id_budget_poll_history.png)

初始约2.16 MB压缩片段复核获得两日54个REAL端点，历史名义90%区间覆盖19/30、8/24。27段的到发偏移均不同，最大差239秒；两条固定行程中续乘S11比换乘S20早到302、233秒。 [初始实测复核](../reconstruction_data_audit/realized_subset.json) · [初始区间覆盖](../reconstruction_data_audit/realized_interval_audit.json) · [固定实测行程](../reconstruction_data_audit/verified_itinerary.json)

当前V4已完成v6铁路96趟四方法回放，使用官方GTFS站内换乘、每决策512次转移，执行无删失；完整声明城市候选的四方法16趟也已终端完成。实际公告、未知事件处理、端到端开销和后续日期校准见V4结果。 [V4最终实测与校准](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/RESULTS.md)

多段精确参照在所有未来状态执行同一H；两个片段的候选损失为2.125和0分钟，H内启发式损失为1.25和2分钟。阶段四现已实现固定有限场景库内的安全删留证书和标签依赖修复，完成两片段精确核对及2,000连接模板的更新验证；策略树仍重新计算。适用条件、完整计费及新颖性边界见证书记录。 [多段参照](../continuation_v1/multileg_reference_results.json) · [证书与局部修复](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/CANDIDATE_CERTIFICATE.md)

原面板入口为 `run_chronological_panel.py`，默认生成35×18。本次修正复跑使用 `rerun_poll_history.py --panel confirmation`、`learning`、`budget128`或`budget512`，逐格压缩回执保留公告接收、WAIT、miss及耗时。`plot_chronological_panel.py`生成修正预算图，原图与旧回执作为历史保留。

四板修正复跑共120趟，使用原日期、OD、seed与预算，未增加独立确认日期；输入与修正实现版本冻结在各 `_protocol.json`。1,837次poll、4,648条消息和828次WAIT的时序/决策引用及逐决策上限均核验通过。旧budget当时未存逐参数protocol，本次依据历史日期/OD/seed/cap/method/budget和同源factorial冻结参数恢复，与旧回执2,246条公开车辆事件逐项一致。 [复跑核验与差异](poll_history_rerun_comparison.json)

V2 `ContinuationRouter` 的12条件归因中，常规条件rescue比首次失败封顶低4.83分钟，压力条件高0.33分钟。压力条件retain比refresh高12.17分钟。旧marginal的mean×cancel×penalty八变体全部12/12失败封顶，零终端差异无法识别组件作用。平台修正后的V2新OD确认，常规rollout/nominal均14.83分钟，压力分别8.33/33.00分钟。首轮按同名站映射选错Stettbach平台的试跑保留并排除。 [V2归因](route_id_factorial_results.json) · [V2新OD确认](final_platform_resolved_confirmation.json)

最终 `CausalPlanningRouter` 的mean×cancel四格在修复后复跑此前冻结的12条件，属于事后诊断。四格均使用full物理菜单、8个基础样本、每决策512次模型转移上限及相同query seed，精确公开车辆事实均保留。48趟损失、到达、工作量、公开事件及决策路径均与旧回执一致。

| route学习 | 常规损失 | 压力损失 | 常规/压力到达数 |
|---|---:|---:|---:|
| mean关，cancel关 | 28.67 | 51.50 | 6/6，5/6 |
| mean关，cancel开 | 28.67 | 51.50 | 6/6，5/6 |
| mean开，cancel关 | 28.67 | 49.00 | 6/6，5/6 |
| mean开，cancel开 | 28.67 | 49.00 | 6/6，5/6 |

均值学习的压力条件平均效应为−2.50分钟，2格改善、1格变差、3格持平。取消学习确实改变全部12条件的概率估计，动作路径及损失均持平。每趟实际转移数常规9,057.67–9,058.83、压力9,628.83–9,659.50，未完成共同轮及回退次数保存在回执。最终规划器累计执行成本并比较rollout均值，未调用marginal的beta罚项。测试将beta从0改到10¹²，动作/value/work相同；先验std仍改变预测分布。 [修正后学习归因](causal_poll_history_learning_factorial_results.json) · [累计成本与选择](/home/erzhu419/mine_code/BAPR-HRO/src/causal_planning_router.py:90) · [分支验证](/home/erzhu419/mine_code/BAPR-HRO/tests/test_rollout_learning_cross.py:49)

最终Causal类生成确认的修正复跑使用原Nov12–14日期、seed和两个明确平台OD，比较同一full菜单的rollout与nominal，均采用8样本设置，rollout预算512。24趟全部到达，常规损失均14.33分钟，压力rollout/nominal为13.67/15.67分钟；仅一个压力格改善12分钟，其余11格持平，结果和路径与旧回执逐项一致。压力日期差值为−6/0/0分钟。此次每趟计算约0.251/0.267秒对0.013/0.013秒，同批8 workers。三个先验预测生成日期及两个OD界定确认范围。 [修正后生成确认](final_causal_poll_history_confirmation.json)
