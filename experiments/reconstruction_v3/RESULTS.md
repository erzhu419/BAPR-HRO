# 线路 ID 面板与实测补证

35 个日期按 33 个精确线路 ID 汇总，得到 1,027,848 条去重预测与 109,335 个已见 trip 状态，其中 237 个声明取消。29 个 ID 有延误预测。策略只使用过去日期构建先验。目标日预测与已见声明用于生成车辆世界，缺失线路回退到过去参数。原图包含多个日历版本，每天在静态 GTFS 中有效的连接为 74–176 条，因此主面板保持原 2,000 连接模板并按生成实验解释。 [身份与日历](../reconstruction_data_audit/graph_identity.json)

| 35 日期 × 18 OD，共 2,520 趟 | 平均截断损失（分钟） | 到达率 | 平均整趟计算（秒） |
|---|---:|---:|---:|
| 边际评分 | 52.00 | 70.95% | 4.56 |
| Durner strict | 61.56 | 63.81% | 4.44 |
| 名义策略 | 27.05 | 95.08% | 3.59 |
| Continuation rollout | 25.88 | 95.87% | 16.24 |

Rollout 相对名义改善 59 格、变差 15 格、持平 556 格，按 35 个生成日期分组重抽样的差值区间为 [−2.49, −0.30] 分钟。在 60、90、120 分钟的压力对照中，两者每档均到达 11/12，rollout 各多 0.67 分钟。六个线路 ID 条件上的全菜单、固定菜单、较小菜单和 UCT 共 48 趟，按每决策 128/512 次模型转移比较。512 档较小菜单与全菜单逐案同损失，平均计算耗时 18.99 对 25.68 秒。128 档较小菜单多 0.17 分钟。等转移预算与端到端时延分开报告。 [主面板](chronological_results.json) · [截止敏感性](deadline_results.json) · [128 档](budget_128_results.json) · [512 档](budget_512_results.json)

从官方档案请求约 2.16 MB 压缩片段，获得两日 54 个可核验 `REAL` 端点。历史名义 90% 区间分别覆盖 19/30、8/24 个端点。27 个完整段的到发偏移均不同，最大差 239 秒。两条固定实测行程中，同车续乘 S11 相对换乘 S20 分别早到 302、233 秒。竞争线路和日期完整性决定城市级实测回放的可行性。 [实测端点](../reconstruction_data_audit/realized_subset.json) · [区间覆盖](../reconstruction_data_audit/realized_interval_audit.json) · [实测行程](../reconstruction_data_audit/verified_itinerary.json)

多段精确参照将同信息、仅知工况、预知分开，在所有未来状态执行同一 H。两个片段的候选损失为 2.125 和 0 分钟，H 内启发式损失为 1.25 和 2 分钟。原型强规划和较小菜单的速度对照支持继续检验计算权衡。低成本质量证书、可推广的安全删留与依赖更新复杂度仍需新的结构结果，阶段四暂缓。 [多段参照](../continuation_v1/multileg_reference_results.json)

运行入口为 `run_chronological_panel.py`，默认生成全 35 × 18 面板。截止对照使用 `--dates 2023-10-12 2023-10-25 2023-10-29 --ods 0 1 5 12 --caps 60 90 120 --stress`。预算对照在同三日期使用 `--ods 1 5 --methods full_rollout fixed_h_rollout material_h_rollout belief_uct --transition-budget 128` 或 `512`，各用单独输出路径。`plot_chronological_panel.py` 从最终汇总生成图。逐格压缩回执保留公告、事件及决策耗时。优化前试跑与最终回执分开，6 个配对复核中的到达、动作、公告、扫描与标签计数一致。
