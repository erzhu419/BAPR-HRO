# 新诊断实施的证据入口

原计划：`md/GPT_diagnosis_v2.md`、`md/BAPR_HRO_OR_diagnosis_plan_20261111.docx`；完整提取在`md/review_20261111_v2/plan_extracted.txt`。前置条件和停止规则在`protocol.json`，最终研究判断在`RESULTS.md`。

| 判断对象 | 实现/证明 | 原始记录与入口 |
|---|---|---|
| 40/30合同反例、本站可见性、车内默认续乘、下车后MCT、步行截止 | `src/platform_transit.py`；`tests/test_platform_transit.py` | `contract_protocol.json`、`contract_results.json` |
| 相同后台发布能力、旧策略继续、过时替换拒绝 | `src/background_policy.py`；`tests/test_background_policy.py` | `contract_results.json`中的30秒局部敏感性；不是整趟算法排名 |
| 静态平台声明与未知物理下界 | `src/platform_transfers.py` | `platform_audit.json`、`platform_pair_declarations.csv.gz`、`platform_audit.py` |
| 原生优化的严格等价、冷/热/观测更新、全部开销 | `src/optimized_native_router.py`、`src/native_order.cpp` | `baseline_compiled_profile_results.json`、`baseline_native_logical.json`、`baseline_RESULTS.md` |
| 完整首退出、逐世界下包络、标准宏组合及先例差异 | `src/intermediate_boundary.py`、`boundary_THEORY.md` | `boundary_results.json`、`boundary_support_audit.json`、`boundary_RESULTS.md` |
| 服务失败正支持、同一全程服务律及先验依赖 | `src/service_state_model.py` | `service_results.json`、`service_summary.json`、`service_RESULTS.md` |

每个主实验保留源文字/配置protocol，未新增摘要hash文件。编译调用为显式`build_native_kernel(path)`，费用单列；`.so`可由源重建。原始全量静态交通数据继续使用工作区已有文件，没有从服务器拉取新大文件。

`*_before_alight`、`*_before_auto_stay`、`*_before_walk_deadline`、`contract_*_before_onboard_default`及`contract_*_before_pose_audit`是合同修正前的记录，均不替代最终主结果。Python RMQ性能负例单独保留。训练日期和所有V1–V6已看日期在本轮属于开发；局部下一站任务、整趟旅程、模型内界及真实风险各有独立范围。
