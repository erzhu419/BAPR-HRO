# `experiments/g2_reset_v1`: G1–G2–G3 协议重构、在车条件校准拆分与有界门控修复

本目录针对 `natural_gap_v2` 结束后项目长期卡在 **G2** 的根因，提供**不触碰任何历史冻结记录、不向原 72 根追加回放预算**的协议重构与核心算法修复模块：

1. [`protocol_v2.json`](./protocol_v2.json)：重构后的 G1–G2–G3 验收协议（移除不可验证的 2023 `ist-daten` 物理轨道列硬门与欠功效的 14 日 `±0.02` 四分类 Bonferroni 等价硬门；拆分初乘可用性与在车条件校准；落地公开/声明站台对 MCT 合同；强制在线策略满足 `CPU <= 2.0s` 停站时间预算与配对置信区间认证）。
2. [`calibration_split.py`](./calibration_split.py)：
   - 精确拆分“初乘车辆取消未上车（`initial_service_availability`）”与“在车条件联合换乘校准（`conditional_onboard_calibration`）”；
   - 支持直接从 `natural_gap_v2/calibration_summary.json` 的紧凑事件组上下界代数无损还原全部 `1,309/1,328` 项已到达在车事件的精确残差与 Brier 分数（无需从服务器重拉原始大数据）；
   - 实现静态 `transfers.txt` + 同站台（`tau_same_platform=90s`）+ 父站回退的三级站台对 MCT 合同审计。
3. [`certified_gate.py`](./certified_gate.py)：修复导致 `full128`/`matched128` 耗时暴涨至 `49.01s` 且产生随机噪声负例的三大代码缺陷：
   - `should_invoke_rollout`：基于解析裕度 `|V_alight - V_continue| <= margin` 过滤无需搜索的明显占优状态（在 `control_identity_v1` 上减少 `71.3%` 的门调用）；
   - `choose_certified`：修复 `reconstruction_v44/matched.py` 第 136 行仅凭 `1e-9` 点估计差（忽略 `17–25s` 蒙特卡洛标准误）覆写 `base` 动作的缺陷，消除 `R04 (t=-0.14, -263s)` 与 `R32 (t=+0.47, -140s)` 等纯抽样噪声倒退；
   - `active_platform_rides` 与 `RecursiveSelectivePlanner`：按当前 `entry` 时刻剩余正概率班次裁剪活跃站台宽度（修复静态 `len(p.rides) <= 6` 导致的 `87.5%` 退回 `full128` 问题），并将 `MatchedContinuationPlanner` 深度为 1 的 `PhysicalPlanner` 尾估值扩展为多换乘递归选择性等待估值。
4. [`offline_audit.py`](./offline_audit.py) 与 [`OFFLINE_AUDIT_SUMMARY.json`](./OFFLINE_AUDIT_SUMMARY.json)：对本地已冻结收据的纯只读复核脚本及完整数值输出。

## 运行与测试

```bash
python3 -m experiments.g2_reset_v1.offline_audit
python3 -m pytest tests/test_g2_reset_v1.py
```
