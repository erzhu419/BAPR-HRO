# BAPR-HRO 代码/算法第二轮 Review

日期：2026-04-29

本轮 review 的对象是上一轮算法优化建议之后的当前代码状态。结论是：代码已经开始落地上一轮最关键的方向，包括分层 route prior、infeasibility/timeout penalty、自适应 top-k/lookahead、Adaptive-beta 共享状态等；但当前实现里仍有几处会直接扭曲算法效果的高风险点。优先修这些，比继续加新算法变体更有价值。

## 主要发现

### P0. `timeout_penalty` 使用了绝对时刻错误，导致分数几乎必然惩罚所有候选

位置：

- `src/bandit_router.py:365-367`
- `src/bandit_router_v2.py:326-328`
- `src/bandit_router_v3.py:170-172`

当前 V1 使用：

```python
p_on_time = label.dest_arrival.prob_le(self.max_time)
timeout_penalty = self.timeout_weight * (1.0 - p_on_time)
```

V2/V3 更严重，直接写死：

```python
p_on_time = label.dest_arrival.prob_le(120)
```

但 `label.dest_arrival` 是分钟制绝对时刻，通常在 480、490、520 这一类“当天分钟数”附近；`max_time=120` 是 journey duration。用 `prob_le(120)` 基本等价于问“是否在凌晨 2:00 前到达”，对 8 点出发的旅程通常为 0。结果是大多数候选都会被加满 `60` penalty，这个 penalty 不再表达 timeout risk，只是常数噪声；如果不同候选的 PMF offset 恰好不同，还会产生错误排序。

应改为：

```python
deadline = current_time + remaining_budget
# 或至少：
deadline = journey_departure_time + max_time
p_on_time = label.dest_arrival.prob_le(deadline)
```

当前 `select_connection()` 没有 `t_depart` 或 elapsed budget，因此建议把 `journey_deadline` 放进 router state：`route(..., t_source)` 时保存 `self.journey_deadline = t_source + max_time`，之后 scoring 用这个绝对 deadline。

验收测试：

- 构造 `PMF.deterministic(520)`，`t_depart=490`，`max_time=120`，应判定 on-time；
- 构造 `PMF.deterministic(620)`，应判定 timeout risk 高；
- V1/V2/V3 都要测，避免 V2/V3 继续写死 `120`。

### P0. `RouteBeliefState.update_cancel(kind=...)` 已支持类型，但调用链完全没有传 kind，typed cancellation 实际未生效

位置：

- `src/bandit_router.py:109-132`
- `src/simulate_bandit.py:105-107`
- `src/simulate_bandit.py:149-154`
- `src/simulate_bandit.py:164-168`

`RouteBeliefState` 增加了 `n_true_cancels`、`n_late_no_shows`，也提供了 `cancel_rate_by_type`，但所有 router 的 `observe_cancel(route)` API 仍不接收 kind，模拟器也一直调用：

```python
router.observe_cancel(o.route)
router.observe_cancel(c.route)
```

因此 `update_cancel()` 默认永远是 `kind='true'`。更关键的是，等待超过 patience 的 late no-show 分支现在走的是：

```python
router.observe_delay(c.route, delay)
```

这会把“等不到车”的事件当作普通 delay 更新，而不是 late no-show。上一轮建议里的 typed cancellation 目前只是计数字段，尚未进入算法。

建议：

- 把 router API 改成 `observe_cancel(route, kind="true")`；
- GTFS-RT 大 delay / 明确取消用 `kind="true"`；
- `actual_dep > current_time + patience` 用 `kind="late_no_show"`；
- feed 缺失单独用 `kind="feed_missing"`，权重应低于 true cancel；
- scoring 里使用 `true_rate, late_rate = belief.cancel_rate_by_type`，不要继续只用 `belief.cancel_rate`。

验收测试：

- late no-show 后 `n_late_no_shows` 增加，`n_true_cancels` 不增加；
- true cancel 后 `n_true_cancels` 增加；
- feed missing 对 score 的影响小于 true cancel。

### P0. V2/V3 的新增 risk penalty 与 V1 不一致，且写死权重和 deadline

位置：

- `src/bandit_router.py:184-215`
- `src/bandit_router_v2.py:323-333`
- `src/bandit_router_v3.py:168-177`

V1 把 `max_time`、`infeasibility_weight`、`timeout_weight` 做成了构造参数，这是正确方向。但 V2/V3 直接写死：

```python
infeasibility_penalty = 60.0 * (1.0 - label.feasibility)
p_on_time = label.dest_arrival.prob_le(120)
timeout_penalty = 60.0 * (1.0 - p_on_time)
```

这会让 V1/V2/V3 的对比不公平：同一篇实验表里，方法之间不仅算法不同，risk penalty 的参数来源和 deadline 语义也不同。

建议提取一个共享 scoring helper：

```python
def hyperpath_risk_penalty(label, deadline, infeasibility_weight, timeout_weight):
    infeas = infeasibility_weight * (1.0 - label.feasibility)
    timeout = timeout_weight * (1.0 - label.dest_arrival.prob_le(deadline))
    return infeas + timeout
```

V1/V2/V3/DRO/SW-LCB 如果要比较“算法差异”，应共享这一套 risk penalty 或明确开关。

### P1. route prior 是 route-level，但上一轮要求的是 route/hour 或 route/global 层级；当前仍会污染时段差异

位置：

- `experiments/swiss_full/build_route_priors.py:1-10`
- `experiments/swiss_full/build_route_priors.py:38-48`
- `src/bandit_router.py:162-178`
- `src/bandit_router_v2.py:179-189`

当前 `route_priors.pkl` 只按 `route_short_name` 聚合 mean/std/cancel_rate。它能解决“所有路线同质 cold start”的一部分问题，但还没有解决早高峰/平峰/晚高峰差异，也没有 OD-local shrinkage。

这会导致两个算法风险：

- normal-day 某些高频路线被全天平均 cancel/delay 过度惩罚；
- disrupted-day 如果异常集中在特定时段，全日 prior 会稀释风险。

建议第二步升级为：

```text
prior_key = (route, hour_bucket)
fallback = route
fallback = global
```

最小实现：

- 在 `build_route_priors.py` 输出 `{route: {"global": ..., "by_hour": ...}}`；
- router 初始化时根据 `t_source` 或 `current_time` 选 hour bucket；
- 数据不足时 fallback 到 route global。

### P1. Adaptive-beta 的共享状态会在实验 cell 之间泄漏，可能污染 normal/disrupted 对比

位置：

- `src/adaptive_bandit_router.py:37-51`
- `src/adaptive_bandit_router.py:69-77`
- `src/adaptive_bandit_router.py:162-182`

把 beta meta-policy 变成 class-level shared state 是朝“跨 journey 学习”迈了一步，但当前没有看到实验 runner 在 scenario、OD、seed、method cell 之间显式 reset。这样会发生：

- 先跑 normal，后跑 disrupted：disrupted 继承 normal 的 beta posterior；
- 先跑 disrupted，后跑 normal：normal 继承 disrupted 的 beta posterior；
- 多进程时每个进程有各自 shared state，结果取决于任务调度，不完全可复现。

如果论文 claim 是“online meta-learning across a day”，共享是合理的；如果实验表格要比较独立 scenario，必须在 cell 开始时 reset。

建议：

- 在 experiment config 中明确 `share_meta_state_scope`：`none` / `per_day` / `per_scenario` / `global`；
- 每个统计 cell 开始前调用 `AdaptiveBetaBanditRouter.reset_shared_state(n_betas)`，除非该实验明确评估跨 cell 学习；
- 结果 JSON 记录该 scope。

同时，当前更新仍是“未观测 beta 用当前 cost 或历史均值代替”的 full-information 启发式，不是标准 EXP3。文档和论文里应避免继续称为严格 EXP3。

### P1. `is_bandit = isinstance(...)` 仍是扩展瓶颈，已有脚本开始 monkey patch

位置：

- `src/simulate_bandit.py:60-63`
- `experiments/swiss_full/run_csa_meat_baseline.py:138-213`

上一轮建议定义 Router Protocol。当前 `simulate_bandit_journey()` 仍然维护一长串 class 列表。已经有 `run_csa_meat_baseline.py` 注释说明需要 patch simulator 来识别 `CSAMEATRouter`，这说明问题已经实际发生。

这会影响算法实验，不只是代码整洁问题：新 baseline 如果没加到列表，就被当成 Static 风格执行，结果会静默错误。

建议尽快改为 capability-based 判断：

```python
is_online_router = all(hasattr(router, name) for name in [
    "observe_delay", "observe_cancel", "select_connection"
])
```

或者用 `typing.Protocol`。这项改动风险低、收益高。

### P1. `hash(sid)` 仍未修复，真实 GTFS 实验跨进程不可稳定复现

位置：

- `src/gtfs_parser.py:39-40`
- `src/gtfs_parser.py:216-229`

`Stop.id = hash(sid) % 1e9` 仍存在。Python hash randomization 会让同一 stop_id 在不同进程得到不同 int id。已有 Swiss multi-day 使用 multiprocessing，这会让可复现性和缓存一致性存在隐患。

建议：

- 读取 stops 后按 GTFS stop_id 排序，分配连续 int；
- 保存 `stop_id_map`，构图和 OD 解析共用；
- 禁止业务 id 使用 `hash()`。

### P2. 新增算法逻辑缺少单元测试，当前 `tests/` 仍只覆盖基础结构和 TopoCSA

验证：

- `python -m pytest -q tests -o cache_dir=/tmp/bapr_hro_pytest_cache` 通过，结果 `10 passed in 0.82s`。

但测试只覆盖 PMF、合成网络和 TopoCSA。新增的关键算法改动没有测试：

- risk penalty deadline；
- typed cancellation kind；
- hierarchical prior loading / fallback；
- adaptive top-k/lookahead；
- Adaptive-beta shared-state reset scope；
- V2/V3 和 V1 scoring 一致性。

建议新增 `tests/test_router_scoring.py`，先用小型 deterministic PMF 和 fake label 测 scoring，不要依赖真实 GTFS。

## 已完成的有效改进

这轮代码相比上一轮建议已经有实质进展：

- V1 增加了 `infeasibility_weight`、`timeout_weight` 和基于 `label.feasibility` / `dest_arrival` 的 risk penalty。
- V1/V2/V3 都开始引入自适应 top-k/lookahead。
- V1/V2 引入了 route-level historical prior，并已有 `experiments/swiss_full/build_route_priors.py` 和 `data/route_priors.pkl`。
- V1 的 disruption gate 从 `max(cancel_score, delay_score)` 改成 bilinear 组合，更平滑。
- Adaptive-beta 改为 class-level shared meta-state，开始回应“单 journey 学不动 beta”的问题。
- 核心测试仍通过。

这些方向是对的，但目前大多是“第一版落地”，还没有完成语义闭环。

## 下一轮最小修复清单

按算法效果优先级排序：

1. 修复 timeout deadline：所有 `prob_le(120)` / `prob_le(max_time)` 改为绝对 deadline。
2. 打通 typed cancellation：`observe_cancel(route, kind)` 从 simulator 到 belief 到 score 全链路生效。
3. 抽出共享 risk scoring helper，让 V1/V2/V3 参数一致。
4. 给 Adaptive-beta 增加 shared-state scope 和 reset，避免 scenario 泄漏。
5. 修复 `hash(sid)` 为稳定 stop id。
6. 新增 `tests/test_router_scoring.py` 覆盖上述 1-4。
7. route prior 升级为 route/hour/global fallback。

如果只做一件事，先修第 1 条。当前 timeout penalty 的绝对时间错误会直接让新加的 reach-risk score 失真。

## 当前判断

上一轮 md 提出的算法效果优化方向已经开始落实，但目前最危险的是“看起来加了 risk penalty，实际 deadline 用错导致 penalty 语义错误”。在这个问题修正前，不建议用当前新结果强化论文结论。

修完 P0 后，再跑 paired OD/seed 的 normal/disrupted 对照，重点看：

- reach 是否不再低于 Static；
- unconditional mean 是否下降；
- conditional mean 是否没有靠牺牲 reach 得到；
- per-OD win/loss 是否改善，而不是只改善 aggregate mean。
