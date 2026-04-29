# BAPR-HRO 代码/算法第三轮 Review

日期：2026-04-29  
对象：当前工作区代码，重点是 `src/bandit_router*.py`、`src/adaptive_bandit_router.py`、`src/simulate_bandit.py`、Swiss multi-day/LOO 实验脚本。

结论：这一轮新增了 leave-one-day-out route prior 的代码入口，方向正确；但上一轮几个会直接扭曲算法效果的 P0 仍未修。当前最需要先修的是 timeout deadline 语义、typed cancellation 链路、Adaptive-beta 共享状态作用域。否则新跑出来的 Swiss 结果仍可能是“分数函数/实验状态污染”的结果，而不是算法真实提升。

## 主要发现

### P0. A7 timeout penalty 仍然把 journey duration 当成绝对到达时刻

位置：

- `src/bandit_router.py:371-374`
- `src/bandit_router_v2.py:331-334`
- `src/bandit_router_v3.py:171-174`
- `src/pmf.py:68-75`

`PMF.prob_le(t)` 的 `t` 是绝对分钟时刻，因为 `PMF.from_delays()` 使用 `scheduled + delay_offset` 作为 offset。实际 Zurich 早高峰出发在 `490` 左右，到达 PMF 通常在 `520+`。但当前 V1 用：

```python
p_on_time = label.dest_arrival.prob_le(self.max_time)
```

V2/V3 仍写死：

```python
p_on_time = label.dest_arrival.prob_le(120)
```

这等价于问“是否在当天 02:00 前到达”，大多数 08:00 出发的 journey 都会得到 `p_on_time = 0`，于是所有候选都吃满 `60` 分 timeout penalty。这个 penalty 不是 timeout risk，而是近似常数；如果某些 PMF offset/截断不同，还会制造错误排序。

建议修法：

- `BanditRouter.route(s_source, s_dest, t_source)` 保存 `self.journey_deadline = t_source + self.max_time`。
- V1/V2/V3 的 A7 penalty 全部调用同一个 helper：`label.dest_arrival.prob_le(self.journey_deadline)`。
- V2/V3 不要再硬编码 `120`。

验收测试最小例：

```python
label.dest_arrival = PMF.deterministic(520)
t_source = 490
max_time = 120
# deadline = 610, p_on_time 应为 1.0，而不是 0.0
```

### P0. typed cancellation 仍未从 simulator 传到 belief，A3 实际没有生效

位置：

- `src/bandit_router.py:109-132`
- `src/bandit_router.py:262-265`
- `src/simulate_bandit.py:105-107`
- `src/simulate_bandit.py:149-168`
- `src/adaptive_bandit_router.py:153-157`

`RouteBeliefState.update_cancel(kind='true')` 已经有 `true` / `late_no_show` / `feed_missing` 的设计，但公开 API 仍是：

```python
def observe_cancel(self, route: str):
    belief.update_cancel()
```

模拟器里所有取消都调用 `router.observe_cancel(route)`，没有 kind。更关键的是等待超过 patience 的分支仍走：

```python
if actual_dep > current_time + 12:
    router.observe_delay(c.route, delay)
```

这会把“乘客等不到车”当成普通 delay 样本，而不是 late no-show。结果是 A3 的 typed counter 字段存在，但不影响 score，也不影响实验。

建议：

- 所有 router 统一 API：`observe_cancel(route, kind="true")`。
- `delay == 999` 和 GTFS-RT 明确取消用 `kind="true"`。
- `actual_dep > current_time + 12` 用 `kind="late_no_show"`，不要写入普通 delay posterior。
- feed gap 单独用 `kind="feed_missing"`，且权重低于 true cancel。
- scoring 使用加权取消率，例如 `60 * (true_rate + 0.5 * late_rate + 0.2 * feed_rate)`，并加测试。

### P0. Adaptive-beta 共享状态仍会跨 OD/day/worker 污染实验结果

位置：

- `src/adaptive_bandit_router.py:37-51`
- `src/adaptive_bandit_router.py:71-79`
- `experiments/swiss_full/run_multi_day.py:108-123`
- `experiments/swiss_full/run_multi_day_loo.py:98-115`
- `experiments/swiss_full/run_swiss_multi_od_v3.py:79-90`

`AdaptiveBetaBanditRouter` 默认使用 class-level shared state。multi-day 脚本没有在 day、OD、seed、scenario 边界 reset，因此 Adaptive-beta 的结果取决于：

- worker 进程拿到 day 的顺序；
- 同一 worker 上之前跑过哪些 OD；
- normal/disrupted/LOO 是否在同一个 Python 进程中连续跑；
- `Pool.imap_unordered` 的调度。

这会让 Adaptive-beta 的 35-day 结果不可严格复现，也很难解释“跨 journey 学习”到底发生在什么统计单元内。

建议：

- 在实验配置中显式记录 `share_meta_state_scope`: `none` / `per_od` / `per_day` / `global`。
- 如果论文表格按 `(day, OD)` cell 统计，建议 cell 开始前 reset；如果要评估 day-level online learning，则每个 day reset 一次并固定 OD 顺序。
- `results/*.json` 写入 scope、reset 边界和 beta final weights。
- 多进程实验不要依赖 class-level state 做跨 day 学习；需要集中式状态就不要用 `Pool` 隐式分片。

### P0. Adaptive-beta convergence 脚本会双重 begin/end journey

位置：

- `experiments/swiss_full/run_adapt_beta_convergence_R15.py:67-73`
- `src/simulate_bandit.py:65-67`
- `src/simulate_bandit.py:193-196`

`run_adapt_beta_convergence_R15.py` 手动调用：

```python
adapt_router.begin_journey()
...
res_a = simulate_bandit_journey(...)
...
adapt_router.end_journey(tt_a)
```

但 `simulate_bandit_journey()` 内部也会对 `AdaptiveBetaBanditRouter` 调用 `begin_journey()` 和 `end_journey()`。所以每个 journey 至少有两个问题：

- 记录的 `current_beta` 可能不是实际 simulation 用的 beta，因为 simulator 里又采样了一次；
- 同一 travel time 被更新两次，beta 权重被重复计入。

这个脚本生成的 convergence 曲线不能作为可靠证据。建议只保留一个生命周期入口：要么 simulator 管 begin/end，要么外层脚本管，不能两边都管。

## 次要但会影响算法结论的问题

### P1. V2/V3 接收了 `route_priors_override`，但没有使用 historical cancel prior

位置：

- `src/bandit_router_v2.py:197-219`
- `src/bandit_router_v2.py:104-113`
- `experiments/swiss_full/build_route_priors_loo.py:40-47`

LOO prior 文件包含 `mean/std/cancel_rate`，V1 会把 `cancel_rate` 转成 Beta prior；但 V2/V3 的 `RouteEnsembleBelief` 只用 historical `mean/std` 初始化 ensemble，`cancel_rate` 仍是：

```python
if self.n_attempts == 0:
    return 0.0
alpha = 1 + self.n_cancels
beta = 99 + ...
```

这意味着 A4 在 V2/V3 上不是完整的 route prior，而只是 delay prior。若论文/实验描述说 V2/V3 都使用 34-day/LOO mean/std/cancel prior，当前代码不匹配。

建议给 `RouteEnsembleBelief` 增加 `cancel_alpha/cancel_beta`，按 V1 同样从 historical cancel rate 初始化。

### P1. LOO 脚本写好了，但结果文件尚未生成，paper 里的 35-day 数字还不是 LOO 数字

位置：

- `experiments/swiss_full/build_route_priors_loo.py:1-10`
- `experiments/swiss_full/run_multi_day_loo.py:143-145`
- 当前 `experiments/swiss_full/results/` 下没有 `swiss_multi_day_loo.json`

新增 LOO prior 代码解决了“target day 泄漏”的方向问题，但目前只看到 `data/route_priors_loo.pkl`，没有 LOO multi-day 结果 JSON。当前 `paper.tex` 仍引用 `49.34 -> 46.91/47.11` 那组结果，无法判断是否来自 strict OOS。

建议先跑：

```bash
python experiments/swiss_full/run_multi_day_loo.py
```

然后补一个 LOO paired analysis，确认 V2/Adaptive 的 delta 和 CI 是否仍成立。没有这步，不应把 A4 称为严格 out-of-sample。

### P1. `simulate_bandit_journey()` 仍用 hard-coded class tuple 判断 online router

位置：

- `src/simulate_bandit.py:60-63`
- `experiments/swiss_full/run_csa_meat_baseline.py:109`

当前仍是：

```python
is_bandit = isinstance(router, (BanditRouter, BanditRouterV2, ...))
```

新 baseline 如果忘记加进 tuple，会被静默当成 StaticRouter 执行，结果错误但不报错。`run_csa_meat_baseline.py` 已经出现了自定义 baseline，这说明扩展点已经在压力下。

建议改成 capability-based protocol：

```python
is_online_router = all(hasattr(router, x) for x in (
    "route", "select_connection", "observe_delay", "observe_cancel"
))
```

或者定义 `typing.Protocol`，并给 simulator 写一个单测：未知 router 只要实现接口就走 online 分支。

### P1. `hash()` 仍用于业务 ID/特征，跨进程不可复现

位置：

- `src/gtfs_parser.py:39-40`
- `src/neural/features.py:83-86`

`hash(sid)` 和 `hash(conn.route)` 受 Python hash randomization 影响，不同进程/不同启动会变。Swiss 实验大量使用 multiprocessing，这会影响 stop id 稳定性、缓存、OD 名称映射和 neural feature reproducibility。

建议：

- GTFS stop id：按原始 `stop_id` 排序后分配连续 int，并保存 `stop_id_map`。
- route feature：用稳定 hash，例如 `hashlib.md5(route.encode()).hexdigest()` 截断，或建立 route vocabulary。

### P1. SW-LCB cancellation 会把 `True` 写进 delay window

位置：

- `src/sw_lcb_router.py:57-65`
- `src/sw_lcb_router.py:72-87`

`observe_cancel()` 当前做了：

```python
self._push(True, self._cancel_window, self.window_size)
self._push(True, self._delay_window, self.window_size)
```

`True` 在 Python 里等于 `1`，所以取消事件会作为 1 分钟 delay 进入 `window_mean/window_std`。这会压低被取消路线的 delay 均值，抵消一部分 cancel penalty。取消和 delay 应分开统计：cancel 只进 cancel window，成功 boarding/no-cancel 才进 delay window。

### P2. V1 historical cancel prior 对低取消率有 1% 下限，可能抹平 route 间差异

位置：

- `src/bandit_router.py:237-247`

当前：

```python
p_cancel = max(min(p['cancel_rate'], 0.5), 1e-4)
cancel_alpha = max(p_cancel * 100, 1.0)
cancel_beta = max(100 - cancel_alpha, 1.0)
```

当 historical cancel rate 小于 1% 时，`cancel_alpha` 被抬到 `1.0`，prior mean 变成约 1%。如果 Swiss normal route 大量低于 1%，A4 的 cancel_rate 区分度会被压平。建议允许 `alpha < 1`，或用更大的 pseudo count 保持均值，同时用单独下限控制数值稳定。

## 已确认的有效改进

1. `route_priors_override` 已接入 V1/V2/V3/Adaptive-beta，LOO evaluation 可以不再依赖全局 `data/route_priors.pkl`。
2. `build_route_priors_loo.py` 能为每个 target date 构造排除目标日的 per-route prior，这是修复 A4 泄漏风险的正确方向。
3. V2 cold-start 已改为使用 `posterior_std`，不再只依赖 cold-start 恒为 0 的 ensemble disagreement。
4. `run_adapt_beta_convergence_R15.py` 至少意识到了 shared state 需要按 scenario reset；这个原则应推广到 multi-day 主实验。
5. 当前基础测试通过：`python -m pytest -q tests -o cache_dir=/tmp/bapr_hro_pytest_cache`，结果 `10 passed in 1.60s`。

## 建议的最小修复顺序

1. 修 A7 deadline：所有 `prob_le(120)` / `prob_le(max_time)` 改成绝对 `t_source + max_time`。
2. 打通 `observe_cancel(route, kind)`，late no-show 不再写入普通 delay posterior。
3. 固化 Adaptive-beta reset scope，主实验 JSON 写明 scope；修掉 convergence 脚本双 begin/end。
4. V2/V3 使用 historical cancel prior，保持 A4 语义一致。
5. 改 simulator online-router 判断为 Protocol/capability-based。
6. 修 `hash()` 不稳定问题。
7. 增加 `tests/test_router_scoring.py`、`tests/test_adaptive_beta_state.py`、`tests/test_swiss_prior_loo.py`。
8. 重新跑 LOO multi-day + paired bootstrap，再更新论文/报告数字。

## 当前判断

代码方向是对的：A7 layered risk、A4 LOO prior、V2 cold-start fix、Adaptive-beta 跨 journey 学习都在往能提升算法效果的方向走。但当前还不能把新结果当成稳固证据，因为分数函数和实验状态边界仍有 P0 级问题。

如果只修一个点，先修 timeout deadline。这个 bug 会直接改变每个候选 connection 的排序，是当前最可能让 reach-rate 结论失真的地方。
