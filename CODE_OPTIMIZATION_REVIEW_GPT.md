# BAPR-HRO 代码项目优化分析

日期：2026-04-29

## 结论概览

BAPR-HRO 当前已经不是单一 transit demo，而是一个研究型实验仓库：`src/` 是核心公交 hyperpath + LCB/DRO/EXP3/BAMCP 路由实现，`experiments/swiss_full/` 是真实 Swiss GTFS 实验，`crew_scheduling/`、`VRP/`、`power_dispatch/`、`sdn_routing/`、`adaptive_sign/` 是跨域压力测试和扩展验证。项目能跑通核心测试，且已有真实多日、多 OD 和跨域结果，但工程层面仍然偏“论文实验脚本集合”，离稳定、可复现、可维护的代码项目还有明显距离。

最值得优先做的不是继续增加算法变体，而是先把实验入口、配置、依赖、随机性、测试边界和数据流水线收紧。否则后续任何新结果都很容易受脚本差异、硬编码参数、环境依赖和随机种子影响。

算法效果层面，最优先的方向是把当前 LCB/DRO 从“优化成功样本的到达时间”调整为“保 reach / 降 timeout 约束下优化 travel time”。后文的“算法效果优化”章节给出了具体改法：分层风险 score、状态校准 beta、typed cancellation posterior、route-hour 层级先验、自适应 top-k/lookahead、换乘 miss-risk penalty，以及对应消融矩阵。

## 当前结构

核心代码：

- `src/durner/topocsa.py`、`src/durner/preprocessing.py`：Durner TopoCSA / successor graph / cycle cutting。
- `src/transit_graph.py`、`src/pmf.py`：网络和离散时间分布基础结构。
- `src/bandit_router.py`、`src/bandit_router_v2.py`、`src/bandit_router_v3.py`：V1/V2/V3 LCB 路由器。
- `src/dro_router.py`、`src/ssp_mdp.py`、`src/bamcp_router.py`、`src/sw_lcb_router.py`、`src/exp3_router.py`、`src/oracle_router.py`：baseline 和对照方法。
- `src/simulator.py`、`src/simulate_bandit.py`：journey 模拟器。
- `src/gtfs_parser.py`、`src/gtfs_rt_parser.py`：GTFS / GTFS-RT 数据处理。

实验与扩展：

- `experiments/run_full_comparison.py`：合成 small/large 网络主实验。
- `experiments/swiss_full/`：Swiss 多 OD、多日、消融、可扩展性实验。
- `crew_scheduling/`、`VRP/`、`power_dispatch/`、`sdn_routing/`：跨域验证。
- `adaptive_sign/`：自适应 sign / beta 跨域整合实验。

验证现状：

- `python -m pytest -q tests -o cache_dir=/tmp/bapr_hro_pytest_cache` 通过，结果为 `10 passed in 1.05s`。
- 直接运行全量 `python -m pytest -q` 会失败，因为 pytest 会收集 `VRP/svrbench/`、`mlopt/`、`power_dispatch/rl4uc/` 等第三方或 vendored 子项目测试，缺少 `rl4co`、`scikits`、`rl4uc` 等依赖，并且当前目录在沙箱内不可写 `.pytest_cache`。

## 优先级 P0：先保证可复现和可运行边界

### 1. 增加仓库级 README 和环境文件

当前仓库根目录没有 `README.md`，也没有统一的 `requirements.txt` / `pyproject.toml`。这会导致两个问题：别人不知道核心入口是什么；测试和实验依赖只能从脚本 import 反推。

建议：

- 新增 `README.md`，明确三类入口：
  - 快速测试：`python -m pytest -q tests -o cache_dir=/tmp/bapr_hro_pytest_cache`
  - 合成实验：`python experiments/run_full_comparison.py`
  - Swiss 实验：`python experiments/swiss_full/run_multi_day.py`
- 新增 `pyproject.toml` 或 `requirements.txt`，至少覆盖核心依赖：`numpy`、`pandas`、`scipy`、`networkx`、`torch`、`protobuf`、`gtfs-realtime-bindings`、`pytest`。
- 将可选依赖拆成 extras，例如 `neural`、`swiss`、`vrp`、`uc`、`sdn`，避免核心 transit 测试被跨域依赖拖垮。

### 2. 配置 pytest 只收集项目测试

全量 pytest 失败不是核心代码失败，而是测试发现边界不清。仓库里有多个克隆参考项目和第三方测试目录，默认收集会污染 CI。

建议在 `pyproject.toml` 中加入：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
norecursedirs = [
  "VRP/svrbench",
  "mlopt",
  "power_dispatch/rl4uc",
  "sdn_routing/DRL-GNN",
  "hybrid-cp-rl-solver",
  "learntocut",
  "public-transport-statistics",
  "data",
  "experiments_log",
]
```

如果跨域模块要保留测试，应单独建 `tests_cross_domain/`，并用 marker 控制，例如 `pytest -m vrp`、`pytest -m uc`。

### 3. 固定 GTFS stop id，不要用 Python `hash`

`src/gtfs_parser.py` 中 `load_stops()` 使用：

```python
id=hash(sid) % (10**9)
```

Python 的字符串 hash 默认跨进程随机化，这会让同一份 GTFS 在不同进程、不同机器上生成不同 stop id。影响包括：pickle 结果不可稳定复现、多进程实验中 name/id 映射可能漂移、日志和结果文件难以比较。

建议改为稳定映射：

- 构图时维护 `stop_id_to_int = {sid: i}`，按文件顺序或排序后的 stop_id 分配连续整数。
- 若需要跨脚本持久化，保存 `stop_id_map.json`。
- 禁止在核心数据结构中使用 `hash()` 生成业务 id。

这是 P0 级问题，优先级高于算法微调。

## 优先级 P1：统一算法接口，减少模拟器耦合

### 4. 定义统一 Router Protocol

`src/simulate_bandit.py` 现在用 `isinstance(router, (...大量类...))` 判断 bandit router，并直接假设对象有 `observe_delay()`、`observe_cancel()`、`select_connection()`。每新增一个 router 都要改模拟器。

建议定义一个轻量协议：

```python
class OnlineRouter(Protocol):
    total_observations: int
    def route(self, s_source: int, s_dest: int, t_source: int) -> HyperpathResult: ...
    def observe_delay(self, route: str, delay: float) -> None: ...
    def observe_cancel(self, route: str) -> None: ...
    def select_connection(self, stop_id: int, current_time: int,
                          rng: np.random.Generator, top_k: int = 5) -> Optional[tuple[StopLabel, float]]: ...
```

然后模拟器只检查 `hasattr` 或直接依赖 Protocol。这样 `simulate_bandit_journey()` 不需要 import 所有 router 类，也不会被跨域/新增 baseline 反复修改。

### 5. 把硬编码参数集中到配置对象

项目里大量关键参数散落在脚本和方法默认值中，例如：

- 出发时间：`480`、`490`
- regime 切换：`490`、`540`、`560`
- 最大时长：`120`、`180`
- 候选窗口：`current_time + 25`
- 乘客耐心：`12`、`10`
- cancel sentinel：`999`
- beta/gamma/cancel penalty：`1.5`、`60`

建议新增 `src/config.py`：

```python
@dataclass(frozen=True)
class SimulationConfig:
    max_time: int = 120
    departure_base: int = 490
    departure_jitter: int = 10
    candidate_lookahead: int = 25
    patience: int = 12
    cancel_delay_threshold: int = 25
    cancel_sentinel: int = 999
```

实验脚本只构造配置，不在业务逻辑中重复写 magic number。这样消融和多日实验也能输出完整 config，便于复现。

### 6. 统一取消事件表达，不要继续扩散 `999`

`src/simulator.py` 用 `999` 表示取消，`src/simulate_bandit.py` 再用 `delay == 999` 判断；另一些实验用 `arrival_time >= 99999` 判断异常。这种 sentinel 容易污染统计。

建议：

- 定义 `SampledOutcome(delay: float | None, canceled: bool)`。
- `sample_actual_delay()` 改为 `sample_connection_outcome()`。
- 所有 router 的 observation API 区分 `observe_cancel()` 和 `observe_delay()`，模拟器不再用数值哨兵传递业务事件。

## 优先级 P1：提升 TopoCSA 和 GTFS 性能

### 7. successor graph 构建需要缓存和更强索引

`src/durner/preprocessing.py` 每次 `topocsa()` 都重新：

- 扫描所有 connections 过滤 active ids；
- 构建 `stop_to_deps`；
- 对每个 active connection 遍历同站后续 departure；
- 重新 DFS cycle cutting。

可扩展性结果显示 routing time 从 100 connections 的 1.6ms 增至 2000 connections 的 476.5ms，3000 connections 还出现找不到 connected pair 的情况。当前性能对真实网络仍偏紧。

建议：

- `TransitGraph` 构建时维护按 `dep_time` 排序的 per-stop departure 索引。
- successor 查询用二分按 `[arr_time - slack, arr_time + max_wait]` 取候选，而不是扫描同站全部 active departures。
- 对 `(graph_version, t_start, t_end, transfer_policy)` 缓存 topological order 和 cuts。
- regime 更新只改变 PMF，不改变拓扑时，复用 successor graph；只有 transfer policy 或时间窗变化时重算。

### 8. `load_stop_times()` 不应反复全文件扫描

`src/gtfs_parser.py` 当前构图流程至少扫描 `stop_times.txt` 两遍，并且每次 scalability benchmark 都重新从 CSV 解析。`scalability.json` 里 build time 基本稳定在 11 秒左右，说明开销主要来自解析而不是 `max_connections`。

建议：

- 第一次解析 GTFS 后生成 normalized parquet/pickle cache：`stop_times.parquet`、`trip_routes.pkl`、`stops.pkl`。
- 以 `(gtfs_dir, region_bbox, time_window, max_connections)` 做 cache key。
- `run_scalability.py` 不应每个 size 重扫原始 CSV，可以先构建全量 region graph，再切片连接数或共享预处理结果。

## 优先级 P1：实验流水线工程化

### 9. 合并重复实验入口

现在有多套相似入口：

- `experiments/run_full_comparison.py`
- `experiments/run_comparison.py`
- `experiments/run_v1_vs_v2.py`
- `experiments/swiss_full/run_multi_od.py`
- `experiments/swiss_full/run_swiss_multi_od_v2.py`
- `experiments/swiss_full/run_swiss_multi_od_v3.py`
- `experiments/swiss_full/run_multi_day.py`

这些脚本重复定义 methods、schedule、seed、max_time、输出格式。建议新增一个统一 runner：

```text
src/experiment/
  registry.py        # method registry
  configs.py         # dataclass configs
  metrics.py         # reach, cond_mean, timeout, paired bootstrap
  runner.py          # run_cell / run_grid
experiments/
  run.py             # CLI: --config configs/swiss_multi_day.yaml
configs/
  synthetic_full.yaml
  swiss_multi_od.yaml
  swiss_multi_day.yaml
  cross_domain.yaml
```

这样 paper 表格和日志都从同一套 metric 生成，避免 V2/V3 脚本间统计口径漂移。

### 10. 结果文件要包含 code version 和 config hash

当前结果 JSON 有部分 config，但没有 git commit、dirty state、Python 版本、依赖版本、输入数据 hash。建议每个实验输出：

```json
{
  "run_meta": {
    "git_commit": "...",
    "git_dirty": true,
    "python": "3.12.x",
    "numpy": "...",
    "config_hash": "...",
    "data_hashes": {
      "zurich_wide.pkl": "...",
      "per_day_distributions.pkl": "..."
    }
  }
}
```

对于 OR/论文审稿，结果可追溯性比再多一个 baseline 更重要。

### 11. 多进程实验避免复制大图对象

`experiments/swiss_full/run_multi_day.py` 中 worker 初始化持有 `_GRAPH`，但每个 method / seed 又频繁 `copy.deepcopy(_GRAPH)`，并且调用 `simulate_bandit_journey(copy.deepcopy(_GRAPH), ri, ...)`，其中 router 本身已持有另一个 deepcopy 的 graph。这会制造大量内存和 CPU 开销。

建议：

- 将 `TransitGraph` 设计为不可变 graph + 独立 mutable belief/router state。
- 对 regime distribution 采用 copy-on-write 或 route-level overlay，不直接修改 graph。
- `simulate_bandit_journey()` 使用 `router.graph`，不要再额外传入另一个 graph 副本。
- 至少先修正 double-copy：`ri = make_router(copy.deepcopy(_GRAPH))` 后调用 `simulate_bandit_journey(ri.graph, ri, ...)`。

## 优先级 P0/P1：算法效果优化

这一节专门回答“怎么提升算法效果”。当前结果里的核心矛盾不是单纯 travel time，而是 reach rate、timeout、conditional mean 之间存在 trade-off：LCB/DRO 在部分实验里能降低成功样本的条件均值，但会牺牲一部分到达率。因此算法优化应先围绕“少 timeout、保 reach，再缩短时间”重新校准目标。

### A1. 把目标从单一 mean travel time 改成分层风险目标

当前统计同时出现：

- Static reach 更高；
- V1/V2/V3/Adaptive 的 conditional mean 更低；
- normal 场景下 LCB 系列可能伤害 reach；
- disrupted 场景下改善不稳定。

如果算法只按 `mean_dest_arrival + beta * uncertainty + cancel_penalty` 排序，容易选到“看起来更短但更容易断路/错过换乘”的候选。建议把打分函数改成分层风险目标：

```text
score = timeout_risk_weight * P(timeout or infeasible)
      + arrival_time_quantile
      + transfer_risk_weight * P(missed_transfer)
      + cancel_risk_weight * P(cancel)
      + uncertainty_penalty
```

优先级应是：

1. 先最大化 reach / minimize timeout；
2. 再优化 reached journeys 的 travel time；
3. 最后优化 transfer count 或等待时间。

具体实现上，可以先不用复杂模型，直接在候选评分中加入：

- `1 - label.feasibility` 作为 infeasibility penalty；
- `P(actual_dep > current_time + patience)` 作为 wait-too-long penalty；
- `P(next transfer missed)` 作为 transfer-risk proxy；
- timeout penalty 使用 `max_time`，不要只在最终统计时才惩罚。

这会直接减少“条件均值好看但 reach 下降”的问题。

### A2. LCB 分数需要从固定 beta 改成按状态校准的 beta

V1 当前用 disruption gate 把 `beta_eff = beta * gate`，V2/V3 用 OOD/topology gate。但 gate 仍偏启发式：几个阈值如 `cancel_threshold=0.05`、`delay_threshold=10.0`、`beta_base=1.0`、`beta_ood=1.0` 没有系统校准。

建议建立 beta 校准流程：

- 离线按 OD、day category、departure window 网格搜索 `beta`、`cancel_penalty_weight`、`patience`。
- 目标函数使用 `unconditional_mean = travel_time if reached else max_time`，并额外约束 reach 不低于 Static。
- 对 normal/mild/severe 三类日子分别学习 beta 先验。
- 在线时 `beta_eff` 从后验 day category 和 local observations 平滑插值，而不是直接阈值开关。

推荐先做一个简单可落地版本：

```text
if normal_probability high:
    beta_eff = 0 or small
elif disruption_probability high:
    beta_eff = tuned_beta_disrupted
else:
    beta_eff = interpolation
```

这比“所有路线同一个 beta”更可能同时保住 normal-day reach 和 disrupted-day robustness。

### A3. 取消模型要区分 true cancel、feed missing、late no-show

现在 cancellation 主要由大 delay 或 `999` sentinel 触发，且 GTFS-RT 中 delays > 30min 被当作 effective cancellation。真实数据里 no-show 可能来自 feed 缺失、vehicle matching 错误、短线调整，不一定是真取消。把这些全部作为 cancel 会让 LCB 过度惩罚某些路线，降低 reach。

建议把取消事件拆成三类 posterior：

- `p_true_cancel`：明确取消或长期无可达车辆；
- `p_feed_missing`：GTFS-RT 缺失/匹配失败；
- `p_late_no_show`：超过 patience 未到。

打分时不要统一乘 `60`：

```text
cancel_penalty =
    w_true_cancel * p_true_cancel
  + w_feed_missing * p_feed_missing
  + w_late_no_show * p_late_no_show
```

其中 `w_true_cancel > w_late_no_show > w_feed_missing`。这样能减少因为数据质量问题误杀路线。

### A4. 冷启动不要让所有候选的 uncertainty 完全同质

V2 已经修过 ensemble cold start：不再只用 `ensemble_std`，因为所有 bootstrap 成员同初始化会导致 std 为 0。但另一个问题是：如果所有路线 prior 完全相同，uncertainty penalty 对排序没有区分度；如果 prior 太大，又会整体推高分数但不改善相对排序。

建议引入 route/time-of-day 层级先验：

- route-level：按历史 route delay mean/std/cancel_rate 初始化；
- hour-level：早高峰、平峰、晚高峰分别初始化；
- OD-local：只用当前 OD hyperpath 中出现过的路线做 shrinkage；
- global fallback：数据少时才退回全局 Swiss prior。

形式上可以用 hierarchical shrinkage：

```text
prior(route, hour) =
    w_route_hour * empirical(route, hour)
  + w_route * empirical(route)
  + w_global * global_prior
```

这样 cold start 时 V1/V2/V3 就能基于历史可靠性区分路线，而不是只能等在线观测。

### A5. 候选集 top-k 和 lookahead 应自适应

当前不同模拟器里存在固定 `TOP_K_ROUTES = 3`、`top_k=5`、`current_time + 25`、patience 10/12 分钟等设置。算法效果很容易被这些值限制：如果正确路线排在 top-k 外，LCB 再聪明也选不到；如果 lookahead 太短，会过早 timeout；太长又会引入慢车。

建议：

- top-k 根据 stop 的 route diversity 和当前 disruption risk 自适应：normal day 用较小 top-k，disrupted day 扩大 top-k。
- lookahead 根据 headway 分布自适应：high-frequency route 用短 patience，low-frequency route 用长 patience。
- 将 `top_k`、lookahead、patience 纳入联合调参，不要只调 beta。

一个简单规则：

```text
top_k = min(max_k, 2 + number_of_recent_cancels_at_stop)
lookahead = base_lookahead + disruption_gate * extra_lookahead
patience = route_headway_quantile(route, 0.75)
```

### A6. 从 per-route belief 升级为 route-stop-time belief

当前 V1/V2 主要按 `route` 维护 belief。公交延误和取消通常强依赖 stop、方向、小时、是否在换乘枢纽。如果只按 route 聚合，可能把远端路段的异常污染当前 OD。

建议 belief key 从：

```text
route
```

升级为分层 key：

```text
(route, direction, hour_bucket)
(route, stop_cluster, hour_bucket)
route fallback
global fallback
```

在线观测时先更新最细粒度；数据不足时向 route/global 回退。这样能减少“全线路一刀切惩罚”，提升 normal-day reach。

### A7. 加入换乘链路级风险，而不只是当前 connection 风险

BAPR-HRO 的核心是 hyperpath rerank，但当前 LCB score 主要看当前 label 的 mean destination arrival 和 route belief。真正导致 timeout 的往往是后续换乘断裂，而不是当前车本身。

建议对每个候选 label 估计：

- 当前车取消概率；
- 当前车晚点导致下一跳不可达概率；
- 后续 hyperpath 是否仍有 backup route；
- 到达下一个 transfer stop 后的 expected slack。

可以先用已有 PMF 做近似：

```text
transfer_slack = next_dep_distribution.mean - arr_distribution.mean - min_transfer_time
miss_prob = P(arrival + transfer_time > next_departure)
```

然后把 `miss_prob` 加入 score。这个改动直接针对 reach rate，优先级高。

### A8. Adaptive-beta 要改成按 OD/场景共享学习

`AdaptiveBetaBanditRouter` 现在每次 journey 采样 beta，journey 结束后更新。单次乘客出行场景里，在线学习样本太少；如果每个 OD/day 都从零开始，EXP3/Hedge 很难在有效时间内学到稳定 beta。

建议：

- 将 Adaptive-beta 学成跨 journey、跨 OD 的 meta-policy；
- 输入特征包括 day category、current stop route diversity、recent cancel/delay、OOD、topology gate；
- 输出 beta 或 beta distribution；
- 每天/每批 OD 共享更新，而不是每个乘客从零开始。

轻量实现可以不是神经网络，而是表格策略：

```text
state_bin = (day_category, route_diversity_bin, cancel_rate_bin, delay_bin)
beta = table[state_bin]
```

用历史实验日志离线拟合初值，再在线微调。

### A9. 用“保 reach 约束下的 travel-time 优化”做调参

建议所有算法调参统一使用以下目标：

```text
minimize   E[travel_time_with_timeout]
subject to reach_rate >= reach_rate_static - epsilon
```

或者用拉格朗日形式：

```text
objective = mean_travel_time
          + lambda_timeout * timeout_rate
          + lambda_reach_drop * max(0, static_reach - method_reach)
```

这样可以防止算法通过放弃困难样本来降低 conditional mean。`lambda_timeout` 应该大于一次普通晚点的代价，例如 60-120 分钟。

### A10. 必做消融矩阵

为了判断算法效果到底来自哪里，建议新增一个固定消融矩阵：

| 组件 | 开/关 |
|---|---|
| route/hour hierarchical prior | on/off |
| disruption gate | on/off |
| typed cancel posterior | on/off |
| transfer miss probability penalty | on/off |
| adaptive top-k/lookahead | on/off |
| Adaptive-beta table/meta-policy | on/off |

每个 cell 报告：

- normal reach / disrupted reach；
- unconditional mean；
- conditional mean；
- timeout rate；
- paired OD difference；
- per-OD win/loss count。

如果只能先做一个最小版本，优先顺序是：

1. 加 `label.feasibility` 和 transfer miss penalty；
2. 加 route/hour hierarchical prior；
3. 加 typed cancel posterior；
4. 调 beta/top-k/lookahead 的 constrained objective。

## 优先级 P2：算法实现层面的稳健性

### 12. V1/V2/V3 的 posterior 更新需要单测覆盖

当前测试只覆盖 PMF、合成网络和 TopoCSA，缺少关键算法测试。建议增加：

- `RouteBeliefState`：连续 delay 更新后 posterior mean/var 是否单调合理。
- cancellation prior：无观测时不产生跨路线偏置；有 cancel 后 penalty 生效。
- disruption gate：`total_observations < 3` 时 beta 为 0，超过阈值后上升。
- V2 cold start：`posterior_std > 0`，防止 ensemble_std 全零。
- V3 topo gate：单候选 route 时 beta 为 0，多候选 route 时 beta 恢复。

这些测试比继续跑大实验更便宜，也能防止“修一个真实数据问题又破坏合成实验”。

### 13. V2 ensemble variance 更新公式应单独校验

`RouteEnsembleBelief.update_delay()` 中 `_vars[k]` 的在线更新写成：

```python
self._vars[k] += ((delay - old_mean) * (delay - self._means[k]) - self._vars[k]) / n
```

这是近似 running variance，但变量名 `_vars` 更像 population variance；同时 prior_n 参与了 count，但 prior variance 与实际样本方差混合方式不完全清晰。建议：

- 改为维护 Welford 的 `M2`，输出时再除以 `n - 1` 或 `n`。
- prior variance 和 bootstrap estimator variance 分开保存。
- 增加数值测试：输入固定序列 `[1, 2, 3]`，校验每个 estimator 的 mean/variance 合理。

### 14. Adaptive-beta 的 EXP3 更新需要更清晰的离线/在线语义

`AdaptiveBetaBanditRouter` 在一个 journey 结束后用已采样 beta 的 travel time 更新权重，但代码里还用每个 beta 历史平均成本来更新所有 beta。这个更像启发式 full-information update，而不是标准 bandit EXP3。

建议：

- 明确实现两种模式：`BanditFeedbackEXP3` 和 `FullInfoHedge`。
- 如果只有选中 beta 的成本，使用 importance-weighted loss。
- 如果能离线回放所有 beta，才更新所有 beta，并在实验名中叫 Hedge/Expert 而不是 EXP3。
- 输出 `beta_probs` 曲线，作为稳定性诊断，不只输出最终均值。

## 优先级 P2：仓库治理和数据管理

### 15. 仓库需要瘦身和边界隔离

当前目录大小约 7.9G。`.gitignore` 已忽略 `data/` 和部分克隆参考项目，但工作树中仍有：

- `gtfs_fp2023_2023-10-25_04-15.zip`
- 多个外部项目目录：`VRP/svrbench/`、`power_dispatch/rl4uc/`、`sdn_routing/DRL-GNN/`、`hybrid-cp-rl-solver/`、`mlopt/`、`learntocut/`
- 大量 `__pycache__` 和实验日志

建议：

- 将外部项目移为 git submodule 或 `external/`，并默认不参与测试。
- `.gitignore` 增加 `*.zip`、`.pytest_cache/`、`experiments_log/**/*.log`，但保留精选 JSON summary。
- 用 `data/README.md` 记录数据下载和生成命令，不提交原始大文件。
- 清理 pycache；以后由 `.gitignore` 防止进入版本控制。

### 16. 不要让论文构建产物和实验代码互相污染

仓库同时包含 `paper/`、`paper-backup-*`、实验结果、外部代码和核心库。建议：

- `src/` 保持可安装库。
- `experiments/` 只放可复现实验入口。
- `paper/` 只读取 frozen results，不反向修改实验代码。
- `archive/` 或单独分支保存历史 backup，不放在主工作树。

## 建议执行路线

第一轮，1 天内完成：

1. 算法侧先加 `label.feasibility` / transfer miss-risk penalty，并把调参目标改成 `travel_time_with_timeout + reach_drop_penalty`。
2. 新增 route/hour 层级先验，替代所有路线同质 cold-start prior。
3. 新增 `README.md`、`pyproject.toml`、pytest 收集配置。
4. 修复 `hash(sid)` 为稳定 id 映射。
5. 新增 `SimulationConfig`，先替换 `simulate_bandit.py` 中的硬编码。
6. 给 V1/V2/V3 belief 和 scoring 增加 8-12 个单元测试。

第二轮，2-3 天完成：

1. 增加 typed cancellation posterior，区分 true cancel、feed missing、late no-show。
2. 对 beta/top-k/lookahead 做 constrained tuning：reach 不低于 Static 的前提下最小化 unconditional mean。
3. 定义 router Protocol，解耦 `simulate_bandit_journey()` 的 `isinstance` 列表。
4. 合并 experiment method registry，减少 `run_*_v2/v3` 脚本分叉。
5. 每个实验输出 `run_meta`、config hash、data hash。
6. 修正 Swiss multi-day double-copy graph 问题。

第三轮，1 周内完成：

1. 把 Adaptive-beta 改成跨 OD/day 共享的表格或轻量 meta-policy。
2. 固化算法消融矩阵：prior、gate、typed cancel、miss-risk、自适应 top-k/lookahead、Adaptive-beta。
3. GTFS parser 增加 cache / parquet 中间层。
4. TopoCSA successor graph 增加 per-stop sorted index 和 query cache。
5. 梳理跨域项目为 optional extras / submodules。
6. 做一次端到端 benchmark：build time、routing time、memory、multi-day wall time、reach/unconditional mean/conditional mean。

## 当前最值得警惕的研究结论风险

真实多 OD paired bootstrap 已经显示：normal 场景下 V1/V2/V3/DRO 相对 Static 的 reach rate 多为负；disrupted 场景下改善区间仍较宽。multi-day summary 中 Static reach 约 78.2%，V1/V2/V3/Adaptive 约 75.3%-75.5%，但 conditional mean 更低。这说明方法更像“牺牲部分 reach，换取成功样本中的更短 travel time”，而不是无条件支配 Static。

因此论文和代码输出应同时报告：

- reach rate
- unconditional mean with timeout penalty
- conditional mean among reached journeys
- paired OD/seed difference
- timeout count
- per-OD heatmap

不要只报告 conditional mean，否则容易高估 LCB/DRO 的实际可用性。

## 最小验收标准

完成上述 P0/P1 后，项目应至少满足：

- normal/disrupted 场景都同时报告 reach、timeout、unconditional mean、conditional mean，且主调参目标不再只看 conditional mean。
- LCB/DRO score 至少包含 `label.feasibility` 和换乘 miss-risk penalty。
- cold start prior 至少按 route/hour 或 route/global 做层级初始化，不再所有路线完全同质。
- beta/top-k/lookahead 有固定 constrained tuning 流程，目标是保 reach 后降 travel time。
- 克隆后按 README 可安装核心环境。
- `python -m pytest` 默认只跑核心测试并通过。
- 每个实验可由一个 config 文件复现。
- 同一 GTFS 输入在不同 Python 进程生成一致 stop id。
- 主要结果 JSON 可追溯到 git commit、config、data hash。
- TopoCSA 在 2000 connections 级别保持稳定，3000+ connections 不因 OD 搜索策略失败而无法报告 routing benchmark。
