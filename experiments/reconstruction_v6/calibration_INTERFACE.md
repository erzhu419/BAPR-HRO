# 当前公开状态下的联合投影接口

```python
from src.trip_marginal_tilt import TripMarginalTilt

# Original public belief stays in the real router; this is a query-local copy.
joint = TripMarginalTilt(public_belief, parameters, {cid: conditional_clock_law})
bank = joint.sample(parameters, rng, n)
times, weights, cancel = joint.event_distribution(cid, 'departure', parameters)
diagnostic = joint.projection_summary({cid: traveler_ready}, deadline=deadline)
```

`targets`可含不同trip的多个cid，但每trip只能一个未知departure。全部actions必须共享这个`joint`。`copy`、`_law(trip, parameters)`、`mean_events`、`sample`、observations/forecast/wait/attempt和`hard_support`均可用。`applied_cids`、`skipped_cids`保留实际投影/未投影原因。单cid形式`TripMarginalTilt(base, parameters, cid, law)`也可用。

未来virtual branch应复制并condition该对象；不能读私有branch事件再改public_belief。真实旅程仍把反馈交给原persistent belief，每个新H再创建投影。冻结网格之外的事实要求新context；不会将它偷偷添加为新prior atom。改变当前context参数也要求新context。

冻结条件预测器`calibration_frozen_model.pkl`是SBB16站、Oct1–2训练的Student-t3 law，`predict(incoming_source, outgoing_source, observed_arrival)`返回秒级CDF/quantile/mean/board_probability。`marginal`里的route基准也属于该铁路人口；其pooled fallback不是城市公交的已校准先验。城市须用自身训练源或明确标为跨域实验。

`projection_summary`返回continuous catch（使用原joint服务质量）、projected catch、差值与两种quantiles。当前calibration date-block envelope只校准预测集合，不改CDF或概率。改造后的有限joint需要另行读出，不能把集合coverage当作概率校准，也不能将pair/group人口的coverage当选择后或宏后悔保证。目标risk functional及旅客访问人口需要各自匹配验证。
