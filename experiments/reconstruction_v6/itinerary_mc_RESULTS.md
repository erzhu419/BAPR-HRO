# 同一编译策略的解析积分与MC误差

固定primary、首次失败后的rescue以及root posterior mean发车阈值；后续按实际到站相位WAIT60秒到菜单/阈值开放，再attempt。成功才进入下一段，根失败切固定rescue，任何更晚失败停至deadline。MC执行器只调用原committed transitions，不读未来world clock决定动作，不更新发车阈值。5项全路径枚举测试通过，覆盖部分取消、同车块、不同车61秒MCT、WAIT相位和首次失败补救；与`CommitmentBounds.itinerary/with_rescue`逐世界加权值一致。

真实图固定July26根0/5、July27根0，各构造名义路径和代表备选，路径构造后才读取实际首段作诊断。每策略8个模拟seed、4/8世界，比较IID和保留原primitive tape的比例分层；共96个配对trial、192个完整policy估计。两方法共用原HMM和同policy，最多96次transition/两policy，未截断、丢弃失败或把restricted family当完整最优。[所有回执/协议](itinerary_mc_results.json) · [方差、RMSE及family-regret汇总](itinerary_mc_summary.json)

| public root | 解析nominal / alternative秒 | IID4错排/8 | 分层4错排/8 | IID8错排/8 | 分层8错排/8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| July26/0 | 2031.071 / 1319.788 | 2 | 2 | 0 | 1 |
| July26/5 | 83.533 / 1022.288 | 0 | 0 | 0 | 0 |
| July27/0 | 2031.071 / 1319.788 | 2 | 2 | 0 | 0 |

初始根两策略的解析差为711.283秒，IID4和分层4的平均family-regret均177.821秒；已观测根的差938.755秒，所有MC选择正确。这里family-regret是两个已固定witness解析成本之差，不是完整POMDP后悔。两个初始根具有相同静态图/过去参数/公开状态，因此解析值相同，不能将它们当三个独立模型证据。

另外15次等口径timing中，每方法只建一次fresh公共context，两policy共享阈值和cache。积分phase耗时12.8–18.8毫秒，IID4为38.7–43.9毫秒、分层4为50.6–56.2毫秒，8世界50.9–87.2毫秒。公共context/阈值准备后，完整耗时为0.15–0.36秒，未显示普遍端到端提速。路径发现另需约3.44/1.00/3.28秒。旧V5重新规划tail数秒的开销不能全归为MC；编译固定policy本身已经省去了反复全图选路。[同口径timing与实际源码](itinerary_mc_timing_results.json)

模型fit仍是独立问题：解析积分在两个初始根都偏好7588，但它实际08:01:23/08:01:37已发走，8341实际08:05:15/08:05:21仍可赶；另一备选11321实际未确定，保留unknown。去掉MC噪声不能修复给定模型的错误偏好。上述积分都是当前网格模型的期望，实际单次出行不是概率truth，更不是独立新日期的政策确认。

复现：`python3 -m pytest -q tests/test_itinerary_mc_evaluator.py`；`itinerary_mc_compare.py`生成固定路径/冻结输入与96trial，`itinerary_mc_timing.py`读取冻结输入做15次fresh同口径计时，`itinerary_mc_summary.py`组装统计。已有协议拒绝跨源码续跑，应另选output保存新版本。
