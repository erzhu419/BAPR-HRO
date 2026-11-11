# 同信息strict-native城市对照

新`NativeEventRouter`将全部Evolving后验事件边际投影到临时minute graph，调用原strict native TopoCSA；实际graph/world仍为seconds，root按原MCT、12min、deadline、pickup执行。使用capped root评分，内部仍为作者conditional-mean严格标签；明确记作`native_capped`，不称原作者同目标算法。3项针对性测试通过。

原July26/27、两个原OD、两种时钟共8趟完成，所有连接、信息和实测覆盖均与已有对照匹配。

|时钟|native平均损失秒|UCT64|analytic_recourse|native平均整趟计算秒|
|---|---:|---:|---:|---:|
|logical|1559.75|1345.75|1396.50|389.26|
|charged|3540.00|1566.00|1825.00|235.42|

logical四趟全到达；两长例从首选9282即不同于UCT8315，所有20段均board，损失2290/2275来自路线选择差异。charged三趟timeout、一趟3360秒到达，共16次wait_late；秒级轨迹证实30秒左右重算期间错过短dwell。最长logical病例607.9秒来自20次全图查询，单query最大31.37秒，并非卡住。

冷root保留27895连接、55790事件、3905300边际格，后验7.19秒、构图.31秒、查询36.50秒。精确概率数组共享复用原native积分缓存：55790数组归并15925种，策略/删环/标签值/PMF逐位相等，查询37.81→28.36秒（CPU35.18→26.41秒）；未舍弃质量或候选。旧运行只停止本人PID，已完成行及停止状态另存，不把未完成当失败。

限制：事件量化最大30秒，query source向下取整最大58秒，MCT本面板无取整误差；该minute输入和独立marginal递推不是完整相关seconds模型。这里每步重建后验的Python实现成本，不能外推原算法复杂度或优化C++速度。CSA–MEAT早已有包含错过换乘后替代车次的[decision graph（2018，§6）](https://ben-strasser.net/paper/connection_scan_algorithm_acm_j_exp_algorithmics.pdf)，本次补齐质量/执行对照，不以backup本身或慢port证明OR创新。

回执：`native_city_interned_results.json`、逐例`native_city_interned_cases/`；表示等价比较`native_shape_results.json`，原始冷成本`native_cold_results.json`。
