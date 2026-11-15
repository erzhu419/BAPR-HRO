# 固定14日校准：未达到放行条件

**本轮处理和核验完成；联合概率与站台验收均为 `unresolved`，暂不进入求解器深度重构。** 日期固定为2023-03-09—22，模型、菜单、MCT、时钟与验收阈值保持原约定。168根、1,328个双端事件全部保留，另有3,126个无后续对应发车的候选对；14根没有双端机会。

- **概率核验。** 1,223个模型分布、1,324个实际标签可确定；完整配对1,222个。105个模型未知来自3/9消息缺档的102个事件及root114初始登乘端点`SKIPPED`的3个事件。实际未知共4个：上述3个，加root150的一个后站发车仅有`PROGNOSE`。
- **冻结主检验。** 保留全部缺测界，按事件→车次连通事件簇→日期等权，使用原20,000次日期bootstrap。四个偏差区间均跨越±2个百分点验收边界；关键“仅上游能赶上”P10区间为 **[-27.60, +11.29]个百分点**。3/9完整性条件也未满足。完整配对事件的P10预测均值 **4.04%**、实测 **7.45%**，只是机会加权描述，不能替代主检验或宣称统计拒绝。全部四分类、分组和逐日结果见[summary.json](summary.json)、[by_day.csv](by_day.csv)、[完整配对描述](complete_pair_description.json)。
- **平台核验。** 2,656个已注册换乘对中，1,156对有明确规则且与固定MCT一致；1,500对缺少明确的同站台规则。实际站台确认 **0对**。决策前平台预告也已逐对保存，不能当作实际使用站台或实测步行时间。[逐对记录](platform_pairs.json.gz) · [决策前预告](platform_announcements.json.gz)

## 本轮修复

镜像目录日期不能代表实时feed使用的静态版。3/16—19的634个车次日经公开时刻表逐字段核验，物理服务与平台完全一一对应，但598个ID改名，38个同字面ID已指向别的服务。已统一翻译为冻结图的车辆身份；8,355帧的原始预测值、端点CID和receipt均保持不变，四日重算的物理状态、注册事件、实际标签及平台记录逐项相同。两版在涉及的716个换乘对上适用规则也完全一致。原错误解析与计算保留在`literal_id_receipts/`，没有按概率或损失挑选映射。[修复回执](source_corrections.json) · [身份核验](static_identity_audit.json) · [逐帧核验](rt_identity_frame_audit.json)

统计实现同时修正了“整日消息缺失就丢弃已知实际标签”的问题：完整分母仍使用有效的事件界，日期缺失仍强制未解。阈值、权重、bootstrap预算均未改变，真实面板只执行一次最终bootstrap。**37项定向测试通过**；统计元数据修正后8项相关测试再次通过。[理论](THEORY.md) · [测试记录](validation.log)

## 来源与限制

端点与静态来自[官方档案](https://archive.opentransportdata.swiss/)，消息来自用户认可的[traines第三方官方feed采集档案](https://github.com/traines-source/public-transport-statistics/blob/master/README.md)。14个RT原档案约2.013GB均留服务器；官方端点在服务器扫描约2,986万行后保留11,523行。本地仅取约2.45MB结果包及小审计。

冻结前的目录元数据只确认了文件存在，没有确认内部早间时钟。3/9整包最早记录为12:02，故保留缺测、不补日。2023端点档案没有实际站台字段。此次是未开发历史日期的回溯校准，模型拟合日期为更晚的10月，不能称为前瞻确认；采集文件名时钟也不是已验证的乘客消息送达时钟。14日现已全部记入[已使用日期](exposure_record.json)。

**下一步先审计V35训练消息的车次身份。** 其准备代码同样复用字面`trip_id`连接；本轮尚未证实旧拟合受影响的样本数，不能据此断言旧拟合已经错配。先按训练日的公开版本证明身份对应，再判断是否需要重建拟合。新的确认面板应在冻结前核对归档内部采集时钟，并补足实际站台证据。当前结果不支持追加大DP、搜索或以本轮修复宣称G2通过。

## 复现

服务器工作目录：`/home/erzhu419/BAPR-HRO/experiments/decision_calibration_v2/source`；Python：`/home/erzhu419/miniconda3/envs/csbapr/bin/python`。入口顺序为`acquire_static`、`run_calibration prepare`、`cache_rt`/`filter_actuals_remote`/`parse_cached_rt`、`evaluate_ready`、`correct_rt_identities`、`finalize`，均位于`experiments.decision_calibration_v2`。日期及来源见[冻结清单](date_manifest.json)，完整公共选择见[selection.json.gz](selection.json.gz)。

本地[events.json.gz](events.json.gz)与[evaluation_manifest.json](evaluation_manifest.json)足以复算冻结统计；全部根的状态、消息历史、标签与工时在`results/`，四日原回执在`literal_id_receipts/results/`。原始大档案和筛后的输入留在服务器。
