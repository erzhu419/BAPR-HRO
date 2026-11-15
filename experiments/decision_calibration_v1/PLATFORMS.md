# 计划站台与换乘规则核验

原连接构造在 `build_dated_replay.py:dated_connections` 用 `stop_id.split(':')[0]` 压到站级，但保留车次和站序。用同版本静态表回接，1712 个车次—站序键全部唯一。对已冻结 580 个开发事件的上下游共 1160 个换乘对，490 对有精确有向 GTFS 最小换乘规则，全部等于既有固定值；670 对是同一 GTFS stop_id，未找到适用规则。保留全部事件，模型 MCT 未调整。

GTFS 的站规则可继承到子站台，必须依据 `location_type=1` 和 `parent_station`，不能用 BPUIC 字符串猜测。[官方 transfers 定义](https://gtfs.org/documentation/schedule/reference/#transferstxt)。此源中裸 `8503000` 是 `location_type=0` 子 stop，真正父站为 `Parent8503000`；后者没有适用 transfers 记录。全部 1160 对的父站继承规则数为 0。因此，裸 `8503000→8503000:0:41/42` 的 420 秒不能当成 `41/42→41/42` 的父站规则。

关键负例：root6/18 上游 `Stadelhofen 2→1` 有 240 秒来源，后站 `HB 41/42→41/42` 无精确或继承 self-pair 规则；root32 的 `Oerlikon 3→6` 与 `HB 32→43/44` 分别有 180/420 秒来源。缺少规则不推出零秒，更不证明真实可赶。

已检开发日 `2023-10-25T07:00:03+02:00` 原始 RT 帧：42 个相关车次、422 个 stop 更新均保留带站台后缀的 `stop_id`，无 `assigned_stop_id`。旧消息提取器丢弃了这些标识，今后可另存公开站台证据。[官方 RT 定义](https://gtfs.org/documentation/realtime/reference/#message-stoptimeproperties) 将 assigned_stop_id 用于实时站点分配；公开站台信息与实际到发确认仍是不同证据。

限制：以上是同版本静态计划与一个已开发 RT 帧的字段核验。实际端点 CSV 没有站台字段，实际站台对确认数仍为 0；未扫描所有历史 RT 帧，也未读取新日期目标结果。`platform_pair_audit.json.gz` 保留全部 1160 对，`rt_platform_schema_audit.json` 保存公开帧字段回执。5 项定向测试覆盖唯一映射、方向、真实父站继承、裸 BPUIC、歧义和缺失规则。
