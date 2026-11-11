# 三份诊断的执行对应

两份Markdown和Word正文、表格、实施规格已完整阅读。Word建议存在条件依赖：阶段四方法开发须由前三阶段的结果决定；明确的停止/转向决定也属于执行。原文在 `assessment_docx_extracted.txt`，公式须结合原Word数学节点读取。以下对应合并重复建议，不把已撤回的结论当作待补强事实。

| 来源 | 工作及当前状态 | 位置或验收依据 |
|---|---|---|
| Word 23、212 | 保留旧稿、旧数据规则、旧结果；当前代码不冒称送审冻结版本 | Git旧版本与既有结果保留；送审系统冻结PDF仍未取得 |
| A.1；Word 27–31 | Normal–Gamma均值后验、观测方差、预测方差分别定义；修正版与旧成绩分开 | `bandit_router.py`、`THEORY.md`；解析后验回归通过 |
| A.2/B.2–3/R3.2、4；Word 32–36 | 一般Lipschitz仅上界；仿射等式为已知特例；无自动校准/整体消失保证 | 正确陈述及反例已写 `THEORY.md`；稿件按用户要求暂缓 |
| A.3/B.1；Word 38 | 正确折扣收缩及SSP边界 | `THEORY.md` 给出sup-norm证明，未套用未折扣问题 |
| A.4–5；Word 39 | 未证旅程/悲观/Adaptive保证退出本轮理论；同一未来H下分解 | `information_planner.py`、`THEORY.md`；Bellman telescoping解析核对通过 |
| B.5；Word 40 | Lean只覆盖实际对象，移补充材料 | 语义缺口已写 `THEORY.md`；既有Lean和稿件保持原样，待证据冻结后处理 |
| R3.3；Word 42 | 清除不可核实Chen条目与依赖论断；Ren作者名 | 已定位；稿件按用户要求暂缓，尚未清理稿件引用 |
| B.4/7；Word 43、268–269 | 收窄主线、删除无证据宣传 | 新实验不纳入Adaptive/neural/跨域；稿件按用户要求暂缓 |
| B.6；Word 30、86、265 | 共轭模型适配需真实留出误差证据 | SBB三日训练、三日留出及额外11个过去校准日已准备；VBZ完整周源的7月25日训练、26/27日校准已执行。经验预测区间与后验均值覆盖分别解释。见 [11日输入](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/rail_calibration_dates_v2.json.gz)、[城市校准](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/vbz_city_calibration_v2.json) |
| Word 46–65 | 取消质量与成功条件PMF分开，避免重复计入取消/换乘失败 | `pmf.py`、`topocsa.py`、`native.py`；回归通过 |
| Word 66–69 | 同车续乘与不同车最小换乘一致 | 两执行器及 `finite_transit.py`；物理换乘回归通过 |
| Word 70–76、213–214 | 固定车辆场景、因果信息、重复消息去重 | `JourneyScenario`、显式公告及trip级学习去重；另修复由未来实现决定提前重选的等待分支 |
| Word 79–81、216 | 恢复Durner strict策略基线，不能以均值去重或all-label近似冒称原生 | `src/durner/native.py`、`src/native_router.py`；`BASELINES.md`；5项原生语义及2项执行接口回归通过 |
| Word 82–84 | 移除逐趟重建Adaptive的学习证据 | 新面板排除该重复列；既有代码/结果保留，稿件暂缓 |
| Word 85–87 | 明确CANCELED、SKIPPED、NO_DATA、严重延误；保留事件身份和发布时间 | 35日corrected汇总及共用 `actual_event_parser.py`；REAL时间与取消标记冲突保留原字段并作未确定状态；17项相关回归通过 |
| Word 76、87 | 不按重复snapshot扩增样本，不跨agency同名线路混池 | latest event revision；按route_id汇总并保留agency/name |
| Word 88–90 | 统一截断旅程损失、失败和超时 | `matched_router.py`、`finite_transit.py`；216次匹配旅程完成，直达发车/到达共享延误已条件化；多段估值明确为近似 |
| Word 91–98 | 新BAPR只借研究组织方式，不迁移可逆动作结论 | 本地BAPR已读；V47结果本地存在，不能沿用“公开快照未找到”当本地缺失事实 |
| Word 99–136、177–191 | 最近方法差异、独立性/状态压缩/界/复杂度的真实边界 | `BASELINES.md`；不预先宣称新颖、次模或多项式 |
| Word 142–175、217 | 精确信息树、共同扰动/时机例、full/H/实际损失分解、不同oracle分开 | 36个构造机制cell与4个公交子网cell已执行，输出位于 `experiments/reconstruction_v2/` |
| Word 218–220 | 少量现有OD×正常/扰动筛查，候选×评分交叉，其余匹配 | 2参数日期、3起点、2种子、2生成工况、9方法完成216次旅程；96个retain/refresh配对损失相同 |
| Word 219、237、264–266 | 独立日期/起点、全部开销、正常日代价、截止敏感性与三类关键图 | 最终Causal生成确认及预算板已按完整空公告历史修正复跑，另完成铁路96趟和城市16趟实测回放，见 [修正生成确认](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v3/final_causal_poll_history_confirmation.json)、[V4结果](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/RESULTS.md) |
| Word 265 | 部署式只用过去日期，LOO只称离线泛化 | 新 `build_route_priors_temporal.py`；LOO文案和corrected输出已修 |
| Word 242–262 | 数据、信息、旅客状态、目标、最优参照和运行记录对应 | 数据字段审计已完成；其余在对应实现中记录 |
| Word 221–236、209、271–273 | 根据结构损失和真实计算瓶颈继续、转向估值或停止压缩路线 | continuation估值及最终学习归因已完成；阶段四已实现有限场景的安全删留证书与标签依赖修复，适用条件及OR新颖性边界见 [证书记录](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/CANDIDATE_CERTIFICATE.md) |

已流读35日本地归档，每30个归档快照抽取1个，输出 `data/per_day_distributions_corrected.pkl` 和 `data/route_priors_temporal_corrected.pkl`。精确trip-ID恢复匹配189个trip和全部2000个连接，得到33个route ID，其中29个有延误预测。1,027,848条去重延误预测、109,335项常规或取消状态、237项取消、20,600项跳站和1,036条严重延误分别统计。10月19–21日覆盖偏低，四个线路ID全期无数值预测。保存的图包含不同服务日的班次，按静态GTFS日历逐日只有6–23个trip和74–176个连接启用，见 `experiments/reconstruction_data_audit/graph_identity.json`。历史pickle保留。

初始54端点复核与96趟SBB回放保留。铁路REAL与取消冲突解析已修为v2，6个回放日期和11个过去校准日的现有样本中，操作状态/时间、两套拟合参数与369个已选登车动作均无变化，v1保存。[铁路v2](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/dated_replay_v2.json.gz) · [版本核对](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/rail_semantics_v1_v2_comparison.json)

完整10月25日官方日源在远端扫描2,226,994行，本地仅取473KB城市峰窗子集。声明矩形内28,448条日期有效公交/电车候选全部保留，26,933条实测、477条明确取消、1,038条未确定，含714个REAL与取消标记冲突端点。[完整日源与匹配](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/city_dated_replay.json.gz)

完整VBZ周源272MB保留在远端，仅小拉三日峰窗gzip；显式16个站名别名映射后，7月25/26/27日分别实测24,051/23,900/23,655条连接，对应全部28,164/27,895/27,895条计划候选。26/27日VBZ本身覆盖95.58%/94.60%，其他运营方及未匹配候选继续作删失；源文件只记录已执行服务，取消结果未提供。完整源城市策略回放由V4 runner记录。[城市v2](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/city_vbz_dated_replay_v2.json.gz)

城市留出校准保留负例。相对提供方原始计划，历史均值在26/27日均使电车MAE约增0.32秒，公交分别减1.04秒、增0.24秒；GTFS分钟计划口径另报。样本按班次和日期聚集。[校准](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/vbz_city_calibration_v2.json)

初始S11/S20固定行程采用自行设定120秒换乘时间。16站中11站有GTFS站内声明值且均高于120秒，HB为420秒，另5站无该项；平台标识保留，单站敏感性可取适用声明值最大值。实测取消发布时间仍缺失，历史预测与实现状态分别处理。[换乘来源](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_data_audit/station_transfer_bounds.json)

## Continuation 后续执行

按用户后续指令，已完成概率/补救分解、与真实执行器一致的 committed 精确参照、公开历史条件化模型及因果 rollout 估值。共同平移精确归因36格、旧旅时模型36格、3个真实历史状态和150次匹配旅程均完成；未知 trip 概率模型/采样误差另行分解。101项测试通过，稿件未动。具体结果、负例、计算成本与剩余候选/尾策略问题见 `experiments/continuation_v1/RESULTS.md`，不把工程改善记为 OR 新颖性已成立。

## 后续交付

- V2 的归档负例及交叉实验见 [V2 结果](../../experiments/continuation_v1/V2_RESULTS.md)。最终Causal类已补mean/cancel四格事后归因与另24趟独立生成确认，见 [线路 ID 结果](../../experiments/reconstruction_v3/RESULTS.md)。此前段落中的测试数和稿件状态为 V1 时点记录。
- 修订线路ID生成面板、SBB回放及完整源城市数据均保留。真实输入含全部声明候选与未确定结果，铁路正常入口96趟和城市16趟均已完成。[铁路最终回放](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/dated_replay_final_results.json) · [城市回放](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/city_replay_results.json)
- 新稿为 [重构源稿](../../paper/paper_reconstruction.tex)。旧稿、历史数字和提交 ZIP 作为版本证据保留。Lean 附录说明及全部模块的类型检查见 [理论记录](THEORY.md)。[送审版本核对](SUBMISSION_VERSION.md) 记录了冻结原件的缺口。
- 有限场景证书及标签局部修复已完成两片段精确核对与2,000连接模板更新验证；保证适用于固定场景库及明确拓扑条件，策略树仍重新计算。适用边界与计费见 [证书记录](/home/erzhu419/mine_code/BAPR-HRO/experiments/reconstruction_v4/CANDIDATE_CERTIFICATE.md)。
