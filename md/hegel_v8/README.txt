BAPR HRO 0022df2 独立诊断与复核材料
审计日期：2026 年 11 月 14 日
固定提交：0022df22e775c677e0cf6ebeb71901b7fe8d9dbd
来源：https://github.com/erzhu419/BAPR-HRO/commit/0022df22e775c677e0cf6ebeb71901b7fe8d9dbd

判断
V41–V45 已实质执行上一轮的关键建议。V45 有带适用条件的规范历史压缩和净计算收益；当前独立 OR 方法主张仍未建立，主要不是执行漏项。精确 absorbing 只消除 κ 的前缀抽样误差，未控制完整 π 的替代误差；现有粗界因 ALIGHT 尾部完全未认证而必含零。

本次新发现
1. 全部 444 次状态估值、360 趟在线回放复算通过。148+135 个比较点的 72,448 个臂样本前缀与 κ 值逐项保持。Markov 相对 matched CPU 减少 20.91995%，相对完整 rollout 减少 61.31710%。
2. 压缩后前缀只占总 CPU 13.05%，局部续程积分占 70.07%。全支持 exact 枚举使独立前缀评分 993→9907、局部 κ 求值 955→3878，增加费用有明确工作量原因。
3. 纯终段平台必要形状覆盖 39/148 冻结状态及 32/135 在线状态，但 11 个改选状态、7 个非零行程关键状态均为零覆盖；关键 CONTINUE 首次平台纯终段质量也为零。单独做 terminal solver 不能处理当前主要缺口。
4. 11 个关键尾部均为 4 个当前候选、最多再登乘两次。四个关键根 4/16/28/52 没有当前候选车次在其他候选尾菜单重现；七个关键根 2/13/14/26/27/62/63 各有一条这样的引用。应先检验这个具体的混合一次换乘结构。
5. 当前 6 胜1负不是历史负例全部消失。root13 的343秒和root62的96秒分别恢复了物理修订与浅层修订后的便宜控制实际损失。相对V40便宜仍有root31/57/63三个负例。所有V45最优损失逐根等于V41 physical_rollout128。

文件导航
metrics/AUDIT.md             全量统计、预算、估值和跨版本谱系分析
metrics/recomputed.json      444+360 原始收据独立复算
metrics/states.json          148 个状态的差值、参照和结构
metrics/online_gates.json    135 个在线比较的逐项记录
metrics/roots.json           72 根收益及成本
metrics/lineage.json         V40→V41→V45 的同根质量谱系
metrics/menu_structure.json  菜单深度、候选、共享物理车次结构
metrics/critical_depths.json  全部关键正概率平台的剩余登乘深度
theory_code/AUDIT.md         代码调用链、理论范围与下一有限原型
theory_code/structure_probe.json  独立运行时结构检查
novelty/AUDIT.md             原始文献核查、贡献定位与停止条件
NEXT_STEPS.txt              建议的下一阶段协议和验收顺序
tests.log                  本环境完整 tests 运行日志
audit_environment.json     固定提交、环境与验证范围
SHA256_MANIFEST.json        本包文件校验

验证范围
只读原收据及固定代码，没有新跑策略实验，没有重新计时或选择最快结果，没有修改项目文件。这里的 CPU/wall 来自原提交归档收据。新增结构探针只加载原图、模型和菜单，不执行新控制器。
本环境 tests 为885通过、1失败；唯一失败因公开检出缺少外部 data/zurich_wide.pkl。项目发布记录为886通过。输入缺失未被当作算法回归，也未声称本环境独立复现886全部通过。

复算
先检出上述固定提交并安装项目依赖，将BAPR_HRO_REPO设为自己的本地路径。下列命令在解压目录执行，输出写入本包各脚本所在目录；建议先保留原结果副本。

export BAPR_HRO_REPO=/absolute/path/BAPR-HRO
python3 metrics/audit_metrics.py
python3 metrics/lineage.py
PYTHONPATH="$BAPR_HRO_REPO" OPENBLAS_NUM_THREADS=1 python3 metrics/menu_structure.py
PYTHONPATH="$BAPR_HRO_REPO" OPENBLAS_NUM_THREADS=1 python3 theory_code/structure_probe.py

限制
全部六日期为开发数据。结构计数是冻结菜单的证据，不能外推全网。纯终段形状不等于完整π精确性证书。一条跨候选引用边不等于求解图树宽为1。条件物理一致不等于现实校准，根内动作保序不等于整趟策略改进。NEXT_STEPS中的新命题均待证；本包不宣称G2或OR发表标准已经满足。
