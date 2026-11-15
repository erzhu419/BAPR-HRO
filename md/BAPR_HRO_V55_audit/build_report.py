"""Build the Chinese V55 audit report from reviewed evidence.

This is a document builder, not a project modification or an experiment runner.
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

HERE = Path(__file__).resolve().parent
OUT = HERE / 'deliverable'
OUT.mkdir(exist_ok=True)
COMMIT = 'a2b82be9c3fc6b61e2a5826d7cd8691d39ff4491'
BASE = 'https://github.com/erzhu419/BAPR-HRO/blob/' + COMMIT + '/'
BLOCKS = []

def h(text): BLOCKS.append(('heading', text))
def p(text): BLOCKS.append(('paragraph', text))
def table(headers, rows, widths=None): BLOCKS.append(('table', headers, rows, widths))

BLOCKS.append(('title','BAPR-HRO V55 深度诊断与下一阶段方案'))
p('固定提交 a2b82be9 | 审计日期 2026-11-15 | 研究判断与可复算证据')
p('核心结论：V55 收束当前分解路线的决定合理。局部约简和实现正确性有支持，但所消除的结构没有覆盖主要混合续程难点；完整策略差距的诊断任务又被绑定到昂贵的完整最优参照。下一轮应先验证具体的候选覆盖与换乘机会概率问题，再用独立配对的可行策略比较识别残留决策缺口。现有证据不保证下一轮能通过 G2，也不能证明 cheap 已经最优。')
p('本次最直接的新证据：root4 恢复一条被计划时刻守卫剔除的公开同车路径后，cheap 从 1721 降到 1386 秒，吸收原 rollout 的 335 秒正例；root62 的 96 秒已由现有 snapshot_csa 捕获。这是两个正例的便宜逐例解释，尚不是统一新规则的全样本验证。')

h('一、审计做了什么，结论适用到哪里')
p('核查了 V50-V55 的新增代码、结果、理论、新颖性说明和原 G2 / 审稿意见。只读复算 4400 份 V53-V55 原查询收据、576 份 V51 整趟回放；独立复测 V55 五个新增测试文件，156/156 通过。1412 是仓库报告的全量通过数，本次没有重新跑完整 1412 项，也没有重跑 V55 的全部 1760 次查询。')
p('另做两次保持原 10 秒上限的插桩诊断；五个已有开发根共 20 条首动作与未来消息消融；root4 的三臂菜单/观察消融；重建三个负根的原 128 世界并逐条复放，两臂共 768 个损失与状态全部复现。原始实验、参数、预算和 Git 工作区未改动；新增诊断全部单列。插桩成本不能用于宣称提速，新增回放不是新日期验证。')
p('所有实际损失均指既有档案物理端点下的回放结果，使用当前 gtfs_station_max 换乘合同与理想逻辑发布钟；它们不是实际乘客轨迹，也不构成真实硬时限性能结论。固定根 P0 是查询中的物理概率律；W 是原便宜控制器的工作评分更新规则。两种求值的区间宽都不是新的实测旅行损失。[S1-S8]')

h('二、前几轮确实完成了什么')
table(['阶段','已经成立的证据','对下一轮的含义'],[
 ['V49','相同保证下增量归约提速；共享接口能被普通静态依赖恢复','缓存、接口与一般合并不能继续承担独立方法主张'],
 ['V50','统一观察后，解除四首车上限仍省 20.1806 秒/趟；信息差贡献为零','这项菜单上限修复已经执行，不应再当新任务'],
 ['V51','同宽菜单：cheap 1390.750 秒，full128 1392.375 秒；2 胜 3 负 67 平','昂贵 rollout 没有整体实测优势；存在正负个案'],
 ['V53','投影、整段积分、根预界和 impact 强对照完成','不能再假设没有普通强参照'],
 ['V54','根额度回流、降低截断阈值已试；所有查询受真实 CPU/状态帽限制','重复建议“再回流”没有证据上的新意'],
 ['V55','目的地评分约简与终端 DP 有严格适用条件；补齐 impact 全面板','局部有效，关键根混合续程未解，独立新颖性未成立']
],[62,225,225])
p('V51 的 full128 只在当前下车/继续动作间做样本改进，未来执行 cheap，估计的是 Q^cheap；在线整趟又会在以后重新调用。它不是完整 Q*。snapshot_csa 也只是现有路线树上的条件均值连接扫描，不能代表已复现全网络 CSA-MEAT 或最新动态重规划方法。[S2-S4]')

h('三、V55 的 17.6% 加速究竟说明什么')
table(['10 秒档，88 状态','V53 impact','V55 priority','判断'],[
 ['最优值平均区间宽','289.992 秒','279.080 秒','4 格收紧，84 格相同'],
 ['最优查询平均 CPU','7.599 秒','6.258 秒','局部成本下降成立'],
 ['P0 cheap 平均区间宽','1021.943 秒','1266.203 秒','计算不确定性扩大'],
 ['W cheap 平均区间宽','1685.427 秒','1859.605 秒','52 宽、21 窄、15 平'],
 ['最优首动作证书','40/88','40/88','没有新增 10 秒首动作证书'],
 ['完整 cheap regret','88/88 未解','88/88 未解','既未证有损，也未证最优']
],[155,105,105,147])
p('逐收据追踪 root_bounds 后发现：V53 的 88 个最终最优值区间全部逐数等于初始化根界的 min 下界 / min 上界；V55 的 84 个仍然如此。真正新增根值信息只有 root 0、12、36、41，均早已认证首动作，均不属于七个关键根。根预界本身已有 30 个严格首动作证书，递归增加 10 个 WAIT 证书；边界相等不能按严格证书计数。递归并非完全无用，但“最优根值收紧”“排除竞争动作”“完整固定策略 regret”是三个不同任务。[A1]')
p('初始化并不免费。它包含物理律准备、目的地下界和至多 16 条可行承诺路线的上界搜索；上界搜索使用约 15% 查询 CPU 的限额。原收据没有分阶段计时，不能把仅根预界的成本当作零。应补这一基线，明确后续递归增加了多少信息。')
p('更重要的是预算停止的归因。88 格合计净省 117.9953 CPU 秒；其中 58 格新版本达到 20000 状态帽、且根值区间逐数不变，合计省 122.0971 秒，超过总净节省，因为其他格合计变慢。其中八格由旧版 CPU 先停变为新版状态帽先停，贡献 42.6144 秒，即总净节省的 36.12%。它们都没有认证动作。')
p('root32 / root57 从约 10 秒降到 2.614 / 2.535 秒，但 20000 个状态里分别有 18263 / 17940 个终端 DP 状态。根最优值区间不变，动作仍未认证，ALIGHT 上界还分别放宽 16.701 / 20.638 秒。更便宜地耗尽异构状态额度，不能解释为完整任务已同质量解决。应使用“达到同一目标质量所需全成本”或真正相同硬时限的质量比较，并保留未完成结果。')

h('四、理论约简为何没有切中主要困难')
p('1199/1486 = 80.69% 是结构上适用的终端平台数，不是访问概率、CPU 占比或对根值区间的贡献。七个关键根立即下车的平台均是混合续程，候选数为 7、8、8、9、9、15、9；语义约简后两个动作仍共享 8-12 个评分车次。许多可积分叶子存在于整棵菜单树，并不代表当前预算内会到达它们。[S7,A2]')
p('物理车次独立也不使控制问题独立。乘客位置、可到达时刻、是否还留在车上、转乘门槛、可观察信息和 W 的后续重选共同耦合动作。即使两个动作都涉及同一辆救援车，若到达站台的时刻不同，可赶乘指示函数就不同，不能消掉相同车次对应的损失。')
p('简单例子：Q(a)=E[1{A_a<=D}g(D)+1{A_a>D}H]。两个动作的差是 E[(1{A_a<=D}-1{A_b<=D})(g(D)-H)]。真正影响方向的是两种可赶乘窗口之间的概率质量及其代价。共享随机变量并不等于共享价值。这个恒等式是解释，不是新方法。')
p('V55 目的地屏障证明保留了后槽预测的反向似然、原排序、P0 实际登乘损失、WAIT 分数及可见发车后的 W BOARD 分数；这些条件是必要的。普通吸收目标支配、依赖消元、子集 DP 和信息松弛已有明确先例。新的有效充分条件有实现价值，但不能仅因它适用于本项目就宣布独立算法创新。[S7,L2-L6]')

h('五、代码里仍有可修的问题，但它们不是 G2 主答案')
table(['位置/问题','现有证据','建议的有限处理'],[
 ['固定策略目标仍求所有根臂','V52/reference.py 最后只取 cheap_action，但先对全部 forced actions 分配预算','分离 evaluate_policy 与 evaluate_forced_actions，分别计费'],
 ['混合平台先生成再筛分支','r6/18/62 的 W 原收据终端准入均为零；产生约 25.36/28.74/14.39 万个分支','对事件生成、条件化、评分分阶段计时；不能只调终端核'],
 ['单观察组创建无用 child law','_joint_trip_outcomes 在最后一项观察后仍造 yes/no 链并立即丢弃','增加单项/末项快路径，验证分布和概率质量相同'],
 ['终端块局部 DFS、部分状态不续用','每次未完成调用重建 _Terminal；内部无根影响调度','仅在 profile 证明主导时试可恢复接口，与普通调度对照'],
 ['W 历史 law 重建昂贵','r6 插桩中相关累计时间约 4.13 秒，存在嵌套','先定位改变的证据及依赖，再做有限增量更新']
],[145,207,160])
p('全部 b10 W 查询里有 46/88 从未准入终端 DP，其中 31 格变宽。因此不能把退化统归于终端核。两个 profile 的累计时间含嵌套，不能相加为 CPU 占比；新测试通过也不证明全局界已形式化验证。本次没有发现足以推翻已声明有限模型界的新增正确性问题。建议把上述工作限定为诊断工具修整，最多一次小范围工程轮，不再以此承诺 G2。[A2]')

h('六、新诊断：当前模型为何偏好那些实际输掉的动作')
p('原 V51 的 136 个 gate 中有 13 次改变 cheap 当前动作，分布于 11 根；其中 11 个开发性普通均值区间正向排零。这些样本已经参与选动作，不能作为独立或同时有效的认证。整趟实际损失非零只有五根，不应因此忽略其他动作分歧。13 个分歧中有一格 root15 / cid127 / clock28844 不在 V52 的 88 状态里：两个 rollout 都选 ALIGHT，而其虚拟 cheap base 选 CONTINUE，原按已执行策略之间分歧的筛选没有捕捉它。应保留原 88 分母，额外单列该开发状态。[A3]')
table(['root','原样本支持替代动作的均值收益','实际 cheap - rollout','解释范围'],[
 ['4','70.313 秒','+335 秒','本次局部候选恢复已吸收'],
 ['6','140.625 秒','-263 秒','模型与单次实现方向相反'],
 ['18','98.906 秒','-145 秒','模型与单次实现方向相反'],
 ['32','102.656 秒','-140 秒','模型与单次实现方向相反'],
 ['62','33.281 秒','+96 秒','该实际收益已由 snapshot_csa 捕获']
],[45,180,130,157])
p('本次对五根做“强制原 cheap/替代首动作 × 后续未来远程消息 live/frozen”四臂回放，随后全部执行 cheap W。10 个 live 臂精确复现原损失、状态、动作及逐事件公开事实；冻结未来远程消息后，全部十条路径的动作与损失仍相同。故这五个已知正负例不支持把未来消息树作为下一轮主要重构方向。它没有证明其他状态或总体的信息价值为零。[A4]')
p('三个负例具有共同交通机制：cheap 在 Stadelhofen 或 Oerlikon 提前登上贯通车辆，留在车内通过 Zürich HB；rollout 留在原车到 HB 再换乘。在当前 420 秒换乘合同下，同一后续车已经分别早于 ready 53、198、48 秒离开。这些实际机会当前 cheap 已抓住，不能把它们包装成 cheap 的新缺口。')
table(['root','上游可赶乘 P0 概率','128 世界出现次数','实际上游余量','HB 错过量'],[
 ['6','0.6205%','0','100 秒','53 秒'],
 ['18','1.7535%','2','11 秒','198 秒'],
 ['32','29.2934%','46','58 秒','48 秒']
],[42,144,106,110,110])
p('这些概率由根时点完整 P0 条件链边缘计算，保留硬观察、当时选择的 soft forecast、取消状态、ready 与截止掩码。原银行按同 seed、根、全图 trip 编号重建；768 个逐 world_index 的两臂损失与状态均相同。原文件没有保存完整物理端点数组，故不声称逐原端点字节验证。[A5]')
p('root6 在 128 次中期望只出现 0.794 次上游可赶事件，一次也不出现的概率约 45.08%；root18 期望 2.244 次，实际样本出现 2 次。因此“再增加搜索精度/更准确积分”不会自动把这些事件的模型概率改对。它可能更稳定地得到当前模型偏好的 CONTINUE。')
p('仍须避免过度推断：这三格是事后因实际输掉而挑出的，不是校准检验的独立分母；一次低概率事件发生不能证明 P0 失校准，也不能证明跨车独立假设错误。下一步应在所有事前合格的换乘机会、新日期和统一筛选规则下检查可赶乘事件概率、损失贡献与排序；原状态上的新随机 seed 只能验证固定模型内的条件结论。')

h('七、已验证的候选覆盖机制：root4 的 335 秒被 cheap 吸收')
p('root4 是原 V51 已测便宜整趟控制中尚未吸收的 full128 实际正例。只读菜单审计发现：S5 的上游连接 cid145 在 Stadelhofen 计划 28920 发车；该根原菜单的计划 ready 为 28980，故被 schedule-only departures 守卫提前剔除。V50 解除的是每站四首车上限，并未解除这个计划时刻过滤。')
p('该车的后段 cid193 已在菜单中，可在 Hardbrücke 搭上；其上游完整同车路径 145→182→193 在图中存在。实际 root4 到 Stadelhofen 后 ready=29181，而 cid145 departure=29219，尚有 38 秒余量。当前 action 菜单和共同 local observation 都没有 145，因此便宜策略根本不能利用这个上游机会。其同车终点时刻 29736 与 rollout 的正例到达时刻相同。')
p('本次进一步完成了合法三臂局部机制消融。新增路径完全来自公开图，验证了同物理车、计划连通性和全部端点标签；该车早已在原后站菜单中，所以车次集合、远程帧和根公共历史不变。只新增 Stadelhofen 的 cid145 现场观察及完整同车路径 145→182→193，新增车程唯一出口为目的地 193；没有顺便增加 HB 的 182 平台候选。[A6]')
table(['root4：全部使用 cheap W','损失','动作结果'],[
 ['旧动作 / 旧观察','1721 秒','逐项复现原 cheap'],
 ['旧动作 / 增加145观察','1721 秒','仅增加观察没有旅行收益'],
 ['增加145同车路径 / 增加145观察','1386 秒','仍首动 ALIGHT；29219 登145；经182到193，29736到达']
],[215,78,219])
p('局部动作恢复已吸收原 rollout 的 335 秒收益，且三臂清楚区分了信息增加和动作覆盖。root62 的 96 秒则由原 snapshot_csa 以完全相同动作序列取得。两个已知实际正例都有便宜的逐例解释；这削弱了以它们支撑新深层算法的论据。')
p('限制同样重要：root4 是事后选根、选候选的机制探针，动作域与原 V55 合同不同。它不证明统一扩域在 72 根上有净收益，不证明新 cheap 支配旧 cheap，也不能作为旧域下同任务的性能对照。新增候选可能在其他根诱发坏选择。下一步应把该诊断变成统一公开规则，然后保留所有正负与未解样本。')
p('建议规则先限制到已存在于剩余菜单的车次：检查沿原可达站点的更早登乘位置，补齐合法同车后段；保留取消、截止、换乘条件和共同信息合同。不能只因计划发车早于计划 ready 就剔除具有正条件可乘支持的连接，也不能根据 actual 时刻逐根补车。若筛选概率昂贵，可先做保守的支持覆盖，再完整计费；不能把极小但正概率的删除默称精确。')

h('八、改写下一轮诊断目标：先检验一个可行改进')
p('当前完整 regret 区间依赖 cheap 与最优值两套宽界：R_c ∈ [max(0,L_c-U_*), max(0,U_c-L_*)]。把它作为唯一入口，会迫使研究先建设一个很难求解的完整参照。要证明存在可改进空间，要求其实更弱。')
p('固定同一历史 h、物理律、损失和信息合同，令 eta 为合法候选策略。因为 J_* <= J_eta，所以 R_c(h)=J_c(h)-J_*(h) >= J_c(h)-J_eta(h)=Delta(h)。只要有效的独立配对下界 L_Delta 为正，就得到固定模型内的正 regret 下界，无需先把 J_* 解完。这是标准可行策略比较逻辑，不是本项目新算法。[L5]')
p('最小 eta 可先用“当前强制另一个合法动作，后续始终 cheap W”。候选必须在生成确认银行之前冻结；平台多动作时，使用旧开发数据或统一事前规则选一个候选。每个策略只读取各自轨迹中的合法历史，不能复制另一臂已经看到的未来信息或共享可变缓存。')
p('当前首动作最优证书并不排除这个 eta 改善：证书比较 Q*，单次偏离比较 Q^cheap。例：Q*(a)=100、Q*(b)=110，但 cheap 的后续值可能为 Q^cheap(a)=200、Q^cheap(b)=120。此时选 b 再 cheap 可以省 80 秒，而 a 仍是最优续程下的正确首动作。必须注明续程目标。')
p('建议使用预先冻结的新验证银行及有限样本批次，例如 512、1024、2048、4096；这些是新诊断任务的预算草案，不是修改 V55 旧比较。执行前用不看效果的成本试跑冻结总 CPU 上限。对状态、候选和批次预分配总 alpha=0.05；使用有效的有界经验 Bernstein 界或置信序列。普通 1.96SE 只能做描述，反复查看后择显著不能用作认证。中断时只使用已完整完成的预定批次，或报告未解；不继续扩预算直到通过。[L8]')
p('D_i=loss_c(world_i)-loss_eta(world_i)，两个控制器共享物理随机世界以降低差值方差，但规划随机带与评价随机带隔离。D 有明确截断范围。报告均值、有效上下界、正负/相同世界数、取消/缺测、全部计算成本。新 bank 只能消除原 128 世界的动作筛选偏差，不会把六个开发日期变成独立现实确认。')
p('正模型信号、正实际个案和现实稳定收益须分开。若固定 P0 中 eta 更优，而跨独立日期的实际效果系统性相反，先处理模型/合同/目标错配；若一种便宜规则能吸收改进，升级基线；只有剩余且有实际幅度的差距，才进入主算法重构。')

h('九、可执行的阶段安排与交付物')
table(['阶段','建议工作量','必须交付','继续/停止条件'],[
 ['0 本次已完成','不重做','根界来源、CPU帽归因、23条机制回放、原bank事件分区','停止把缺对照/未接回流当作当前解释'],
 ['1 覆盖与概率机制','2-3工作日','统一更早登乘规则；72根共同菜单/信息；全分母换乘事件账本','覆盖已吸收就留强基线；MCT/概率错配先修模型合同'],
 ['2 可行策略独立验证','2-3工作日','固定候选、独立bank、新旧面板分别冻结、配对差值与成本','有剩余缺口才启动；不显著/超成本为未解'],
 ['3 排除便宜解释','1-2工作日','单/双目标、共同信息 rollout、快照/等待规则，同预算对照','能吸收则收束该方法假设'],
 ['4 方法立项','最多3工作日初审','最小因果反例族、一个待证结构命题、最近邻差异表','没有不可被现有技术直接解释的增量就停止'],
 ['5 算法开发与确认','仅条件触发','同质量全成本曲线、硬时限回放、新日期确认','G2方法与计算；G3独立事件证据分别验收']
],[70,75,220,147])
p('阶段 1 的换乘事件账本必须包含所有事前合格机会：日期、OD、incoming / outgoing trip 与连接、公开历史指纹、计划/实际/预测时刻及各自身份、ready、MCT 来源、可赶概率、取消质量、实际是否可赶、两臂损失与所走路线。实际标签只用于评价，不参与候选选择或可见信息。按日期/事件聚类报告，不能把同日大量片段视为独立数据。')
p('对 root4 的事后单例扩域只作机制证明；后续规则须在打开确认结果之前冻结。若采用 gtfs_station_max，它仍是该声明合同，不能把 420 秒当成所有平台对的实测行走时长。只有需要外推真实运行时，才针对具体相关平台对核实 MCT；不要根据胜负调 buffer。')
p('阶段 2 不应只追七个旧正负根。旧合同保留原 88 状态所有结果，并将遗漏的 root15 第二分歧单列为第 89 个开发诊断状态。统一扩域后，由共同新菜单/观察的实际回放重新冻结新面板；它的状态与分母单独记录，不能把旧 88/89 个状态证书直接移植到新动作域。只有确认完整历史、动作域、物理律和目标均相同的状态才可复用。')
p('开发时可按机制分组，但不能按新验证结果决定是否纳入主要汇总。不得把局部条件优势简单平均后叫整趟优势；策略会改变后续状态访问分布，整趟收益必须重新执行完整控制器。如果统一覆盖和便宜控制已经吸收已知信号，允许直接停止方法立项，不以完成更多 bank 作为目的。')
p('阶段 3 的便宜对照包含 V51 原 cheap、single、pair、one_step、条件均值快照；如果新问题超出现有路线树，应在共同扩展动作域中配置真正相关的图搜索/重规划对照。不能给新方法更多候选、更多现场观察或免费预处理。r62 已有便宜 snapshot_csa 的完整动作复现，是应保留的否证案例。')

h('十、只有什么证据出现后，才值得深度重构')
p('建议围绕“同一后续车次在不同站的登乘机会，和不可逆的留车/下车选择如何交互”建立最小机制族。它与现有具体轨迹相关，但当前负例是 cheap 成功，root4 的收益又能由局部候选恢复吸收，故尚未证明这是 cheap 无法解决的机制。最小例子应保留同车有序链、跨车假设、平台对 MCT、合法观察次序、至少一次后续真实选择、取消与截断损失；逐步删去因素并观察优势是否消失。')
p('可立项的理论内容必须比通用恒等式多一步，例如：在可验证的交通条件下，证明只需保留哪些跨站机会边界就能充分表达两策略差值；或给出不枚举所有混合续程仍能有效认证的差值界；或证明一个特殊内层问题确实可多项式求解。要计入构造界、判断适用性和退化回退的成本，并与普通变量消元、rollout / anytime DP 等共享同一内核比较。')
p('不能预设“共享车次数少，所以复杂度低”：复杂度取决于消元后的策略/观察交互宽度。不能预设“每车鞅惩罚，所以方法新”：信息松弛及弱耦合分解已存在，且当前乘客位置、登乘/等待与截断到达损失不自动满足可分条件。不能预设“消息触发重规划，所以新颖”：近年交通算法已有明确研究。[L2-L7]')
p('若最后仍只是普通依赖删除、缓存、少造两个条件链或更好的部分求值调度，应作为工程贡献保留。若收益由统一候选补齐或一个便宜规则全部吸收，应更新强基线并关闭该主方法。若只有不确定区间，保留未知，不将其解释为最优或有必要创造新算法。')

h('十一、G2 还缺什么，OR 的要求是什么')
table(['门槛','当前判断','下一份有效证据'],[
 ['真实且剩余的决策缺口','尚未确立；已知两正例已有便宜的逐例解释','同域同信息、独立配对，机制在新日期重现'],
 ['独立方法增量','未建立；V55约简原则有先例','一个可明确验证的新结构/复杂度/有效界贡献'],
 ['整体净计算价值','局部CPU节省，完整策略不确定性更宽','仅根预界强对照；达到同质量的全成本曲线'],
 ['真正在线执行价值','当前逻辑钟回放不等同硬deadline','统一发布/回退，计算导致错车也进入损失'],
 ['现实泛化与校准','六开发日期不能支持任意条件保证','独立日期/事故、决策前定义分组、相应区间']
],[125,195,192])
p('原 G2 要求方法增量与实际计算价值，并没有要求先把 88 个完整最优 regret 全解；60 秒是项目特定判据，1412 个测试也不是 OR 录用门槛。OR 官方交通方向强调问题精确定义、充分分析、现实相关性、显著贡献以及可核验的新意。它允许机制合理的适用域与负例，不要求所有工况全部胜出。[S9,L1]')
p('原审稿人对 DRO 等价的主要质疑仍未被 V55 自动解决：把现有评分重写成已有优化形式不产生方法增量，正确证明和形式化也不能替代贡献；固定 P0 中的证书更不等于真实任意条件风险校准。算法方向应围绕一个清楚的决策问题和机制重写主张；在剩余缺口尚未找到时，不宜开始新论文式的理论堆叠。[S10]')
p('最终建议：保留 V49/V55 的实现和负证据；先完成候选覆盖与换乘机会模型的有限诊断，再用独立配对的可行策略差识别残留空间。两周初审若仍找不到不被便宜规则吸收、具有现实幅度的机制，就正式结束当前算法主线，评估以建模、复现和经验发现为中心的研究定位。这是资源与证据的停止规则，不是对项目未来价值作否定判断。')

h('十二、复现与文件索引')
p('metrics/recompute.py 只读读取固定提交的原收据，输出 metrics.json 与逐格 CSV；theory_code/counter_audit.py 提供准入/工作计数；profile_queries.py 复现两个带插桩的局部查询。strategy/extract_existing_banks.py 提取136个原gate；extract_actual_mechanisms.py 提取五根交通轨迹；frozen_live_probe.py 运行20条局部消息消融；reconstruct_catch_events.py 重建原随机带并复核768个结果；root4_menu_probe.py 复现三臂菜单机制。环境路径可通过脚本声明的 BAPR_HRO_REPO 调整。')
p('本包报告及统计表是本次分析的派生文件。GitHub 固定提交仍是原始数据与代码来源，包中不复制整个仓库。实验入口、账单、完整正负案例和未知状态保留在相应 JSON/CSV 中。')

REFERENCES = [
 ('S1','V55 完整结果',BASE+'experiments/reconstruction_v55/RESULTS.md'),
 ('S2','V50 菜单与共同观察结果',BASE+'experiments/reconstruction_v50/RESULTS.md'),
 ('S3','V51 控制器比较与理论',BASE+'experiments/reconstruction_v51/RESULTS.md'),
 ('S4','V51 rollout 目标实现',BASE+'experiments/reconstruction_v51/rollout.py'),
 ('S5','V53 根界与强 impact 实现',BASE+'experiments/reconstruction_v53/reference.py'),
 ('S6','V54 回流结果',BASE+'experiments/reconstruction_v54/RESULTS.md'),
 ('S7','V55 新颖性核查',BASE+'experiments/reconstruction_v55/NOVELTY.md'),
 ('S8','当前逻辑发布时钟',BASE+'experiments/reconstruction_v41/run.py'),
 ('S9','原 G2 及停止规则',BASE+'md/review_20261111_v2/plan_extracted.txt'),
 ('S10','原审稿意见 B',BASE+'md/review_20261110/report_B_OR_hyperpath_DRO.txt'),
 ('L1','Operations Research 交通方向编辑声明','https://pubsonline.informs.org/page/opre/editorial-statement/area-editors-statements'),
 ('L2','McMahan, Likhachev & Gordon: Bounded RTDP','https://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf'),
 ('L3','Liu, Blandin & Samaranayake: Transit SOTA','https://arxiv.org/pdf/1808.04360'),
 ('L4','Rish & Dechter: probabilistic inference / elimination','https://ics.uci.edu/~dechter/publications/r42.pdf'),
 ('L5','Brown, Smith & Sun: Information Relaxations and Duality','https://people.duke.edu/~psun/bio/785.pdf'),
 ('L6','Ye, Zhu & Zhou: weakly coupled dynamic programs','https://arxiv.org/pdf/1405.3363'),
 ('L7','Abuaisha et al.: Dynamic Replanning, IJCAI 2025','https://ijcai.org/proceedings/2025/0937.pdf'),
 ('L8','Maurer & Pontil: Empirical Bernstein Bounds','https://www.cs.mcgill.ca/~colt2009/papers/012.pdf'),
]
LOCAL = [
 ('A1','metrics/metrics.json；optimal_cpu_attribution.json；paired_raw_recomputed.csv'),
 ('A2','theory_code/AUDIT.md；counter_audit.json；profile_root*.txt/.json'),
 ('A3','strategy/existing_bank_signals.json；changed_gate_signals.csv；changed_gate_panel_mapping.json'),
 ('A4','strategy/frozen_live_probe.json；frozen_live_receipts/'),
 ('A5','strategy/catch_event_partitions.json；reconstruct_catch_events.py'),
 ('A6','strategy/root4_menu_probe.json；root4_menu_receipts/；root4_menu_probe.py'),
]

def build():
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    navy=colors.HexColor('#173B51'); teal=colors.HexColor('#117D87'); muted=colors.HexColor('#506471')
    styles={
      'title':ParagraphStyle('title',fontName='STSong-Light',fontSize=22,leading=30,textColor=navy,spaceAfter=16,wordWrap='CJK'),
      'heading':ParagraphStyle('heading',fontName='STSong-Light',fontSize=14.5,leading=21,textColor=teal,spaceBefore=16,spaceAfter=8,keepWithNext=True,wordWrap='CJK'),
      'paragraph':ParagraphStyle('paragraph',fontName='STSong-Light',fontSize=10.5,leading=17,spaceAfter=8,wordWrap='CJK'),
      'cell':ParagraphStyle('cell',fontName='STSong-Light',fontSize=9.2,leading=13.5,wordWrap='CJK'),
      'headcell':ParagraphStyle('headcell',fontName='STSong-Light',fontSize=9.2,leading=13.5,textColor=colors.white,wordWrap='CJK'),
      'ref':ParagraphStyle('ref',fontName='STSong-Light',fontSize=9,leading=14,spaceAfter=5,wordWrap='CJK'),
    }
    story=[]; text_parts=[]
    for b in BLOCKS:
        if b[0] in ('title','heading','paragraph'):
            story.append(Paragraph(escape(b[1]),styles[b[0]]));text_parts.append(b[1])
        elif b[0]=='table':
            _,headers,rows,widths=b
            data=[[Paragraph(escape(str(c)),styles['headcell']) for c in headers]]
            data += [[Paragraph(escape(str(c)),styles['cell']) for c in row] for row in rows]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),navy),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#EFF5F7'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,navy),('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#D2E0E4'))]))
            story += [KeepTogether([t,Spacer(1,10)])]
            text_parts.append('\n'.join([' | '.join(headers)]+[' | '.join(map(str,row)) for row in rows]))
    story.append(Paragraph('来源与复算证据',styles['heading']))
    text_parts.append('来源与复算证据')
    for key,title,url in REFERENCES:
        story.append(Paragraph('['+key+'] <link href="'+escape(url,{'"':'&quot;'})+'" color="#146B8A">'+escape(title)+'</link>',styles['ref']))
        text_parts.append('['+key+'] '+title+'\n'+url)
    for key,title in LOCAL:
        story.append(Paragraph('['+key+'] '+escape(title),styles['ref']));text_parts.append('['+key+'] '+title)
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#BCD1DB'));canvas.line(41,39,A4[0]-41,39)
        canvas.setFont('STSong-Light',8);canvas.setFillColor(muted)
        canvas.drawString(41,26,'BAPR-HRO V55 审计 | a2b82be9 | 2026-11-15')
        canvas.drawRightString(A4[0]-41,26,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(OUT/'V55_diagnosis_and_plan.pdf'),pagesize=A4,rightMargin=41,leftMargin=41,topMargin=38,bottomMargin=51,title='BAPR-HRO V55 深度诊断与下一阶段方案',author='ChatGPT')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'V55_diagnosis_and_plan.txt').write_text('\n\n'.join(text_parts)+'\n')
    (OUT/'report_content.json').write_text(json.dumps({'commit':COMMIT,'blocks':BLOCKS,'references':REFERENCES,'local_evidence':LOCAL},ensure_ascii=False,indent=2)+'\n')
    print(OUT/'V55_diagnosis_and_plan.pdf')

if __name__=='__main__':build()
