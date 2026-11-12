# 86ba89c 独立审计与问题发现

固定仓库提交：86ba89c327e5230f1290e8678bebcb589b2465a9。
仓库：https://github.com/erzhu419/BAPR-HRO
日期：2026-11-12。

这是本轮分析的原始探针、选择清单、逐根回执与汇总。它们未修改原仓库，全部是开发期证据；既有日期不因此重新成为独立确认。原脚本原样保留，MANIFEST.json 记录每个文件的 SHA256。

## 主要入口

- current/SUMMARY.json：61 根的名义及完整条件 BOARD 精确积分、动作边际、最大残余和数值误差核查。
- current/residual_full_2023-07-17_2_31200_origin.json：一个非预知的多未来机会反例、全部 Bellman 归因、两目标控制及 CDF 稳定性核查。
- board_gap/AUDIT.md：恢复目录、所有候选下车点、尾站控制、公开历史传播、途中滚动控制及缺测账本。
- board_gap/headroom_45.json：唯一 45 秒回顾路径空间的端点与公开信息边界；不是已实现的策略收益。
- algorithm/broad_probe_selection.json：在读取实际结果之前按静态条件选择的 150 新 OD 筛选清单。
- algorithm/broad_probe_summary.json：10 根新 OD 的完整结果，6 完整实际结果及 4 缺测。
- verification.txt：本轮 52 项针对性测试回执。

## 复现

使用该提交项目的 Python 环境与依赖。解压本包，提供已有只读 checkout 的绝对路径。replay.py 检查提交身份，仅重定向原探针的路径变量；不改科学计算逻辑。它将新结果写入指定目录，不覆盖本包原始证据。

    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced small --law segment --mode nominal --max-k 8
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced small --law segment --mode full --max-k 8
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced residual --mode full
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced board
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced overlap
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced conditional_overlap
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced rolling_alight
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced overlap_witnesses
    python replay.py --repo /absolute/path/BAPR-HRO --out /absolute/path/reproduced broad

rolling_alight 会读取之前保存的 overlap/board 结果，建议先依次运行 board、overlap，再运行 rolling_alight。原始脚本中有本次工作区的绝对路径；直接运行原脚本前需要调整路径，推荐统一用 replay.py。

## 解释边界

1. 精确指固定有限模型和固定续程内的期望，不是现实任意历史校准或一般网络最优。
2. 61 根没有按收益挑选，覆盖全部既有 K≤8 非空根；六个大菜单未全解。
3. 两目标吸收 4.322 秒的完整归因针对选中的最大残余根；新 OD 10 根也均核查两目标，但其他旧五个残余根未全做两目标实验。
4. BOARD 目录扩展仍保留原候选窗口、至多一次换乘、至少两辆计划可接尾车等限制。
5. 交叠恢复的 473/501 秒收益只相对弱 first-offer，强基线乘同一最终车辆，损失相同。
6. 已选根车的 274 个下车候选中 24 缺标签；不能把缺测当失败或无改善。
7. 新 OD 选择从 701 站、28164 连接随机抽 150 对，静态筛出 10 根；这是探索机制覆盖，不是全市频率估计。
8. 运行时间是审计端到端开销，不能作为生产算法质量—计算优势。
