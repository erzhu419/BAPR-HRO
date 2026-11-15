BAPR-HRO 6e46267 / V46–V47 诊断与复核材料
固定commit：6e46267b56656595d2f996bc206ab65c5ee222d8
日期：2026年11月14日

请先阅读DIAGNOSIS_AND_PLAN.txt。完整源仓库仍以该固定GitHub提交为准，本包没有修改仓库，也不包含原实验仓库的重复副本。

主要结论
V47额度回流成立，但单臂仍是局部概率排序的DFS，造成根影响极小的子树占据预算。root27/63回流任务全部根概率小于1e-50；root62剩余183秒宽度的99.995%来自7个未展开算术event。原模型/策略/2000额度的全局前沿顺序消融在root27/63/62均得到排零区间，但朴素实现很慢。随后增量队列在root27单次验证保持同一最终区间，完整query CPU从朴素110.77秒降到7.74秒；首个保存证书约5.88秒。该证据支持修复强普通求值器，尚不证明独立OR方法贡献。

文件导航
DIAGNOSIS_AND_PLAN.txt       完整中文诊断、结果解释、实施与G2判断方案
audit_environment.json      固定commit、环境和验证范围
metrics/AUDIT.md            原63调用/330检查点复算、root62/13前沿、动态结构准入
metrics/recomputed.json     全部归档重算及每个根的值/费用
metrics/diagnostic_summary.json  root62/13的原内核诊断摘要
metrics/diagnostic_replay/   原内核同2000额度的观测重放
metrics/frontier_shape.json  剩余菜单静态准入，未声称动态批量接入已验证
theory_code/AUDIT.md        代码与证明审查、root27/63诊断及三格顺序消融
theory_code/priority_probe.py  朴素全图扫描的合法优先顺序原型
theory_code/priority_diagnostic/  三格全部结果、成本和负结果
theory_code/new_tests.log   本次67项新增测试，非953全套重跑
theory_code/heap_adversarial_checks.*  独立多父/bulk增量核对
frontier_efficiency/        增量权重与版本堆原型、校验、root27单次计费
strategy/AUDIT.md           方法贡献、文献对照与停止条件
SHA256_MANIFEST.json        本包文件校验

证据范围
原项目发布结果：V47仍0/7、953测试通过。
本次独立验证：读取归档并复算；新增V46/V47测试67项全部通过；4个原内核同额度观测重放；3个朴素优先顺序诊断；1个增量队列自然状态诊断。没有扩大2000单次额度，没有新增在线旅行回放，没有重新采集数据。
新诊断不属于原63正式调用，单独保存。区间排零是固定根P0和原π的结论，不是全网最优保证、整趟策略改进或真实条件风险校准。浮点区间宽归零不当作所有数学分支都精确枚举完。

复现准备
检出上述commit，安装项目依赖；在解压目录执行。所有脚本应支持BAPR_HRO_REPO，建议显式设置，并将同一仓库加入PYTHONPATH。

export BAPR_HRO_REPO=/absolute/path/BAPR-HRO
export PYTHONPATH="$BAPR_HRO_REPO"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

读取原归档复算：python3 metrics/recompute.py
原内核诊断：python3 metrics/diagnostic_replay.py
原root27/63前沿：python3 theory_code/frontier_probe.py
朴素优先顺序：python3 theory_code/priority_probe.py 55 122 120
增量优先顺序：python3 frontier_efficiency/efficient_priority.py 55 --output /absolute/path/new_root27_result.json

注意：55/122/120是状态bank索引，对应root27/63/62，不是根编号。原型会保留或拒绝覆盖已有结果，若复跑请使用新的输出位置，不删除本包原证据来选择更快结果。原始完整计时范围和解释见对应AUDIT，代码导入/输入载入/query范围不得混算。

本包旨在把下一步收紧为可验收的实现和贡献判断。认证成功不自动等于G2通过；如果强普通方法吸收所有优势，应重新评估论文定位。
