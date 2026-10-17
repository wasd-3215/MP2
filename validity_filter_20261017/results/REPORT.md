# 最终选择前排除解析无效候选：固定模型配对实验

日期：2026-10-17（北京时间）。复用固定20轮模型，不训练、不调参、不重新校准。

完成18个基础预测器的过滤前后配对，并增加冻结回退的过滤前后配对：共38次独立新进程预测与官方评分。

## 主要发现

- 普通S2五成员集成：735→737题正确（92.105%→92.356%），两次替换均纠正；带解析特征的集成734→735；一致性集成740→740，没有触发替换。
- 所有19对中，没有新增错误；本验证集发生的替换均恰好修复错误。后一观察不保证新题上每次替换都正确，理论只保证按官方指标不降分。
- 单模型五种子均值：普通S2过滤增加0.476pp，解析模型增加0.201pp，一致性模型增加0.025pp。解析特征相对普通模型的优势从0.401pp缩至0.125pp；后者95%描述性区间[-0.301,+0.551]。收益中的较大部分可由最终约束补齐，仍不能单凭差值将学习机制完全归因。
- 冻结回退加过滤达到740/798=92.732%；相对统一过滤基线737题，它纠正5题、改错2题，净增加3题（+0.376pp，95%描述性区间[-0.251,+1.003]）。因此这条回退的收益并非全部来自解析修复，额外收益的区间仍包含零。
- 冻结回退加过滤与一致性集成整体同为740题，但单正确题为74/117对63/117，错误共识题为35/60对24/60。这个并列比较是观察结果后的描述，未重新选择模型，也不是独立测试优越性结论；两个子组重叠，不能累加。
- 过滤没有修复一致性集成的错误共识弱点：该集成原来没有无效选择，过滤后错误共识仍为40%。普通集成过滤后为61.667%，原有取舍依然存在。

## 固定规则

四个候选完整进入原网络。成员sigmoid分数先按原规则平均，原回退照常执行；仅当最终选择的官方parse_answer结果为None且存在有效候选时，改选原聚合分数最高的有效候选。有效选择原样保留；全部无效保留原选择；并列选原顺序最早候选。未修改官方解析器。

所有主结果来自原版predict.py与score_predictions.py。诊断读取验证参考答案，推理源码不读取参考答案或保存预测。没有测试输入、预测、评分或训练。

## 五种子单模型结果

| 输入 | 原始均值±样本SD % | 过滤后均值±样本SD % | 平均替换题数 | 平均修正题数 | 平均增益pp |
|---|---:|---:|---:|---:|---:|
| s2 | 91.253 ± 0.722 | 91.729 ± 0.698 | 3.8 | 3.8 | +0.476 |
| s2_parse | 91.654 ± 0.521 | 91.855 ± 0.524 | 1.6 | 1.6 | +0.201 |
| s2_consistency | 92.130 ± 0.380 | 92.155 ± 0.392 | 0.2 | 0.2 | +0.025 |

各行是五个单模型的统计，不是可部署集成分数；逐种子结果见filter_pairs.csv。

## 固定五成员集成与回退

| 预测器 | 原正确数/798 | 过滤后正确数/798 | 原无效选择 | 替换 | 修正 | 替换后仍错 | 新增错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s2_ensemble5_original | 735 | 737 | 2 | 2 | 2 | 0 | 0 |
| s2_parse_ensemble5_original | 734 | 735 | 1 | 1 | 1 | 0 | 0 |
| s2_consistency_ensemble5_original | 740 | 740 | 0 | 0 | 0 | 0 | 0 |
| s2_ensemble5_route_original | 739 | 740 | 1 | 1 | 1 | 0 | 0 |

## 同协议关键比较

以下区间为10000次按题目配对bootstrap的描述性95%区间；单模型先对五种子正确性平均。区间条件于冻结预测器，不包含历史验证开发/选择不确定性，未校正多重比较。

| 比较 | 子组 | 差值pp [95%区间] |
|---|---|---:|
| s2: filter - original (five-seed mean) | all | +0.476 [+0.150, +0.927] |
| s2_parse: filter - original (five-seed mean) | all | +0.201 [+0.025, +0.501] |
| s2_consistency: filter - original (five-seed mean) | all | +0.025 [+0.000, +0.075] |
| s2_parse - s2: original (five-seed mean) | all | +0.401 [-0.050, +0.927] |
| s2_consistency - s2_parse: original (five-seed mean) | all | +0.476 [-0.677, +1.629] |
| s2_parse - s2: filtered (five-seed mean) | all | +0.125 [-0.301, +0.551] |
| s2_consistency - s2_parse: filtered (five-seed mean) | all | +0.301 [-0.827, +1.454] |
| B-A: filter | all | +0.251 [+0.000, +0.627] |
| C-A: frozen route | all | +0.501 [-0.125, +1.253] |
| D-B: frozen route after equal final filtering | all | +0.376 [-0.251, +1.003] |
| D-C: filter after route | all | +0.125 [+0.000, +0.376] |

## 困难子组：固定集成

| 预测器 | 子组 | 题数 | 原准确率 % | 过滤后 % | 修正题数 |
|---|---|---:|---:|---:|---:|
| s2_ensemble5_original | mixed | 419 | 84.964 | 85.442 | 2 |
| s2_ensemble5_original | one_correct | 117 | 64.103 | 64.957 | 1 |
| s2_ensemble5_original | wrong_unique_plurality | 60 | 61.667 | 61.667 | 0 |
| s2_ensemble5_original | multiple_valid_answers | 355 | 82.817 | 82.817 | 0 |
| s2_ensemble5_original | any_invalid_parse | 126 | 88.095 | 89.683 | 2 |
| s2_parse_ensemble5_original | mixed | 419 | 84.726 | 84.964 | 1 |
| s2_parse_ensemble5_original | one_correct | 117 | 63.248 | 63.248 | 0 |
| s2_parse_ensemble5_original | wrong_unique_plurality | 60 | 60.000 | 60.000 | 0 |
| s2_parse_ensemble5_original | multiple_valid_answers | 355 | 82.254 | 82.254 | 0 |
| s2_parse_ensemble5_original | any_invalid_parse | 126 | 88.095 | 88.889 | 1 |
| s2_consistency_ensemble5_original | mixed | 419 | 86.158 | 86.158 | 0 |
| s2_consistency_ensemble5_original | one_correct | 117 | 53.846 | 53.846 | 0 |
| s2_consistency_ensemble5_original | wrong_unique_plurality | 60 | 40.000 | 40.000 | 0 |
| s2_consistency_ensemble5_original | multiple_valid_answers | 355 | 83.662 | 83.662 | 0 |
| s2_consistency_ensemble5_original | any_invalid_parse | 126 | 87.302 | 87.302 | 0 |
| s2_ensemble5_route_original | mixed | 419 | 85.919 | 86.158 | 1 |
| s2_ensemble5_route_original | one_correct | 117 | 62.393 | 63.248 | 1 |
| s2_ensemble5_route_original | wrong_unique_plurality | 60 | 58.333 | 58.333 | 0 |
| s2_ensemble5_route_original | multiple_valid_answers | 355 | 83.662 | 83.662 | 0 |
| s2_ensemble5_route_original | any_invalid_parse | 126 | 88.889 | 89.683 | 1 |

## 回退归因

A=原集成；B=原集成加过滤；C=原冻结回退；D=原冻结回退加过滤。阈值固定model_agreement≤0.8且唯一最高票至少3票；不重新计算过滤后的置信度或拟合阈值。D−B衡量这条冻结策略在统一最终过滤下的额外贡献。

| 比较 | 原选择分层 | 题数 | 答案变化 | 修正 | 改错 | 净修正 |
|---|---|---:|---:|---:|---:|---:|
| C-A | all | 798 | 8 | 6 | 2 | +4 |
| C-A | base_choice_valid | 796 | 7 | 5 | 2 | +3 |
| C-A | base_choice_invalid | 2 | 1 | 1 | 0 | +1 |
| D-B | all | 798 | 7 | 5 | 2 | +3 |
| D-B | base_choice_valid | 798 | 7 | 5 | 2 | +3 |
| D-B | base_choice_invalid | 0 | 0 | 0 | 0 | +0 |

## 完整fresh验证预测加独立评分及资产

| 预测器 | 秒 | 解包后资产MiB |
|---|---:|---:|
| s2_ensemble5_original | 5.161 | 5.242 |
| s2_ensemble5_filtered | 5.048 | 5.242 |
| s2_parse_ensemble5_original | 5.002 | 5.243 |
| s2_parse_ensemble5_filtered | 5.082 | 5.243 |
| s2_consistency_ensemble5_original | 5.072 | 5.249 |
| s2_consistency_ensemble5_filtered | 5.050 | 5.249 |
| s2_ensemble5_route_original | 5.277 | 5.242 |
| s2_ensemble5_route_filtered | 5.100 | 5.242 |

纯过滤（包含官方解析）19个预测器的中位耗时均值：0.019484秒/798题；每个重复15次。这是诊断微计时，不替代完整fresh计时。

推理资产计入每个方案全部成员的权重、标准化统计、配置和流水线manifest；不累加互斥实验方案。完整测量包含启动、导入、原始输入加载和全部集成成员；进程串行，Torch intra-op4/inter-op1，OpenBLAS1。

本机：AMD Ryzen 7 4800U with Radeon Graphics         ；Python 3.12.14，NumPy 2.5.3，PyTorch 2.7.1+cpu。未在官方Xeon及完整测试集测量，不据此认证30秒资格。

## 验证与解释边界

- 192组掩码/原选择/分数边界测试通过，包括全部无效与并列；有效的原回退选择严格保留。
- 19个无过滤预测器复现原方案或原冻结成员分数聚合；每对过滤前后分数相同，改变选择恰好等于可替换题集合；全部配对新增错误为0。
- 38个预测器通过新ID、乱序13题、单题和原生int检查；完整记录见new_bank_checks.json。
- 验证集含解析失败候选126题，不等于模型选中无效候选的题数。单正确题与错误共识题重叠，不能相加。
- 理论上的逐题不降分仅来自当前官方数值解析评分规则；不代表推理过程正确或所有人工语义标准下正确。
- 过滤不改变有效候选之间的分数；解析特征在统一过滤后仍可能通过训练改变排序。过滤后仍有差异不能直接证明具体因果机制。
- 复用的所有模型均来自同一固定20轮协议；本轮没有训练成本、超参数搜索或模型优胜者选择。
- 当前验证已反复用于历史开发；罕见替换带来的小幅增益应按题数解释，不能当作独立测试显著性。
- 代码与分析由Codex AI辅助；未使用外部数据、预训练权重或API。原文件保护结果见verification.json与protected_verification.json。

## 产物与复现

filter_pairs.csv/JSON包含全部19对；subgroups.csv/JSON包含诊断子组；changed_questions.csv/JSON逐题列出替换；paired_comparisons.json保存配对区间；baseline_replay.json保存复现来源；pipelines下各预测器包含官方预测、评分及外部计时。

新实验目录的README.md提供运行命令。pipelines中的原始成员路径用于复现实验；portable目录中的独立bundle可脱离历史实验目录加载，复核见verification.json。
