# Interaction stage experiments

本轮比较一次候选注意力前后如何分配逐候选处理。
所有模型统一四头、允许自身、隐藏维度128，从随机初始化共同训练。

定义：`A = LN_A(h + Dropout(MHA(h)))`，`P_i = h + Dropout(FFN_i(h))`，
`N_i = LayerNorm_i`，完整块 `F_i = N_i(P_i)`。FFN 为128→256→128。
编码器1024→128和评分头128→64→1固定，以下省略它们。

| 配置 | 顺序 | 参数 |
|---|---|---:|
| s0_after | A → F1 | 272001 |
| s1_before | F1 → A | 272001 |
| s2_before_keep_norm_after | P1 → A → N1 | 272001 |
| s3_before_before | F1 → F2 → A | 338177 |
| s4_before_after | F1 → A → F2 | 338177 |
| s5_after_after | A → F1 → F2 | 338177 |

S2 区分部分非线性加工位置与归一化位置影响。
同容量组比较才能直接检验阶段；两块模型与S0之间同时有容量变化。
评分头也有逐候选非线性，因此“交互前/后”专指上述独立块的位置。

## 训练与分析协议

- 主实验固定学习率0.0003，六组×种子17/29/43/71/101，共30次训练。
- 敏感性实验只用seed17、学习率0.001，每组一次，共6次；不参与主学习率/最终模型选择。
- 全部运行60轮、1140次更新，无早停；第10/20/40/60轮保存预测并调用原版官方评分脚本。
- 最佳检查点按60轮内官方验证正确数、较低验证BCE、较早轮次选择。
- 原有全部题BCE、训练集归一化、dropout0.1、AdamW衰减0.001、batch256、梯度裁剪1。
- 公共E/A/F1/H先按原模型初始化，S3-5再追加相同初始化的F2；只改变前向顺序。
- 主实验使用新预算的S0对照；不把旧早停91.90%直接当成本轮基线。
- 同一eval检查点测训练/验证BCE和正确性，避免将随机dropout损失当作泛化差距。
- 主要比较同容量布局；最佳检查点和固定轮次均有配对bootstrap，10000次按题目抽样，先在五种子间平均。

验证集重复用于实验与检查点选择，区间有条件且未做多重比较校正，不能替代独立测试。
测试集没有训练、选择、预测或评分。只允许官方完整性检查核对其哈希/对齐。

## 文件

- `protocol.json`：训练前固定的结构、预算、主要比较和限制。
- `reference_models.py`：原模型逐字副本；`models.py`仅调整模块执行顺序并按需增加F2。
- `check_models.py`：模块调用顺序、匹配容量/初始化、S0前向和梯度完全一致、有限梯度、排列等变等检查。
- `run_suite.py`：36次完整训练；固定轮次、最佳检查点的官方评分与源代码哈希。
- `summarize.py`：固定轮次/最佳模型配对比较、敏感性、拟合差距、学习曲线与报告。
- `verify_delivery.py`：选定模型独立进程重评；六种布局seed17的新ID、重排输入和单题验证。
- `results/REPORT.md` / `summary.csv`：主结果及解释。
- `results/fixed_epochs.csv`：相同更新次数下的表现；`paired_comparisons.json`为所有配对区间。
- `results/baseline_prefix_replay.json`：S0在旧早停终点前的轨迹复现，以及新60轮最佳结果。
- `results/selected_bundle`：只根据30个主运行选择配置，固定seed17的单模型。

## 复现

从code目录执行。训练Python为 `D:/anaconda/envs/mp1/python.exe`，依赖见requirements.txt。
完整重训使用新输出路径；源码、协议、配置变化时拒绝复用旧缓存。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B interaction_stage_20261017/check_models.py
python -B interaction_stage_20261017/run_suite.py --output interaction_stage_20261017/reproduction
python -B interaction_stage_20261017/summarize.py --output interaction_stage_20261017/reproduction
python -B interaction_stage_20261017/verify_delivery.py --output interaction_stage_20261017/reproduction
```

绘图使用装有matplotlib的Python，本机为 `D:/anaconda/python.exe`；只读取已保存的轨迹：

```powershell
python -B interaction_stage_20261017/summarize.py --plot-only --output interaction_stage_20261017/reproduction
```

重评已有模型：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B predict.py --method interaction_stage_20261017/method.py --checkpoint interaction_stage_20261017/results/selected_bundle --split validation --output interaction_stage_20261017/replayed-validation.json
python -B score_predictions.py --split validation --predictions interaction_stage_20261017/replayed-validation.json --output interaction_stage_20261017/replayed-score.json
```

CPU串行运行，PyTorch intra-op4/inter-op1，BLAS1。推理只读取输入和完整bundle，不读取标签或按ID查表。
原有数据、官方脚本及此前实验全部保留。新增实验代码由Codex AI辅助编写运行，需用户审阅理解。
