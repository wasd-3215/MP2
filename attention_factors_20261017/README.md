# Attention heads × self inclusion

8 个配置：头数 1 / 2 / 4 / 8，与允许 / 排除自身 key/value 交叉。
原模型是 `include_h4`。全部配置固定总宽度 128、FFN 256、评分头 64、
dropout 0.1、weight decay 0.001、全部题 BCE；参数量均为 272001。

排除自身时对角线 mask 为 True，禁止读取自身 key/value。
query 仍由自身表示生成，自身残差路径保留；这不意味着模型丢失自己的信息。
每头维度依次为 128、64、32、16；同种子初始权重张量在八组中逐项一致。

## 协议与分析

每组 seed 17 搜索学习率 0.001 / 0.0003，再按所选学习率运行 29、43、71、101。
最多 60 轮、patience 10、每轮 19 次更新。共 48 次训练，40 个检查点进入五种子主表。
按官方验证正确数、较低验证 BCE、较早轮次选择检查点。

分析包含：

- 每个头数下，排除自身减允许自身。
- 每种掩码下，各头数相对四头。
- 将另一个因素各水平等权平均后的主效应。
- 排除自身的效果在不同头数间是否变化，以四头为交互作用参照。
- 所有配置相对原模型的修正/新增错误与训练—验证差距。
- 两个学习率下 seed 17 的完整交叉对照，辅助识别选定学习率不同带来的混淆。

主结果比较相同调参预算下的配置，所选学习率可能不同。
题目 bootstrap 先在五种子间平均正确性，不能把种子×题目当成独立样本。
区间条件于已选择模型，未包含搜索/早停不确定性，未校正多重比较。
验证集已被前几轮使用，因此所有结果仍是开发证据。

## 文件

- `protocol.json`：训练前固定的八组协议与分析规则。
- `models.py`：原模型增加 `num_heads` 和 `exclude_self` 配置。
- `reference_models.py`：上一轮模型原样副本，仅用于检查原四头行为一致。
- `check_models.py`：参数/初始权重一致、显式 QK-softmax-V 参考、掩码、梯度、排列等变与 batch 独立检查。
- `run_suite.py`：训练、eval 模式拟合诊断、官方训练/验证评分、资产/代码哈希。
- `summarize.py`：因素主效应、交互作用、配对 bootstrap、错误变化、报告与图。
- `verify_delivery.py`：最终模型独立进程评分；八组 seed 17 的新 ID、重排子集和单题推理检查。
- `results/REPORT.md` / `summary.csv` / `runs.csv`：报告、五种子汇总与全部 48 次运行。
- `results/factorial_contrasts.json`：主要因素与交互作用分析，附探索性单元对照。
- `results/same_lr_seed17_contrasts.csv`：同学习率的单种子对照。
- 每个运行的 `validation-attention-weights.npy`：形状 `[798, heads, 4, 4]`，不属于推理资产。
- `results/selected_bundle`：五种子均值最高配置的固定 seed 17 单模型。

注意力权重与归一化熵只是读取分布的描述，不能直接作为正确性、因果解释或贡献大小。
候选位置没有编码，预测与 ID 无关；推理只读取输入和 bundle，不读取参考答案。

## 复现

从 `code` 目录执行。当前训练 Python 为 `D:/anaconda/envs/mp1/python.exe`，
依赖见 `requirements.txt`。完整重训请使用新输出路径；源码/协议/配置变化时拒绝复用缓存。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B attention_factors_20261017/check_models.py
python -B attention_factors_20261017/run_suite.py --output attention_factors_20261017/reproduction
python -B attention_factors_20261017/summarize.py --output attention_factors_20261017/reproduction
python -B attention_factors_20261017/verify_delivery.py --output attention_factors_20261017/reproduction
```

绘图需装有 matplotlib 的 Python；本机为 `D:/anaconda/python.exe`。只读取已保存统计量：

```powershell
python -B attention_factors_20261017/summarize.py --plot-only --output attention_factors_20261017/reproduction
```

已有模型的预测与官方评分：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B predict.py --method attention_factors_20261017/method.py --checkpoint attention_factors_20261017/results/selected_bundle --split validation --output attention_factors_20261017/replayed-validation.json
python -B score_predictions.py --split validation --predictions attention_factors_20261017/replayed-validation.json --output attention_factors_20261017/replayed-score.json
```

全部实验串行运行，PyTorch intra-op 4、inter-op 1、BLAS 1。资产按单模型完整解包后大小计算；
本机时间不替代官方硬件检查。测试集不训练、不选择、不预测、不评分。
数据、官方代码、此前实验全部保留。新增代码由 Codex AI 辅助编写运行，需用户审阅理解。
