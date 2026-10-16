# Independent judgment plus residual correction

本轮实验将分数分解为 `s_i = b_i + delta_i`，或门控形式 `s_i = b_i + g_i * delta_i`。
`b_i` 为已训练的独立 MLP 分数；修正分支通过同题候选注意力获得上下文。
修正头最后一层零初始化，使第二阶段初始预测精确等于同种子独立模型。

## 七组对照

| 组 | 独立分支 | 修正信息 | 用途 |
|---|---|---|---|
| independent | 从头训练 | 无 | 基线与预训练来源 |
| attention | 无独立分数直达路径 | 原单层注意力 | 与原最佳方法对照 |
| finetune | 独立预训练后继续训练 | 无 | 额外训练对照 |
| residual_self | 冻结且固定 eval | 仅自身 | 新增修正路径/参数对照 |
| residual_context | 冻结且固定 eval | 四候选 | 核心跨候选修正 |
| residual_gated | 冻结且固定 eval | 四候选＋门控 | 自适应修正幅度 |
| residual_joint | 允许一起微调 | 四候选＋门控 | 冻结是否必要 |

两个学习率先用 seed 17 搜索，其余四种子用选定学习率。共 42 次实际训练，35 个主比较检查点。
所有组使用全部题 BCE、同一固定输入、训练集归一化。阶段二第 0 轮状态允许被选中作为回退，
但这种验证驱动回退并不保证新数据上不退步。

冻结基线的参数及输出在整个训练中保持不变；最终选择仍可能被修正改变。
门控初始为 0.1。门控值不是校准置信度，实际贡献要结合修正量一起看。
仅自身注意力与跨候选注意力的存储参数量相同，但仅一个 key 时 Q/K 匹配退化。
没有 out-of-fold 预训练预测；独立预训练与修正训练使用同一训练集，报告说明其限制。

## 文件

- `protocol.json`：训练前固定的完整协议与主要比较。
- `baseline_models.py`：上一轮原模型代码的原样副本。
- `models.py`：包装原模型、零初始化修正、冻结与门控。
- `check_models.py`：初始保持、冻结、学习能力、信息依赖、排列等变与 batch 独立检查。
- `run_suite.py`：两阶段训练、来源哈希、官方评分、独立分支消融与完整资产导出。
- `summarize.py`：主表、配对 bootstrap、修正/新增错误、成本与中文报告。
- `verify_delivery.py`：导出模型重新加载、官方评分、新 ID/重排输入与文件完整性检查。
- `results/REPORT.md`、`summary.csv`、`paired_comparisons.json`、`repair_summary.csv`：主要结果。
- 各运行的 `validation-components.npz`：`base`、`raw_delta`、`gate`、`applied_delta`、`final`。
- 各运行的 `base-only-score.json`：关闭修正、只用当前独立路径的官方评分。
- `results/selected_bundle/`：固定 seed 17 单模型，已包含原独立路径全部权重；不依赖额外检查点。

## 复现

从 `code` 目录执行；训练解释器为 `D:/anaconda/envs/mp1/python.exe`，依赖见 `requirements.txt`。
完整重训使用新输出目录；源码、配置、协议或来源检查点不一致时不复用训练缓存。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B residual_experiments_20261017/check_models.py
python -B residual_experiments_20261017/run_suite.py --output residual_experiments_20261017/reproduction
python -B residual_experiments_20261017/summarize.py --output residual_experiments_20261017/reproduction
python -B residual_experiments_20261017/verify_delivery.py --output residual_experiments_20261017/reproduction
```

绘图使用已安装 matplotlib 的 Python，本机为 `D:/anaconda/python.exe`，仅处理保存的统计量：

```powershell
python -B residual_experiments_20261017/summarize.py --plot-only --output residual_experiments_20261017/reproduction
```

重新预测、评分：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B predict.py --method residual_experiments_20261017/method.py --checkpoint residual_experiments_20261017/results/selected_bundle --split validation --output residual_experiments_20261017/replayed-validation.json
python -B score_predictions.py --split validation --predictions residual_experiments_20261017/replayed-validation.json --output residual_experiments_20261017/replayed-score.json
```

验证集仍用于搜索、早停和方案选择。五种子与题目 bootstrap 不消除重复使用验证集的偏差。
测试集不训练、不选择、不预测、不评分。原数据、官方代码和此前实验均保持不变。
实验代码由 Codex AI 辅助编写运行；所有模型仅使用现有训练数据，无外部数据或预训练模型。
