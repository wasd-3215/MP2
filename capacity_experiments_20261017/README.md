# Capacity and regularization experiments

本目录保存独立 MLP、候选注意力的容量/正则化对照，原有文件和前两轮实验全部保留。

| 配置 | 隐藏宽度 | Dropout | AdamW weight decay |
|---|---:|---:|---:|
| baseline | 128 | 0.1 | 0.001 |
| width64 | 64 | 0.1 | 0.001 |
| width32 | 32 | 0.1 | 0.001 |
| dropout30 | 128 | 0.3 | 0.001 |
| decay01 | 128 | 0.1 | 0.01 |

每种结构都训练五种配置；每组搜索两个学习率，按 seed 17 选择后再运行四个种子。
总计 60 次训练，50 个检查点用于五种子对照。

所有组统一使用全部题 BCE、全部训练集拟合的同一归一化和原始固定 embedding。
本轮没有同时加入混合题采样，防止将训练目标/数据分配与容量变化混淆。
缩小隐藏宽度时评分头中间层保持 64、注意力头数保持 4。
Dropout 变体改变已有各 dropout 位置，包含注意力权重 dropout。

## 诊断方法

每轮优化后用 `model.eval()` 在同一检查点计算完整训练集和验证集指标，
关闭 dropout，记录 BCE、混合题 BCE 和正确性诊断。
另外保存初始化与共同第 10 轮（190 次更新）的指标。
选定检查点的训练/验证准确率均由未修改的官方评分脚本计算；训练准确率只用于衡量拟合。

比较：

- 验证准确率是否上升，以及五种子同题配对区间。
- 训练准确率减验证准确率的差距。
- 验证 BCE 减训练 BCE 的差距。
- 混合题上的准确率和 BCE，排除全对题对选择指标的稀释。
- 参数量、推理资产、训练成本与本机完整验证耗时。

更小的差距不必然意味着更好泛化：如果训练拟合变差而验证准确率不升，
更符合欠拟合/优化变化的解释。不同数据难度和概率校准也会影响 BCE 差距。
各组按同样预算独立选学习率，主结果是调参后表现；另报同学习率 seed 17 的有限对照。

## 文件

- `protocol.json`：训练前固定的配置、选择规则、诊断与限制。
- `models.py`、`features.py`、`runtime.py`、`method.py`、`support.py`：复制上一轮实现。
- `measurements.py`：关闭 dropout 的完整数据集测量。
- `check_models.py`：宽度/正则化配置、有限梯度、等变性、测量不改变状态和随机数检查。
- `run_suite.py`：60 次训练、逐轮日志、官方验证、训练集官方评分、代码/资产哈希。
- `summarize.py`：汇总、配对 bootstrap、学习曲线、基线复现、检查点导出与报告。
- `verify_delivery.py`：最终模型重新加载、官方验证、新 ID/重排子集与文件完整性验证。
- `results/REPORT.md`：完整结果及解释；`summary.csv` / `runs.csv` / `learning_curves.csv` 为分析数据。
- `results/selected_bundle`：按五种子均值选择配置，固定 seed 17 的单模型。

## 复现

从 `code` 目录运行。本机训练 Python 为 `D:/anaconda/envs/mp1/python.exe`，版本 3.12，
依赖见 `requirements.txt`。使用新目录完整重训；源码、协议、配置变化时拒绝复用旧训练缓存。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B capacity_experiments_20261017/check_models.py
python -B capacity_experiments_20261017/run_suite.py --output capacity_experiments_20261017/reproduction
python -B capacity_experiments_20261017/summarize.py --output capacity_experiments_20261017/reproduction
python -B capacity_experiments_20261017/verify_delivery.py --output capacity_experiments_20261017/reproduction
```

绘图需有 matplotlib 的 Python。当前绘图使用 `D:/anaconda/python.exe`，
仅读取保存的数值，不训练或选择模型；环境另存 `plot_environment.json`。

```powershell
python -B capacity_experiments_20261017/summarize.py --plot-only --output capacity_experiments_20261017/reproduction
```

重新评估已有模型：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B predict.py --method capacity_experiments_20261017/method.py --checkpoint capacity_experiments_20261017/results/selected_bundle --split validation --output capacity_experiments_20261017/replayed-validation.json
python -B score_predictions.py --split validation --predictions capacity_experiments_20261017/replayed-validation.json --output capacity_experiments_20261017/replayed-score.json
```

## 边界

验证集用于搜索和早停，且已被前两轮重复使用。五种子/配对 bootstrap 仍是开发证据；
未使用测试集训练、选择、预测或评分。无外部数据或预训练模型。
不是严格的非劣实验，也不能单凭本轮诊断认定 embedding 已达到信息上限。
本机计时不替代官方硬件的时间限制检查。代码由 Codex AI 辅助编写运行，需用户审阅理解。
