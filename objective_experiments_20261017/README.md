# Training-objective experiment

本目录独立保存本轮实验；不修改上一轮 `experiments_20261017`、数据、解析器或评分器。
实验以原有独立 MLP、四候选注意力两种结构为基础，对照四种目标。

| 目标 | 训练题池 | 每题损失 |
|---|---|---|
| `bce_all` | 全部 4755 题 | 四个候选 BCE 的平均 |
| `bce_mixed` | 混合 2513 题 | 四个候选 BCE 的平均 |
| `pairwise` | 混合 2513 题 | 所有正负对的 `softplus(s_negative - s_positive)` 平均 |
| `set_mass` | 混合 2513 题 | `logsumexp(s_all) - logsumexp(s_positive)` |

每题可有多个正确候选。先题内平均再按题平均，防止正负对数量改变题目权重。
全对题不提供排序梯度。禁止将多正确标签强行转换成单一目标下标。

每个预算轮次都抽取 4755 题、更新 19 次；混合题组用打乱的题池循环补齐。
所有组最大 60 预算轮，同一验证频率和 patience 10；早停的实际步数另行记录。
混合题每轮约重复曝光 1.892 次，这是固定更新预算下重分配训练样本的实验。

仅训练集拟合归一化；仅训练集用于反向传播。验证集用于学习率、检查点和最终模型选择。
测试集没有参与选择或评分。官方数据完整性检查允许检查所有 split 的哈希与对齐。

## Files

- `protocol.json`：训练前固定的实验协议。
- `losses.py`：损失与对齐采样器。
- `check_objectives.py`：手算、梯度、极端输入、多正确标签、采样检查。
- `run_suite.py`：48 次训练（8 组 × 两个搜索运行 + 四个额外种子），新进程官方验证。
- `support.py`：从上一轮复制的数据标签、官方评测、完整性等辅助函数。
- `models.py` / `features.py` / `runtime.py` / `method.py`：逐字复制上一轮，确保结构和推理接口一致。
- `summarize.py`：汇总、配对 bootstrap、目标与结构的交互分析、中文报告及固定种子模型导出。
- `verify_delivery.py`：复制后模型重跑官方验证、重排子集和新 ID 推理、原文件及训练源码哈希检查。
- `results/REPORT.md`：结果、结论、限制、成本。
- `results/selected_bundle/`：按五种子平均选择组合、固定 seed 17 的单模型。

## Run

Python 3.12，依赖见 `requirements.txt`；本机使用 `D:/anaconda/envs/mp1/python.exe`。
以下命令从 `code` 目录执行。已完成的训练在源码、协议、配置及资产哈希相同时可以恢复；
修改实验后请使用新的输出路径。源代码禁用 Python 字节码写入，环境变量需在解释器启动前设置。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B objective_experiments_20261017/check_objectives.py
python -B objective_experiments_20261017/run_suite.py --output objective_experiments_20261017/reproduction
python -B objective_experiments_20261017/summarize.py --output objective_experiments_20261017/reproduction
python -B objective_experiments_20261017/verify_delivery.py --output objective_experiments_20261017/reproduction
```

使用本轮已选检查点重新预测和评分（下面只评验证集）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B predict.py --method objective_experiments_20261017/method.py --checkpoint objective_experiments_20261017/results/selected_bundle --split validation --output objective_experiments_20261017/replayed-validation.json
python -B score_predictions.py --split validation --predictions objective_experiments_20261017/replayed-validation.json --output objective_experiments_20261017/replayed-score.json
```

模型预测只读取调用方输入和 bundle，不读取标签，不依赖 ID，也不缓存题目选择。
所有模型独立提交时均需计入自身完整资产；本实验没有集成。
训练与评分串行运行，PyTorch intra-op 4、inter-op 1，BLAS 1。

结果为开发集证据。seed 17 参与学习率搜索，另报其余四种子的均值；这些种子仍在同一验证集早停。
题目 bootstrap 区间是有条件的探索性统计，并未计入调参或多重比较影响。
排名损失的输出不等于校准的单候选正确概率；不同目标的损失值不能直接比较。

代码由 Codex AI 辅助编写和运行；模型从随机初始化训练，没有引入外部数据或预训练权重。
