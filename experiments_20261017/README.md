# 四候选实验实现

本目录是本次用户授权新增的独立实验目录。原有脚本、评分器、解析器、数据和配置均不修改；
`submission.py` 保持原样，官方 runner 通过 `--method experiments_20261017/method.py` 调用这里的算法。

## 文件

| 文件 | 用途 |
|---|---|
| `protocol.json` | 训练前固定的模型、种子、搜索范围、早停与选择规则 |
| `models.py` | 线性、独立 MLP、平均融合、自注意力、容量对照、自身注意力及题目分支 |
| `features.py` | 仅在训练集拟合的向量标准化和题目 TF-IDF |
| `run_suite.py` | 训练、验证选 checkpoint、独立进程官方预测/评分、资源记录 |
| `method.py` | 离线 `predict(bank, arrays, checkpoint)`，只读取模型资产 |
| `summarize.py` | 汇总官方结果、配对比较、分组诊断与中文报告 |
| `check_models.py` | 梯度、候选排列等变性、跨候选依赖和题目非线性融合检查 |
| `runtime.py` | 控制 CPU 线程与确定性设置 |
| `protected_before.json` | 开始前全部现有文件的 SHA256 |
| `results/REPORT.md` | 本次实际结果与限制 |

## 环境与复现

实际使用现有 `D:/anaconda/envs/mp1/python.exe`（Python 3.12.14、NumPy 2.5.3、PyTorch 2.7.1 CPU）。
没有修改该环境或下载预训练权重。新环境需要 Python 3.12 和本目录 `requirements.txt` 中的依赖。

从仓库的 `code/` 目录执行，以下 `python` 应指向上述环境或已准备好的等价环境：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
python -B experiments_20261017/check_models.py
python -B experiments_20261017/run_suite.py --output experiments_20261017/reproduction
```

全套包含每组两个学习率的 seed 17 搜索，然后以选定学习率完成其余四种子，共 48 次训练。
已完成且配置相同的 run 会跳过；更改代码或协议后应使用新的输出目录，避免混合版本。
训练程序仅加载训练/验证参考标签；官方 `check_data.py` 会读取测试文件做完整性与对齐检查，
但训练、模型选择和结果汇总不使用测试标签或测试分数。

不重新训练即可复现已选模型的验证分数：

```powershell
python -B predict.py --method experiments_20261017/method.py --checkpoint experiments_20261017/results/selected_bundle --split validation --output experiments_20261017/reproduced-validation.json
python -B score_predictions.py --split validation --predictions experiments_20261017/reproduced-validation.json --output experiments_20261017/reproduced-score.json
```

也可以将 `--checkpoint` 换成某次运行的 `results/runs/<model>/lr_<value>/seed_<seed>/bundle`。
每个 bundle 都包含未压缩 NPZ 权重、训练统计量及所需配置；运行时不依赖训练脚本或保存的预测。
方法接受任意题数的传入 bank/arrays，没有样本数、题目 ID 或测试集的硬编码。

## 评价解释

- 所有完整验证分数由原样的官方 scorer 产生；逐轮 checkpoint 选择使用其 `common.score_choices` 实现。
- 报告中的随机期望是从正确标签比例计算的数学期望，分组/配对结果是对官方预测的诊断聚合。
- 5 种子标准差反映训练随机性；验证集同时用于学习率选择和早停，不能把它当独立测试集。
- 运行时间为新预测进程加独立评分进程的外部墙钟时间。当前仅测验证集与本机 CPU，未认证官方参考 CPU 上的限制。
- 注意力仅自身消融与全注意力有相同存储参数，但前者 Q/K 匹配退化；容量对照同时保留。
- 题目模型增加参数且只采用浅层文本表示，不能把其结果直接解释为模型理解或不理解题意。
- 原始数据和官方工具来自课程包；新增代码由 Codex 辅助设计、编写和执行，需要在最终提交中披露。

## 设计背景

- 结果监督验证：<https://arxiv.org/abs/2110.14168>
- 集合上下文排序：<https://arxiv.org/abs/1912.05891>
- 注意力模块：<https://arxiv.org/abs/1706.03762>

这里使用 PyTorch 的基础模块重新编写轻量模型，没有复用上述论文的训练数据或预训练参数。
