# 显式共识与候选交互：3×2对照

I/Pool/S2三种结构×解析P/解析及一致性C两种输入，六组各五个种子。全部固定20轮，最后检查点；原始与统一最终有效性过滤分别评测。科学主要比较为过滤后的 `(S2_C-I_C)-(S2_P-I_P)`，不是挑选最高分方案。

## 运行

从仓库根目录执行；当前解释器 `D:/anaconda/envs/mp1/python.exe`。依赖Python 3.12、NumPy 2.5.3、PyTorch 2.7.1+cpu。绘图另使用本机含matplotlib的解释器，不影响训练与选择。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B code/interaction_consensus_20261017/check_models.py
python -B code/interaction_consensus_20261017/train.py
python -B code/interaction_consensus_20261017/evaluate_suite.py
python -B code/interaction_consensus_20261017/analysis/summarize.py
python -B code/interaction_consensus_20261017/analysis/plot_results.py
python -B code/interaction_consensus_20261017/analysis/verify_delivery.py
```

运行时禁止字节码写入。runtime限制Torch intra-op4/interop1、OpenBLAS1；串行训练和评估，不同时启动模型作业。30次训练、72次官方新进程预测及独立评分，另有六次部署包复现。没有额外OOF、阈值或学习率搜索。

缓存只在代码、协议与权重哈希一致时复用。完整重跑应复制到新工作区或使用新目录，保留原研究记录；protected_before.json记录原工作区开始实验前的所有原文件哈希，迁移后应另外记录文件保护基准。

## 预测

从code目录运行以下命令，可复现一个研究预测器：

```powershell
python -B predict.py --method interaction_consensus_20261017/method.py --checkpoint interaction_consensus_20261017/results/pipelines/S2_C_ensemble5_filtered --split validation --output interaction_consensus_20261017/reproduced-validation.json
python -B score_predictions.py --split validation --predictions interaction_consensus_20261017/reproduced-validation.json --output interaction_consensus_20261017/reproduced-score.json
```

六个 `portable/<配置>_ensemble5_filtered` 是各自独立完整的推理权重包，配合 `portable/inference/method.py` 使用，不依赖历史实验目录。继续使用官方code目录与输入；不需要重训练或研究预测缓存。每个方案独立计算全部成员资产，不能累加互斥的六个部署方案。支持同样的audit入口和新题数/ID。

## 主要产物

`results/REPORT.md`中文报告；`primary_result.json`预设主要交互项；`summary.csv`六组均值及集成；`paired_comparisons.json`完整主次比较；`subgroups.csv`子组；`repairs.csv`纠正与新增错误；`PC_choice_changes.json`区分换候选与换答案；`learning_curves.csv`完整逐轮曲线；`official_results.json`72份官方分数和fresh计时；`cost.json`成本；`verification.json`独立包、新ID和原文件保护复核。

P/C均为六个特征槽位，P在标准化后屏蔽五个共识槽位；特征权重零初始化。I-C仍通过显式特征使用同题信息，只在embedding处理上独立。原始参考模块来自项目前期实验，新实现与报告由Codex AI辅助。未修改官方数据、解析器或评分器，未使用外部数据、预训练权重或API；测试集仅被文件哈希保护扫描按字节核对，不用于训练、选择、预测或评分。
