# 实验报告配套复现说明

报告日期为2026-10-28，主候选是评分前冻结的V。PDF位于`output/pdf/DASE7506_MP2_Report.pdf`，可编辑正文为`report.md`。报告中的实验编号见末页参考记录，来源与V便携包文件哈希见`source_manifest.json`；该清单不取代原测试冻结记录。

## 环境与官方重放

从`code/`目录运行，使用Python 3.12，历史实际版本为3.12.14。依赖为NumPy 2.5.3和PyTorch 2.7.1+cpu；在新环境中安装匹配Windows/Python的CPU版PyTorch。网络仅用于环境安装，正式推理离线。

```powershell
python -m pip install numpy==2.5.3
python -m pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
python -B check_data.py
$v='arith_relation_audit_20261018/portable'
python -B predict.py --method "$v/inference/method.py" --checkpoint "$v/pipelines/V" --split test --output report_replay_001/test-predictions.json
python -B score_predictions.py --split test --predictions report_replay_001/test-predictions.json --output report_replay_001/test-score.json
```

复现输出目录应尚不存在，避免覆盖保存的记录。预期正确数374/467，官方accuracy为80.08565310492506。只生成预测的运行器不读取参考答案；评分器独立读取参考。官方测试原结果位于`final_test_20261028/results/V/score.json`。本次写报告没有重新运行以上训练、预测或评分命令。

验证复现将`--split test`改为`--split validation`，同时使用新的validation输出文件；预期742/798。新的audit输入按原接口传入`--split audit --data-dir <输入目录>`，不能使用缓存预测或按题目ID查表。

## 训练过程及来源链

不重训的评分复现只需官方接口、数据与冻结便携包。训练复现涉及以下来源链；不同目录脚本存在对历史输出路径的引用，完整重训应在独立副本中执行，不能直接覆盖原始证据目录。

1. `fusion_confidence_20261017/train.py`从随机初始化训练S2等全量模型，并生成按题五折OOF。每折只使用其余四折拟合归一化，固定20轮。配置及原训练源码哈希见该目录`protocol.json`及results来源文件。
2. `interaction_consensus_20261017`提供后续使用的匹配P/C全量专家；S2-C与历史OOF复用关系由`method_complementarity_20261017`逐折重放核对。具体顺序和命令以各目录README为准。
3. `text_evidence_20261017/check_extractor.py`、`audit_train.py`、训练审计输入、`fit.py`、`evaluate.py`、`analyze.py`、`verify.py`依序建立格式/算术校正；`analysis/interpretation.py`只拆解冻结结果。人工审阅输入需要保留，不能伪称程序自动重做语义判断。
4. `arith_relation_audit_20261018/reproduce.py --output arith_relation_audit_reproduction_001`在新同级目录复现B/N/V/NV/H算术评价与关系审计；它复用并校验既有主模型、OOF及算术特征，**不等于从零重新训练基础网络**。

所有训练步骤只用训练输入和训练标签。验证已反复用于开发；测试仅用于冻结后的最终评价。报告中的577次神经拟合累计包含重跑基线与折内运行，额外六次线性文本残差、阈值与门控拟合不计入该神经网络次数。

## 资产与版本

V的正式入口为`arith_relation_audit_20261018/portable/inference/method.py`，checkpoint为同portable目录的`pipelines/V`。便携目录包含共享五成员及其它比较配置；资产计数引用的是V实际依赖的5.24880 MiB，不应把全部研究目录视为单套资产。部署包不应包含OOF、参考答案、审计标注或预测缓存。

最终六方案完整命令、资产计数、源代码和权重冻结哈希见`final_test_20261028/frozen_before_test.json`。最新实验在原Git历史中未全部提交，所以报告以具体文件哈希识别版本。本报告没有创建提交、公开仓库或作业提交记录。

## 报告验证

报告构建只读取既有报告、配置与评分，不读取测试逐题错因。`verification.json`核对六套官方测试数值、页数及既有文件完整性。`qa/`保存排版检查材料，不属于课程提交正文。Markdown中的分页标记仅用于固定PDF排版。
