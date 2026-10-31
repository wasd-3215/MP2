# Pool-P 与 S2-C 代表方法互补

本目录是用户授权的独立实验，原实验和官方数据、解析器、评分器均只读。全部新增训练及报告写入本目录。

协议在 protocol.json。主对照为8特征线性条件选择对固定S2-C；4特征条件选择是去掉显式票数及答案结构的消融。另比较Pool-P、等权概率平均、训练OOF选择的全局权重。

固定20轮、5种子×5折，补训25个Pool-P折模型。S2-C复用既有OOF，但逐折复核划分、标准化、权重和预测。完整训练集的部署权重沿用 interaction_consensus_20261017，复用不会将参考答案带入推理。

在PowerShell设置 `$env:PYTHONDONTWRITEBYTECODE='1'`，依次用Python3.12和 `-B` 执行：

1. train.py
2. fit_selector.py
3. evaluate.py
4. analyze.py
5. verify.py

环境见results/environment.json。数值工作串行，Torch4线程、interop1、OpenBLAS1。门控参数不在验证集拟合；验证集已反复用于开发，相关区间仅作描述。

结果见results/REPORT.md与results/summary.json，分歧题短索引见results/disagreement_index.csv。results/selector_fit.json中门控训练表现属于拟合内诊断，不是无偏的门控层OOF评价。results/frozen_before_validation.json记录验证前冻结的源码、权重、OOF和选择规则哈希。

portable/只含推理源码、10个共享专家成员与6个流水线配置，不包含训练OOF、候选输入、预测缓存或参考答案。使用官方predict.py，--method指向portable/inference/method.py，--checkpoint指向portable/pipelines/下具体方法，支持新输入。每种方法都以全新预测和独立官方评分运行，并做便携重放。

保护快照protected_before.json记录本组开始时所有既有文件。最终results/verification.json保留任何差异；不会通过更新快照使校验通过。
