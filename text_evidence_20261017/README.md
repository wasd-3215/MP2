# 输出结构与保守算术证据

用户授权的新独立实验。原S2-C主模型冻结，训练OOF上分别拟合输出结构、算术、两者合并的候选分数残差；与原S2-C共四组。新增参数最多7个，主模型训练0次。协议见protocol.json。

全部原实验、数据、解析器和评分器只读。最初训练审计发现百分号、外置单位和Markdown结尾误报，在训练阶段修正，原审计保留audit_history/v0。最终训练审计人工复核输入单独保存。未知表达式不当错，不执行候选代码，也不以过程错误硬淘汰候选。

Python3.12，先设置PYTHONDONTWRITEBYTECODE=1与PYTHONIOENCODING=utf-8，全部命令带-B。依次执行check_extractor.py、audit_train.py、复核训练审计卡并保存固定人工审计、fit.py、evaluate.py、analyze.py、verify.py。数值任务串行，Torch4/interop1、OpenBLAS1。

结果见results/REPORT.md、summary.json、answer_changes.csv。训练覆盖和示例见training_audit.json、training_audit_samples.jsonl、manual_training_audit.json。拟合内训练表现不是校正器无偏OOF评价。验证前哈希见frozen_before_validation.json。

portable/仅包括五个共享S2-C成员、四组配置及推理源码。使用官方predict.py，--method指向portable/inference/method.py，--checkpoint指向portable/pipelines/下具体组。资产没有OOF、预测缓存或参考答案，可接受新输入。四组均做完整新进程官方预测评分和便携重放。

最后运行analysis/interpretation.py，复现冻结评分后的得失解释、分数贡献拆解及总阶段成本。该脚本不再训练或评价新预测器；其哈希单独保存在verification.json，不修改验证前冻结的顶层源码。cost_accounting.json包含全部三版训练提取审计成本。

原文件保护快照protected_before.json保持开始时状态。结果verification.json保留任何差异；不会更新旧快照使检查通过。未读取测试集语义内容。
