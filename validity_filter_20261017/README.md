# 最终选择前排除解析无效候选

复用 `fusion_confidence_20261017` 的固定20轮权重，执行18个基础预测器×过滤前后，以及冻结回退×过滤前后，共38次官方预测与评分。无需重新训练或阈值搜索。科学对照完整保留，不重新选择优胜模型。

## 运行

从仓库根目录执行。当前实验解释器是 `D:/anaconda/envs/mp1/python.exe`，Python 3.12、NumPy 2.5.3、PyTorch 2.7.1+cpu。设置禁止字节码写入；runtime自动设置Torch4/interop1、OpenBLAS1。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B code/validity_filter_20261017/check_implementation.py
python -B code/validity_filter_20261017/run_suite.py
python -B code/validity_filter_20261017/analysis/summarize.py
python -B code/validity_filter_20261017/analysis/verify_delivery.py
```

已有计时结果会复用；如需重新测量，在复制的完整仓库中使用新实验输出，不覆盖原研究记录。`protected_before.json`是本机开始实验前的原文件快照，保护核验面向原工作区；迁移后应另外记录新工作区的基准。

## 独立加载过滤方案

`portable/inference/`包含本轮推理源码；`portable/<方案>/`包含独立完整的五成员权重、预处理和配置，不依赖历史实验目录。保留官方code目录、NumPy、PyTorch及输入数据即可使用。不要把研究用保存预测当作部署输入。

从code目录执行：

```powershell
python -B predict.py --method validity_filter_20261017/portable/inference/method.py --checkpoint validity_filter_20261017/portable/s2_ensemble5_filtered --split validation --output validity_filter_20261017/reproduced-validation.json
python -B score_predictions.py --split validation --predictions validity_filter_20261017/reproduced-validation.json --output validity_filter_20261017/reproduced-score.json
```

其它独立bundle：`s2_parse_ensemble5_filtered`、`s2_consistency_ensemble5_filtered`、`s2_ensemble5_route_filtered`。全部方案继续支持官方audit入口；过滤只用输入候选的官方解析结果。官方规则下无法解析不等于人工无法理解，也不代表推理过程必定错误。

## 记录

`results/REPORT.md`为中文报告；`official_results.json`为38个官方结果及完整fresh计时；`filter_pairs.csv`为19对；`five_seed_summary.json`为单模型统计；`subgroups.csv`为子组；`routing_attribution.json`为回退归因；`changed_questions.json`为逐题替换；`verification.json`为复现与文件保护记录。

`reference_method.py`和其它参考模块原样复制此前项目实现，`method.py`只增加最终有效性约束。代码与实验由Codex AI辅助完成，所有模型来自本项目训练集，未引入外部数据、预训练权重或API。本轮不读取或评测测试集。
