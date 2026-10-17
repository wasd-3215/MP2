# 分歧题训练与早期共识条件化

八组：All/D训练范围×P/C特征×Late/Early位置；五种子，共40次训练。每次固定380次更新、95100次题目抽样，最后检查点。48次独立官方预测评分，另有8个独立集成包复现。全部实验记录保留，没有根据验证结果另挑提交模型。

All使用4755道训练题；D由输入候选及官方解析器筛出2114题，不使用参考答案决定入池。所有组统一在只有一个有效答案组时直接选择第一个有效候选，其余才调用模型并排除无效选择。验证798题中443题旁路、355题由模型判断。

## 复现

从仓库根目录执行。当前训练解释器为D:/anaconda/envs/mp1/python.exe；依赖Python3.12、NumPy2.5.3、PyTorch2.7.1+cpu。图表可使用本机安装matplotlib的D:/anaconda/python.exe。runtime设置Torch4/interop1、OpenBLAS1；所有模型作业串行。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B code/early_consensus_20261017/check_models.py
python -B code/early_consensus_20261017/train.py
python -B code/early_consensus_20261017/evaluate_suite.py
python -B code/early_consensus_20261017/analysis/summarize.py
python -B code/early_consensus_20261017/analysis/plot_results.py
python -B code/early_consensus_20261017/analysis/verify_delivery.py
```

缓存只在代码、协议和权重哈希一致时复用；完整重跑应使用另一个完整工作区，保留本实验记录。protected_before.json是原工作区开始前的原文件快照；迁移后请另建文件保护基准。

本机最后严格核验发现唯一原文件差异为 `code/.git/FETCH_HEAD`，其它8573个原文件哈希一致；本实验未执行Git操作，变化来源未确定。保留严格核验失败，不覆盖Git文件或更新基准。模型、新输入和独立包检查完成后，额外运行 `python -B code/early_consensus_20261017/analysis/finalize_verification.py` 分项核对并披露；`verification.json` 的 `experiment_checks_passed=true`，但严格 `passed=false`。该脚本不修改训练/推理/统计定义，也不重新运行官方评分。

## 独立预测

从code目录执行：

```powershell
python -B predict.py --method early_consensus_20261017/method.py --checkpoint early_consensus_20261017/results/pipelines/D_C_Early_ensemble5 --split validation --output early_consensus_20261017/reproduced-validation.json
python -B score_predictions.py --split validation --predictions early_consensus_20261017/reproduced-validation.json --output early_consensus_20261017/reproduced-score.json
```

portable/inference提供推理源码；portable/<单元名>_ensemble5各自包含独立完整的五成员权重和预处理，不依赖其它实验目录或预测缓存。配合官方code目录、环境及输入即可离线预测，支持audit和任意新题数/ID。

Early在编码器后、FFN及注意力前加入6→128投影特征；Late在交互归一化后加入相同投影。投影零初始化、无偏置；P屏蔽五个共识维度。所有八组使用相同标准化和参数量。位置对照有意采用匹配加法结构，不能用历史拼接头分数作为本轮Late分母。

## 记录

results/REPORT.md为中文报告；summary.csv为八组；primary_results.json为预设比较；paired_comparisons.json为全部简单效应与交互；repairs.csv为纠正/改错；learning_curves.csv为初始和20个预算轮；各run的draw-counts.npy为训练逐题曝光；official_results.json为48份官方结果与fresh资源计时；verification.json为独立包、新输入与原文件保护记录。

相同更新预算下D会增加困难题重复曝光，结论是训练分配方案的整体效果。早期注入也改变特征影响FFN与归一化的路径，不能唯一归因于注意力权重。验证集已有大量开发使用，区间为条件性描述证据。

由Codex AI辅助设计、实现和运行。仅复用项目内从头训练的模型组件，未引入外部数据、预训练权重、API或新候选。官方数据、parser和scorer保持不变；测试文件只按字节进行哈希保护，不用于训练、选择、预测或评分。
