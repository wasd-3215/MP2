# 集成、显式特征和置信度回退

三个独立实验组及预先规定的组合，完整配置见 `protocol.json`。所有结果使用原版官方验证评分器；未使用测试输入或测试参考答案。

## 协议

- 40 次全训练集拟合：8 配置 × 5 种子。配置为 S2、S0、池化、S2+解析、S2+解析与一致性、S2+解析与风格、S2+全部、仅显式特征。
- 100 次训练集内部拟合：S2/S0/池化/S2+一致性 × 5 种子 × 5 折。按整道题划分，四候选不会跨折。各折仅用折内训练题拟合标准化。
- 均训练20轮，LR=0.0003，BCE，batch256，AdamW decay0.001。不早停、不按验证选择检查点，也不按 OOF 留出折选择检查点。
- 20轮依据之前开发结果预先固定。这轮重置 dropout RNG 以匹配特征模型与无特征模型，因此不能拿历史最佳检查点或历史第20轮结果作本轮配对基线。
- 同种子新特征列初始权重为零，其他共有初始权重一致；新增特征不改变初始 embedding 评分函数。所有分支从头联合训练。
- 集成：S2五种子、S2/S0/池化十五成员、S2+一致性五种子。等权 sigmoid 分数平均；答案投票为对照，票数并列用平均候选分数打破。
- 回退：候选分差、不同答案组分差、模型答案一致率。每种信号分别比较唯一最高票且至少2票/至少3票。阈值仅按训练集OOF正确数选择，禁用回退始终为候选。另报告无置信度筛选的投票控制组。
- 阈值来自预设 OOF 信号分位点；正确数并列优先更少答案改动，再优先较小阈值。单模型一致率通常退化为常量，保留为负对照。
- 答案组用官方 `parse_answer`。组内最高分只定义答案分差，不改变最高分选择；没有隐式加入组大小奖励。
- 额外拟合一维 logistic 校准器，用 OOF 选中候选是否正确作监督，仅评价验证集 Brier/ECE；不参与选择或回退。未经校准的 sigmoid/平均分不解释为正确概率。
- 最终按验证正确数在预定义且符合16MiB限制的方案中选方法；完整单模型只允许固定seed17参与交付选择。其余种子用于报告波动。无条件投票控制不参与选择。

## 重现

在项目根目录的 PowerShell 中：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/train.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/check_implementation.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/analyze.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/verify.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/routing_diagnostics.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/tradeoff_diagnostics.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/fusion_confidence_20261017/report.py
& 'D:/anaconda/python.exe' -B code/fusion_confidence_20261017/plot_results.py
```

训练可按结果文件恢复；恢复前检查协议与训练源码哈希。推理资产的权重、配置和统计量均为普通文件，没有题目答案查表，也不需要原实验目录。只将 `results/selected_bundle` 与本目录推理源码一起交付；不要将研究用的所有折内权重纳入推理资产。

```powershell
& 'D:/anaconda/envs/mp1/python.exe' -B code/predict.py --method code/fusion_confidence_20261017/method.py --checkpoint code/fusion_confidence_20261017/results/selected_bundle --split validation --output code/fusion_confidence_20261017/results/reproduced-validation.json
& 'D:/anaconda/envs/mp1/python.exe' -B code/score_predictions.py --split validation --predictions code/fusion_confidence_20261017/results/reproduced-validation.json
```

## 结果与解释边界

详见 `results/REPORT.md`、`feature_summary.csv`、`pipelines.csv`、`routing_fits.json`、`routing_search.json`、`calibration.json`、`subgroups.csv`、`resources.json` 和 `verification.json`。

所有主准确率都有独立官方评分文件。子组、OOF阈值搜索、修复/破坏计数和互补上限为诊断统计，不是新增官方基准分数。

五个种子组成的是一个五成员集成，不能把五个成员当成五次独立集成。配对区间按题目重采样；特征实验先对种子平均，未将种子×题目当独立样本。所有区间条件于当前训练、配置与验证选择，未校正多重比较。验证集已用于大量既往开发，不能将其当成未见测试集。

本机fresh-process验证计时不代表官方Xeon参考机认证。所有计算在CPU顺序执行；Torch最多4线程、interop1、OpenBLAS1，训练与子进程评分不同时进行数值计算。

源码与原有架构由AI协助编写；S2/S0/池化沿用项目内已有实现。运行前记录全部原有文件哈希，交付时复核，原实验与官方文件不改动。
