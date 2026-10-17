# 困难训练子组重加权：2×3实验

研究问题：提高错误共识题的训练关注，能否减少一致性特征对少数正确答案的压制，同时保留整体收益？

## 固定设计

- P：S2＋解析有效性，C：S2＋解析与答案一致性；沿用 `fusion_confidence_20261017` 的模型、特征与推理源码，不改原目录。
- U：均匀权重。H：错误共识题相对其它混合题2倍，混合题总权重归一回原题数，全对题权重1。
- K：在每一种正确候选数量内均匀赋权，并严格匹配H在该数量下的总权重；因此K/H的候选正负标签总权重、全对/混合题总权重、单正确/两正确/三正确题总权重均匹配，仅单正确题内部的分配不同。
- 权重只在训练集拟合。训练4755题；全对2242题，混合2513题，其中错误共识367题，其它单正确344题。正确共识/错误共识均指官方解析后**唯一最高票**，不要求超过半数。
- 6配置×5种子17/29/43/71/101，共30次。每题每轮一次，20轮，LR0.0003，AdamW decay0.001，batch256，dropout0.1，clip1。
- 损失为每题四候选BCE均值乘全局问题权重，再在batch取均值；不按batch权重和重新归一化。每种方案的全局平均权重均为1。
- 六组均重新训练，统一损失归约与训练预算。同种子相同样本顺序，模型构建后重置dropout随机数；新增特征列初始为零。
- 无早停、不选择验证最佳轮次、不搜索权重倍数。正式验证不参与权重估计。推理统一argmax，不新增有效性过滤、校准或回退。
- 主要比较：C-H减C-K，在错误共识60题上的五种子配对差值。次要比较及交互固定于 `protocol.json`。
- 实用目标：C-H相对C-U，错误共识提高至少10个百分点，整体损失不超过0.5个百分点。它是点估计工程目标，不是显著性标准。
- 每组另有一个固定五成员等权sigmoid分数集成，单独报告；不是五次独立集成。加权BCE分数不解释为校准后的正确概率。
- 新目录交付选择按预设整体指标选择，不修改任何历史选中模型。科学主比较与提交式整体最高分选择分开记录。

## 重现

项目根目录PowerShell：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& 'D:/anaconda/envs/mp1/python.exe' -B code/reweighting_experiments_20261017/check_implementation.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/reweighting_experiments_20261017/train.py
& 'D:/anaconda/envs/mp1/python.exe' -B code/reweighting_experiments_20261017/finish.py
```

训练支持检查源码与资产哈希后的恢复。研究权重及训练标签仅用于训练/分析，不放进推理bundle。预测只需本目录的 `runtime.py`、`reference_models.py`、`scorers.py`、`explicit_features.py`、`decisions.py`、`method.py` 和选定bundle，配合官方代码。

```powershell
& 'D:/anaconda/envs/mp1/python.exe' -B code/predict.py --method code/reweighting_experiments_20261017/method.py --checkpoint code/reweighting_experiments_20261017/results/selected_bundle --split validation --output code/reweighting_experiments_20261017/results/reproduced-validation.json
& 'D:/anaconda/envs/mp1/python.exe' -B code/score_predictions.py --split validation --predictions code/reweighting_experiments_20261017/results/reproduced-validation.json
```

结果见 `results/REPORT.md`、`group_summary.csv`、`subgroup_summary.csv`、`paired_comparisons.json`、`interactions.json`、`ensemble_summary.json`、`practical_target.json`、`resources.json`、`verification.json`。`learning_curves.csv`记录每轮清洁未加权训练/验证BCE，避免把不同加权训练目标的数值直接比较。

## 边界

官方主准确率均有原版评分器输出。子组、训练正确率和配对修复/破坏计数为诊断，不是新增官方基准分数。bootstrap先对五种子正确性平均再按题目抽样，不将种子×题目当独立观测；区间条件于本轮与历史开发，次要比较未校正多重检验。

验证集已被反复用于开发，预设本轮主要比较不能使它重新变成未见测试。未用测试输入/参考答案训练、调参、预测或评分，原文件保护只按字节核验哈希。最高整体分与困难子组改善可能冲突，应同时报告。

所有训练CPU顺序执行，Torch最多4数值线程、interop1、OpenBLAS1。冷启动本机验证计时包含进程、加载、预处理、全部成员与独立评分，不是官方参考CPU认证。

由AI协助设计、编码和运行，沿用项目内模型与特征实现；全部原有文件保持只读并在结束时复核哈希。
