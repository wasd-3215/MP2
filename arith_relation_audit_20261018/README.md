# 算术代理消融与训练集关系审计

用户授权按既定设计顺序执行的新独立实验。结果入口为`results/REPORT.md`。

算术：B740、N739、V742、NV741、历史H741，分母798，均为原版官方评分。V修复2题、损害0题；配对描述区间接触0，不认定为独立确认的泛化提升。只新增3次小残差拟合，未重训主模型或替换默认入口。

关系：开发48题；冻结检查86道去重正例加20道负对照，共106题、424候选最终上下文，审阅107条提取判断。字面正确与语义绑定分别标注；先冻结盲审，再读取训练参考答案。四类当前探针均不满足预设建模门槛，故没有训练关系选择器。审阅者为Codex，未声称独立人类复核或逐步验证所有完整候选。

## 运行环境

Python3.12，依赖见requirements.txt。实际解释器`D:/anaconda/envs/mp1/python.exe`，NumPy2.5.3、PyTorch2.7.1+cpu；现有环境已安装，无需下载。若安装CPU版Torch，使用PyTorch官方CPU wheel索引选择匹配版本。离线推理不访问网络。

从`code/`目录执行，先设置：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
```

## 使用已经冻结的V便携包

使用新的输出位置，避免覆盖现有记录：

```powershell
python -B predict.py --method arith_relation_audit_20261018/portable/inference/method.py --checkpoint arith_relation_audit_20261018/portable/pipelines/V --split validation --output arith_relation_audit_20261018/user_replay/validation-predictions.json
python -B score_predictions.py --split validation --predictions arith_relation_audit_20261018/user_replay/validation-predictions.json --output arith_relation_audit_20261018/user_replay/validation-score.json
```

`portable/`包含推理代码、共享五成员及B/N/V/NV/H配置。没有训练OOF、参考答案、审计标签或缓存预测。新银行按官方`--split audit --data-dir ...`接口传入，模型只读取当前bank/arrays和checkpoint。

## 全部复现

```powershell
python -B arith_relation_audit_20261018/reproduce.py --output arith_relation_audit_reproduction_001
```

输出必须为`code/`下尚不存在的同级目录。依次执行算术拟合/官方评价/分析、训练拆分与开发提取、冻结关系探针、全训练提取、回放固定阅读标注、揭示训练标签统计、验证和报告。复用本仓库已有S2-C权重、训练OOF及原算术特征；这些来源会核对哈希。不重新训练基础网络，不打开测试语义或标签。

`development_review.json`及`blind_review_input.json`是本次真实阅读的固定人工输入。复现是回放这些输入，并不自动独立重做阅读。原样记录冻结检查中的错误；不得在同一确认样本上修规则后重新报通过。

## 关键证据

- `protocol.json`：拟合前冻结的研究协议。
- `protected_before.json`、`results/protected_verification.json`：11,906个旧文件的前后核验。
- `arithmetic/results/fit_results.json`、`frozen_before_validation.json`：3次拟合与验证前冻结哈希。
- `arithmetic/results/pipelines/*/score.json`：5份原版官方评分，另有5次便携回放。
- `arithmetic/results/answer_changes.jsonl`：所有数值变化、原文、公式及分数贡献。
- `results/arithmetic_interpretation.json`：V的两修复与一次错换错、N的得失解释。
- `relation/results/split_and_sampling.json`：输入确定的拆分、抽样顺序及筛选分母。
- `relation/results/development_review.json`、`audit_history/v0/`：开发阅读、修订和边界。
- `relation/results/frozen_before_confirmation.json`：确认阅读之前冻结的代码。
- `relation/results/confirmation_cards.jsonl`：106题所有候选、提取结果及源位置。
- `relation/results/confirmation_review_blind.json`、`blind_review_frozen_before_labels.json`：标签揭示前固定的107条判断。
- `relation/results/relation_summary.json`：预设门槛、训练标签关联及逐族停止结论。
- `results/verification.json`、`portable_package.json`、`environment.json`、`cost_accounting.json`：交付、资源及完整性证据。

全部设计、实现与审阅由Codex AI辅助。验证集已被反复开发，区间仅作描述。当地CPU测时不替代官方Xeon完整评测认证。
