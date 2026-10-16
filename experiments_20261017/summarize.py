"""Generate tables and a Chinese experiment report from official validation artifacts."""
import runtime

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DATA, load_bank, parse_answer, read_jsonl

NAMES = {"linear": "L：线性", "independent": "I：独立 MLP", "pool": "P：平均上下文",
         "attention": "A：候选注意力", "independent_large": "I-large：容量对照",
         "attention_self": "A-self：仅关注自身", "independent_question": "I-Q：独立＋题目",
         "attention_question": "A-Q：注意力＋题目"}
PAIRS = [("attention", "independent"), ("pool", "independent"), ("attention", "pool"),
         ("attention", "independent_large"), ("attention", "attention_self"),
         ("independent_question", "independent"), ("attention_question", "attention"),
         ("attention_question", "independent_question")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(HERE / "protocol.json")
    runs = read(output / "selected_runs.json")
    all_runs = read(output / "all_runs.json")
    seeds = protocol["seeds"]
    models = protocol["models"]
    bank = load_bank("validation")
    labels = np.load(output / "validation-labels.npy", allow_pickle=False)
    ids = [row["id"] for row in bank]
    # Reconstruct labels independently through the official numeric parser.
    refs = read_jsonl(DATA / "gsm8k_validation.jsonl")
    reconstructed = []
    for row, ref in zip(bank, refs):
        assert row["id"] == ref["id"]
        gold = parse_answer(ref["answer"])
        reconstructed.append([int((a := parse_answer(c["text"])) is not None and a == gold)
                              for c in row["candidates"]])
    assert np.array_equal(labels, reconstructed)
    correctness, choices, summary = {}, {}, []
    for name in models:
        group = sorted([r for r in runs if r["model"] == name], key=lambda r: seeds.index(r["seed"]))
        assert [r["seed"] for r in group] == seeds
        matrices, selected_choices = [], []
        for r in group:
            folder = HERE / r["run_dir"]
            predictions = read(folder / "validation-predictions.json")
            official = read(folder / "validation-score.json")
            assert predictions["ids"] == ids and predictions["split"] == "validation"
            c = np.asarray(predictions["selected"], dtype=np.int64)
            v = labels[np.arange(len(bank)), c]
            assert int(v.sum()) == official["correct"] == r["correct"]
            assert official["accuracy"] == r["accuracy"]
            matrices.append(v)
            selected_choices.append(c)
        correctness[name] = np.asarray(matrices)
        choices[name] = selected_choices
        acc = np.asarray([r["accuracy"] for r in group])
        summary.append({"model": name, "learning_rate": group[0]["learning_rate"],
                        "parameters": group[0]["parameters"], "mean_accuracy": float(acc.mean()),
                        "std_accuracy": float(acc.std(ddof=1)),
                        "non_search_seeds_mean_accuracy": float(acc[1:].mean()),
                        "seed17_accuracy": float(acc[0]),
                        "min_accuracy": float(acc.min()), "max_accuracy": float(acc.max()),
                        "median_best_epoch": float(np.median([r["best_epoch"] for r in group])),
                        "mean_train_seconds": float(np.mean([r["training_seconds"] for r in group])),
                        "mean_validation_wall_seconds": float(np.mean([r["validation_total_wall_seconds"] for r in group])),
                        "uncompressed_asset_mib": group[0]["uncompressed_asset_bytes"] / 1024**2})
    csv_write(output / "summary.csv", summary)
    write(output / "summary.json", summary)
    flat_keys = ["model", "seed", "learning_rate", "accuracy", "correct", "best_epoch", "epochs_run",
                 "parameters", "training_seconds", "validation_total_wall_seconds", "uncompressed_asset_bytes", "run_dir"]
    csv_write(output / "runs.csv", [{key: r[key] for key in flat_keys} for r in all_runs])
    rng = np.random.default_rng(20261017)
    bootstrap = rng.integers(0, len(bank), size=(5000, len(bank)))
    paired = []
    for a, b in PAIRS:
        difference = correctness[a].mean(0) - correctness[b].mean(0)
        distribution = 100 * difference[bootstrap].mean(axis=1)
        av, bv = correctness[a][0], correctness[b][0]
        paired.append({"comparison": f"{a} - {b}", "mean_difference_pp": float(100 * difference.mean()),
                       "question_bootstrap_ci95_low": float(np.quantile(distribution, .025)),
                       "question_bootstrap_ci95_high": float(np.quantile(distribution, .975)),
                       "non_search_seeds_difference_pp": float(100 * (correctness[a][1:].mean() - correctness[b][1:].mean())),
                       "seed17_fixed_errors": int(((av == 1) & (bv == 0)).sum()),
                       "seed17_new_errors": int(((av == 0) & (bv == 1)).sum())})
    csv_write(output / "paired_comparisons.csv", paired)
    write(output / "paired_comparisons.json", paired)
    masks = {"exactly_one_correct": labels.sum(1) == 1, "multiple_correct": labels.sum(1) > 1}
    majority_correct, majority_wrong, tied, disagreement, parse_failure = [], [], [], [], []
    for row, lab in zip(bank, labels):
        answers = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = Counter(a for a in answers if a is not None)
        leaders = [a for a, count in counts.items() if count == max(counts.values())]
        unique = len(leaders) == 1
        right = unique and any(lab[i] and answers[i] == leaders[0] for i in range(4))
        majority_correct.append(unique and right)
        majority_wrong.append(unique and not right)
        tied.append(not unique)
        disagreement.append(len(counts) >= 2)
        parse_failure.append(any(a is None for a in answers))
    masks.update({"unique_plurality_correct": np.asarray(majority_correct),
                  "unique_plurality_wrong": np.asarray(majority_wrong), "plurality_tie": np.asarray(tied),
                  "at_least_two_valid_answers": np.asarray(disagreement),
                  "any_parse_failure": np.asarray(parse_failure)})
    subgroup_rows = []
    for group_name, mask in masks.items():
        subgroup_rows.append({"subgroup": group_name, "n": int(mask.sum()),
                              **{name: float(100 * correctness[name][:, mask].mean()) if mask.any() else None
                                 for name in models}})
    csv_write(output / "subgroups.csv", subgroup_rows)
    best = max(summary, key=lambda r: (r["mean_accuracy"], -r["parameters"]))
    selected = next(r for r in runs if r["model"] == best["model"] and r["seed"] == 17)
    source_bundle = HERE / selected["run_dir"] / "bundle"
    target_bundle = output / "selected_bundle"
    target_bundle.mkdir(exist_ok=True)
    for file in source_bundle.iterdir():
        shutil.copy2(file, target_bundle / file.name)
    write(output / "selection.json", {"policy": protocol["final_seed_policy"], "model": best["model"],
          "seed": 17, "learning_rate": selected["learning_rate"], "source_run": selected["run_dir"],
          "validation_accuracy": selected["accuracy"], "five_seed_mean_accuracy": best["mean_accuracy"],
          "test_evaluated": False, "bundle_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                      for p in target_bundle.iterdir()}})
    question_rows = []
    for i, row in enumerate(bank):
        question_rows.append({"id": row["id"], "question": row["question"], "labels": labels[i].tolist(),
                              "seed17_choices": {name: int(choices[name][0][i]) for name in models},
                              "seed17_correct": {name: int(correctness[name][0, i]) for name in models}})
    write(output / "per_question_validation.json", question_rows)
    baseline = read(output / "baselines.json")
    env = read(output / "environment.json")
    integrity = read(output / "protected_verification.json")
    total_train = sum(r["training_seconds"] for r in all_runs)
    total_eval = sum(r["validation_total_wall_seconds"] for r in all_runs)
    lines = ["# 四候选答案选择：训练与验证实验记录", "", "日期：2026-10-17（北京时间）。", "",
             "## 范围与结论边界", "",
             "完成八组模型、每组两种学习率的 seed 17 搜索，以及选定学习率下五个种子的训练。"
             "合计 48 次训练，其中 40 个结果进入五种子比较。全部正式验证分数来自未修改的 `score_predictions.py`。",
             "没有测试集预测、测试分数或测试驱动调参。官方完整性检查读取各 split 仅用于哈希和对齐校验。"
             "本报告是开发集结果，不是最终泛化评估或排行榜成绩。", "",
             "## 协议", "",
             "- 训练/验证题数：4755 / 798。标签按官方 parse_answer 构造，允许多个正确候选。",
             "- 每维标准化仅拟合训练候选；所有组使用相同标准化。",
             "- 种子：17、29、43、71、101；学习率候选：0.001、0.0003。",
             "- AdamW，BCEWithLogitsLoss，batch 256，weight decay 0.001，dropout 0.1，梯度裁剪 1.0。",
             "- 最多 60 轮，验证正确数（再按 BCE）连续 10 轮未改善则停止；保存最佳验证 checkpoint。",
             "- 先用 seed 17 选择学习率，再训练其余四个种子；同分按验证 BCE，再按预设顺序。",
             "- 五种子平均分仍含参与搜索的 seed 17，因此同时报告另外四种子的均值；它们也使用验证集早停。",
             "- 所有模型逐候选共享参数；注意力不使用候选位置编码。",
             "- 题目分支使用训练集文档频率前 1024 个词/相邻词组的 TF-IDF，投影到 32 维，"
             "与候选表示通过非线性层融合；无额外预训练权重。",
             "- 题目分支增加容量且仅提供词袋语义，不能将任何差值完全归因于理解题目条件；"
             "若结果不提升，也不能证明题目信息无用。", "",
             "## 网络", "",
             "| 组 | 结构 |", "|---|---|",
             "| L | 1024→1，共享线性读出 |",
             "| I | 1024→128→64→1，GELU；各候选独立 |",
             "| P | 1024→128；自身与四候选平均表示拼接；256→64→1 |",
             "| A | 1024→128；一层四头自注意力、残差/LayerNorm、128→256→128 FFN；128→64→1 |",
             "| I-large | I 加逐候选 128→512→128 残差块，参数量接近 A |",
             "| A-self | 与 A 相同，但注意力仅允许读取自己；Q/K 在单键 softmax 下不提供有效匹配能力 |",
             "| I-Q / A-Q | 对 I / A 加相同题目条件化分支 |", "",
             "## 验证结果", "",
             "| 模型 | 学习率 | 参数量 | 五种子准确率 %（均值±样本标准差） | 另外四种子均值 % | seed 17 % | 资产 MiB | 预测+评分秒（均值） |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {NAMES[r['model']]} | {r['learning_rate']} | {r['parameters']:,} | "
                     f"{r['mean_accuracy']:.3f} ± {r['std_accuracy']:.3f} | {r['non_search_seeds_mean_accuracy']:.3f} | "
                     f"{r['seed17_accuracy']:.3f} | {r['uncompressed_asset_mib']:.3f} | {r['mean_validation_wall_seconds']:.3f} |")
    lines += ["", f"固定选第一个：{baseline['always_first']['accuracy']:.3f}%；"
              f"答案多数投票：{baseline['answer_majority']['accuracy']:.3f}%；"
              f"固定 seed 17 随机选择：{baseline['random_seed17']['accuracy']:.3f}%。",
              f"均匀随机选择的解析期望为 {baseline['uniform_random_expectation_percent']:.3f}%；"
              "这是标签比例推导出的期望，不是一次官方评测分数。多数投票忽略不可解析答案，平票选最早候选；它额外使用答案文本。", "",
              "## 配对比较", "",
              "下表差值单位为百分点。95% 区间由 5000 次按题目配对 bootstrap 得到，"
              "对五种子正确性先求均值，区间条件于这些已选择模型，不包含调参不确定性，亦未做多重比较校正。"
              "不能把它当成独立测试集显著性结论。修正/新增错误对应 seed 17。", "",
              "| 比较 | 五种子平均差 | 题目 bootstrap 95% 区间 | 另外四种子差 | 修正错误 | 新增错误 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['mean_difference_pp']:+.3f} | "
                     f"[{r['question_bootstrap_ci95_low']:+.3f}, {r['question_bootstrap_ci95_high']:+.3f}] | "
                     f"{r['non_search_seeds_difference_pp']:+.3f} | {r['seed17_fixed_errors']} | {r['seed17_new_errors']} |")
    primary = paired[0]
    lines += ["", f"主比较 A−I = {primary['mean_difference_pp']:+.3f} 个百分点。"
              + ("该配对区间包含零，当前验证证据不足以支持稳定提升。" if primary['question_bootstrap_ci95_low'] <= 0 <= primary['question_bootstrap_ci95_high']
                 else "该区间不包含零，但仍需要冻结后的独立测试确认。"),
              f"五种子平均准确率最高的是 {NAMES[best['model']]}（{best['mean_accuracy']:.3f}%）。"
              f"按事先规则保留它的 seed 17 checkpoint（{selected['accuracy']:.3f}%），不挑选五个种子中的最高分，不集成。", "",
              "## 分组结果", "",
              "以下为同一官方解析/正确性规则下的诊断分组聚合，非独立官方 benchmark 分数。"
              "unique_plurality 表示可解析答案有唯一最高频项，不一定超过半数；tie 表示频次平票。",
              "", "| 分组 | 题数 | I % | P % | A % | I-Q % | A-Q % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in subgroup_rows:
        cells = ["—" if r[m] is None else f"{r[m]:.3f}" for m in
                 ("independent", "pool", "attention", "independent_question", "attention_question")]
        lines.append(f"| {r['subgroup']} | {r['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## 资源与检查", "",
              f"- 本机：{env['processor']}；CPU 训练与推理；PyTorch {env['torch']}，NumPy {env['numpy']}，Python {env['python'].split()[0]}。",
              "- PyTorch intra-op 4、inter-op 1；BLAS 1；串行训练/串行启动评估，无并行模型作业。",
              f"- 48 次训练累计 {total_train:.1f} 秒（{total_train/60:.1f} 分钟，含逐轮官方验证函数调用）；"
              f"48 次新进程预测+独立评分累计 {total_eval:.1f} 秒。另有数据检查、预处理、基线、报告耗时。",
              "- 表中计时为完整 798 题验证集的新预测进程加评分进程外部墙钟时间，包含启动、导入和加载。"
              "尚未测完整测试集，也未在官方 Xeon 上运行；不据此声称已通过官方 30 秒资格检查。",
              "- 每个模型是独立可选提交，资产不累加为一个集成；大小统计计入 NPZ 解包后各数组及配置和预处理。",
              "- 每次训练后检查候选排列等变性、不同 batch 大小、自身模型对其他候选的独立性；"
              "从新进程加载 bundle 的预测必须逐项匹配训练选中的 checkpoint。",
              f"- 原有文件 SHA256 复核：{integrity['files_checked']} 个，变更 {len(integrity['changed'])} 个。",
              "- test/audit 方法仅使用传入 bank/arrays 和 bundle，不读取任何参考答案，不按 ID 查表。", "",
              "## 复现", "", "在 code 目录执行（PowerShell）：", "", "```powershell",
              "$env:PYTHONDONTWRITEBYTECODE='1'", "$env:OMP_NUM_THREADS='4'", "$env:MKL_NUM_THREADS='4'", "$env:OPENBLAS_NUM_THREADS='1'",
              "python -B experiments_20261017/check_models.py",
              "python -B experiments_20261017/run_suite.py --output experiments_20261017/reproduction",
              "python -B predict.py --method experiments_20261017/method.py --checkpoint experiments_20261017/results/selected_bundle --split validation --output experiments_20261017/reproduced-validation.json",
              "python -B score_predictions.py --split validation --predictions experiments_20261017/reproduced-validation.json --output experiments_20261017/reproduced-score.json",
              "```", "", "当前实际解释器为 `D:/anaconda/envs/mp1/python.exe`；新环境使用 Python 3.12 并安装实验目录的 requirements.txt。"
              "全套复现使用新输出目录；已有完成记录会按配置检查后跳过。", "",
              "## 产物与来源", "",
              "- `summary.csv`：八组五种子汇总；`runs.csv`：全部 48 次训练及搜索成本。",
              "- `runs/<model>/<lr>/<seed>/`：逐轮 history、官方 predictions/score、检查记录、checkpoint、资产哈希。",
              "- `paired_comparisons.csv`、`subgroups.csv`、`per_question_validation.json`：配对与错误分析依据。",
              "- `selected_bundle/`、`selection.json`：预设规则选出的单模型和来源。",
              "- `source_hashes.json`、`environment.json`、`protected_verification.json`：代码版本、环境和原文件保护记录。",
              "- 实验设计依据此前讨论；新增训练/推理/汇总代码由 Codex AI 辅助编写与运行，用户需审阅理解。"
              "官方 parser/scorer/data 原样使用；模型使用 PyTorch 基础层，从随机初始化训练，未引入预训练权重或外部训练数据。",
              "- 本报告不替代最终课程报告、独立测试、公开仓库发布和正式提交。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_model": best["model"], "mean_validation_accuracy": best["mean_accuracy"],
                      "selected_checkpoint_validation_accuracy": selected["accuracy"], "runs": len(all_runs)}, indent=2))


if __name__ == "__main__":
    main()
