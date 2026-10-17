"""Prespecified paired contrasts and subgroup diagnostics after official scoring."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
from support import HERE, CODE, OUT, SEEDS, VARIANTS, ARCHITECTURES, read, write, sha, source_hashes
from collections import Counter
import csv
import json
import statistics
import numpy as np
from common import load_inputs, read_jsonl, parse_answer, DATA, check_data


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    if np.all(values == values[0]):
        return [float(values[0] * 100)] * 2
    samples = []
    for _ in range(50):
        indices = rng.integers(0, len(values), size=(200, len(values)))
        samples.extend(values[indices].mean(axis=1) * 100)
    return np.quantile(samples, [.025, .975]).tolist()


def text_ci(row):
    return f'{row["difference_pp"]:+.3f} [{row["ci95_pp"][0]:+.3f}, {row["ci95_pp"][1]:+.3f}]'


def main():
    write(OUT / 'analysis_sources_before.json', {str(p.relative_to(HERE)): sha(p) for p in (HERE / 'analysis').glob('*.py')})
    assert read(OUT / 'training_sources.json') == source_hashes()
    records = read(OUT / 'official_results.json')
    assert len(records) == 72
    bank, _ = load_inputs('validation')
    check_data(DATA, ['validation'])
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    answers, labels = [], []
    for row, ref in zip(bank, refs):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        gold = parse_answer(ref['answer'])
        assert gold is not None
        a = [parse_answer(c['text']) for c in row['candidates']]
        answers.append(a)
        labels.append([v is not None and v == gold for v in a])
    y = np.asarray(labels, bool)
    valid = np.asarray([[v is not None for v in a] for a in answers], bool)
    assert np.array_equal(y, np.load(OUT / 'validation-labels.npy'))
    assert y.shape == (len(bank), 4) and (y.sum(axis=1) > 0).all()
    wrong, correct, ties, multi = [], [], [], []
    for q, a in enumerate(answers):
        counts = Counter(v for v in a if v is not None)
        maximum = max(counts.values(), default=0)
        winners = [v for v, count in counts.items() if count == maximum]
        unique = len(winners) == 1
        gold = parse_answer(refs[q]['answer'])
        wrong.append(unique and winners[0] != gold)
        correct.append(unique and winners[0] == gold and y[q].sum() < 4)
        ties.append(not unique and y[q].sum() < 4)
        multi.append(len(counts) >= 2)
    masks = {'all': np.ones(len(bank), bool), 'mixed': y.sum(axis=1) < 4,
             'multiple_valid_answers': np.asarray(multi), 'one_correct': y.sum(axis=1) == 1,
             'wrong_unique_plurality': np.asarray(wrong), 'correct_plurality_mixed': np.asarray(correct),
             'plurality_tie_mixed': np.asarray(ties), 'any_invalid_parse': ~valid.all(axis=1)}
    write(OUT / 'subgroup_definitions.json', {k: int(v.sum()) for k, v in masks.items()})
    choice, ok = {}, {}
    replay = []
    train_runs = read(OUT / 'training_runs.json')
    for r in records:
        p = read(Path(r['checkpoint']) / 'predictions.json')
        assert p['ids'] == [row['id'] for row in bank]
        selected = np.asarray(p['selected'], dtype=np.int64)
        choice[r['name']] = selected
        ok[r['name']] = y[np.arange(len(bank)), selected].astype(float)
        assert ok[r['name']].sum() == r['correct']
        logits = np.stack([np.load(OUT / 'runs' / r['variant'] / f'seed_{s}/validation-logits.npy') for s in r['seeds']])
        prob = 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))
        scores = prob.mean(axis=0)
        expected = scores.argmax(axis=1)
        if r['mode'] == 'filtered':
            change = valid.any(axis=1) & ~valid[np.arange(len(bank)), expected]
            expected[change] = np.where(valid[change], scores[change], -np.inf).argmax(axis=1)
        assert np.array_equal(selected, expected), r['name']
        replay.append({'name': r['name'], 'fresh_matches_saved_final_weights_inference': True})
    write(OUT / 'fresh_replay.json', replay)
    history_rows, filter_rows = [], []
    for run in train_runs:
        for row in read(HERE / run['run_dir'] / 'history.json'):
            history_rows.append({'variant': run['variant'], 'seed': run['seed'], 'epoch': row['epoch'],
                'optimizer_steps': row['optimizer_steps'], 'stochastic_train_bce': row['stochastic_train_bce'],
                **{f'{split}_{metric}': value for split in ('train', 'validation') for metric, value in row[split].items()}})
    csv_write(OUT / 'learning_curves.csv', history_rows)
    for r in records:
        if r['mode'] != 'filtered':
            continue
        b = r['name']; a = b.removesuffix('filtered') + 'raw'
        changed = choice[a] != choice[b]
        invalid = ~valid[np.arange(len(bank)), choice[a]]
        repaired = ok[b] > ok[a]; harmed = ok[b] < ok[a]
        assert np.array_equal(changed, invalid & valid.any(axis=1)) and not harmed.any()
        filter_rows.append({'raw': a, 'filtered': b, 'replacements': int(changed.sum()),
            'repairs': int(repaired.sum()), 'new_errors': int(harmed.sum()), 'gain_pp': float((ok[b] - ok[a]).mean() * 100)})
    csv_write(OUT / 'filter_pairs.csv', filter_rows)
    write(OUT / 'filter_pairs.json', filter_rows)
    summary, subgroups = [], []
    matrices = {}
    for mode in ('filtered', 'raw'):
        for variant in VARIANTS:
            single = np.stack([ok[f'{variant}_seed{s}_{mode}'] for s in SEEDS])
            ensemble = ok[f'{variant}_ensemble5_{mode}']
            matrices[(variant, mode, 'single_mean')] = single
            matrices[(variant, mode, 'ensemble')] = ensemble[None]
            runs = [r for r in train_runs if r['variant'] == variant]
            summary.append({'variant': variant, 'mode': mode,
                'mean_accuracy': float(single.mean() * 100),
                'sample_sd': float(single.mean(axis=1).std(ddof=1) * 100),
                'seed_accuracies': (single.mean(axis=1) * 100).tolist(),
                'ensemble_correct': int(ensemble.sum()), 'ensemble_accuracy': float(ensemble.mean() * 100),
                'parameters': runs[0]['parameters'],
                'train_bce': statistics.mean(r['final_metrics']['train']['bce'] for r in runs),
                'validation_bce': statistics.mean(r['final_metrics']['validation']['bce'] for r in runs),
                'train_accuracy_diagnostic': statistics.mean(r['final_metrics']['train'][f'accuracy_{"filtered" if mode == "filtered" else "raw"}'] for r in runs)})
            for kind, values in [('single_mean', single), ('ensemble', ensemble[None])]:
                for group, mask in masks.items():
                    subgroups.append({'variant': variant, 'mode': mode, 'kind': kind, 'subgroup': group,
                        'questions': int(mask.sum()), 'accuracy': float(values[:, mask].mean() * 100),
                        'correct_mean': float(values[:, mask].sum(axis=1).mean())})
    write(OUT / 'summary.json', summary)
    csv_write(OUT / 'summary.csv', [{**r, 'seed_accuracies': str(r['seed_accuracies'])} for r in summary])
    write(OUT / 'subgroups.json', subgroups)
    csv_write(OUT / 'subgroups.csv', subgroups)
    definitions = []
    for feature in ('P', 'C'):
        for left, right in [('S2', 'I'), ('Pool', 'I'), ('S2', 'Pool')]:
            definitions.append((f'{left}-{right} at {feature}', {f'{left}_{feature}': 1, f'{right}_{feature}': -1}, 'structure'))
    for arch in ARCHITECTURES:
        definitions.append((f'C-P at {arch}', {f'{arch}_C': 1, f'{arch}_P': -1}, 'features'))
    for left, right in [('S2', 'I'), ('Pool', 'I'), ('S2', 'Pool')]:
        definitions.append((f'interaction ({left}-{right}): C-P',
             {f'{left}_C': 1, f'{right}_C': -1, f'{left}_P': -1, f'{right}_P': 1}, 'interaction'))
    rng = np.random.default_rng(20261017)
    contrasts, repairs = [], []
    for mode in ('filtered', 'raw'):
        for kind in ('single_mean', 'ensemble'):
            for label, coefficients, category in definitions:
                differences = sum(coef * matrices[(variant, mode, kind)] for variant, coef in coefficients.items())
                for group, mask in masks.items():
                    by_seed = differences[:, mask].mean(axis=1) * 100
                    point = differences[:, mask].mean(axis=0)
                    contrasts.append({'contrast': label, 'category': category, 'mode': mode, 'kind': kind,
                       'subgroup': group, 'questions': int(mask.sum()), 'difference_pp': float(point.mean() * 100),
                       'ci95_pp': interval(point, rng), 'by_seed_pp': by_seed.tolist(),
                       'positive_seeds': int((by_seed > 1e-10).sum()), 'negative_seeds': int((by_seed < -1e-10).sum()),
                       'primary': label == 'interaction (S2-I): C-P' and mode == 'filtered' and kind == 'single_mean' and group == 'all'})
                    if category != 'interaction':
                        left = next(v for v, c in coefficients.items() if c == 1)
                        right = next(v for v, c in coefficients.items() if c == -1)
                        l = matrices[(left, mode, kind)][:, mask]
                        r = matrices[(right, mode, kind)][:, mask]
                        repairs.append({'contrast': label, 'mode': mode, 'kind': kind, 'subgroup': group,
                           'repairs_mean': float((l > r).sum(axis=1).mean()),
                           'new_errors_mean': float((l < r).sum(axis=1).mean()),
                           'net_repairs_mean': float((l - r).sum(axis=1).mean())})
    write(OUT / 'paired_comparisons.json', contrasts)
    csv_write(OUT / 'paired_comparisons.csv', [{**r, 'ci95_pp': str(r['ci95_pp']), 'by_seed_pp': str(r['by_seed_pp'])} for r in contrasts])
    write(OUT / 'repairs.json', repairs)
    csv_write(OUT / 'repairs.csv', repairs)
    changes = []
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            p = choice[f'{arch}_P_seed{seed}_filtered']; c = choice[f'{arch}_C_seed{seed}_filtered']
            index = p != c
            answer = np.asarray([answers[q][p[q]] != answers[q][c[q]] for q in range(len(bank))])
            changes.append({'architecture': arch, 'seed': seed, 'candidate_changes': int(index.sum()),
                'answer_changes': int(answer.sum()), 'same_answer_candidate_changes': int((index & ~answer).sum()),
                'repairs': int((ok[f'{arch}_C_seed{seed}_filtered'] > ok[f'{arch}_P_seed{seed}_filtered']).sum()),
                'new_errors': int((ok[f'{arch}_C_seed{seed}_filtered'] < ok[f'{arch}_P_seed{seed}_filtered']).sum())})
    write(OUT / 'PC_choice_changes.json', changes)
    np.savez(OUT / 'per_question_validation.npz', labels=y, valid=valid, **choice)
    write(OUT / 'question_ids.json', [r['id'] for r in bank])
    # Historical reproduction is descriptive: P now has a padded six-slot head.
    historical = []
    source = CODE / 'fusion_confidence_20261017/results/runs'
    for feature, old in [('P', 's2_parse'), ('C', 's2_consistency')]:
        for seed in SEEDS:
            old_z = np.load(source / old / f'seed_{seed}/full/evaluation-logits.npy')
            new_z = np.load(OUT / 'runs' / f'S2_{feature}' / f'seed_{seed}/validation-logits.npy')
            old_selected = read(source / old / f'seed_{seed}/full/validation-predictions.json')['selected']
            new_selected = choice[f'S2_{feature}_seed{seed}_raw']
            historical.append({'feature_mode': feature, 'seed': seed,
               'max_abs_logit_difference': float(np.max(np.abs(new_z - old_z))),
               'same_all_choices': new_selected.tolist() == old_selected,
               'old_correct': int(y[np.arange(len(bank)), old_selected].sum()),
               'new_correct': int(ok[f'S2_{feature}_seed{seed}_raw'].sum()),
               'note': 'P padded six-slot head differs from old one-slot head; all primary comparisons use this round'})
    write(OUT / 'historical_replay.json', historical)
    costs = {'runs': 30, 'epochs_per_run': 20, 'updates_per_run': 380,
       'total_updates': 11400, 'training_question_exposures': 30 * 20 * 4755,
       'optimization_seconds': sum(r['fit_seconds'] for r in train_runs),
       'diagnostic_seconds': sum(r['diagnostic_seconds'] for r in train_runs),
       'training_plus_diagnostics_seconds': sum(r['total_training_and_diagnostics_seconds'] for r in train_runs),
       'fresh_official_evaluation_seconds': sum(r['fresh_total_seconds'] for r in records),
       'official_evaluations': 72, 'additional_searches': 0, 'test_used': False}
    write(OUT / 'cost.json', costs)
    main_result = next(r for r in contrasts if r['primary'])
    write(OUT / 'primary_result.json', main_result)
    def all_contrast(label):
        return next(r for r in contrasts if r['contrast'] == label and r['mode'] == 'filtered'
                    and r['kind'] == 'single_mean' and r['subgroup'] == 'all')
    gp, gc = all_contrast('S2-I at P'), all_contrast('S2-I at C')
    feature_effects = [all_contrast(f'C-P at {a}') for a in ARCHITECTURES]
    if main_result['ci95_pp'][0] > 0:
        interpretation = '主要交互项的描述性区间位于零以上：本轮共识条件下S2相对I的优势扩大，未支持简单的替代预期。这是当前固定预算与具体实现下的互补迹象，仍需独立评价确认。'
    elif main_result['ci95_pp'][1] < 0:
        interpretation = '主要交互项的描述性区间位于零以下：本轮共识条件下S2相对I的优势收窄，与部分替代解释一致；仍应区分I改善和S2退步，也需独立评价确认。'
    else:
        interpretation = '主要交互项的描述性区间包含零：本轮尚不能确认共识与S2交互是互补还是替代，不能把不确定结果解释为性能等价。'
    lines = ['# 显式共识是否替代候选交互：3×2固定预算实验', '',
       '日期：2026-10-17（北京时间）。完成六组×五种子共30次训练；每次固定20轮、380次更新，使用最后一轮；验证只用于诊断，没有早停、搜索或模型优胜者选择。', '',
       '## 主要比较', '',
       '主要结果统一使用最终解析有效性过滤。定义 G_P=Acc(S2-P)−Acc(I-P)，G_C=Acc(S2-C)−Acc(I-C)，主要交互项 Δ=G_C−G_P。', '',
       f'**主要Δ：{text_ci(main_result)}个百分点；逐种子Δ：{", ".join(f"{v:+.3f}" for v in main_result["by_seed_pp"])}。**', '',
       interpretation, '',
       f'P条件下S2−I为{gp["difference_pp"]:+.3f}pp；C条件下为{gc["difference_pp"]:+.3f}pp。C−P在I、Pool、S2上的效果依次为' + '、'.join(f'{r["difference_pp"]:+.3f}pp' for r in feature_effects) + '。这组分解用于判断交互项来自哪些单元变化，不能只依据一个最高分配置。', '',
       '负值表示C条件下S2相对I的准确率优势缩小，正值表示扩大；必须结合六个单元分数与困难子组解释，不能只凭交互项断定模型学会了何种推理或形成完全替代。', '',
       '## 六组结果', '',
       '| 结构/特征 | 原始单模型均值±SD % | 过滤后单模型均值±SD % | 原始集成正确数/798 | 过滤后集成正确数/798 | 参数量 |',
       '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        raw = next(r for r in summary if r['variant'] == variant and r['mode'] == 'raw')
        filtered = next(r for r in summary if r['variant'] == variant and r['mode'] == 'filtered')
        lines.append(f'| {variant} | {raw["mean_accuracy"]:.3f} ± {raw["sample_sd"]:.3f} | {filtered["mean_accuracy"]:.3f} ± {filtered["sample_sd"]:.3f} | {raw["ensemble_correct"]} | {filtered["ensemble_correct"]} | {raw["parameters"]:,} |')
    lines += ['', '单模型均值是五次拟合的统计；五成员集成是独立固定预测器，两者不能混作同一种分数。标准差为样本标准差。', '',
       '## 结构与特征：过滤后单模型配对比较', '',
       '| 比较 | 整体差值pp [95%描述性区间] | 正向种子/5 |', '|---|---:|---:|']
    for r in contrasts:
        if r['mode'] == 'filtered' and r['kind'] == 'single_mean' and r['subgroup'] == 'all':
            lines.append(f'| {r["contrast"]} | {text_ci(r)} | {r["positive_seeds"]}/5 |')
    lines += ['', '## 图表', '',
       f'![Factorial results]({OUT.as_posix()}/factorial_results.png)', '',
       '左图误差线是五种子样本标准差；中图是错误共识子组准确率；右图是按题目配对的描述性区间。它们不具有相同统计含义。', '',
       f'![Learning curves]({OUT.as_posix()}/learning_curves.png)', '',
       '曲线为完整固定预算下五种子均值；训练与验证BCE均为eval模式，最后第20轮为唯一正式检查点。曲线仅用于分析优化与拟合差距，不用于重新挑选轮次。', '',
       '## 困难子组：过滤后单模型五种子均值', '',
       '| 配置 | 混合题419 % | 多有效答案355 % | 单正确117 % | 错误共识60 % | 正确共识混合271 % | 并列88 % |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        vals = [next(r['accuracy'] for r in subgroups if r['variant'] == variant and r['mode'] == 'filtered' and r['kind'] == 'single_mean' and r['subgroup'] == group)
                for group in ['mixed', 'multiple_valid_answers', 'one_correct', 'wrong_unique_plurality', 'correct_plurality_mixed', 'plurality_tie_mixed']]
        lines.append('| ' + variant + ' | ' + ' | '.join(f'{v:.3f}' for v in vals) + ' |')
    lines += ['', '单正确与错误共识题有重叠，不能将两组得失相加。错误共识、正确共识混合与并列三组构成混合题拆分；全对379题对任意选择器均正确。', '',
       '## 错误共识上的关键次要比较', '', '| 比较 | 差值pp [95%描述性区间] |', '|---|---:|']
    for r in contrasts:
        if r['mode'] == 'filtered' and r['kind'] == 'single_mean' and r['subgroup'] == 'wrong_unique_plurality':
            lines.append(f'| {r["contrast"]} | {text_ci(r)} |')
    lines += ['', '## 训练拟合与资源', '',
       '| 配置 | 训练BCE | 验证BCE | 训练过滤准确率（诊断）% | 验证过滤准确率 % |', '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['mode'] == 'filtered':
            lines.append(f'| {r["variant"]} | {r["train_bce"]:.4f} | {r["validation_bce"]:.4f} | {r["train_accuracy_diagnostic"]:.3f} | {r["mean_accuracy"]:.3f} |')
    lines += ['', 'BCE和训练准确率均在关闭dropout的同一最后检查点测量；训练准确率为按官方标签规则聚合的拟合诊断，未作为独立官方benchmark。', '',
       f'30次训练及逐轮诊断合计{costs["training_plus_diagnostics_seconds"]:.1f}秒，其中优化{costs["optimization_seconds"]:.1f}秒、诊断{costs["diagnostic_seconds"]:.1f}秒；72次新进程官方预测加评分合计{costs["fresh_official_evaluation_seconds"]:.1f}秒。预处理、实现检查、统计、图表、独立包复核与哈希另计。', '',
       '| 过滤后集成 | 官方准确率 % | 完整fresh预测+评分秒 | 解包后资产MiB |', '|---|---:|---:|---:|']
    for r in records:
        if r['members'] == 5 and r['mode'] == 'filtered':
            lines.append(f'| {r["variant"]} | {r["accuracy"]:.3f} | {r["fresh_total_seconds"]:.3f} | {r["asset_bytes"]/1024**2:.3f} |')
    environment = read(OUT / 'environment.json')
    lines += ['', f'本机：{environment["cpu"]}；Python {environment["python"].split()[0]}、NumPy {environment["numpy"]}、PyTorch {environment["torch"]}；CPU串行，Torch4/interop1、OpenBLAS1。计时包含启动、导入、输入和资产加载、预处理、全部成员与独立评分。没有官方Xeon或完整测试集计时，不据此认证30秒资格。', '',
       '## 实现与公平性控制', '',
       '- I逐候选独立处理embedding；Pool拼接自身与四候选平均表示；S2使用原FFN残差前置、四头注意力、归一化布局。I-C通过显式特征访问组信息，不是完全独立预测器。',
       '- 六组均在最后评分头接入六个槽位；P只保留解析有效性，其余五维在标准化后严格置零；C使用全部六维。特征权重零初始化，同结构P/C初始权重与输出逐项一致。',
       '- 15对P/C共享初始权重、逐题采样顺序与预算。P未用槽位梯度与权重均为零。不同结构参数量和运算路径不同，跨结构不主张容量严格匹配或逐次dropout掩码相同。',
       '- 全部训练题BCE，AdamW lr0.0003、weight_decay0.001、batch256、dropout0.1、clip1；标准化只拟合训练集，固定最后轮次，不因验证指标更换设置。',
       '- 诊断eval不消耗训练Torch随机状态；每轮记录训练与验证指标，但从不选择最佳轮次。',
       '- 全部36个基础预测器分别执行原始与最终过滤两种官方评测，共72份。新进程预测与保存最后权重推理逐项一致；36对过滤没有新增错误。',
       '- 历史S2-P/C复现见historical_replay.json。P由原一槽位改成六槽位，数值路径可能变化，不能要求旧轨迹完全一致；主要比较仅使用本轮训练。', '',
       '## 统计与解释边界', '',
       '按题目配对bootstrap10000次，单模型先对同种子正确性求平均。逐种子差值另外报告，不把5×798个观测当独立样本。区间条件于冻结模型和反复验证开发背景，未校正多重比较；关键次要子组也不是独立测试。', '',
       '主要交互项描述准确率收益变化，不直接证明交互模型在识别共识，更不证明掌握推理验证。负向交互需要区分I受益与S2受损；整体改善不替代困难子组可靠性。参数/有效容量不同、固定20轮优化程度不同等限制仍然存在。', '',
       '所有训练权重来自本项目训练集，无外部数据、预训练权重或API。本轮无置信度回退、重加权、风格特征、新损失或学习融合权重；未读取测试输入或评分参考答案。文件保护仅按字节核对原文件。', '',
       '代码、设计实现与报告由Codex AI辅助。官方parser/scorer/data保持原样。', '',
       '## 产物', '',
       'summary.csv/JSON为六组均值、种子与集成；paired_comparisons.json为预设主次比较及子组；repairs.csv为修正/改错；PC_choice_changes.json区分答案变化与同答案候选变化；filter_pairs.csv为过滤对照；learning_curves.csv和runs各自history包含逐轮轨迹；cost.json为训练及评估成本。', '',
       'verification.json记录新ID/乱序/单题、独立包复现及原文件哈希；README.md提供命令。portable目录提供六个过滤后集成的独立权重包，不依赖历史实验目录。没有按本轮验证结果重新选定提交模型。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PRIMARY', json.dumps(main_result, ensure_ascii=False), flush=True)
    for r in summary:
        if r['mode'] == 'filtered':
            print('SUMMARY', json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
