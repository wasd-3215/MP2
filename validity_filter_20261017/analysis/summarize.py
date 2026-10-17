"""Post-prediction validation diagnostics. This file is never imported at inference."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime
from run_suite import HERE, CODE, SOURCE, OUT, SEEDS, VARIANTS, read, write, sha
sys.path.insert(0, str(CODE))
from collections import Counter
import csv
import json
import statistics
import time
import numpy as np
from common import load_inputs, read_jsonl, parse_answer, DATA, check_data


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(difference, rng):
    values = np.asarray(difference, dtype=float)
    if np.all(values == values[0]):
        return [float(values[0] * 100)] * 2
    samples = []
    for _ in range(50):
        indices = rng.integers(0, len(values), size=(200, len(values)))
        samples.extend(values[indices].mean(axis=1) * 100)
    return np.quantile(samples, [.025, .975]).tolist()


def fmt_interval(row):
    return f'{row["difference_pp"]:+.3f} [{row["ci95_pp"][0]:+.3f}, {row["ci95_pp"][1]:+.3f}]'


def main():
    write(OUT / 'analysis_sources_before.json', {str(p.relative_to(HERE)): sha(p) for p in (HERE / 'analysis').glob('*.py')})
    records = read(OUT / 'official_results.json')
    assert len(records) == 38 and read(OUT / 'implementation_checks.json')['passed']
    bank, arrays = load_inputs('validation')
    check_data(DATA, ['validation'])
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert len(refs) == len(bank)
    truth = []
    answers = []
    for row, ref in zip(bank, refs):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        correct = parse_answer(ref['answer'])
        assert correct is not None
        parsed = [parse_answer(c['text']) for c in row['candidates']]
        answers.append(parsed)
        truth.append([a is not None and a == correct for a in parsed])
    y = np.asarray(truth, dtype=bool)
    valid = np.asarray([[a is not None for a in row] for row in answers])
    n = len(bank)
    assert (y.sum(axis=1) >= 1).all()
    wrong_consensus, correct_consensus, ties, multiple_answers = [], [], [], []
    for q, row in enumerate(answers):
        counts = Counter(a for a in row if a is not None)
        largest = max(counts.values(), default=0)
        winners = [a for a, count in counts.items() if count == largest]
        unique = len(winners) == 1
        gold = parse_answer(refs[q]['answer'])
        wrong_consensus.append(unique and winners[0] != gold)
        correct_consensus.append(unique and winners[0] == gold and y[q].sum() < 4)
        ties.append(not unique and y[q].sum() < 4)
        multiple_answers.append(len(counts) >= 2)
    masks = {'all': np.ones(n, bool), 'mixed': y.sum(axis=1) < 4,
             'one_correct': y.sum(axis=1) == 1,
             'wrong_unique_plurality': np.asarray(wrong_consensus),
             'correct_plurality_mixed': np.asarray(correct_consensus),
             'plurality_tie_mixed': np.asarray(ties),
             'multiple_valid_answers': np.asarray(multiple_answers),
             'any_invalid_parse': ~valid.all(axis=1)}
    write(OUT / 'data_summary.json', {'questions': n, 'subgroups': {k: int(v.sum()) for k, v in masks.items()},
          'invalid_candidates': int((~valid).sum()), 'all_invalid_questions': int((~valid.any(axis=1)).sum()),
          'test_used': False})
    choices = {}
    correctness = {}
    replay = []
    for record in records:
        name = record['name']
        prediction = read(Path(record['checkpoint']) / 'predictions.json')
        assert prediction['ids'] == [r['id'] for r in bank]
        selected = np.asarray(prediction['selected'], dtype=np.int64)
        choices[name] = selected
        correctness[name] = y[np.arange(n), selected].astype(float)
        assert correctness[name].sum() == record['correct']
        if not record['filtered']:
            if record['routing']:
                path = SOURCE / 'results/pipelines/s2_5_route_model_agreement_support3/validation-predictions.json'
            elif record['members'] == 1:
                path = SOURCE / 'results/runs' / record['variant'] / f'seed_{record["seeds"][0]}/full/validation-predictions.json'
            elif record['variant'] == 's2':
                path = SOURCE / 'results/pipelines/s2_5/validation-predictions.json'
            elif record['variant'] == 's2_consistency':
                path = SOURCE / 'results/pipelines/consistency_5/validation-predictions.json'
            else:
                path = CODE / 'reweighting_experiments_20261017/results/pipelines/P_U_ensemble5/validation-predictions.json'
            # P ensemble did not have an official standalone result in fusion.
            # Its original reference is the exact frozen full-member logits below.
            if record['variant'] == 's2_parse' and record['members'] == 5:
                raw = np.stack([np.load(SOURCE / 'results/runs/s2_parse' / f'seed_{s}/full/evaluation-logits.npy') for s in SEEDS])
                prob = 1 / (1 + np.exp(-np.clip(raw.astype(np.float64), -40, 40)))
                expected = prob.mean(axis=0).argmax(axis=1).tolist()
                reference = 'frozen fusion member logits, original aggregation'
            else:
                expected = read(path)['selected']
                reference = str(path)
            assert selected.tolist() == expected, name
            replay.append({'name': name, 'matches': True, 'reference': reference})
    write(OUT / 'baseline_replay.json', replay)
    pair_rows, subgroup_rows, changed_rows = [], [], []
    for pair in read(OUT / 'pairs.json'):
        a, b = pair['original'], pair['filtered']
        before, after = choices[a], choices[b]
        bad = ~valid[np.arange(n), before]
        replaceable = bad & valid.any(axis=1)
        changed = before != after
        repair = (correctness[b] > correctness[a])
        harm = (correctness[b] < correctness[a])
        assert np.array_equal(changed, replaceable)
        assert not harm.any()
        assert np.array_equal(before[~bad], after[~bad])
        row = {'original': a, 'filtered': b, 'original_correct': int(correctness[a].sum()),
               'filtered_correct': int(correctness[b].sum()), 'original_invalid': int(bad.sum()),
               'replaceable': int(replaceable.sum()), 'replacements': int(changed.sum()),
               'repairs': int(repair.sum()), 'remaining_wrong_after_replacement': int((changed & ~repair).sum()),
               'new_errors': int(harm.sum()), 'repair_success_percent': float(100 * repair.sum() / changed.sum()) if changed.any() else None,
               'gain_pp': float(100 * repair.sum() / n)}
        pair_rows.append(row)
        for group, mask in masks.items():
            subgroup_rows.append({'pair': a, 'subgroup': group, 'questions': int(mask.sum()),
               'original_correct': int(correctness[a][mask].sum()), 'filtered_correct': int(correctness[b][mask].sum()),
               'original_accuracy': float(100 * correctness[a][mask].mean()),
               'filtered_accuracy': float(100 * correctness[b][mask].mean()),
               'repairs': int((repair & mask).sum()), 'new_errors': int((harm & mask).sum())})
        for q in np.flatnonzero(changed):
            changed_rows.append({'pair': a, 'id': bank[q]['id'], 'row': int(q),
              'original_index': int(before[q]), 'filtered_index': int(after[q]),
              'original_answer': answers[q][before[q]], 'filtered_answer': answers[q][after[q]],
              'repaired': bool(repair[q]), 'one_correct': bool(masks['one_correct'][q]),
              'wrong_unique_plurality': bool(masks['wrong_unique_plurality'][q])})
    write(OUT / 'filter_pairs.json', pair_rows)
    csv_write(OUT / 'filter_pairs.csv', pair_rows)
    write(OUT / 'subgroups.json', subgroup_rows)
    csv_write(OUT / 'subgroups.csv', subgroup_rows)
    write(OUT / 'changed_questions.json', changed_rows)
    csv_write(OUT / 'changed_questions.csv', changed_rows)
    np.savez(OUT / 'diagnostic_choices.npz', labels=y, valid=valid, **choices)
    means = []
    for variant in VARIANTS:
        for suffix in ['original', 'filtered']:
            xs = [100 * correctness[f'{variant}_seed{s}_{suffix}'].mean() for s in SEEDS]
            pairset = [p for p in pair_rows if p['original'].startswith(f'{variant}_seed')]
            means.append({'variant': variant, 'mode': suffix, 'mean_accuracy': statistics.mean(xs),
                'sample_sd': statistics.stdev(xs), 'seed_accuracies': xs,
                'mean_replacements': statistics.mean(p['replacements'] for p in pairset),
                'mean_repairs': statistics.mean(p['repairs'] for p in pairset),
                'mean_gain_pp': statistics.mean(p['gain_pp'] for p in pairset)})
    write(OUT / 'five_seed_summary.json', means)
    rng = np.random.default_rng(20261017)
    comparisons = []
    def compare(label, left, right, groups=('all',)):
        for group in groups:
            mask = masks[group]
            delta = np.asarray(left)[mask] - np.asarray(right)[mask]
            comparisons.append({'comparison': label, 'subgroup': group, 'questions': int(mask.sum()),
                'left_accuracy': float(np.asarray(left)[mask].mean() * 100),
                'right_accuracy': float(np.asarray(right)[mask].mean() * 100),
                'difference_pp': float(delta.mean() * 100), 'ci95_pp': interval(delta, rng)})
    def mean_correct(variant, mode):
        return np.stack([correctness[f'{variant}_seed{s}_{mode}'] for s in SEEDS]).mean(axis=0)
    for variant in VARIANTS:
        compare(f'{variant}: filter - original (five-seed mean)', mean_correct(variant, 'filtered'), mean_correct(variant, 'original'), masks.keys())
        compare(f'{variant}: filter - original (ensemble)', correctness[f'{variant}_ensemble5_filtered'], correctness[f'{variant}_ensemble5_original'], masks.keys())
    for mode in ['original', 'filtered']:
        for left, right in [('s2_parse', 's2'), ('s2_consistency', 's2_parse')]:
            compare(f'{left} - {right}: {mode} (five-seed mean)', mean_correct(left, mode), mean_correct(right, mode), masks.keys())
            compare(f'{left} - {right}: {mode} (ensemble)', correctness[f'{left}_ensemble5_{mode}'], correctness[f'{right}_ensemble5_{mode}'], masks.keys())
    a = correctness['s2_ensemble5_original']; b = correctness['s2_ensemble5_filtered']
    c = correctness['s2_ensemble5_route_original']; d = correctness['s2_ensemble5_route_filtered']
    for name, left, right in [('B-A: filter', b, a), ('C-A: frozen route', c, a),
                              ('D-B: frozen route after equal final filtering', d, b), ('D-C: filter after route', d, c)]:
        compare(name, left, right, masks.keys())
    write(OUT / 'paired_comparisons.json', comparisons)
    csv_write(OUT / 'paired_comparisons.csv', [{**r, 'ci95_pp': str(r['ci95_pp'])} for r in comparisons])
    routing = []
    for label, left, right in [('C-A', 's2_ensemble5_route_original', 's2_ensemble5_original'),
                               ('D-B', 's2_ensemble5_route_filtered', 's2_ensemble5_filtered')]:
        old, new = choices[right], choices[left]
        changed = np.asarray([answers[q][old[q]] != answers[q][new[q]] for q in range(n)])
        original_valid = valid[np.arange(n), old]
        for title, mask in [('all', np.ones(n, bool)), ('base_choice_valid', original_valid), ('base_choice_invalid', ~original_valid)]:
            repair = (correctness[left] > correctness[right]) & mask
            harm = (correctness[left] < correctness[right]) & mask
            routing.append({'comparison': label, 'stratum': title, 'questions': int(mask.sum()),
                'answer_changes': int((changed & mask).sum()), 'repairs': int(repair.sum()), 'new_errors': int(harm.sum()),
                'net_repairs': int(repair.sum() - harm.sum())})
    write(OUT / 'routing_attribution.json', routing)
    # Import inference only after all official subprocess runs have finished.
    from method import evaluate, predict, validity_filter
    micro = []
    for pair in read(OUT / 'pairs.json'):
        original = next(r for r in records if r['name'] == pair['original'])
        actual, scores, _ = evaluate(bank, arrays, Path(original['checkpoint']))
        assert actual == choices[pair['original']].tolist()
        filtered, _ = validity_filter(bank, scores, np.asarray(actual))
        assert filtered.tolist() == choices[pair['filtered']].tolist()
        times = []
        for _ in range(15):
            start = time.perf_counter()
            validity_filter(bank, scores, np.asarray(actual))
            times.append(time.perf_counter() - start)
        micro.append({'original': pair['original'], 'repetitions': 15,
                      'median_filter_seconds': statistics.median(times),
                      'mean_filter_seconds': statistics.mean(times), 'includes_parser': True})
    write(OUT / 'filter_microtiming.json', micro)
    ids = np.array([97, 3, 400, 11, 777, 25, 0, 48, 200, 17, 700, 56, 98])
    subset = [{**bank[i], 'id': f'fresh-audit:{j}'} for j, i in enumerate(ids)]
    inputs = {'embeddings': arrays['embeddings'][ids]}
    checks = []
    for record in records:
        actual = predict(subset, inputs, Path(record['checkpoint']))
        assert actual == choices[record['name']][ids].tolist(), record['name']
        assert predict(subset[:1], {'embeddings': inputs['embeddings'][:1]}, Path(record['checkpoint'])) == actual[:1]
        assert all(type(i) is int and 0 <= i < 4 for i in actual)
        checks.append(record['name'])
    write(OUT / 'new_bank_checks.json', {'passed': True, 'predictors': checks, 'reordered_questions': True,
          'new_ids': True, 'thirteen_and_single_question': True, 'native_int': True})
    environment = read(OUT / 'environment.json')
    import torch
    environment.update({'numpy': np.__version__, 'torch': torch.__version__,
                        'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads()})
    write(OUT / 'environment.json', environment)
    lines = ['# 最终选择前排除解析无效候选：固定模型配对实验', '',
      '日期：2026-10-17（北京时间）。复用固定20轮模型，不训练、不调参、不重新校准。', '',
      '完成18个基础预测器的过滤前后配对，并增加冻结回退的过滤前后配对：共38次独立新进程预测与官方评分。', '',
      '## 主要发现', '',
      '- 普通S2五成员集成：735→737题正确（92.105%→92.356%），两次替换均纠正；带解析特征的集成734→735；一致性集成740→740，没有触发替换。',
      '- 所有19对中，没有新增错误；本验证集发生的替换均恰好修复错误。后一观察不保证新题上每次替换都正确，理论只保证按官方指标不降分。',
      '- 单模型五种子均值：普通S2过滤增加0.476pp，解析模型增加0.201pp，一致性模型增加0.025pp。解析特征相对普通模型的优势从0.401pp缩至0.125pp；后者95%描述性区间[-0.301,+0.551]。收益中的较大部分可由最终约束补齐，仍不能单凭差值将学习机制完全归因。',
      '- 冻结回退加过滤达到740/798=92.732%；相对统一过滤基线737题，它纠正5题、改错2题，净增加3题（+0.376pp，95%描述性区间[-0.251,+1.003]）。因此这条回退的收益并非全部来自解析修复，额外收益的区间仍包含零。',
      '- 冻结回退加过滤与一致性集成整体同为740题，但单正确题为74/117对63/117，错误共识题为35/60对24/60。这个并列比较是观察结果后的描述，未重新选择模型，也不是独立测试优越性结论；两个子组重叠，不能累加。',
      '- 过滤没有修复一致性集成的错误共识弱点：该集成原来没有无效选择，过滤后错误共识仍为40%。普通集成过滤后为61.667%，原有取舍依然存在。', '',
      '## 固定规则', '',
      '四个候选完整进入原网络。成员sigmoid分数先按原规则平均，原回退照常执行；仅当最终选择的官方parse_answer结果为None且存在有效候选时，改选原聚合分数最高的有效候选。有效选择原样保留；全部无效保留原选择；并列选原顺序最早候选。未修改官方解析器。', '',
      '所有主结果来自原版predict.py与score_predictions.py。诊断读取验证参考答案，推理源码不读取参考答案或保存预测。没有测试输入、预测、评分或训练。', '',
      '## 五种子单模型结果', '',
      '| 输入 | 原始均值±样本SD % | 过滤后均值±样本SD % | 平均替换题数 | 平均修正题数 | 平均增益pp |',
      '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        original = next(r for r in means if r['variant'] == variant and r['mode'] == 'original')
        filtered = next(r for r in means if r['variant'] == variant and r['mode'] == 'filtered')
        lines.append(f'| {variant} | {original["mean_accuracy"]:.3f} ± {original["sample_sd"]:.3f} | {filtered["mean_accuracy"]:.3f} ± {filtered["sample_sd"]:.3f} | {filtered["mean_replacements"]:.1f} | {filtered["mean_repairs"]:.1f} | {filtered["mean_gain_pp"]:+.3f} |')
    lines += ['', '各行是五个单模型的统计，不是可部署集成分数；逐种子结果见filter_pairs.csv。', '',
      '## 固定五成员集成与回退', '',
      '| 预测器 | 原正确数/798 | 过滤后正确数/798 | 原无效选择 | 替换 | 修正 | 替换后仍错 | 新增错误 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in pair_rows:
        if 'ensemble5' in row['original']:
            lines.append(f'| {row["original"]} | {row["original_correct"]} | {row["filtered_correct"]} | {row["original_invalid"]} | {row["replacements"]} | {row["repairs"]} | {row["remaining_wrong_after_replacement"]} | {row["new_errors"]} |')
    lines += ['', '## 同协议关键比较', '',
      '以下区间为10000次按题目配对bootstrap的描述性95%区间；单模型先对五种子正确性平均。区间条件于冻结预测器，不包含历史验证开发/选择不确定性，未校正多重比较。', '',
      '| 比较 | 子组 | 差值pp [95%区间] |', '|---|---|---:|']
    for row in comparisons:
        if row['subgroup'] == 'all' and ('five-seed mean' in row['comparison'] or row['comparison'].startswith(('B-A', 'C-A', 'D-B', 'D-C'))):
            lines.append(f'| {row["comparison"]} | all | {fmt_interval(row)} |')
    lines += ['', '## 困难子组：固定集成', '',
      '| 预测器 | 子组 | 题数 | 原准确率 % | 过滤后 % | 修正题数 |', '|---|---|---:|---:|---:|---:|']
    for row in subgroup_rows:
        if 'ensemble5' in row['pair'] and row['subgroup'] in ['mixed', 'one_correct', 'wrong_unique_plurality', 'multiple_valid_answers', 'any_invalid_parse']:
            lines.append(f'| {row["pair"]} | {row["subgroup"]} | {row["questions"]} | {row["original_accuracy"]:.3f} | {row["filtered_accuracy"]:.3f} | {row["repairs"]} |')
    lines += ['', '## 回退归因', '',
      'A=原集成；B=原集成加过滤；C=原冻结回退；D=原冻结回退加过滤。阈值固定model_agreement≤0.8且唯一最高票至少3票；不重新计算过滤后的置信度或拟合阈值。D−B衡量这条冻结策略在统一最终过滤下的额外贡献。', '',
      '| 比较 | 原选择分层 | 题数 | 答案变化 | 修正 | 改错 | 净修正 |', '|---|---|---:|---:|---:|---:|---:|']
    for row in routing:
        lines.append(f'| {row["comparison"]} | {row["stratum"]} | {row["questions"]} | {row["answer_changes"]} | {row["repairs"]} | {row["new_errors"]} | {row["net_repairs"]:+d} |')
    lines += ['', '## 完整fresh验证预测加独立评分及资产', '',
      '| 预测器 | 秒 | 解包后资产MiB |', '|---|---:|---:|']
    for record in records:
        if record['members'] == 5:
            lines.append(f'| {record["name"]} | {record["fresh_total_seconds"]:.3f} | {record["asset_bytes"] / 1024**2:.3f} |')
    lines += ['', f'纯过滤（包含官方解析）19个预测器的中位耗时均值：{statistics.mean(r["median_filter_seconds"] for r in micro):.6f}秒/798题；每个重复15次。这是诊断微计时，不替代完整fresh计时。', '',
      '推理资产计入每个方案全部成员的权重、标准化统计、配置和流水线manifest；不累加互斥实验方案。完整测量包含启动、导入、原始输入加载和全部集成成员；进程串行，Torch intra-op4/inter-op1，OpenBLAS1。', '',
      f'本机：{environment["cpu"]}；Python {sys.version.split()[0]}，NumPy {np.__version__}，PyTorch {torch.__version__}。未在官方Xeon及完整测试集测量，不据此认证30秒资格。', '',
      '## 验证与解释边界', '',
      '- 192组掩码/原选择/分数边界测试通过，包括全部无效与并列；有效的原回退选择严格保留。',
      '- 19个无过滤预测器复现原方案或原冻结成员分数聚合；每对过滤前后分数相同，改变选择恰好等于可替换题集合；全部配对新增错误为0。',
      '- 38个预测器通过新ID、乱序13题、单题和原生int检查；完整记录见new_bank_checks.json。',
      '- 验证集含解析失败候选126题，不等于模型选中无效候选的题数。单正确题与错误共识题重叠，不能相加。',
      '- 理论上的逐题不降分仅来自当前官方数值解析评分规则；不代表推理过程正确或所有人工语义标准下正确。',
      '- 过滤不改变有效候选之间的分数；解析特征在统一过滤后仍可能通过训练改变排序。过滤后仍有差异不能直接证明具体因果机制。',
      '- 复用的所有模型均来自同一固定20轮协议；本轮没有训练成本、超参数搜索或模型优胜者选择。',
      '- 当前验证已反复用于历史开发；罕见替换带来的小幅增益应按题数解释，不能当作独立测试显著性。',
      '- 代码与分析由Codex AI辅助；未使用外部数据、预训练权重或API。原文件保护结果见verification.json与protected_verification.json。', '',
      '## 产物与复现', '',
      'filter_pairs.csv/JSON包含全部19对；subgroups.csv/JSON包含诊断子组；changed_questions.csv/JSON逐题列出替换；paired_comparisons.json保存配对区间；baseline_replay.json保存复现来源；pipelines下各预测器包含官方预测、评分及外部计时。', '',
      '新实验目录的README.md提供运行命令。pipelines中的原始成员路径用于复现实验；portable目录中的独立bundle可脱离历史实验目录加载，复核见verification.json。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'five_seed': means, 'ensemble_pairs': [r for r in pair_rows if 'ensemble5' in r['original']], 'routing': routing}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
