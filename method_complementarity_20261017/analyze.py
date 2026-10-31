"""Descriptive paired analysis of already frozen official predictions."""
import runtime
from utils import HERE, OUT, OLD, read, write
import csv
import numpy as np
from common import load_inputs, read_jsonl, DATA, parse_answer
from decisions import layout

def interval(d):
    rng = np.random.default_rng(20261017)
    samples = np.empty(10000)
    for start in range(0, 10000, 500):
        samples[start:start+500] = d[rng.integers(0, len(d), size=(500, len(d)))].mean(1) * 100
    return np.percentile(samples, [2.5, 97.5]).tolist()

def main():
    records = read(OUT / 'official_results.json')
    bank, _ = load_inputs('validation')
    refs = read_jsonl(DATA / 'gsm8k_validation.jsonl')
    assert [r['id'] for r in refs] == [r['id'] for r in bank]
    answers, valid = layout(bank)
    gold = [parse_answer(r['answer']) for r in refs]
    y = np.asarray([[a is not None and a == g for a in row] for row, g in zip(answers, gold)])
    choices, correct, numeric = {}, {}, {}
    for r in records:
        name = r['name']
        pred = read(OUT / 'pipelines' / name / 'predictions.json')
        assert pred['ids'] == [row['id'] for row in bank]
        choices[name] = np.asarray(pred['selected'])
        correct[name] = y[np.arange(len(bank)), choices[name]]
        numeric[name] = [a[int(i)] for a, i in zip(answers, choices[name])]
        assert int(correct[name].sum()) == r['correct']
    for name in ('S2_C', 'Pool_P'):
        old = read(OLD / 'results/pipelines' / f'{name}_ensemble5_filtered/predictions.json')
        assert old['selected'] == choices[name].tolist()
    disagreement = np.asarray([a != b for a, b in zip(numeric['Pool_P'], numeric['S2_C'])])
    active = np.asarray([len(set(a for a in row if a is not None)) >= 2 for row in answers])
    one_correct = y.sum(1) == 1
    wrong_plurality = []
    for a, g in zip(answers, gold):
        from collections import Counter
        counts = Counter(v for v in a if v is not None)
        top = max(counts.values(), default=0)
        leaders = [k for k, v in counts.items() if v == top]
        wrong_plurality.append(len(leaders) == 1 and leaders[0] != g)
    masks = {'all': np.ones(len(bank), bool), 'active_multi_answer': active,
             'expert_answer_disagreement': disagreement, 'one_correct_candidate': one_correct,
             'wrong_unique_plurality': np.asarray(wrong_plurality)}
    result = []
    for r in records:
        name = r['name']
        c, base = correct[name], correct['S2_C']
        result.append({**r, 'repairs_vs_C': int((c & ~base).sum()), 'harms_vs_C': int((~c & base).sum()),
          'net_questions_vs_C': int(c.sum() - base.sum()), 'paired_delta_pp': float((c.astype(int) - base.astype(int)).mean() * 100),
          'paired_bootstrap_95ci_pp': interval(c.astype(int) - base.astype(int)),
          'numeric_answer_changes_vs_C': sum(a != b for a, b in zip(numeric[name], numeric['S2_C'])),
          'subgroups': {k: {'n': int(m.sum()), 'correct': int(c[m].sum())} for k, m in masks.items()}})
    pair_delta = correct['gate8'].astype(int) - correct['gate4'].astype(int)
    summary = {'n': len(bank), 'active_questions': int(active.sum()), 'expert_answer_disagreements': int(disagreement.sum()),
      'Pool_only_correct': int((correct['Pool_P'] & ~correct['S2_C']).sum()),
      'C_only_correct': int((correct['S2_C'] & ~correct['Pool_P']).sum()),
      'both_wrong': int((~correct['Pool_P'] & ~correct['S2_C']).sum()),
      'oracle_union_correct_diagnostic_only': int((correct['Pool_P'] | correct['S2_C']).sum()),
      'gate8_minus_gate4': {'net_questions': int(pair_delta.sum()), 'delta_pp': float(pair_delta.mean()*100), 'ci95_pp': interval(pair_delta)},
      'results': result, 'intervals_descriptive_development_set': True, 'test_used': False}
    write(OUT / 'summary.json', summary)
    with (OUT / 'disagreement_index.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['id', 'source_line', 'answers', 'gold', 'Pool_P', 'S2_C', 'equal_average', 'fixed_weight', 'gate8', 'gate4', 'gate8_repairs', 'gate8_harms']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in np.flatnonzero(disagreement):
            row = {'id': bank[q]['id'], 'source_line': int(q+1), 'answers': '|'.join(str(a) for a in answers[q]), 'gold': gold[q],
              'gate8_repairs': int(correct['gate8'][q] and not correct['S2_C'][q]), 'gate8_harms': int(not correct['gate8'][q] and correct['S2_C'][q])}
            row.update({n: f'{numeric[n][q]}:{int(correct[n][q])}' for n in choices})
            writer.writerow(row)
    lines = ['# 代表方法互补实验', '', '比较冻结的 Pool-P 与 S2-C 五种子集成。选择规则仅由训练集折外预测建立，验证集只在规则冻结后评价。', '',
        '| 方法 | 正确/798 | 准确率 | 修复/损害（相对S2-C） | 净增 | 配对95%区间（百分点） |',
        '|---|---:|---:|---:|---:|---|']
    for r in result:
        lo, hi = r['paired_bootstrap_95ci_pp']
        lines.append(f'| {r["name"]} | {r["correct"]}/798 | {100*r["correct"]/798:.3f}% | {r["repairs_vs_C"]}/{r["harms_vs_C"]} | {r["net_questions_vs_C"]:+d} | [{lo:.3f}, {hi:.3f}] |')
    fit = read(OUT / 'selector_fit.json')
    train = read(OUT / 'training_verification.json')
    lines += ['', '## 实验设计与训练约束', '',
      '主对照为 gate8 对 S2-C；机制消融为 gate8 对 gate4。gate8 为8个输入的线性收益回归：两方法各自答案分数间隔、种子答案一致率、所选答案票数，再加不同有效答案数与解析无效比例。gate4仅保留分数间隔与种子一致率。', '',
      '在训练集OOF的数值答案分歧题上拟合 Pool 正确性减 S2 正确性；双方均错样本目标为0。特征仅在这些训练行上标准化，岭惩罚10，不惩罚截距，收益>0才切换到Pool。同数值答案保留S2；最终统一过滤解析无效候选，全无效回退原最高分。没有新神经门控，没有阈值搜索，没有验证集拟合。', '',
      f'补训Pool-P 5折×5种子共25次；复用S2-C 25个折模型，并逐折核对划分、标准化、保存的OOF输出及现模型重放，最大logit差{train["reused_C_max_logit_error"]:.3g}。每次固定20轮，训练折3804题、留出951题，共300次更新；全量部署权重沿用原实验。', '',
      f'训练OOF共有{fit["answer_disagreements"]}道数值答案分歧，Pool独有正确{fit["Pool_only_correct"]}，C独有正确{fit["C_only_correct"]}，双方均错分歧{fit["both_wrong_disagreements"]}。固定权重在21个预定候选中选择，Pool权重={fit["selected_pool_weight"]:.2f}。门控拟合后的训练OOF表现只是拟合内诊断，不是门控层的无偏交叉验证估计；没有声称完成嵌套交叉验证。', '',
      '## 互补空间与分组', '',
      f'验证集两个代表有{summary["expert_answer_disagreements"]}道数值答案分歧；Pool独有正确{summary["Pool_only_correct"]}题，C独有正确{summary["C_only_correct"]}题。已知正确性才能实现的二者并集为{summary["oracle_union_correct_diagnostic_only"]}/798，仅为诊断上限。443题只有一个有效答案组，355题需要区分多个答案组。', '',
      '| 方法 | 分歧38题正确 | 多答案355题正确 | 单正确候选题正确 | 错误唯一多数题正确 |', '|---|---:|---:|---:|---:|']
    for r in result:
        g = r['subgroups']
        lines.append(f'| {r["name"]} | {g["expert_answer_disagreement"]["correct"]}/{g["expert_answer_disagreement"]["n"]} | {g["active_multi_answer"]["correct"]}/355 | {g["one_correct_candidate"]["correct"]}/{g["one_correct_candidate"]["n"]} | {g["wrong_unique_plurality"]["correct"]}/{g["wrong_unique_plurality"]["n"]} |')
    lines += ['', f'gate8−gate4净差{int(pair_delta.sum()):+d}题；区间与详细分组见summary.json。所有修复/损害与答案变化记录在disagreement_index.csv，不需要把候选全文送入对话。', '',
      '## 成本与验证', '',
      f'新增25次训练纯优化{train["fit_seconds"]:.1f}秒，含OOF复用校验等每轮工作合计{train["total_run_seconds"]:.1f}秒。门控与权重拟合{fit["selector_fit_seconds"]:.2f}秒。各方法完整新进程预测+独立官方评分：', '',
      '| 方法 | 本机总秒数 | 解压后推理资产MiB |', '|---|---:|---:|']
    for r in result:
        lines.append(f'| {r["name"]} | {r["fresh_total_seconds"]:.3f} | {r["asset_bytes_uncompressed"]/1024**2:.3f} |')
    lines += ['', '每种方法均另做便携资产的新进程官方重放。共享资产包不包含任何OOF预测、验证预测或参考答案；诊断文件不属于部署资产。资产计数展开NPZ内部文件，未用压缩后体积替代。线程限制为Torch4/interop1，OpenBLAS1；数值工作串行。30秒要求仍需在官方参考CPU确认，本机结果不是官方机器认证。', '',
      '当前验证集已多次用于开发，区间只是本组配对描述，不能替代新数据上的确认。全部分数来自未修改的官方评分器；测试集输入与标签未用于本实验。文件保护及新输入检查结果见verification.json；原文件快照不会被更新来掩盖差异。', '',
      '## 复现', '', '使用Python3.12，依次运行 train.py、fit_selector.py、evaluate.py、analyze.py、verify.py，均带-B；运行前设置PYTHONDONTWRITEBYTECODE=1。已有原实验文件均只读；全部新输出限制在本目录。协议见protocol.json，环境见results/environment.json，冻结哈希见results/frozen_before_validation.json。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print([(r['name'], r['correct'], r['repairs_vs_C'], r['harms_vs_C']) for r in result], flush=True)

if __name__ == '__main__':
    main()
