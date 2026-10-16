"""Official-score summaries and clearly separated exploratory diagnostics."""
import runtime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import load_bank, read_jsonl, parse_answer, DATA


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results')
    output = parser.parse_args().output.resolve()
    protocol = read(HERE/'protocol.json')
    runs,all_runs = read(output/'selected_runs.json'),read(output/'all_runs.json')
    assert len(all_runs)==48 and len(runs)==40
    seeds = protocol['seeds']
    labels = np.load(output/'validation-labels.npy',allow_pickle=False)
    bank = load_bank('validation')
    ids = [row['id'] for row in bank]
    reconstructed = []
    for row,ref in zip(bank,read_jsonl(DATA/'gsm8k_validation.jsonl')):
        assert row['id']==ref['id'] and row['question']==ref['question']
        gold = parse_answer(ref['answer'])
        reconstructed.append([int((a:=parse_answer(c['text'])) is not None and a==gold) for c in row['candidates']])
    assert np.array_equal(labels,reconstructed)
    groups = [f'{model}__{obj}' for model in protocol['models'] for obj in protocol['objectives']]
    correct,choices,summary,diagnostics = {},{},{},[]
    for group in groups:
        rows = sorted([r for r in runs if r['group']==group],key=lambda r:seeds.index(r['seed']))
        assert [r['seed'] for r in rows] == seeds
        hits,picks,top2 = [],[],[]
        for r in rows:
            folder = HERE/r['run_dir']
            prediction = read(folder/'validation-predictions.json')
            official = read(folder/'validation-score.json')
            assert prediction['ids']==ids and prediction['split']=='validation'
            pick = np.asarray(prediction['selected'])
            hit = labels[np.arange(len(bank)),pick]
            assert hit.sum()==official['correct']==r['correct'] and official['accuracy']==r['accuracy']
            logits = np.load(folder/'validation-logits.npy',allow_pickle=False)
            assert np.array_equal(logits.argmax(1),pick)
            top = np.argsort(-logits,axis=1,kind='stable')[:,:2]
            top2.append(labels[np.arange(len(bank))[:,None],top].max(1))
            hits.append(hit)
            picks.append(pick)
        correct[group],choices[group] = np.asarray(hits),np.asarray(picks)
        acc = np.asarray([r['accuracy'] for r in rows])
        summary[group] = {'group':group,'model':rows[0]['model'],'objective':rows[0]['objective'],
            'mean_accuracy':float(acc.mean()),'std_accuracy':float(acc.std(ddof=1)),
            'non_search_seeds_mean_accuracy':float(acc[1:].mean()),'seed17_accuracy':float(acc[0]),
            'learning_rate':rows[0]['learning_rate'],'parameters':rows[0]['parameters'],
            'mean_best_epoch':float(np.mean([r['best_epoch'] for r in rows])),
            'mean_steps_run':float(np.mean([r['optimizer_steps'] for r in rows])),
            'mean_train_seconds':float(np.mean([r['training_seconds'] for r in rows])),
            'mean_validation_wall_seconds':float(np.mean([r['validation_total_wall_seconds'] for r in rows])),
            'asset_mib':rows[0]['uncompressed_asset_bytes']/1024**2}
        diagnostics.append({'group':group,
            'mixed_n':int((labels.sum(1)<4).sum()),'mixed_accuracy':float(100*correct[group][:,labels.sum(1)<4].mean()),
            'one_correct_n':int((labels.sum(1)==1).sum()),'one_correct_accuracy':float(100*correct[group][:,labels.sum(1)==1].mean()),
            'two_correct_accuracy':float(100*correct[group][:,labels.sum(1)==2].mean()),
            'three_correct_accuracy':float(100*correct[group][:,labels.sum(1)==3].mean()),
            'all_positive_accuracy':float(100*correct[group][:,labels.sum(1)==4].mean()),
            'top2_contains_correct_percent':float(100*np.mean(top2))})
    csv_write(output/'summary.csv',list(summary.values()))
    write(output/'summary.json',list(summary.values()))
    csv_write(output/'diagnostics.csv',diagnostics)
    write(output/'diagnostics.json',diagnostics)
    fields = ['group','seed','learning_rate','accuracy','correct','best_epoch','epochs_run','optimizer_steps',
              'best_optimizer_steps','eligible_training_questions','training_draws','training_seconds','validation_total_wall_seconds','run_dir']
    csv_write(output/'runs.csv',[{k:r[k] for k in fields} for r in all_runs])
    pairs = []
    for model in protocol['models']:
        for a,b in [('bce_mixed','bce_all'),('pairwise','bce_mixed'),('set_mass','bce_mixed')]:
            pairs.append((f'{model}__{a}',f'{model}__{b}'))
    pairs.extend((f'attention__{obj}',f'independent__{obj}') for obj in protocol['objectives'])
    bootstrap = np.random.default_rng(20261017).integers(0,len(bank),size=(10000,len(bank)))
    paired = []
    for a,b in pairs:
        diff = correct[a]-correct[b]
        question_mean = diff.mean(0)
        distribution = 100*question_mean[bootstrap].mean(1)
        seed_diff = 100*diff.mean(1)
        paired.append({'comparison':f'{a} - {b}','mean_difference_pp':float(seed_diff.mean()),
            'seed_difference_std_pp':float(seed_diff.std(ddof=1)),
            'non_search_difference_pp':float(seed_diff[1:].mean()),
            'bootstrap_ci95_low':float(np.quantile(distribution,.025)),
            'bootstrap_ci95_high':float(np.quantile(distribution,.975)),
            'positive_seeds':int((seed_diff>0).sum()),'seed_differences_pp':seed_diff.tolist(),
            'seed17_fixed_errors':int((diff[0]>0).sum()),'seed17_new_errors':int((diff[0]<0).sum())})
    write(output/'paired_comparisons.json',paired)
    csv_write(output/'paired_comparisons.csv',paired)
    # Interaction: does changing the objective alter the attention advantage?
    interactions = []
    base = correct['attention__bce_mixed']-correct['independent__bce_mixed']
    for obj in ['pairwise','set_mass']:
        diff = correct[f'attention__{obj}']-correct[f'independent__{obj}']-base
        distribution = 100*diff.mean(0)[bootstrap].mean(1)
        interactions.append({'objective':obj,'attention_advantage_change_vs_mixed_bce_pp':float(100*diff.mean()),
            'bootstrap_ci95_low':float(np.quantile(distribution,.025)),
            'bootstrap_ci95_high':float(np.quantile(distribution,.975))})
    write(output/'interactions.json',interactions)
    best = max(summary.values(),key=lambda r:(r['mean_accuracy'],-r['parameters']))
    selected = next(r for r in runs if r['group']==best['group'] and r['seed']==17)
    bundle = output/'selected_bundle'
    bundle.mkdir(exist_ok=True)
    for file in (HERE/selected['run_dir']/'bundle').iterdir():
        shutil.copy2(file,bundle/file.name)
    selection = {'policy':protocol['final_selection'],'group':best['group'],'model':selected['model'],
        'objective':selected['objective'],'seed':17,'learning_rate':selected['learning_rate'],
        'source_run':selected['run_dir'],'validation_accuracy':selected['accuracy'],
        'five_seed_mean_accuracy':best['mean_accuracy'],'test_evaluated':False,
        'bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}}
    write(output/'selection.json',selection)
    write(output/'per_question_validation.json',[{'id':row['id'],'labels':labels[i].tolist(),
        'choices_by_seed':{g:choices[g][:,i].tolist() for g in groups}} for i,row in enumerate(bank)])
    # All-BCE should exactly replay the prior seed/LR experiments if sources/settings agree.
    old = read(HERE.parent/'experiments_20261017/results/selected_runs.json')
    baseline_replay = []
    for r in runs:
        if r['objective']!='bce_all':
            continue
        previous = next(x for x in old if x['model']==r['model'] and x['seed']==r['seed'])
        old_choices = read(HERE.parent/'experiments_20261017'/previous['run_dir']/'validation-predictions.json')['selected']
        new_choices = read(HERE/r['run_dir']/'validation-predictions.json')['selected']
        baseline_replay.append({'model':r['model'],'seed':r['seed'],'same_learning_rate':r['learning_rate']==previous['learning_rate'],
            'identical_predictions':old_choices==new_choices,'previous_accuracy':previous['accuracy'],'new_accuracy':r['accuracy']})
    write(output/'baseline_replay.json',baseline_replay)
    env,integrity = read(output/'environment.json'),read(output/'protected_verification.json')
    train_seconds = sum(r['training_seconds'] for r in all_runs)
    eval_seconds = sum(r['validation_total_wall_seconds'] for r in all_runs)
    names = {'bce_all':'O1 全部题 BCE','bce_mixed':'O2 混合题 BCE','pairwise':'O3 成对排序','set_mass':'O4 正确集合概率质量'}
    lines = ['# 四候选实验：训练目标对照','',
        '完成 2 种网络 × 4 种训练目标。每组先用 seed 17 搜索两种学习率，再按选定学习率训练其余四个种子；共 48 次训练，40 个结果进入主表。正式准确率来自未修改的官方评分脚本。','',
        f"五种子均值最高：**{best['group']}，{best['mean_accuracy']:.3f}% ± {best['std_accuracy']:.3f}%**。固定保留它的 seed 17 单模型（{selected['accuracy']:.3f}%），没有挑选最高分种子或做集成。",'',
        '## 数据、目标与公平性','',
        '- 训练 4755 题：混合题 2513 题，四候选全对 2242 题。验证 798 题：混合题 419 题，全对 379 题。标签沿用官方数值解析规则，允许多个正确答案。',
        '- 输入固定为每题 4×1024 向量；归一化仅拟合全部训练候选，八组完全相同。独立 MLP 和候选注意力模型逐字复制上一轮代码。',
        '- O1：所有题、所有候选的 BCE。O2：仅混合题 BCE。O3：每题对所有正负候选对计算 softplus(s负−s正)，先题内平均，再按题平均。O4：每题 logsumexp(全部分数)−logsumexp(正确候选分数)，再按题平均。',
        '- O2/O3/O4 均只采样混合题；O3/O4 在全对题上没有排序监督。O3 希望每个正确候选胜过每个错误候选；O4 只要求正确候选集合获得较大总概率，可偏重其中一个正确候选。',
        '- 每个“预算轮次”均抽取 4755 题，执行 19 次更新（18×256 + 147）。O1 每轮完整打乱训练集；其他组在混合题池内逐轮打乱并循环取样，保留未消费的索引。O2/O3/O4 同种子看到相同的样本顺序。',
        '- 每个预算轮次相当于混合题池 1.892 次遍历，避免少一半训练题就少一半更新的混淆；但混合题每题曝光次数相应增加。O1→O2 因而检验的是固定预算下将训练分配给混合题的整体效果。',
        '- O1→O2 同时改变了候选标签比例（正例约 75.46%→53.57%）和题目分布。全对题对 BCE 仍有绝对正确性监督，不能由本实验推断它们“没有信息”；收益也不能单独归因于去除无用梯度。',
        '- 最大预算与验证频率相同；早停发生的实际轮数可不同，并非强行让最终所有模型执行相同次数更新。逐轮 draws、steps、eligible_pool_passes 均有记录。',
        '- AdamW，weight decay 0.001，dropout 0.1，梯度裁剪 1；最多 60 预算轮，patience 10。学习率候选 0.001 / 0.0003，种子 17、29、43、71、101。',
        '- 检查点先按官方验证正确数、再按各自的验证目标损失选择，完全相同则保留更早轮次；学习率使用相同规则。不同目标损失的数值不可横向比较。',
        '- 五种子均值包含用于选择学习率的 seed 17，另报其余四种子的均值；所有种子仍使用同一验证集早停。','',
        '## 官方验证结果','',
        '| 训练目标 | 独立 MLP：均值±样本标准差 % | 候选注意力：均值±样本标准差 % | 注意力−独立（百分点） |',
        '|---|---:|---:|---:|']
    for obj in protocol['objectives']:
        i,a = summary[f'independent__{obj}'],summary[f'attention__{obj}']
        lines.append(f"| {names[obj]} | {i['mean_accuracy']:.3f} ± {i['std_accuracy']:.3f} | {a['mean_accuracy']:.3f} ± {a['std_accuracy']:.3f} | {a['mean_accuracy']-i['mean_accuracy']:+.3f} |")
    lines += ['', '| 组 | 学习率 | 其余四种子均值 % | seed 17 % | 最佳预算轮均值 | 实际更新次数均值 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['group']} | {r['learning_rate']} | {r['non_search_seeds_mean_accuracy']:.3f} | {r['seed17_accuracy']:.3f} | {r['mean_best_epoch']:.1f} | {r['mean_steps_run']:.1f} |")
    lines += ['', '## 配对比较与解释','',
        '每个差值比较同一题、同一种子。95% 描述性区间由 10000 次按题目配对 bootstrap 得到，先在五种子间平均正确性。没有将 5×798 个观测视为相互独立；区间不包含调参不确定性，未校正多重比较。','',
        '| 比较 | 平均差（百分点） | 95% 描述性区间 | 正向种子数/5 | seed 17 修正/新增错误 |',
        '|---|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['mean_difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] | {r['positive_seeds']}/5 | {r['seed17_fixed_errors']}/{r['seed17_new_errors']} |")
    lines += ['', 'O2−O1 对应训练数据分配；O3−O2 和 O4−O2 才是在相同混合题、采样顺序和更新预算下比较目标。注意力−独立用于检验排序目标与候选交互是否互补。',
        '独立 MLP 即使用 O3/O4 训练，推理时每个候选仍独立打分；候选之间的相对关系仅进入训练损失。注意力模型则在推理时也读取同题其他候选。两者是不同层面的比较机制。','']
    for r in paired[:6]:
        qualifier = '描述性区间包含零，当前证据不足以确认稳定差异' if r['bootstrap_ci95_low']<=0<=r['bootstrap_ci95_high'] else '描述性区间不含零，方向仍需独立测试确认'
        lines.append(f"- {r['comparison']}：{r['mean_difference_pp']:+.3f} 个百分点；{qualifier}。")
    for r in interactions:
        lines.append(f"- 改用 {r['objective']} 后，相对于混合 BCE 的注意力优势变化为 {r['attention_advantage_change_vs_mixed_bce_pp']:+.3f} 个百分点，描述性区间 [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}]；这是差值之差，不能仅凭单个模型最高分判断互补性。")
    lines += ['', '## 诊断分组','',
        '下面依据已核对的官方正确性标签进行诊断聚合，不是独立官方 benchmark 分数。全对题占验证集 47.49%，任意选择都正确，因此混合题结果能更直接反映区分能力。Top-2 是前两名包含正确候选的比例，只是使用标签的诊断上界，不是可提交的选择准确率。','',
        '| 组 | 混合题 419 题 % | 仅一正确 117 题 % | 两正确 123 题 % | 三正确 179 题 % | Top-2 包含正确 % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in diagnostics:
        lines.append(f"| {r['group']} | {r['mixed_accuracy']:.3f} | {r['one_correct_accuracy']:.3f} | {r['two_correct_accuracy']:.3f} | {r['three_correct_accuracy']:.3f} | {r['top2_contains_correct_percent']:.3f} |")
    lines += ['', '## 验证与复现记录','',
        f"- 全部 BCE 与上一轮基线比较：{sum(r['identical_predictions'] for r in baseline_replay)}/{len(baseline_replay)} 个选定种子预测完全一致，详见 baseline_replay.json。",
        '- 损失检查：手算公式、正负候选梯度方向、有限差分梯度、候选置换、排名损失整体平移不变、极端分数、全对题零损失/排除、无正确候选拒绝、采样对齐均通过。',
        '- 每次最终检查点检查候选排列等变、batch 大小独立、独立模型不受其他候选改变影响；新进程加载后的预测与保存前完全一致。',
        f"- 原有文件 SHA256 复核 {integrity['files_checked']} 个，变更 {len(integrity['changed'])} 个，包括上一轮全部实验记录。",
        f"- 环境：{env['processor']}；Python {env['python'].split()[0]}，PyTorch {env['torch']}，NumPy {env['numpy']}。CPU 串行实验，PyTorch intra-op 4 / inter-op 1，BLAS 1。",
        f"- 48 次训练累计 {train_seconds:.1f} 秒（{train_seconds/60:.1f} 分钟，含逐轮验证）；独立新进程预测和评分累计 {eval_seconds:.1f} 秒。额外预处理、完整性检查、报告与交付检查未包含。",'',
        '| 组 | 参数量 | 资产 MiB（解包后） | 训练秒/次均值 | 验证预测+评分秒/次均值 |',
        '|---|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['group']} | {r['parameters']} | {r['asset_mib']:.3f} | {r['mean_train_seconds']:.2f} | {r['mean_validation_wall_seconds']:.2f} |")
    lines += ['', '计时覆盖验证集 798 题，含新进程启动、导入、数据和资产加载；没有在官方 Xeon 上测试，不据此声称通过官方 30 秒资格检查。每个单模型资产均小于 16 MiB。','',
        '## 结论边界','',
        '- 本轮检验固定输入、固定结构下的目标选择；没有加入题目文本或更换 embedding。',
        '- 同一验证集曾用于上一轮选择，本轮又用于学习率、早停和模型选择；结果是开发证据，不能作为无偏泛化成绩。没有生成测试预测或计算测试分数。',
        '- 两个学习率与当前正则化范围只构成有限预算实验；负结果不代表某种目标普遍无效。O4 的集合概率目标也不是 top-1 准确率的完全等价替代。',
        '- 本轮未比较置信度校准；特别是排序分数不应直接当作单候选正确概率。',
        '- 新增实验代码由 Codex AI 辅助编写并运行；用户需审阅理解，官方解析器、评分器、数据均未改动。','',
        '## 复现','', '在 code 目录执行；当前解释器为 D:/anaconda/envs/mp1/python.exe。完整重训请使用新输出目录。','',
        '```powershell', "$env:PYTHONDONTWRITEBYTECODE='1'",
        'python -B objective_experiments_20261017/check_objectives.py',
        'python -B objective_experiments_20261017/run_suite.py --output objective_experiments_20261017/reproduction',
        'python -B objective_experiments_20261017/summarize.py --output objective_experiments_20261017/reproduction',
        'python -B objective_experiments_20261017/verify_delivery.py --output objective_experiments_20261017/reproduction',
        '```','',
        '核心产物：summary.csv / runs.csv 为汇总与全部训练；paired_comparisons.json 与 interactions.json 为配对分析；每个 runs 子目录保存 history、检查点、官方预测/评分、耗时和哈希；selected_bundle 是固定 seed 17 的可独立加载模型。源文件变化会阻止复用训练缓存。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected':selection,'summary':list(summary.values())},indent=2))


if __name__=='__main__':
    main()
