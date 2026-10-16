"""Summarize official scores, clean learning curves and paired ablations."""
import runtime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.plot_only:
        plot_curves(output)
        return
    protocol=read(HERE/'protocol.json')
    runs,all_runs=read(output/'selected_runs.json'),read(output/'all_runs.json')
    assert len(runs)==50 and len(all_runs)==60
    seeds=protocol['seeds']
    labels=np.load(output/'validation-labels.npy',allow_pickle=False)
    train_labels=np.load(output/'train-labels.npy',allow_pickle=False)
    groups=[f'{m}__{v}' for m in protocol['models'] for v in protocol['variants']]
    correct,summary,curves={},{},[]
    for group in groups:
        rows=sorted([r for r in runs if r['group']==group],key=lambda r:seeds.index(r['seed']))
        assert [r['seed'] for r in rows]==seeds
        hits=[]
        for r in rows:
            folder=HERE/r['run_dir']
            pred=read(folder/'validation-predictions.json')
            score=read(folder/'validation-score.json')
            picks=np.asarray(pred['selected'])
            hit=labels[np.arange(len(labels)),picks]
            assert hit.sum()==score['correct']==r['correct'] and score['accuracy']==r['accuracy']
            assert np.array_equal(np.load(folder/'validation-logits.npy').argmax(1),picks)
            train_pred=read(folder/'train-predictions.json')
            train_score=read(folder/'train-score.json')
            train_hit=train_labels[np.arange(len(train_labels)),train_pred['selected']]
            assert train_hit.sum()==train_score['correct'] and train_score['accuracy']==r['train_accuracy']
            hits.append(hit)
            for epoch in read(folder/'history.json'):
                curves.append({'group':group,'seed':r['seed'],'epoch':epoch['budget_epoch'],
                    'train_eval_bce':epoch['train_eval']['bce'],'validation_eval_bce':epoch['validation_eval']['bce'],
                    'train_mixed_bce':epoch['train_eval']['mixed_bce'],'validation_mixed_bce':epoch['validation_eval']['mixed_bce'],
                    'train_accuracy_diagnostic':epoch['train_eval']['selection_accuracy_diagnostic'],
                    'official_validation_accuracy':epoch['official_accuracy'],
                    'train_mixed_accuracy_diagnostic':epoch['train_eval']['mixed_selection_accuracy_diagnostic'],
                    'validation_mixed_accuracy_diagnostic':epoch['validation_eval']['mixed_selection_accuracy_diagnostic']})
        correct[group]=np.asarray(hits)
        avg=lambda key:float(np.mean([r[key] for r in rows]))
        metric=lambda split,key:float(np.mean([r[split][key] for r in rows]))
        accuracy=np.asarray([r['accuracy'] for r in rows])
        summary[group]={'group':group,'model':rows[0]['model'],'variant':rows[0]['variant'],
            'hidden':rows[0]['hidden'],'dropout':rows[0]['dropout'],'weight_decay':rows[0]['weight_decay'],
            'learning_rate':rows[0]['learning_rate'],'parameters':rows[0]['parameters'],
            'mean_accuracy':float(accuracy.mean()),'std_accuracy':float(accuracy.std(ddof=1)),
            'other_four_seeds_mean_accuracy':float(accuracy[1:].mean()),'seed17_accuracy':float(accuracy[0]),
            'train_accuracy':avg('train_accuracy'),'accuracy_gap_pp':avg('accuracy_gap_pp'),
            'train_bce':metric('train_eval','bce'),'validation_bce':metric('validation_eval','bce'),
            'eval_bce_gap':avg('eval_bce_gap'),
            'train_mixed_accuracy':metric('train_eval','mixed_selection_accuracy_diagnostic'),
            'validation_mixed_accuracy':metric('validation_eval','mixed_selection_accuracy_diagnostic'),
            'mixed_accuracy_gap_pp':metric('train_eval','mixed_selection_accuracy_diagnostic')-metric('validation_eval','mixed_selection_accuracy_diagnostic'),
            'train_mixed_bce':metric('train_eval','mixed_bce'),'validation_mixed_bce':metric('validation_eval','mixed_bce'),
            'epoch10_train_bce':float(np.mean([r['epoch10_metrics']['train_eval']['bce'] for r in rows])),
            'epoch10_validation_bce':float(np.mean([r['epoch10_metrics']['validation_eval']['bce'] for r in rows])),
            'epoch10_validation_accuracy':float(np.mean([r['epoch10_metrics']['official_accuracy'] for r in rows])),
            'mean_best_epoch':avg('best_epoch'),'mean_epochs_run':avg('epochs_run'),
            'decay_only_factor_at_best':float(np.mean([(1-r['learning_rate']*r['weight_decay'])**r['best_optimizer_steps'] for r in rows])),
            'mean_train_seconds':avg('training_seconds'),'mean_validation_wall_seconds':avg('validation_total_wall_seconds'),
            'asset_mib':rows[0]['uncompressed_asset_bytes']/1024**2}
    csv_write(output/'summary.csv',list(summary.values()))
    write(output/'summary.json',list(summary.values()))
    csv_write(output/'learning_curves.csv',curves)
    keys=['group','seed','learning_rate','accuracy','train_accuracy','accuracy_gap_pp','eval_bce_gap','best_epoch',
          'epochs_run','parameters','training_seconds','validation_total_wall_seconds','train_scoring_wall_seconds','run_dir']
    csv_write(output/'runs.csv',[{k:r[k] for k in keys} for r in all_runs])
    boot=np.random.default_rng(20261017).integers(0,len(labels),size=(10000,len(labels)))
    paired=[]
    lr_matched=[]
    for model in protocol['models']:
        baseline=f'{model}__baseline'
        baseline_lr=summary[baseline]['learning_rate']
        fixed_base=next(r for r in all_runs if r['group']==baseline and r['seed']==17 and r['learning_rate']==baseline_lr)
        for variant in protocol['variants']:
            if variant=='baseline':
                continue
            group=f'{model}__{variant}'
            difference=correct[group]-correct[baseline]
            delta=100*difference.mean(1)
            distribution=100*difference.mean(0)[boot].mean(1)
            paired.append({'comparison':f'{group} - {baseline}','group':group,'baseline':baseline,
                'difference_pp':float(delta.mean()),'bootstrap_ci95_low':float(np.quantile(distribution,.025)),
                'bootstrap_ci95_high':float(np.quantile(distribution,.975)),
                'positive_seeds':int((delta>0).sum()),'seed_differences_pp':delta.tolist(),
                'other_four_seeds_difference_pp':float(delta[1:].mean()),
                'accuracy_gap_change_pp':summary[group]['accuracy_gap_pp']-summary[baseline]['accuracy_gap_pp'],
                'bce_gap_change':summary[group]['eval_bce_gap']-summary[baseline]['eval_bce_gap'],
                'mixed_accuracy_gap_change_pp':summary[group]['mixed_accuracy_gap_pp']-summary[baseline]['mixed_accuracy_gap_pp'],
                'parameter_ratio':summary[group]['parameters']/summary[baseline]['parameters'],
                'seed17_fixed_errors':int((difference[0]>0).sum()),'seed17_new_errors':int((difference[0]<0).sum())})
            matched=next(r for r in all_runs if r['group']==group and r['seed']==17 and r['learning_rate']==baseline_lr)
            lr_matched.append({'group':group,'seed':17,'learning_rate':baseline_lr,
                'baseline_accuracy':fixed_base['accuracy'],'variant_accuracy':matched['accuracy'],
                'difference_pp':matched['accuracy']-fixed_base['accuracy'],
                'baseline_best_epoch':fixed_base['best_epoch'],'variant_best_epoch':matched['best_epoch']})
    csv_write(output/'paired_comparisons.csv',paired)
    write(output/'paired_comparisons.json',paired)
    csv_write(output/'same_learning_rate_seed17.csv',lr_matched)
    best=max(summary.values(),key=lambda r:(r['mean_accuracy'],-r['parameters']))
    selected=next(r for r in runs if r['group']==best['group'] and r['seed']==17)
    bundle=output/'selected_bundle'
    bundle.mkdir(exist_ok=True)
    for p in (HERE/selected['run_dir']/'bundle').iterdir():
        shutil.copy2(p,bundle/p.name)
    selection={'policy':protocol['final_selection'],'model':selected['model'],'variant':selected['variant'],
        'group':best['group'],'seed':17,'learning_rate':selected['learning_rate'],
        'source_run':selected['run_dir'],'validation_accuracy':selected['accuracy'],
        'five_seed_mean_accuracy':best['mean_accuracy'],'test_evaluated':False,
        'bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}}
    write(output/'selection.json',selection)
    previous_root=HERE.parent/'objective_experiments_20261017'
    previous=read(previous_root/'results/selected_runs.json')
    replay=[]
    for r in runs:
        if r['variant']!='baseline':
            continue
        p=next(p for p in previous if p['model']==r['model'] and p['objective']=='bce_all' and p['seed']==r['seed'])
        before=read(previous_root/p['run_dir']/'validation-predictions.json')['selected']
        after=read(HERE/r['run_dir']/'validation-predictions.json')['selected']
        replay.append({'model':r['model'],'seed':r['seed'],'identical_predictions':before==after,
            'same_learning_rate':r['learning_rate']==p['learning_rate'],'previous_accuracy':p['accuracy'],'new_accuracy':r['accuracy']})
    write(output/'baseline_replay.json',replay)
    report(output,protocol,summary,paired,lr_matched,selection,all_runs,replay)
    print(json.dumps({'selection':selection,'summary':list(summary.values())},indent=2))


def report(output,protocol,summary,paired,lr_matched,selection,all_runs,replay):
    env=read(output/'environment.json')
    integrity=read(output/'protected_verification.json')
    train_seconds=sum(r['training_seconds'] for r in all_runs)
    eval_seconds=sum(r['validation_total_wall_seconds']+r['train_scoring_wall_seconds'] for r in all_runs)
    names={'baseline':'基线 h128','width64':'缩小至 h64','width32':'缩小至 h32',
           'dropout30':'dropout 0.3','decay01':'weight decay 0.01'}
    lines=['# 缩小模型与增强正则化：泛化瓶颈实验','',
        '完成独立 MLP / 候选注意力 × 5 种配置。每组 seed 17 搜索两档学习率，再训练其余四个种子；共 60 次训练，50 个结果进入五种子汇总。每个检查点的训练和验证准确率都由原版官方评分脚本核对。','',
        f"本轮五种子均值最高：**{selection['group']}，{selection['five_seed_mean_accuracy']:.3f}%**。按预定规则保留 seed 17 单模型，验证准确率 **{selection['validation_accuracy']:.3f}%**。",'',
        '## 实验问题与控制','',
        '假设：若当前误差主要受过拟合影响，减少模型容量或增强正则化可能降低同一检查点的训练—验证差距，同时提高验证选择准确率。差距缩小但验证准确率不升，只能说明拟合行为改变，不能单独证明泛化改善。','',
        '- 每题 4×1024 固定向量；训练 4755 题、验证 798 题；训练集拟合归一化，无题目文本分支。',
        '- 全部配置统一用全部训练题的 BCE，允许多正确候选。这是对容量和正则化的控制实验，没有与上一轮独立 MLP 的混合题 BCE 同时组合。',
        '- 基线：隐藏维度 128，dropout 0.1，weight decay 0.001。每个变体只改变宽度、dropout 或衰减中的一项。',
        '- h64/h32 改变输入投影和注意力/FFN 宽度；评分头中间层固定 64，注意力头数固定 4。dropout 0.3 同时作用于原结构各 dropout 位置，包含注意力权重 dropout。',
        '- AdamW，batch 256，梯度裁剪 1；每轮完整打乱 4755 题，共 19 次更新。每组最多 60 轮，patience 10；种子 17、29、43、71、101。',
        '- 每组在 seed 17 搜索学习率 0.001 / 0.0003；按官方验证正确数、验证 BCE、预设顺序选择。检查点按正确数、BCE、较早轮次选择。其余四种子也使用验证集早停。',
        '- 主表比较相同搜索预算下调参后的配置，因此部分组学习率不同。另列已有 seed 17 搜索记录中匹配基线学习率的结果；它只有一个种子，不能替代主实验。',
        '- 不使用测试集训练、选择、预测或评分；完整性检查可以检查其文件哈希和对齐。','',
        '## 验证准确率','',
        '表中为五种子均值 ± 样本标准差；正式分数均来自未修改的 score_predictions.py。','',
        '| 配置 | 独立 MLP % | 候选注意力 % | MLP 参数量 | 注意力参数量 |',
        '|---|---:|---:|---:|---:|']
    for variant in protocol['variants']:
        i,a=summary[f'independent__{variant}'],summary[f'attention__{variant}']
        lines.append(f"| {names[variant]} | {i['mean_accuracy']:.3f} ± {i['std_accuracy']:.3f} | {a['mean_accuracy']:.3f} ± {a['std_accuracy']:.3f} | {i['parameters']:,} | {a['parameters']:,} |")
    lines += ['', '## 相对各自基线的配对比较','',
        '同题、同种子配对；95% 描述性区间为 10000 次题目 bootstrap，先对五种子正确性求平均。区间条件于已选择模型，不包含超参数搜索或早停的不确定性，未做多重比较校正。','',
        '| 变体 | 准确率差（百分点） | 95% 描述性区间 | 正向种子数/5 | 准确率差距变化（百分点） | BCE 差距变化 | 参数比 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['group']} | {r['difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] | {r['positive_seeds']}/5 | {r['accuracy_gap_change_pp']:+.3f} | {r['bce_gap_change']:+.4f} | {r['parameter_ratio']:.3f} |")
    lines += ['', '准确率差距定义为训练准确率−验证准确率；BCE 差距定义为验证 BCE−训练 BCE。差距变化为变体−基线，负值表示差距缩小。','']
    for r in paired:
        if r['difference_pp']>0 and r['accuracy_gap_change_pp']<0:
            conclusion='验证均值上升且准确率差距缩小，与缓解过拟合的假设一致'
        elif r['difference_pp']<=0 and r['accuracy_gap_change_pp']<0:
            conclusion='差距缩小但验证均值没有上升，不能作为泛化改善的证据，需考虑欠拟合或优化变化'
        elif r['difference_pp']>0:
            conclusion='验证均值上升，但准确率差距未缩小，不能将收益简单归因于减轻过拟合'
        else:
            conclusion='本设置没有显示验证均值改善'
        uncertainty='区间包含零' if r['bootstrap_ci95_low']<=0<=r['bootstrap_ci95_high'] else '描述性区间不含零，仍需独立数据确认'
        lines.append(f"- **{r['group']}**：{conclusion}；{uncertainty}。")
    lines += ['', '## 同一检查点的泛化差距','',
        '以下为各自选定检查点上的五种子均值。训练集和验证集都在 eval 模式、关闭 dropout 后计算；没有把随机 minibatch 训练损失与验证损失直接相减。训练准确率也经过官方脚本评分，但它是拟合程度指标，不是泛化成绩。','',
        '| 组 | 训练准确率 % | 验证准确率 % | 准确率差距（百分点） | 训练 BCE | 验证 BCE | BCE 差距 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['group']} | {r['train_accuracy']:.3f} | {r['mean_accuracy']:.3f} | {r['accuracy_gap_pp']:.3f} | {r['train_bce']:.4f} | {r['validation_bce']:.4f} | {r['eval_bce_gap']:.4f} |")
    lines += ['', '全对题无法区分选择方法。下面单独报告至少一个错误候选的混合题：训练 2513 题、验证 419 题。它们是官方正确性标签的诊断聚合，不是额外官方 benchmark。','',
        '| 组 | 训练混合题准确率 % | 验证混合题准确率 % | 差距（百分点） | 训练混合 BCE | 验证混合 BCE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['group']} | {r['train_mixed_accuracy']:.3f} | {r['validation_mixed_accuracy']:.3f} | {r['mixed_accuracy_gap_pp']:.3f} | {r['train_mixed_bce']:.4f} | {r['validation_mixed_bce']:.4f} |")
    lines += ['', '## 共同更新次数与学习率检查','',
        '所有运行都至少经历 10 轮，因此第 10 轮（190 次更新）不受不同早停时点的影响。它并不保证各模型都已收敛，尤其小模型或强 dropout 可能需要更多更新。','',
        '| 组 | 学习率 | 第10轮训练 BCE | 第10轮验证 BCE | 第10轮验证准确率 % | 最佳轮次均值 | 其余四种子最佳均值 % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['group']} | {r['learning_rate']} | {r['epoch10_train_bce']:.4f} | {r['epoch10_validation_bce']:.4f} | {r['epoch10_validation_accuracy']:.3f} | {r['mean_best_epoch']:.1f} | {r['other_four_seeds_mean_accuracy']:.3f} |")
    lines += ['', '匹配基线学习率的 seed 17 对照（各自按既定规则选择检查点；仅一个种子）：','',
        '| 组 | 相同学习率 | 基线准确率 % | 变体准确率 % | 差值（百分点） |','|---|---:|---:|---:|---:|']
    for r in lr_matched:
        lines.append(f"| {r['group']} | {r['learning_rate']} | {r['baseline_accuracy']:.3f} | {r['variant_accuracy']:.3f} | {r['difference_pp']:+.3f} |")
    lines += ['', '权重衰减的额外尺度检查：固定学习率下，忽略梯度更新，仅累乘 AdamW 的直接衰减项得到 (1−lr×weight_decay)^steps。它不是实际权重范数变化，但可判断本轮衰减设置的直接强度。']
    for model in protocol['models']:
        base=summary[f'{model}__baseline']['decay_only_factor_at_best']
        strong=summary[f'{model}__decay01']['decay_only_factor_at_best']
        lines.append(f'- {model}：最佳检查点的衰减项乘积均值，基线 {base:.6f}，weight decay 0.01 为 {strong:.6f}。若两者都接近 1，负结果只能说明这个有限强度设置未改善，不能排除更强衰减的作用。')
    lines += ['', '## 学习曲线','',
        '图展示 seed 17、每组所选学习率的完整轨迹。上排实线为验证 BCE、虚线为关闭 dropout 的训练 BCE；下排为官方验证准确率，圆点标出选定检查点。只画一个种子以避免早停后不同运行数量造成均值曲线偏差；五种子结论以表格为准。','',
        f"![Learning curves]({(output/'learning_curves.png').as_posix()})",'',
        '全部五种子逐轮数值保存在 learning_curves.csv；原始每个运行还保留初始化指标、逐轮历史和随机训练损失。','',
        '## 验证、成本与边界','',
        f"- 基线复现：{sum(r['identical_predictions'] for r in replay)}/{len(replay)} 个选定种子与上一轮全部题 BCE 的预测完全一致。",
        '- 结构检查涵盖 h128/h64/h32、dropout、有限梯度、候选排列等变、batch 独立、独立模型的候选独立性；诊断计算不修改参数或随机数状态。',
        '- 每个最终验证检查点由新进程加载推理并独立官方评分，预测与保存前逐项一致；全部推理资产均小于 16 MiB。',
        f"- 原有文件复核 {integrity['files_checked']} 个，变更 {len(integrity['changed'])} 个；包括前两轮全部代码和结果。",
        f"- 环境：{env['processor']}；Python {env['python'].split()[0]}，PyTorch {env['torch']}，NumPy {env['numpy']}。CPU 串行训练与评分，PyTorch intra-op 4、inter-op 1，BLAS 1。",
        f"- 60 次训练累计 {train_seconds:.1f} 秒（{train_seconds/60:.1f} 分钟，含逐轮训练/验证诊断）；独立验证预测、验证评分和训练评分累计 {eval_seconds:.1f} 秒。检查、预处理、导出、报告和最终交付验证另计。",'',
        '| 组 | 资产 MiB | 训练秒/次均值 | 完整验证预测+评分秒/次均值 |','|---|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['group']} | {r['asset_mib']:.3f} | {r['mean_train_seconds']:.2f} | {r['mean_validation_wall_seconds']:.2f} |")
    lines += ['', '计时包括验证集 798 题的新进程启动、导入、数据和模型加载；未在官方 Xeon 测试，不能据此认证官方 30 秒限制。','',
        '- 验证集已经多次用于搜索、早停和方案选择，配对区间只提供开发证据，不是独立测试显著性结论。',
        '- 损失差距也可能受数据难度、标签分布和概率校准影响；准确率与 BCE 不一定同步改善。缩小差距本身不是目标。',
        '- 当前实验只检查两个宽度、一个 dropout 强度、一个额外衰减值，以及有限学习率范围；负结果不能排除其他正则化或优化方案。',
        '- 若缩小模型后分数接近，只能描述本验证集上的观察；未预先定义非劣界限，因此不宣称统计意义上的性能等价。',
        '- 未检验数据规模学习曲线或更换表示，不能由本轮直接证明瓶颈就是 embedding 信息不足，也不能证明所有泛化误差都来自过拟合。',
        '- 训练/推理框架复用上一轮代码；新增测量、实验编排和报告由 Codex AI 辅助编写运行。无外部数据或预训练权重，官方 parser/scorer/data 未修改。','',
        '## 复现与文件','',
        '从 code 目录执行。当前训练解释器为 D:/anaconda/envs/mp1/python.exe；完整重训请使用新输出目录。','',
        '```powershell',"$env:PYTHONDONTWRITEBYTECODE='1'",
        'python -B capacity_experiments_20261017/check_models.py',
        'python -B capacity_experiments_20261017/run_suite.py --output capacity_experiments_20261017/reproduction',
        'python -B capacity_experiments_20261017/summarize.py --output capacity_experiments_20261017/reproduction',
        'python -B capacity_experiments_20261017/verify_delivery.py --output capacity_experiments_20261017/reproduction',
        '```','',
        '绘图另用已安装 matplotlib 的 Python 运行 summarize.py --plot-only --output <同一结果目录>；图像运行不涉及训练或选择。',
        'summary.csv 是五种子汇总；runs.csv 为全部 60 个运行；paired_comparisons.json 为配对结果；learning_curves.csv 为逐轮指标；selected_bundle 与 selection.json 保存固定 seed 17 单模型及来源。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def plot_curves(output):
    import platform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs=read(output/'selected_runs.json')
    variants=list(read(HERE/'protocol.json')['variants'])
    colors=dict(zip(variants,plt.get_cmap('tab10').colors))
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex='col')
    for col,model in enumerate(['independent','attention']):
        for variant in variants:
            r=next(r for r in runs if r['model']==model and r['variant']==variant and r['seed']==17)
            history=read(HERE/r['run_dir']/'history.json')
            x=[h['budget_epoch'] for h in history]
            c=colors[variant]
            axes[0,col].plot(x,[h['validation_eval']['bce'] for h in history],color=c,label=f'{variant}, lr={r["learning_rate"]}')
            axes[0,col].plot(x,[h['train_eval']['bce'] for h in history],color=c,linestyle='--',alpha=.7)
            axes[1,col].plot(x,[h['official_accuracy'] for h in history],color=c)
            axes[1,col].scatter([r['best_epoch']],[r['accuracy']],color=c,s=28,zorder=3)
        axes[0,col].set_title(model+' (seed 17)')
        axes[0,col].set_ylabel('BCE (solid: validation; dashed: train eval)')
        axes[0,col].legend(fontsize=8)
        axes[1,col].set_ylabel('Official validation accuracy (%)')
        axes[1,col].set_xlabel('Epoch (19 optimizer updates per epoch)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Capacity / regularization ablations: clean learning curves')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png',dpi=180)
    fig.savefig(output/'learning_curves.pdf')
    plt.close(fig)
    write(output/'plot_environment.json',{'python':platform.python_version(),'numpy':np.__version__,
        'matplotlib':matplotlib.__version__,'source':'selected_runs.json and saved history.json; no training or selection'})
    print('Saved learning_curves.png and learning_curves.pdf')


if __name__=='__main__':
    main()
