"""Eight-cell summaries, paired factorial contrasts and descriptive interaction plot."""
import runtime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent
HEADS=(1,2,4,8)
BASE='include_h4'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def group(mask,heads):
    return f'{mask}_h{heads}'


def contrast_specs():
    specs=[]
    for heads in HEADS:
        specs.append(('mask_at_head',f'exclude - include at h{heads}',
                      {group('exclude',heads):1,group('include',heads):-1}))
    for mask in ('include','exclude'):
        for heads in (1,2,8):
            specs.append(('head_within_mask',f'h{heads} - h4 within {mask}',
                          {group(mask,heads):1,group(mask,4):-1}))
    weights={group(mask,heads):(.25 if mask=='exclude' else -.25)
             for mask in ('include','exclude') for heads in HEADS}
    specs.append(('mask_main','exclude - include averaged over heads',weights))
    for heads in (1,2,8):
        specs.append(('head_main',f'h{heads} - h4 averaged over masks',
                      {group('include',heads):.5,group('exclude',heads):.5,
                       'include_h4':-.5,'exclude_h4':-.5}))
        specs.append(('interaction',f'mask effect at h{heads} - mask effect at h4',
                      {group('exclude',heads):1,group('include',heads):-1,
                       'exclude_h4':-1,'include_h4':1}))
    return specs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args(); output=args.output.resolve()
    if args.plot_only:
        plot(output); return
    protocol=read(HERE/'protocol.json')
    runs,all_runs=read(output/'selected_runs.json'),read(output/'all_runs.json')
    assert len(runs)==40 and len(all_runs)==48
    labels=np.load(output/'validation-labels.npy',allow_pickle=False)
    train_labels=np.load(output/'train-labels.npy',allow_pickle=False)
    assert np.array_equal(labels,np.load(HERE.parent/'capacity_experiments_20261017/results/validation-labels.npy'))
    variants=list(protocol['variants'])
    hits,picks,summary,curve_rows={},{},{},[]
    expected_ids=None
    for variant in variants:
        rows=sorted([r for r in runs if r['variant']==variant],key=lambda r:protocol['seeds'].index(r['seed']))
        assert [r['seed'] for r in rows]==protocol['seeds']
        matrices,choices=[],[]
        for r in rows:
            folder=HERE/r['run_dir']
            prediction=read(folder/'validation-predictions.json')
            official=read(folder/'validation-score.json')
            if expected_ids is None: expected_ids=prediction['ids']
            assert prediction['ids']==expected_ids and prediction['split']=='validation'
            choice=np.asarray(prediction['selected'])
            hit=labels[np.arange(len(labels)),choice]
            assert hit.sum()==official['correct']==r['correct'] and official['accuracy']==r['accuracy']
            assert np.array_equal(np.load(folder/'validation-logits.npy').argmax(1),choice)
            train_pred=read(folder/'train-predictions.json')['selected']
            train_score=read(folder/'train-score.json')
            assert train_labels[np.arange(len(train_labels)),train_pred].sum()==train_score['correct']
            assert train_score['accuracy']==r['train_accuracy']
            assert r['parameters']==272001
            matrices.append(hit); choices.append(choice)
            for h in read(folder/'history.json'):
                curve_rows.append({'variant':variant,'seed':r['seed'],'epoch':h['budget_epoch'],
                    'train_eval_bce':h['train_eval']['bce'],'validation_eval_bce':h['validation_eval']['bce'],
                    'official_validation_accuracy':h['official_accuracy'],
                    'train_accuracy_diagnostic':h['train_eval']['selection_accuracy_diagnostic']})
        hits[variant],picks[variant]=np.asarray(matrices),np.asarray(choices)
        a=np.asarray([r['accuracy'] for r in rows])
        avg=lambda key:float(np.mean([r[key] for r in rows]))
        diag=lambda key:float(np.mean([r['attention_diagnostics'][key] for r in rows]))
        summary[variant]={'variant':variant,'num_heads':rows[0]['num_heads'],
            'exclude_self':rows[0]['exclude_self'],'mean_accuracy':float(a.mean()),
            'total_correct_across_seeds':sum(r['correct'] for r in rows),
            'std_accuracy':float(a.std(ddof=1)),'other_four_seeds_mean_accuracy':float(a[1:].mean()),
            'seed17_accuracy':float(a[0]),'learning_rate':rows[0]['learning_rate'],'parameters':272001,
            'train_accuracy':avg('train_accuracy'),'accuracy_gap_pp':avg('accuracy_gap_pp'),
            'train_bce':float(np.mean([r['train_eval']['bce'] for r in rows])),
            'validation_bce':avg('best_validation_loss'),'eval_bce_gap':avg('eval_bce_gap'),
            'mixed_accuracy_diagnostic':float(100*hits[variant][:,labels.sum(1)<4].mean()),
            'one_correct_accuracy_diagnostic':float(100*hits[variant][:,labels.sum(1)==1].mean()),
            'epoch10_validation_accuracy':float(np.mean([r['epoch10_metrics']['official_accuracy'] for r in rows])),
            'mean_best_epoch':avg('best_epoch'),'mean_epochs_run':avg('epochs_run'),
            'mean_self_attention_weight':diag('mean_self_attention_weight'),
            'mean_normalized_attention_entropy':diag('mean_normalized_attention_entropy'),
            'mean_train_seconds':avg('training_seconds'),
            'mean_validation_wall_seconds':avg('validation_total_wall_seconds'),
            'asset_mib':rows[0]['uncompressed_asset_bytes']/1024**2}
    write(output/'summary.json',list(summary.values())); csv_write(output/'summary.csv',list(summary.values()))
    csv_write(output/'learning_curves.csv',curve_rows)
    fields=['variant','seed','learning_rate','accuracy','correct','train_accuracy','accuracy_gap_pp','eval_bce_gap',
            'best_epoch','epochs_run','parameters','training_seconds','validation_total_wall_seconds','train_scoring_wall_seconds','run_dir']
    csv_write(output/'runs.csv',[{k:r[k] for k in fields} for r in all_runs])
    boot=np.random.default_rng(20261017).integers(0,len(labels),size=(10000,len(labels)))
    contrasts=[]
    specs=contrast_specs()
    for variant in variants:
        if variant!=BASE:
            specs.append(('cell_vs_original',f'{variant} - {BASE}',{variant:1,BASE:-1}))
    for kind,label,weights in specs:
        d=sum(weight*hits[g] for g,weight in weights.items())
        seed_deltas=100*d.mean(1)
        distribution=100*d.mean(0)[boot].mean(1)
        contrasts.append({'kind':kind,'contrast':label,'difference_pp':float(seed_deltas.mean()),
            'bootstrap_ci95_low':float(np.quantile(distribution,.025)),
            'bootstrap_ci95_high':float(np.quantile(distribution,.975)),
            'positive_seeds':int((seed_deltas>0).sum()),'seed_differences_pp':seed_deltas.tolist(),
            'other_four_seeds_difference_pp':float(seed_deltas[1:].mean())})
    write(output/'factorial_contrasts.json',contrasts); csv_write(output/'factorial_contrasts.csv',contrasts)
    same_lr=[]
    for lr in protocol['learning_rates']:
        grid={}
        for variant in variants:
            r=next(r for r in all_runs if r['variant']==variant and r['seed']==17 and r['learning_rate']==lr)
            prediction=read(HERE/r['run_dir']/'validation-predictions.json')['selected']
            grid[variant]=labels[np.arange(len(labels)),prediction]
        for kind,label,weights in contrast_specs():
            delta=sum(weight*grid[g] for g,weight in weights.items())
            same_lr.append({'seed':17,'learning_rate':lr,'kind':kind,'contrast':label,'difference_pp':float(100*delta.mean())})
    write(output/'same_lr_seed17_contrasts.json',same_lr); csv_write(output/'same_lr_seed17_contrasts.csv',same_lr)
    repairs=[]
    for variant in variants:
        d=hits[variant]-hits[BASE]
        repairs.append({'variant':variant,'mean_changed_choices':float((picks[variant]!=picks[BASE]).sum(1).mean()),
            'mean_fixed_errors':float((d>0).sum(1).mean()),'mean_new_errors':float((d<0).sum(1).mean()),
            'mean_net_fixes':float(d.sum(1).mean())})
    csv_write(output/'repairs_vs_original.csv',repairs); write(output/'repairs_vs_original.json',repairs)
    # All runs have the same N: integer totals implement exact mean ties reliably.
    best=max(summary.values(),key=lambda r:r['total_correct_across_seeds'])
    selected=next(r for r in runs if r['variant']==best['variant'] and r['seed']==17)
    bundle=output/'selected_bundle'; bundle.mkdir(exist_ok=True)
    for p in (HERE/selected['run_dir']/'bundle').iterdir(): shutil.copy2(p,bundle/p.name)
    selection={'policy':protocol['final_selection'],'model':'attention','variant':best['variant'],
        'num_heads':best['num_heads'],'exclude_self':best['exclude_self'],'seed':17,
        'learning_rate':selected['learning_rate'],'source_run':selected['run_dir'],
        'validation_accuracy':selected['accuracy'],'five_seed_mean_accuracy':best['mean_accuracy'],
        'test_evaluated':False,'bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}}
    write(output/'selection.json',selection)
    old_root=HERE.parent/'capacity_experiments_20261017'
    old_runs=read(old_root/'results/selected_runs.json'); replay=[]
    for r in runs:
        if r['variant']!=BASE: continue
        prev=next(p for p in old_runs if p['model']=='attention' and p['variant']=='baseline' and p['seed']==r['seed'])
        same=read(old_root/prev['run_dir']/'validation-predictions.json')['selected']==read(HERE/r['run_dir']/'validation-predictions.json')['selected']
        replay.append({'seed':r['seed'],'identical_predictions':same,'same_learning_rate':r['learning_rate']==prev['learning_rate']})
    write(output/'baseline_replay.json',replay)
    report(output,protocol,summary,contrasts,same_lr,repairs,selection,all_runs,replay)
    print(json.dumps({'selection':selection,'summary':list(summary.values()),
                      'main_effects':[c for c in contrasts if c['kind'] in ('mask_main','head_main','interaction')]},indent=2))


def report(output,protocol,summary,contrasts,same_lr,repairs,selection,all_runs,replay):
    env=read(output/'environment.json'); integrity=read(output/'protected_verification.json')
    train_seconds=sum(r['training_seconds'] for r in all_runs)
    eval_seconds=sum(r['validation_total_wall_seconds']+r['train_scoring_wall_seconds'] for r in all_runs)
    lines=['# 注意力头数 × 是否允许关注自身：八组交叉实验','',
        '完成 8 个配置、48 次训练，其中 40 个结果进入五种子主比较。正式训练/验证准确率均由未修改的官方评分脚本计算。没有测试集预测或评分。','',
        f"本轮最高五种子均值为 **{selection['variant']}：{selection['five_seed_mean_accuracy']:.3f}%**。按预定规则保留该配置的 seed 17 单模型（{selection['validation_accuracy']:.3f}%），不挑选最高分种子或集成。",'']
    if selection['variant']!=BASE:
        win=next(c for c in contrasts if c['kind']=='cell_vs_original' and c['contrast']==f"{selection['variant']} - {BASE}")
        meaning='区间包含零，当前尚不能确认它稳定优于原模型' if win['bootstrap_ci95_low']<=0<=win['bootstrap_ci95_high'] else '描述性区间不含零，但仍需独立测试确认'
        lines += [f"相对原模型的均值差为 {win['difference_pp']:+.3f} 个百分点，95% 描述性区间 [{win['bootstrap_ci95_low']:+.3f}, {win['bootstrap_ci95_high']:+.3f}]；{meaning}。",'']
    lines += [
        '## 问题与控制','',
        '考察两个因素：头数 1/2/4/8，以及注意力是否允许将自身作为 key/value。排除自身时，只屏蔽 4×4 注意力矩阵对角线；query 仍由自身表示产生，自身残差连接也保留。','',
        '- 总隐藏宽度固定 128，每头维度依次 128/64/32/16。八组参数量均为 272001；同种子的初始权重张量逐项一致。',
        '- 保留原模型的一层注意力、FFN 256、评分头 64、LayerNorm 与残差连接；没有位置编码。',
        '- 所有组为全部题 BCE、dropout 0.1、AdamW weight decay 0.001、batch 256、梯度裁剪 1。',
        '- 训练/验证 4755/798 题，固定输入 4×1024。仅训练集拟合标准化，无题目分支、外部数据或额外预训练模型。',
        '- 每组 seed 17 搜索学习率 0.001 / 0.0003，再用选定学习率运行 29、43、71、101；最多 60 轮，每轮 19 次更新，patience 10。',
        '- 检查点按官方验证正确数、较低验证 BCE、较早轮次选择；学习率同规则，完全同分优先预设顺序。',
        '- 主比较是相同调参预算下各配置的效果，选中的学习率可能不同。因此另列全部已运行的 seed 17 同学习率交叉对照，它只有一个种子，不能替代五种子证据。','',
        '## 官方验证结果','',
        '准确率为五种子均值 ± 样本标准差。原模型为允许自身、4 个头。','',
        '| 头数 | 每头维度 | 允许自身 % | 排除自身 % | 排除−允许（百分点） |','|---:|---:|---:|---:|---:|']
    for h in HEADS:
        a,b=summary[group('include',h)],summary[group('exclude',h)]
        lines.append(f"| {h} | {128//h} | {a['mean_accuracy']:.3f} ± {a['std_accuracy']:.3f} | {b['mean_accuracy']:.3f} ± {b['std_accuracy']:.3f} | {b['mean_accuracy']-a['mean_accuracy']:+.3f} |")
    lines += ['', '| 配置 | LR | 其余四种子均值 % | seed 17 % | 最佳轮次均值 | 第10轮均值 % |','|---|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['variant']} | {r['learning_rate']} | {r['other_four_seeds_mean_accuracy']:.3f} | {r['seed17_accuracy']:.3f} | {r['mean_best_epoch']:.1f} | {r['epoch10_validation_accuracy']:.3f} |")
    lines += ['', f"![Factor comparison]({(output/'factorial_accuracy.png').as_posix()})",'',
        '图中误差线是五种子样本标准差，不是独立测试置信区间。','',
        '## 两因素及交互作用','',
        '按题目配对 bootstrap 10000 次，对五种子正确性先求平均；没有将 5×798 个观测当作独立样本。所有区间均为条件于已选择模型的探索性区间，不包含调参不确定性，未做多重比较校正。','',
        '主效应将另一个因素的各水平等权平均。交互作用以四头为参照，计算“某头数下排除自身的效果−四头下排除自身的效果”；它判断掩码效果是否随头数改变。','',
        '| 对比 | 差值（百分点） | 95% 描述性区间 | 正向种子数/5 | 其余四种子差值 |','|---|---:|---:|---:|---:|']
    for r in contrasts:
        if r['kind'] not in ('mask_at_head','mask_main','head_main','interaction'): continue
        lines.append(f"| {r['contrast']} | {r['difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] | {r['positive_seeds']}/5 | {r['other_four_seeds_difference_pp']:+.3f} |")
    lines += ['']
    for r in contrasts:
        if r['kind'] not in ('mask_main','head_main','interaction'): continue
        text='区间包含零，当前证据不足以确认稳定差异' if r['bootstrap_ci95_low']<=0<=r['bootstrap_ci95_high'] else '描述性区间不含零，仍需冻结方案后的独立测试确认'
        lines.append(f"- {r['contrast']}：{r['difference_pp']:+.3f} 个百分点；{text}。")
    lines += ['', '各掩码下头数相对四头的完整比较也保存在 factorial_contrasts.csv。下表补充各单元相对原模型的比较；其中最佳单元由本验证集选出，不能将其区间视为未经过选择的显著性检验。','',
        '| 配置−原模型 | 差值（百分点） | 95% 描述性区间 |','|---|---:|---:|']
    for r in contrasts:
        if r['kind']=='cell_vs_original':
            lines.append(f"| {r['contrast']} | {r['difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] |")
    lines += ['', '## 同学习率的 seed 17 对照','',
        '以下全部来自既有搜索运行，没有追加训练。先列各配置的两个学习率结果；均衡主效应与交互作用点估计随后给出，完整各头数对照保存在 same_lr_seed17_contrasts.csv。','',
        '| 配置 | LR=0.001 准确率 % | LR=0.0003 准确率 % |','|---|---:|---:|']
    for variant in protocol['variants']:
        values=[next(r['accuracy'] for r in all_runs if r['variant']==variant and r['seed']==17 and r['learning_rate']==lr)
                for lr in protocol['learning_rates']]
        lines.append(f'| {variant} | {values[0]:.3f} | {values[1]:.3f} |')
    lines += ['',
        '| 对比 | LR=0.001 差值（百分点） | LR=0.0003 差值（百分点） |','|---|---:|---:|']
    for r in same_lr:
        if r['learning_rate']!=.001 or r['kind'] not in ('mask_main','head_main','interaction'): continue
        other=next(x for x in same_lr if x['learning_rate']==.0003 and x['contrast']==r['contrast'])
        lines.append(f"| {r['contrast']} | {r['difference_pp']:+.3f} | {other['difference_pp']:+.3f} |")
    lines += ['', '## 错误变化与拟合差距','',
        '修正/新增错误按同种子的原四头模型比较，每个数字为五种子平均题数。更换两个都正确的候选不改变准确率。混合题（419 题）、单正确题（117 题）的结果是官方标签规则下的诊断聚合，不是额外 benchmark。','',
        '| 配置 | 修正错误 | 新增错误 | 净修正 | 混合题准确率 % | 单正确题准确率 % |','|---|---:|---:|---:|---:|---:|']
    for r in repairs:
        s=summary[r['variant']]
        lines.append(f"| {r['variant']} | {r['mean_fixed_errors']:.1f} | {r['mean_new_errors']:.1f} | {r['mean_net_fixes']:+.1f} | {s['mixed_accuracy_diagnostic']:.3f} | {s['one_correct_accuracy_diagnostic']:.3f} |")
    lines += ['', '| 配置 | 训练准确率 % | 训练−验证差距（百分点） | 训练 BCE | 验证 BCE | 平均自身注意力 | 归一化熵 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['variant']} | {r['train_accuracy']:.3f} | {r['accuracy_gap_pp']:.3f} | {r['train_bce']:.4f} | {r['validation_bce']:.4f} | {r['mean_self_attention_weight']:.4f} | {r['mean_normalized_attention_entropy']:.4f} |")
    lines += ['', '训练和验证均在同一检查点的 eval 模式测量。注意力熵除以允许 key 数的对数（允许自身为 log4，排除自身为 log3），再对题目、头与 query 求平均。权重和熵仅描述读取分布，不能直接证明因果机制。','',
        '## 验证与成本','',
        f"- 原模型复现：{sum(r['identical_predictions'] for r in replay)}/{len(replay)} 个选定种子的预测与上一轮完全一致。",
        '- 八组结构均通过参数量/初始权重一致、显式 QK/softmax/V 参考计算、对角掩码、有限梯度、排列等变、batch 独立和跨候选依赖检查。',
        '- 每个最终检查点由新进程加载并通过官方评分，与训练时选中的预测逐项一致。完整注意力权重单独保存于运行目录，不属于推理资产。',
        f"- 原有文件哈希复核 {integrity['files_checked']} 个，变更 {len(integrity['changed'])} 个。",
        f"- 环境：{env['processor']}，Python {env['python'].split()[0]}、PyTorch {env['torch']}、NumPy {env['numpy']}；CPU 串行实验，PyTorch intra-op4、inter-op1，BLAS1。",
        f'- 48 次训练累计 {train_seconds:.1f} 秒（{train_seconds/60:.1f} 分钟，含逐轮训练/验证诊断）；新进程验证预测、验证评分和训练评分累计 {eval_seconds:.1f} 秒。预处理、检查、注意力导出和最终交付检查另计。','',
        '| 配置 | 资产 MiB | 训练秒/次均值 | 完整验证预测+评分秒/次均值 |','|---|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['variant']} | {r['asset_mib']:.3f} | {r['mean_train_seconds']:.2f} | {r['mean_validation_wall_seconds']:.2f} |")
    lines += ['', '本机计时含验证集 798 题的新进程启动、导入、输入与模型加载；未在官方 Xeon 上运行，不能据此认证官方 30 秒限制。所有单模型完整资产小于 16 MiB。','',
        '## 结论边界与复现','',
        '- 本轮固定总宽度、训练目标与容量，检验的是头数与自身 key/value 通路；头数也会改变每头宽度及注意力 dropout 的组织方式，不能只解释为抽象“比较角度”的数量。',
        '- 验证集已经多次用于实验设计、调参和早停；五种子与配对区间不消除选择偏差。排名第一的配置仍需冻结后在独立数据上验证。',
        '- 本轮不包含第二层注意力、额外候选独立 MLP 或相似度形式的变化，因此不能推断这些方向无效。',
        '- 新增代码由 Codex AI 辅助编写并运行；官方数据、解析器、评分器和此前全部实验保持原样。','',
        '从 code 目录执行，训练解释器为 D:/anaconda/envs/mp1/python.exe；完整重训使用新输出目录。','',
        '```powershell',"$env:PYTHONDONTWRITEBYTECODE='1'",
        'python -B attention_factors_20261017/check_models.py',
        'python -B attention_factors_20261017/run_suite.py --output attention_factors_20261017/reproduction',
        'python -B attention_factors_20261017/summarize.py --output attention_factors_20261017/reproduction',
        'python -B attention_factors_20261017/verify_delivery.py --output attention_factors_20261017/reproduction',
        '```','',
        '绘图另用装有 matplotlib 的 Python 执行 summarize.py --plot-only --output <结果目录>。summary.csv / runs.csv 保存主结果和全部搜索；factorial_contrasts.json 保存因素与交互分析；learning_curves.csv 保存所有主运行逐轮指标；selected_bundle 是固定 seed17 的独立可加载模型。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def plot(output):
    import platform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows={r['variant']:r for r in read(output/'summary.json')}
    fig,ax=plt.subplots(figsize=(8.5,5))
    x=np.arange(4)
    for mask,label,color,marker in [('include','Self included','#4273ad','o'),('exclude','Self excluded','#b35b49','s')]:
        cells=[rows[group(mask,h)] for h in HEADS]
        ax.errorbar(x,[r['mean_accuracy'] for r in cells],yerr=[r['std_accuracy'] for r in cells],
                    marker=marker,color=color,label=label,capsize=4,linewidth=1.6)
    ax.axhline(rows[BASE]['mean_accuracy'],color='gray',linestyle='--',alpha=.5,label='Original 4-head mean')
    ax.set_xticks(x,[str(h) for h in HEADS]); ax.set_xlabel('Number of attention heads (total width = 128)')
    ax.set_ylabel('Official validation accuracy (%)')
    ax.set_title('Attention heads × self inclusion; mean ± sample SD over five seeds')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    fig.savefig(output/'factorial_accuracy.png',dpi=180); fig.savefig(output/'factorial_accuracy.pdf'); plt.close(fig)
    write(output/'plot_environment.json',{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__})
    print('Saved factorial_accuracy.png and factorial_accuracy.pdf')


if __name__=='__main__': main()
