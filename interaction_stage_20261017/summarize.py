"""Main fixed-LR stage comparisons, fixed-update snapshots and LR sensitivity."""
import runtime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent
BASE='s0_after'
ORDERS={'s0_after':'A → F1','s1_before':'F1 → A','s2_before_keep_norm_after':'P1 → A → N1',
        's3_before_before':'F1 → F2 → A','s4_before_after':'F1 → A → F2','s5_after_after':'A → F1 → F2'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args(); output=args.output.resolve()
    if args.plot_only:
        plot(output); return
    protocol=read(HERE/'protocol.json')
    runs,all_runs=read(output/'selected_runs.json'),read(output/'all_runs.json')
    assert len(runs)==30 and len(all_runs)==36
    assert all(r['learning_rate']==protocol['main_learning_rate'] and r['role']=='main' for r in runs)
    assert all(r['epochs_run']==60 and r['optimizer_steps']==1140 for r in all_runs)
    labels=np.load(output/'validation-labels.npy',allow_pickle=False)
    train_labels=np.load(output/'train-labels.npy',allow_pickle=False)
    variants=list(protocol['variants'])
    times=['best']+[str(e) for e in protocol['fixed_epochs']]
    correct={t:{} for t in times}; choices={}; summaries={}; fixed=[]; curves=[]
    expected_ids=None
    for variant in variants:
        rows=sorted([r for r in runs if r['variant']==variant],key=lambda r:protocol['seeds'].index(r['seed']))
        assert [r['seed'] for r in rows]==protocol['seeds']
        matrices={t:[] for t in times}; picks=[]
        for r in rows:
            folder=HERE/r['run_dir']
            prediction=read(folder/'validation-predictions.json'); official=read(folder/'validation-score.json')
            if expected_ids is None: expected_ids=prediction['ids']
            assert prediction['ids']==expected_ids and prediction['split']=='validation'
            pick=np.asarray(prediction['selected']); hit=labels[np.arange(len(labels)),pick]
            assert hit.sum()==official['correct']==r['correct'] and official['accuracy']==r['accuracy']
            assert np.array_equal(np.load(folder/'validation-logits.npy').argmax(1),pick)
            train_pred=read(folder/'train-predictions.json')['selected']; train_score=read(folder/'train-score.json')
            assert train_labels[np.arange(len(train_labels)),train_pred].sum()==train_score['correct']
            assert train_score['accuracy']==r['train_accuracy']
            matrices['best'].append(hit); picks.append(pick)
            history=read(folder/'history.json'); assert len(history)==60
            for h in history:
                curves.append({'variant':variant,'seed':r['seed'],'epoch':h['budget_epoch'],
                    'train_eval_bce':h['train_eval']['bce'],'validation_eval_bce':h['validation_eval']['bce'],
                    'official_validation_accuracy':h['official_accuracy'],
                    'train_accuracy_diagnostic':h['train_eval']['selection_accuracy_diagnostic'],
                    'train_mixed_accuracy_diagnostic':h['train_eval']['mixed_selection_accuracy_diagnostic'],
                    'validation_mixed_accuracy_diagnostic':h['validation_eval']['mixed_selection_accuracy_diagnostic']})
            for epoch in protocol['fixed_epochs']:
                sub=folder/f'epoch_{epoch}'
                p=read(sub/'validation-predictions.json'); score=read(sub/'validation-score.json')
                assert p['ids']==expected_ids
                h=labels[np.arange(len(labels)),p['selected']]
                assert int(h.sum())==score['correct']==history[epoch-1]['official_correct']
                matrices[str(epoch)].append(h)
        for t in times: correct[t][variant]=np.asarray(matrices[t])
        choices[variant]=np.asarray(picks)
        acc=np.asarray([r['accuracy'] for r in rows]); avg=lambda key:float(np.mean([r[key] for r in rows]))
        summaries[variant]={'variant':variant,'order':ORDERS[variant],'parameters':rows[0]['parameters'],
            'mean_accuracy':float(acc.mean()),'std_accuracy':float(acc.std(ddof=1)),
            'total_correct_across_seeds':sum(r['correct'] for r in rows),
            'other_four_seeds_mean_accuracy':float(acc[1:].mean()),'seed17_accuracy':float(acc[0]),
            'train_accuracy':avg('train_accuracy'),'accuracy_gap_pp':avg('accuracy_gap_pp'),
            'train_bce':float(np.mean([r['train_eval']['bce'] for r in rows])),
            'validation_bce':avg('best_validation_loss'),'eval_bce_gap':avg('eval_bce_gap'),
            'mixed_accuracy_diagnostic':float(100*correct['best'][variant][:,labels.sum(1)<4].mean()),
            'mean_best_epoch':avg('best_epoch'),'best_after_epoch30_seeds':sum(r['best_epoch']>30 for r in rows),
            'mean_train_seconds_including_fixed_scoring':avg('training_seconds'),
            'mean_fixed_scoring_seconds_in_training':avg('fixed_epoch_scoring_seconds_in_training'),
            'mean_validation_wall_seconds':avg('validation_total_wall_seconds'),
            'asset_mib':rows[0]['uncompressed_asset_bytes']/1024**2}
        for epoch in protocol['fixed_epochs']:
            a=100*correct[str(epoch)][variant].mean(1)
            hs=[read(HERE/r['run_dir']/'history.json')[epoch-1] for r in rows]
            fixed.append({'variant':variant,'epoch':epoch,'optimizer_steps':epoch*19,
                'mean_accuracy':float(a.mean()),'std_accuracy':float(a.std(ddof=1)),
                'train_eval_bce':float(np.mean([h['train_eval']['bce'] for h in hs])),
                'validation_eval_bce':float(np.mean([h['validation_eval']['bce'] for h in hs]))})
    write(output/'summary.json',list(summaries.values())); csv_write(output/'summary.csv',list(summaries.values()))
    write(output/'fixed_epochs.json',fixed); csv_write(output/'fixed_epochs.csv',fixed)
    csv_write(output/'learning_curves.csv',curves)
    fields=['variant','role','seed','learning_rate','accuracy','correct','train_accuracy','accuracy_gap_pp',
            'best_epoch','epochs_run','optimizer_steps','parameters','training_seconds',
            'fixed_epoch_scoring_seconds_in_training','validation_total_wall_seconds','train_scoring_wall_seconds','run_dir']
    csv_write(output/'runs.csv',[{k:r[k] for k in fields} for r in all_runs])
    specs=[('primary',c) for c in protocol['primary_comparisons']]
    specs.extend(('capacity_secondary',f'{v} - {BASE}') for v in variants[3:])
    boot=np.random.default_rng(20261017).integers(0,len(labels),size=(10000,len(labels)))
    paired=[]
    for t in times:
        for kind,comparison in specs:
            a,b=comparison.split(' - '); d=correct[t][a]-correct[t][b]
            sd=100*d.mean(1); distribution=100*d.mean(0)[boot].mean(1)
            paired.append({'checkpoint':t,'kind':kind,'comparison':comparison,
                'difference_pp':float(sd.mean()),'bootstrap_ci95_low':float(np.quantile(distribution,.025)),
                'bootstrap_ci95_high':float(np.quantile(distribution,.975)),
                'positive_seeds':int((sd>0).sum()),'seed_differences_pp':sd.tolist(),
                'other_four_seeds_difference_pp':float(sd[1:].mean())})
    write(output/'paired_comparisons.json',paired); csv_write(output/'paired_comparisons.csv',paired)
    sensitivity=[]
    for variant in variants:
        r=next(r for r in all_runs if r['variant']==variant and r['role']=='sensitivity')
        base=next(r for r in all_runs if r['variant']==BASE and r['role']=='sensitivity')
        sensitivity.append({'variant':variant,'main_seed17_accuracy':summaries[variant]['seed17_accuracy'],
            'main_seed17_difference_vs_s0_pp':summaries[variant]['seed17_accuracy']-summaries[BASE]['seed17_accuracy'],
            'sensitivity_seed17_accuracy':r['accuracy'],'sensitivity_difference_vs_s0_pp':r['accuracy']-base['accuracy'],
            'sensitivity_best_epoch':r['best_epoch']})
    write(output/'sensitivity.json',sensitivity); csv_write(output/'sensitivity.csv',sensitivity)
    repairs=[]
    for v in variants:
        d=correct['best'][v]-correct['best'][BASE]
        repairs.append({'variant':v,'mean_changed_choices':float((choices[v]!=choices[BASE]).sum(1).mean()),
            'mean_fixed_errors':float((d>0).sum(1).mean()),'mean_new_errors':float((d<0).sum(1).mean()),
            'mean_net_fixes':float(d.sum(1).mean())})
    write(output/'repairs_vs_s0.json',repairs); csv_write(output/'repairs_vs_s0.csv',repairs)
    best=max(summaries.values(),key=lambda r:(r['total_correct_across_seeds'],-r['parameters']))
    selected=next(r for r in runs if r['variant']==best['variant'] and r['seed']==17)
    bundle=output/'selected_bundle'; bundle.mkdir(exist_ok=True)
    for p in (HERE/selected['run_dir']/'bundle').iterdir(): shutil.copy2(p,bundle/p.name)
    selection={'policy':protocol['final_selection'],'model':'attention','variant':best['variant'],'seed':17,
        'learning_rate':protocol['main_learning_rate'],'source_run':selected['run_dir'],
        'validation_accuracy':selected['accuracy'],'five_seed_mean_accuracy':best['mean_accuracy'],
        'test_evaluated':False,'sensitivity_used_for_selection':False,
        'bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}}
    write(output/'selection.json',selection)
    old_root=HERE.parent/'capacity_experiments_20261017'; old=read(old_root/'results/all_runs.json'); replay=[]
    for r in all_runs:
        if r['variant']!=BASE: continue
        previous=next(p for p in old if p['model']=='attention' and p['variant']=='baseline'
                      and p['seed']==r['seed'] and p['learning_rate']==r['learning_rate'])
        before=read(old_root/previous['run_dir']/'history.json'); after=read(HERE/r['run_dir']/'history.json')
        matches=all(a['official_correct']==b['official_correct'] and abs(a['validation_bce']-b['validation_bce'])<=1e-7
                    for a,b in zip(before,after))
        assert matches,'Original baseline training prefix changed'
        replay.append({'seed':r['seed'],'learning_rate':r['learning_rate'],'prefix_epochs_checked':len(before),
            'prefix_matches':matches,'old_best_accuracy':previous['accuracy'],'full60_best_accuracy':r['accuracy'],
            'old_best_epoch':previous['best_epoch'],'full60_best_epoch':r['best_epoch']})
    write(output/'baseline_prefix_replay.json',replay)
    report(output,protocol,summaries,fixed,paired,sensitivity,repairs,selection,all_runs,replay)
    print(json.dumps({'selection':selection,'summary':list(summaries.values()),
                      'primary_best':[r for r in paired if r['checkpoint']=='best' and r['kind']=='primary']},indent=2))


def report(output,protocol,summary,fixed,paired,sensitivity,repairs,selection,all_runs,replay):
    env=read(output/'environment.json'); integrity=read(output/'protected_verification.json')
    train_time=sum(r['training_seconds'] for r in all_runs)
    fixed_time=sum(r['fixed_epoch_scoring_seconds_in_training'] for r in all_runs)
    external_time=sum(r['validation_total_wall_seconds']+r['train_scoring_wall_seconds'] for r in all_runs)
    lines=['# 信息交互发生在哪个阶段：六组结构实验','',
        '完成 30 次主实验（固定 LR=0.0003，六组×五种子）与 6 次敏感性实验（seed17，LR=0.001）。每次都完整训练 60 轮、1140 次更新，没有早停。正式分数均由原版官方评测核对。','',
        f"只按主实验选择的最高五种子均值为 **{selection['variant']}：{selection['five_seed_mean_accuracy']:.3f}%**。固定 seed17 模型验证准确率为 **{selection['validation_accuracy']:.3f}%**；敏感性结果没有参与选择。",'']
    s2=next(r for r in paired if r['checkpoint']=='best' and r['comparison']=='s2_before_keep_norm_after - s0_after')
    lines += [f"S2 相对 S0 平均提高 {s2['difference_pp']:+.3f} 个百分点，五个种子的方向一致；95% 描述性区间为 [{s2['bootstrap_ci95_low']:+.3f}, {s2['bootstrap_ci95_high']:+.3f}]，仍包含零。这是值得复核的开发结果，尚不足以确认稳定提升。",'',
        '整块前置 S1 没有出现同等收益；S2 则只前置 FFN 残差、保留其归一化在交互后。因此本轮更支持继续区分“非线性加工位置”与“归一化位置”，不能概括成“交互越晚越好”。两者直接对比的区间也包含零，归一化机制尚未得到确认。','',
        '在两个独立块的同容量组中，前后各一个 S4 没有可靠胜过集中在一侧的 S3/S5。增加独立块没有超过参数更少的 S2；这些布局也都没有消除训练BCE继续下降、验证BCE逐步上升的现象。','']
    lines += [
        '## 操作定义与控制','',
        'A(h)=LN_A(h+Dropout(MHA(h)))；P_i(h)=h+Dropout(FFN_i(h))；N_i 为该 FFN 对应的 LayerNorm；完整逐候选块 F_i=N_i(P_i)。编码器 E:1024→128 与评分头 H:128→64→1 固定，表格省略 E/H。','',
        '| 组 | 顺序 | 参数量 | 对照目的 |','|---|---|---:|---|',
        '| S0 | A → F1 | 272001 | 原架构在新60轮预算下的基线 |',
        '| S1 | F1 → A | 272001 | 整个逐候选块前置 |',
        '| S2 | P1 → A → N1 | 272001 | 前置 FFN 残差，保留其归一化在后 |',
        '| S3 | F1 → F2 → A | 338177 | 两个独立块都在比较前 |',
        '| S4 | F1 → A → F2 | 338177 | 比较前后各一个 |',
        '| S5 | A → F1 → F2 | 338177 | 两个独立块都在比较后 |','',
        '- 一次四头注意力，允许自身 key/value，隐藏宽度128，FFN中间宽度256；同一容量组只改执行顺序。',
        '- 先按原模型顺序构建 E/A/F1/H，再为 S3-5 追加 F2；同种子的共有权重逐项相同，S3-5 的 F2 初始权重也相同。',
        '- S1−S0 同时移动 P1/N1；S2−S0 在 N1 保持最后的情况下移动 P1；S1−S2 在 P1 前置的条件下检查 N1 的位置。归一化影响仍是条件于这些具体结构的比较。',
        '- 所有模型从随机初始化共同训练，没有冻结分支。评分头本身也有逐候选非线性处理，因此“在前/在后”专指列出的128维独立块。',
        '- 训练/验证 4755/798 题，输入4×1024固定向量；仅训练集拟合归一化。无题目分支、外部数据或额外预训练权重。',
        '- 全部题 BCE、dropout0.1、AdamW weight_decay0.001、batch256、梯度裁剪1；主学习率0.0003固定，没有根据敏感性结果重新选择。',
        '- 同种子样本顺序一致，每轮19次更新。保存10/20/40/60轮的官方评分，并从60轮中按验证正确数、较低BCE、较早轮次选最佳检查点。',
        '- 跑满60轮与过去早停协议不同，因此比较使用本轮 S0；旧的91.90%不能直接作为本轮同预算对照。','',
        '## 主实验最佳检查点','',
        '| 组 | 顺序 | 五种子准确率 %（均值±标准差） | 非seed17四种子均值 % | 最佳轮次均值 | 30轮后达到最佳的种子数 |',
        '|---|---|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['variant']} | {r['order']} | {r['mean_accuracy']:.3f} ± {r['std_accuracy']:.3f} | {r['other_four_seeds_mean_accuracy']:.3f} | {r['mean_best_epoch']:.1f} | {r['best_after_epoch30_seeds']} |")
    lines += ['', '## 相同更新次数的表现','',
        '| 组 | 第10轮 % | 第20轮 % | 第40轮 % | 第60轮 % |','|---|---:|---:|---:|---:|']
    for v in summary:
        cells=[next(r['mean_accuracy'] for r in fixed if r['variant']==v and r['epoch']==epoch) for epoch in protocol['fixed_epochs']]
        lines.append(f"| {v} | "+' | '.join(f'{c:.3f}' for c in cells)+' |')
    lines += ['', f"![Stage learning curves]({(output/'stage_learning_curves.png').as_posix()})",'',
        '各曲线为固定学习率主实验的五种子均值，所有运行都有完整60轮，不存在早停后参与均值的运行数减少。上排为验证准确率，下排实线为验证BCE、虚线为关闭dropout的训练BCE。同一子图内参数量相同。主表先选择每个种子各自的最佳轮次再平均，因此不等于图中某一共同轮次均值的最高点。','',
        '## 同容量配对比较','',
        '按题目配对 bootstrap 10000 次，先在五种子间平均正确性。区间条件于这些已选择检查点，不包含调参/检查点选择不确定性，未校正多重比较；种子×题目没有作为独立样本。','',
        '| 最佳检查点比较 | 差值（百分点） | 95% 描述性区间 | 正向种子数/5 | 非seed17四种子差值 |','|---|---:|---:|---:|---:|']
    for r in paired:
        if r['checkpoint']!='best' or r['kind']!='primary': continue
        lines.append(f"| {r['comparison']} | {r['difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] | {r['positive_seeds']}/5 | {r['other_four_seeds_difference_pp']:+.3f} |")
    lines += ['']
    for r in paired:
        if r['checkpoint']!='best' or r['kind']!='primary': continue
        meaning='区间包含零，尚不能确认稳定差异' if r['bootstrap_ci95_low']<=0<=r['bootstrap_ci95_high'] else '描述性区间不含零，仍需独立数据确认'
        lines.append(f"- {r['comparison']}：{r['difference_pp']:+.3f} 个百分点；{meaning}。")
    lines += ['', '以下比较前后各一个独立块 S4 与两块集中在一侧的模型，在各固定轮次的差值；完整六组主比较在各轮次的区间见 paired_comparisons.json。','',
        '| 比较 | 第10轮差值 | 第20轮差值 | 第40轮差值 | 第60轮差值 |','|---|---:|---:|---:|---:|']
    for comp in ('s4_before_after - s3_before_before','s4_before_after - s5_after_after'):
        values=[next(r['difference_pp'] for r in paired if r['comparison']==comp and r['checkpoint']==str(epoch)) for epoch in protocol['fixed_epochs']]
        lines.append(f'| {comp} | '+' | '.join(f'{x:+.3f}' for x in values)+' |')
    lines += ['', 'S3/S4/S5 与 S0 的差异同时包含容量变化，属于补充观察，不能单独据此归因于交互阶段。','',
        '| 额外容量模型−S0 | 最佳均值差（百分点） | 95% 描述性区间 |','|---|---:|---:|']
    for r in paired:
        if r['checkpoint']=='best' and r['kind']=='capacity_secondary':
            lines.append(f"| {r['comparison']} | {r['difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] |")
    lines += ['', '## 学习率敏感性：只有一个种子','',
        '全部按原定计划运行，不用于主学习率或最终模型选择；差值均相对同学习率、同预算的 S0。','',
        '| 组 | LR0.0003 seed17 % | 相对S0差值 | LR0.001 seed17 % | 相对S0差值 |','|---|---:|---:|---:|---:|']
    for r in sensitivity:
        lines.append(f"| {r['variant']} | {r['main_seed17_accuracy']:.3f} | {r['main_seed17_difference_vs_s0_pp']:+.3f} | {r['sensitivity_seed17_accuracy']:.3f} | {r['sensitivity_difference_vs_s0_pp']:+.3f} |")
    lines += ['', '## 拟合差距与错误变化','',
        '以下均为最佳检查点；训练/验证在同一eval模式测量。混合题为有错误候选的419道验证题，是诊断聚合。修正/新增错误相对同种子的本轮S0，数字为五种子平均题数。','',
        '| 组 | 训练准确率 % | 训练−验证差距 | 训练 BCE | 验证 BCE | 混合题准确率 % | 修正/新增错误 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.values():
        rep=next(x for x in repairs if x['variant']==r['variant'])
        lines.append(f"| {r['variant']} | {r['train_accuracy']:.3f} | {r['accuracy_gap_pp']:.3f} | {r['train_bce']:.4f} | {r['validation_bce']:.4f} | {r['mixed_accuracy_diagnostic']:.3f} | {rep['mean_fixed_errors']:.1f} / {rep['mean_new_errors']:.1f} |")
    lines += ['', '## 实现检查、成本与边界','',
        f"- S0 的训练轨迹前缀与过去早停运行对照：{sum(r['prefix_matches'] for r in replay)}/{len(replay)} 个运行完全复现官方正确数，BCE差异不超过1e-7；完整记录见 baseline_prefix_replay.json。",
        '- 操作顺序由模块hook核对；同容量参数量、公共初始化、S0训练/评估输出及梯度与原实现完全一致、排列等变、batch独立、跨候选依赖和有限梯度检查均通过。',
        '- 每个最终检查点新进程加载并官方评分，预测与保存前逐项一致；每个固定轮次另存预测并调用原版评分脚本。',
        f"- 原有文件哈希复核 {integrity['files_checked']} 个，变更 {len(integrity['changed'])} 个。所有此前实验和官方数据/脚本保留。",
        f"- 环境：{env['processor']}，Python {env['python'].split()[0]}、PyTorch {env['torch']}、NumPy {env['numpy']}；CPU串行，PyTorch intra-op4/inter-op1，BLAS1。",
        f'- 36次完整训练与逐轮测量累计 {train_time:.1f} 秒，其中固定轮次官方评分子进程占 {fixed_time:.1f} 秒，已包含在前者中；最佳检查点的新预测/验证评分/训练评分另计 {external_time:.1f} 秒。敏感性运行已计入总成本，不重复计算固定轮次评分。','',
        '| 组 | 资产 MiB | 每次完整训练秒均值（含固定轮次评分） | 最佳模型完整验证预测+评分秒均值 |','|---|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {r['variant']} | {r['asset_mib']:.3f} | {r['mean_train_seconds_including_fixed_scoring']:.2f} | {r['mean_validation_wall_seconds']:.2f} |")
    lines += ['', '计时含验证798题的新进程启动、导入和数据/资产加载；未在官方Xeon上测试，不据此认证官方30秒限制。所有单模型资产均低于16MiB。','',
        '- 比较的是当前四头、允许自身的架构及具体FFN/归一化放置；结论不能直接推广到任意交互层、其他头数或其他归一化形式。',
        '- 固定60轮使更新预算一致，但未必保证每种布局完全收敛。不同操作顺序也会改变优化路径；不能仅凭最终分数断定表示中“包含更多信息”。',
        '- 验证集已多次用于设计与检查点选择；最佳60轮结果也具有选择偏差，固定轮次结果是辅助证据，五种子/区间并不替代独立测试。',
        '- 测试集没有训练、选择、预测或评分。完整性检查可以核验其文件对齐与哈希。',
        '- 新代码与报告由Codex AI辅助编写运行；仅使用现有数据，模型全部随机初始化并共同训练。','',
        '## 复现与文件','',
        '从code目录执行；训练解释器为D:/anaconda/envs/mp1/python.exe；完整重训使用新输出目录。','',
        '```powershell',"$env:PYTHONDONTWRITEBYTECODE='1'",
        'python -B interaction_stage_20261017/check_models.py',
        'python -B interaction_stage_20261017/run_suite.py --output interaction_stage_20261017/reproduction',
        'python -B interaction_stage_20261017/summarize.py --output interaction_stage_20261017/reproduction',
        'python -B interaction_stage_20261017/verify_delivery.py --output interaction_stage_20261017/reproduction',
        '```','',
        '绘图由装有matplotlib的Python执行 summarize.py --plot-only --output <结果目录>。',
        'summary.csv为主结果；fixed_epochs.csv为固定预算；paired_comparisons.json为所有配对；sensitivity.csv为单种子第二学习率；learning_curves.csv为完整主实验轨迹；runs.csv含全部36次运行与成本；selected_bundle为仅依主实验选出的固定seed17单模型。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def plot(output):
    import platform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocol=read(HERE/'protocol.json'); runs=read(output/'selected_runs.json')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    groups=[list(protocol['variants'])[:3],list(protocol['variants'])[3:]]
    for col,variants in enumerate(groups):
        for j,v in enumerate(variants):
            rows=[r for r in runs if r['variant']==v]
            histories=[read(HERE/r['run_dir']/'history.json') for r in rows]
            x=np.arange(1,61); color=plt.get_cmap('tab10')(j)
            acc=np.mean([[h['official_accuracy'] for h in hist] for hist in histories],axis=0)
            tr=np.mean([[h['train_eval']['bce'] for h in hist] for hist in histories],axis=0)
            va=np.mean([[h['validation_eval']['bce'] for h in hist] for hist in histories],axis=0)
            axes[0,col].plot(x,acc,label=v,color=color)
            axes[1,col].plot(x,va,color=color)
            axes[1,col].plot(x,tr,color=color,linestyle='--',alpha=.75)
        axes[0,col].set_title('One FFN: 272,001 parameters' if col==0 else 'Two FFNs: 338,177 parameters')
        axes[0,col].set_ylabel('Official validation accuracy (%)'); axes[0,col].legend(fontsize=8)
        axes[1,col].set_ylabel('BCE (solid: validation; dashed: train eval)')
        axes[1,col].set_xlabel('Epoch (19 updates per epoch)')
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('Interaction stage: five-seed mean, fixed LR=0.0003, no early stopping')
    fig.tight_layout(); fig.savefig(output/'stage_learning_curves.png',dpi=180)
    fig.savefig(output/'stage_learning_curves.pdf'); plt.close(fig)
    write(output/'plot_environment.json',{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__})
    print('Saved stage_learning_curves.png and stage_learning_curves.pdf')


if __name__=='__main__': main()
