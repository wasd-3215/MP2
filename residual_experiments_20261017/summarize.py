"""Official-score comparisons and residual repair/harm diagnostics."""
import runtime
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

HERE=Path(__file__).resolve().parent
NAMES={'independent':'独立 MLP','attention':'原注意力','finetune':'独立 MLP 继续训练',
       'residual_self':'冻结独立＋仅自身修正','residual_context':'冻结独立＋跨候选修正',
       'residual_gated':'冻结独立＋门控跨候选修正','residual_joint':'门控修正＋联合微调'}


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
    assert len(runs)==35 and len(all_runs)==42
    labels=np.load(output/'validation-labels.npy',allow_pickle=False)
    train_labels=np.load(output/'train-labels.npy',allow_pickle=False)
    correct,choices,summary,repairs={},{},{},[]
    for name in protocol['models']:
        rows=sorted([r for r in runs if r['model']==name],key=lambda r:protocol['seeds'].index(r['seed']))
        assert [r['seed'] for r in rows]==protocol['seeds']
        hits,picks=[],[]
        for r in rows:
            folder=HERE/r['run_dir']
            prediction=read(folder/'validation-predictions.json')
            official=read(folder/'validation-score.json')
            pick=np.asarray(prediction['selected'])
            hit=labels[np.arange(len(labels)),pick]
            assert hit.sum()==official['correct']==r['correct'] and official['accuracy']==r['accuracy']
            assert np.array_equal(np.load(folder/'validation-logits.npy').argmax(1),pick)
            t=read(folder/'train-predictions.json')['selected']
            ts=read(folder/'train-score.json')
            assert train_labels[np.arange(len(train_labels)),t].sum()==ts['correct'] and ts['accuracy']==r['train_accuracy']
            hits.append(hit); picks.append(pick)
            if r['configuration']['anchor'] is not None:
                anchor=read(HERE/r['configuration']['anchor']['run_dir']/'validation-predictions.json')
                assert anchor['ids']==prediction['ids']
                original=np.asarray(anchor['selected'])
                original_hit=labels[np.arange(len(labels)),original]
                fixed=(hit==1)&(original_hit==0)
                harmed=(hit==0)&(original_hit==1)
                assert fixed.sum()-harmed.sum()==int(hit.sum()-original_hit.sum())
                parts=np.load(folder/'validation-components.npz',allow_pickle=False)
                base_pick=parts['base'].argmax(1)
                base_hit=labels[np.arange(len(labels)),base_pick]
                base_score=read(folder/'base-only-score.json')
                assert base_hit.sum()==base_score['correct']
                if name in ('residual_self','residual_context','residual_gated'):
                    assert np.array_equal(original,base_pick)
                repairs.append({'model':name,'seed':r['seed'],'anchor_accuracy':100*float(original_hit.mean()),
                    'final_accuracy':r['accuracy'],'changed_choices':int((pick!=original).sum()),
                    'fixed_errors':int(fixed.sum()),'new_errors':int(harmed.sum()),'net_fixes':int(fixed.sum()-harmed.sum()),
                    'changed_but_both_correct':int(((pick!=original)&(original_hit==1)&(hit==1)).sum()),
                    'changed_but_both_wrong':int(((pick!=original)&(original_hit==0)&(hit==0)).sum()),
                    'current_base_only_accuracy':base_score['accuracy'],
                    'correction_fixed_current_base_errors':int(((hit==1)&(base_hit==0)).sum()),
                    'correction_harmed_current_base':int(((hit==0)&(base_hit==1)).sum()),
                    'base_max_abs_logit_drift':r['correction_diagnostics']['base_max_abs_logit_drift'],
                    'mean_abs_applied_delta':float(np.abs(parts['applied_delta']).mean()),
                    'mean_gate':float(parts['gate'].mean()),
                    'mean_gate_on_fixed':float(parts['gate'][fixed].mean()) if fixed.any() else None,
                    'mean_gate_on_harmed':float(parts['gate'][harmed].mean()) if harmed.any() else None,
                    'fallback_epoch_zero':r['best_epoch']==0})
        correct[name],choices[name]=np.asarray(hits),np.asarray(picks)
        acc=np.asarray([r['accuracy'] for r in rows])
        avg=lambda key:float(np.mean([r[key] for r in rows]))
        summary[name]={'model':name,'mean_accuracy':float(acc.mean()),'std_accuracy':float(acc.std(ddof=1)),
            'other_four_seeds_mean_accuracy':float(acc[1:].mean()),'seed17_accuracy':float(acc[0]),
            'learning_rate':rows[0]['learning_rate'],'parameters':rows[0]['parameters'],
            'trainable_parameters':rows[0]['trainable_parameters'],'train_accuracy':avg('train_accuracy'),
            'accuracy_gap_pp':avg('accuracy_gap_pp'),'eval_bce_gap':avg('eval_bce_gap'),
            'validation_bce':avg('best_validation_loss'),
            'mixed_accuracy_diagnostic':float(100*correct[name][:,labels.sum(1)<4].mean()),
            'mean_best_epoch':avg('best_epoch'),'fallback_seeds':sum(r['best_epoch']==0 for r in rows),
            'mean_stage_train_seconds':avg('training_seconds'),
            'mean_required_anchor_train_seconds':avg('anchor_training_seconds'),
            'mean_validation_wall_seconds':avg('validation_total_wall_seconds'),
            'asset_mib':rows[0]['uncompressed_asset_bytes']/1024**2}
    write(output/'summary.json',list(summary.values())); csv_write(output/'summary.csv',list(summary.values()))
    write(output/'repairs_by_seed.json',repairs); csv_write(output/'repairs_by_seed.csv',repairs)
    repair_summary=[]
    for name in protocol['models'][2:]:
        rows=[r for r in repairs if r['model']==name]
        repair_summary.append({'model':name,**{k:float(np.mean([r[k] for r in rows])) for k in
            ['changed_choices','fixed_errors','new_errors','net_fixes','current_base_only_accuracy',
             'correction_fixed_current_base_errors','correction_harmed_current_base','base_max_abs_logit_drift',
             'mean_abs_applied_delta','mean_gate']}})
    write(output/'repair_summary.json',repair_summary); csv_write(output/'repair_summary.csv',repair_summary)
    boot=np.random.default_rng(20261017).integers(0,len(labels),size=(10000,len(labels)))
    paired=[]
    for comparison in protocol['primary_comparisons']:
        a,b=comparison.split(' - ')
        d=correct[a]-correct[b]
        seed_difference=100*d.mean(1)
        distribution=100*d.mean(0)[boot].mean(1)
        paired.append({'comparison':comparison,'difference_pp':float(seed_difference.mean()),
            'bootstrap_ci95_low':float(np.quantile(distribution,.025)),
            'bootstrap_ci95_high':float(np.quantile(distribution,.975)),
            'positive_seeds':int((seed_difference>0).sum()),'seed_differences_pp':seed_difference.tolist(),
            'other_four_seeds_difference_pp':float(seed_difference[1:].mean()),
            'seed17_fixed_errors':int((d[0]>0).sum()),'seed17_new_errors':int((d[0]<0).sum())})
    write(output/'paired_comparisons.json',paired); csv_write(output/'paired_comparisons.csv',paired)
    keys=['model','seed','learning_rate','accuracy','train_accuracy','best_epoch','epochs_run','parameters',
          'trainable_parameters','training_seconds','anchor_training_seconds','validation_total_wall_seconds',
          'train_scoring_wall_seconds','base_only_scoring_wall_seconds','run_dir']
    csv_write(output/'runs.csv',[{k:r[k] for k in keys} for r in all_runs])
    best=max(summary.values(),key=lambda r:(r['mean_accuracy'],-r['parameters']))
    selected=next(r for r in runs if r['model']==best['model'] and r['seed']==17)
    bundle=output/'selected_bundle'; bundle.mkdir(exist_ok=True)
    for p in (HERE/selected['run_dir']/'bundle').iterdir():
        shutil.copy2(p,bundle/p.name)
    selection={'policy':protocol['final_selection'],'model':best['model'],'seed':17,'source_run':selected['run_dir'],
        'learning_rate':selected['learning_rate'],'validation_accuracy':selected['accuracy'],
        'five_seed_mean_accuracy':best['mean_accuracy'],'test_evaluated':False,
        'bundle_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.iterdir()}}
    write(output/'selection.json',selection)
    old_root=HERE.parent/'capacity_experiments_20261017'
    old=read(old_root/'results/selected_runs.json')
    replay=[]
    for r in runs:
        if r['model'] not in ('independent','attention'): continue
        previous=next(x for x in old if x['model']==r['model'] and x['variant']=='baseline' and x['seed']==r['seed'])
        same=read(old_root/previous['run_dir']/'validation-predictions.json')['selected']==read(HERE/r['run_dir']/'validation-predictions.json')['selected']
        replay.append({'model':r['model'],'seed':r['seed'],'identical_predictions':same})
    write(output/'baseline_replay.json',replay)
    report(output,protocol,summary,repair_summary,paired,selection,all_runs,replay)
    print(json.dumps({'selection':selection,'summary':list(summary.values()),'repairs':repair_summary},indent=2))


def report(output,protocol,summary,repairs,paired,selection,all_runs,replay):
    env=read(output/'environment.json'); integrity=read(output/'protected_verification.json')
    total_train=sum(r['training_seconds'] for r in all_runs)
    total_eval=sum(r['validation_total_wall_seconds']+r['train_scoring_wall_seconds']+r['base_only_scoring_wall_seconds'] for r in all_runs)
    lines=['# 保留独立判断，让交互负责修正','',
        '完成 7 组配置、42 次实际训练，其中 35 个检查点进入五种子主比较。训练目标、输入归一化和数据划分一致；正式准确率来自原版官方评分脚本。','',
        f"按五种子均值选择的是 **{NAMES[selection['model']]}：{selection['five_seed_mean_accuracy']:.3f}%**；固定 seed 17 单模型验证准确率为 **{selection['validation_accuracy']:.3f}%**。没有挑最高分种子或集成。",'',
        '## 设计','',
        '先训练独立分数 b_i=f(x_i)，再增加修正量：s_i=b_i+δ_i；门控版本为 s_i=b_i+g_iδ_i。δ 来自候选表示上的一层四头注意力与 FFN，g=sigmoid(Linear([h_i,c_i]))。最终仍用四个分数的 argmax 选择答案。','',
        '| 组 | 用途 |','|---|---|',
        '| 独立 MLP | 原始独立基线与同种子预训练来源 |',
        '| 原注意力 | 与此前最佳交互架构比较，从随机初始化训练 |',
        '| 独立 MLP 继续训练 | 同一个预训练检查点、新优化器、相同第二阶段搜索预算，控制额外训练 |',
        '| 冻结独立＋仅自身修正 | 相同修正块，但注意力每个 query 只能读自身；控制新增路径与存储参数量 |',
        '| 冻结独立＋跨候选修正 | 核心实验：原判断完全固定，仅学习相对原分数的调整 |',
        '| 冻结独立＋门控跨候选修正 | 增加候选级门控，初始 g=0.1 |',
        '| 门控修正＋联合微调 | 相同门控架构，但允许原独立分支变化，检验冻结是否必要 |','',
        '- 修正头最后一层的权重与偏置初始化为零，因此阶段二初始分数逐项等于原独立模型；门控不能改变这一性质。',
        '- 冻结版本不仅关闭独立分支梯度，也在训练期间强制该分支 eval，使其 dropout 不改变原判断。冻结参数与原分数在训练完成后再次逐项核对。',
        '- 所有新模型保留独立分数的直接加法路径；但只有冻结版本保证它始终是原判断。联合微调版本的独立分数可能漂移。',
        '- 仅自身修正和跨候选修正的存储参数相同，但单 key 的注意力使 Q/K 匹配退化，不能声称有效容量严格相同。门控只增加 257 个参数，同时改变初始修正梯度尺度。',
        '- 门控值不是置信度，也不能单独解释贡献大小：g 与 δ 可以互相补偿缩放，因此另记录实际 gδ 的幅度与预测变化。','',
        '## 训练协议','',
        '- 固定 4×1024 向量，训练 4755 题、验证 798 题。仅训练集拟合归一化；不引入题目文本、外部数据或预训练权重。',
        '- 统一对全部训练题使用最终分数的 BCE，多正确标签不转换成单一正确下标；没有加入额外辅助损失。',
        '- 隐藏维度 128、dropout 0.1、AdamW weight decay 0.001、batch 256、梯度裁剪 1。每轮 19 次更新，最多 60 轮，patience 10。',
        '- 各组在 seed 17 搜索学习率 0.001 / 0.0003，然后用选定学习率训练 29、43、71、101。按官方验证正确数、BCE、较早轮次选检查点。',
        '- 阶段二全部使用同种子独立模型的同一个选定检查点；该来源在训练后续组前已经固定。独立模型本身也使用验证集选择。',
        '- 阶段二的第 0 轮预训练状态也可被选中：修正无益时回退。它是验证驱动选择，不保证独立测试上不退步。',
        '- 初始/逐轮/最终训练与验证诊断都关闭 dropout。阶段二重置优化器与随机种子，额外预算一致；早停后实际步数可能不同。',
        '- 两阶段模型比从头训练的注意力有额外预训练成本。继续训练与仅自身修正两组分别控制部分额外训练和容量因素，仍不能消除所有架构/优化差异。','',
        '## 官方验证结果','',
        '| 模型 | 五种子准确率 %（均值±标准差） | 其余四种子均值 % | LR | 总参数 / 本阶段可训练参数 | 最佳阶段轮次均值 | 回退种子数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {NAMES[r['model']]} | {r['mean_accuracy']:.3f} ± {r['std_accuracy']:.3f} | {r['other_four_seeds_mean_accuracy']:.3f} | {r['learning_rate']} | {r['parameters']:,} / {r['trainable_parameters']:,} | {r['mean_best_epoch']:.1f} | {r['fallback_seeds']} |")
    lines += ['', '## 配对比较','',
        '同题、同种子配对；10000 次按题目 bootstrap，先在五种子上平均正确性。区间是条件于这些已选择模型的描述性区间，不含调参与早停不确定性，未校正多重比较。','',
        '| 比较 | 差值（百分点） | 95% 描述性区间 | 正向种子数/5 | 其余四种子差值 |','|---|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['difference_pp']:+.3f} | [{r['bootstrap_ci95_low']:+.3f}, {r['bootstrap_ci95_high']:+.3f}] | {r['positive_seeds']}/5 | {r['other_four_seeds_difference_pp']:+.3f} |")
    lines += ['']
    for r in paired:
        conclusion='区间包含零，不能确认稳定差异' if r['bootstrap_ci95_low']<=0<=r['bootstrap_ci95_high'] else '描述性区间不含零，但仍需独立测试确认'
        lines.append(f"- {r['comparison']}：{r['difference_pp']:+.3f} 个百分点；{conclusion}。")
    lines += ['', '跨候选修正−仅自身修正最直接检验修正分支是否受益于其他候选；跨候选修正−继续训练用于排除部分单纯增加训练的解释；门控−无门控与联合−冻结分别检验门控、保留原判断的作用。','',
        '## 修正还是破坏','',
        '以下为验证集 798 题上每种方法相对同种子预训练独立模型的平均题数。正确候选可能不止一个：更换候选不一定改变得分。修正/新增错误按官方正确性规则聚合；它们是诊断，不是独立 benchmark。','',
        '| 模型 | 改变选择数 | 修正错误 | 新增错误 | 净修正 | 当前独立分支单独评分 % | 独立分数最大漂移均值 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in repairs:
        lines.append(f"| {NAMES[r['model']]} | {r['changed_choices']:.1f} | {r['fixed_errors']:.1f} | {r['new_errors']:.1f} | {r['net_fixes']:+.1f} | {r['current_base_only_accuracy']:.3f} | {r['base_max_abs_logit_drift']:.6f} |")
    lines += ['', '冻结版本的独立分数漂移必须为零。联合微调需区分“相对最初独立模型的总改善”与“修正分支在当前独立分数上额外带来的改善”，详细计数均保存在 repairs_by_seed.csv。','',
        f"![Repair and harm counts]({(output/'repair_tradeoff.png').as_posix()})",'',
        '## 拟合差距与资源','',
        '| 模型 | 训练准确率 % | 训练−验证差距（百分点） | 验证 BCE | 验证混合题准确率 % | 资产 MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {NAMES[r['model']]} | {r['train_accuracy']:.3f} | {r['accuracy_gap_pp']:.3f} | {r['validation_bce']:.4f} | {r['mixed_accuracy_diagnostic']:.3f} | {r['asset_mib']:.3f} |")
    lines += ['', '混合题为有正确也有错误候选的 419 道验证题，结果为诊断聚合。资产包含独立路径和修正路径全部权重、配置与预处理，部署不需要再读取外部预训练检查点。','',
        '| 模型 | 本阶段训练秒/次均值 | 所需独立预训练秒/次均值 | 完整验证预测+评分秒/次均值 |','|---|---:|---:|---:|']
    for r in summary.values():
        lines.append(f"| {NAMES[r['model']]} | {r['mean_stage_train_seconds']:.2f} | {r['mean_required_anchor_train_seconds']:.2f} | {r['mean_validation_wall_seconds']:.2f} |")
    lines += ['', f'42 次实际训练共 {total_train:.1f} 秒，独立验证预测、训练/验证/独立分支官方评分共 {total_eval:.1f} 秒。共享预训练在总成本中只计算一次；上表展示部署方案所需的预训练，不包含额外学习率搜索运行，完整搜索已计入总成本。',
        f"本机：{env['processor']}，Python {env['python'].split()[0]}、PyTorch {env['torch']}、NumPy {env['numpy']}。CPU 串行，PyTorch intra-op 4 / inter-op 1，BLAS 1。计时含进程启动、导入、输入/资产加载；不据此认证官方 Xeon 的 30 秒限制。",'',
        '## 验证与限制','',
        f"- 原独立/注意力基线：{sum(r['identical_predictions'] for r in replay)}/{len(replay)} 个选定种子与上一轮预测完全一致。",
        '- 检查零初始化精确复现、冻结参数不变、冻结分数不变、仅自身修正无跨候选依赖、跨候选修正可学习、候选排列等变、不同 batch 一致、有限梯度和独立进程重载预测。',
        f"- 原有 {integrity['files_checked']} 个文件哈希复核，变更 {len(integrity['changed'])} 个。未修改任何此前实验、官方解析器/评分器或数据。",
        '- 验证集已多次用于实验设计、学习率与检查点选择；五种子和 bootstrap 不消除这一选择偏差。没有测试集预测或评分。',
        '- 修正训练与独立预训练使用同一训练集，原模型在训练题上的误差分布可能比新题更乐观；没有使用 out-of-fold 独立分数。这限制两阶段结果的外推。',
        '- 冻结版本也冻结了共享的 1024→128 候选编码器，修正分支只能读取原独立模型的表示，不能重新提取输入特征。负结果不能推广到带独立可训练交互编码器的所有残差结构。',
        '- 冻结保留的是原分数，不是最终选择：修正仍可能推翻一个正确答案。门控也没有“不伤害正确选择”的保证。',
        '- 基线来自全部题 BCE；本轮没有同时换成上一轮混合题目标，避免将来源模型/训练目标变化混入架构对照。',
        '- 实验与汇总代码由 Codex AI 辅助编写运行；模型仅用现有训练数据从随机初始化/本实验独立预训练获得。','',
        '## 复现与产物','',
        '从 code 目录执行，当前训练解释器为 D:/anaconda/envs/mp1/python.exe。完整重训使用新输出目录。','',
        '```powershell',"$env:PYTHONDONTWRITEBYTECODE='1'",
        'python -B residual_experiments_20261017/check_models.py',
        'python -B residual_experiments_20261017/run_suite.py --output residual_experiments_20261017/reproduction',
        'python -B residual_experiments_20261017/summarize.py --output residual_experiments_20261017/reproduction',
        'python -B residual_experiments_20261017/verify_delivery.py --output residual_experiments_20261017/reproduction',
        '```','',
        '绘图由装有 matplotlib 的 Python 执行 summarize.py --plot-only --output <结果目录>；图像不影响训练与选择。',
        'summary.csv / runs.csv 保存主结果与所有搜索运行；paired_comparisons.json 保存配对分析；repairs_by_seed.csv / repair_summary.csv 保存修正与破坏计数；每个运行包含 history、独立分数/修正量/门控/最终分数、官方预测与评分。selected_bundle 是按预定规则选出的固定 seed 17 单模型。','']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def plot(output):
    import platform
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(output/'repair_summary.json')
    x=np.arange(len(rows)); width=.24
    fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(x-width,[r['fixed_errors'] for r in rows],width,label='Fixed errors',color='#2b8c69')
    ax.bar(x,[r['new_errors'] for r in rows],width,label='New errors',color='#ce6550')
    ax.bar(x+width,[r['net_fixes'] for r in rows],width,label='Net fixes',color='#477db3')
    ax.set_xticks(x,[r['model'].replace('residual_','') for r in rows])
    ax.set_ylabel('Questions (mean over five seeds; validation N=798)')
    ax.set_title('Repairs and harms relative to each seed\'s pretrained independent scorer')
    ax.axhline(0,color='black',linewidth=.7); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.legend(); fig.tight_layout()
    fig.savefig(output/'repair_tradeoff.png',dpi=180); fig.savefig(output/'repair_tradeoff.pdf'); plt.close(fig)
    write(output/'plot_environment.json',{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__})
    print('Saved repair_tradeoff.png and repair_tradeoff.pdf')


if __name__=='__main__':
    main()
