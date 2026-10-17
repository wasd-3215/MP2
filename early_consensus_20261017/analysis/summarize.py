"""Prespecified factorial contrasts, paired diagnostics and full cost records."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
from support import HERE,CODE,OUT,VARIANTS,SEEDS,SCOPES,FEATURES,POSITIONS,read,write,sha,source_hashes
from collections import Counter
import csv,json,statistics
import numpy as np
from common import load_inputs,read_jsonl,parse_answer,DATA,check_data

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(values,rng):
    values=np.asarray(values,float)
    if np.all(values==values[0]):return [float(values[0]*100)]*2
    samples=[]
    for _ in range(50):
        index=rng.integers(0,len(values),size=(200,len(values)))
        samples.extend(values[index].mean(axis=1)*100)
    return np.quantile(samples,[.025,.975]).tolist()
def ci(r):return f'{r["difference_pp"]:+.3f} [{r["ci95_pp"][0]:+.3f}, {r["ci95_pp"][1]:+.3f}]'

def definitions():
    ds=[]
    def add(name,coefs,primary=False):ds.append((name,coefs,primary))
    add('scope_main: D-All',{f'{s}_{f}_{p}':(.25 if s=='D' else -.25) for s in SCOPES for f in FEATURES for p in POSITIONS},True)
    add('early_at_C_main',{f'{s}_C_{p}':(.5 if p=='Early' else -.5) for s in SCOPES for p in POSITIONS},True)
    add('early_feature_interaction',{f'{s}_{f}_{p}':.5*(1 if f=='C' else -1)*(1 if p=='Early' else -1) for s in SCOPES for f in FEATURES for p in POSITIONS},True)
    add('early_scope_interaction',{f'{s}_{f}_{p}':.5*(1 if s=='D' else -1)*(1 if p=='Early' else -1) for s in SCOPES for f in FEATURES for p in POSITIONS},True)
    add('position_main: Early-Late',{f'{s}_{f}_{p}':(.25 if p=='Early' else -.25) for s in SCOPES for f in FEATURES for p in POSITIONS})
    add('feature_main: C-P',{f'{s}_{f}_{p}':(.25 if f=='C' else -.25) for s in SCOPES for f in FEATURES for p in POSITIONS})
    add('early_at_P_main',{f'{s}_P_{p}':(.5 if p=='Early' else -.5) for s in SCOPES for p in POSITIONS})
    add('scope_feature_interaction',{f'{s}_{f}_{p}':.5*(1 if s=='D' else -1)*(1 if f=='C' else -1) for s in SCOPES for f in FEATURES for p in POSITIONS})
    add('three_way_interaction',{f'{s}_{f}_{p}':(1 if s=='D' else -1)*(1 if f=='C' else -1)*(1 if p=='Early' else -1) for s in SCOPES for f in FEATURES for p in POSITIONS})
    for f in FEATURES:
        for p in POSITIONS:add(f'D-All at {f}/{p}',{f'D_{f}_{p}':1,f'All_{f}_{p}':-1})
    for s in SCOPES:
        for f in FEATURES:add(f'Early-Late at {s}/{f}',{f'{s}_{f}_Early':1,f'{s}_{f}_Late':-1})
    for s in SCOPES:
        for p in POSITIONS:add(f'C-P at {s}/{p}',{f'{s}_C_{p}':1,f'{s}_P_{p}':-1})
    add('combined D_C_Early - All_C_Late',{'D_C_Early':1,'All_C_Late':-1})
    return ds

def main():
    write(OUT/'analysis_sources_before.json',{str(p.relative_to(HERE)):sha(p) for p in (HERE/'analysis').glob('*.py')})
    assert read(OUT/'training_sources.json')==source_hashes()
    records=read(OUT/'official_results.json');runs=read(OUT/'training_runs.json');assert len(records)==48 and len(runs)==40
    bank,_=load_inputs('validation');check_data(DATA,['validation']);refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    answers=[];truth=[]
    for row,ref in zip(bank,refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        gold=parse_answer(ref['answer']);assert gold is not None
        a=[parse_answer(c['text']) for c in row['candidates']];answers.append(a);truth.append([v is not None and v==gold for v in a])
    y=np.asarray(truth,bool);n=len(y);valid=np.asarray([[v is not None for v in a] for a in answers],bool)
    assert np.array_equal(y,np.load(OUT/'validation-labels.npy'))
    group_count=np.array([len(set(v for v in a if v is not None)) for a in answers]);hard=group_count>=2;easy=group_count==1
    assert hard.sum()==355 and easy.sum()==443 and group_count.min()>=1
    wrong=[];correct=[];ties=[]
    for q,a in enumerate(answers):
        counts=Counter(v for v in a if v is not None);maximum=max(counts.values());wins=[v for v,c in counts.items() if c==maximum]
        unique=len(wins)==1;gold=parse_answer(refs[q]['answer'])
        wrong.append(unique and wins[0]!=gold);correct.append(unique and wins[0]==gold and y[q].sum()<4);ties.append(not unique and y[q].sum()<4)
    masks={'all':np.ones(n,bool),'disagreement':hard,'mixed':y.sum(axis=1)<4,'one_correct':y.sum(axis=1)==1,
        'wrong_unique_plurality':np.asarray(wrong),'correct_plurality_mixed':np.asarray(correct),
        'plurality_tie_mixed':np.asarray(ties),'any_invalid_parse':~valid.all(axis=1)}
    write(OUT/'subgroup_definitions.json',{k:int(v.sum()) for k,v in masks.items()})
    choice={};ok={};replay=[]
    for r in records:
        prediction=read(Path(r['checkpoint'])/'predictions.json');assert prediction['ids']==[row['id'] for row in bank]
        selected=np.asarray(prediction['selected'],int);choice[r['name']]=selected;ok[r['name']]=y[np.arange(n),selected].astype(float)
        assert ok[r['name']].sum()==r['correct'] and ok[r['name']][easy].all()
        logits=np.stack([np.load(OUT/'runs'/r['variant']/f'seed_{s}/validation-logits.npy') for s in r['seeds']])
        scores=(1/(1+np.exp(-np.clip(logits.astype(np.float64),-40,40)))).mean(axis=0)
        expected=scores.argmax(axis=1);change=valid.any(axis=1)&~valid[np.arange(n),expected]
        expected[change]=np.where(valid[change],scores[change],-np.inf).argmax(axis=1)
        expected[easy]=valid[easy].argmax(axis=1)
        assert np.array_equal(selected,expected),r['name']
        replay.append({'name':r['name'],'matches_full_bank_saved_logits_plus_fixed_decision':True,'easy_correct':443,'neural_questions':355})
    write(OUT/'fresh_replay.json',replay)
    matrices={};summary=[];subgroups=[]
    for variant in VARIANTS:
        singles=np.stack([ok[f'{variant}_seed{s}'] for s in SEEDS]);ensemble=ok[f'{variant}_ensemble5']
        matrices[(variant,'single_mean')]=singles;matrices[(variant,'ensemble')]=ensemble[None]
        rs=[r for r in runs if r['variant']==variant];s,f,p=variant.split('_')
        summary.append({'variant':variant,'scope':s,'feature':f,'position':p,
            'mean_accuracy':float(singles.mean()*100),'sample_sd':float(singles.mean(axis=1).std(ddof=1)*100),
            'D_mean_accuracy':float(singles[:,hard].mean()*100),'D_sample_sd':float(singles[:,hard].mean(axis=1).std(ddof=1)*100),
            'seed_accuracies':(singles.mean(axis=1)*100).tolist(),'ensemble_correct':int(ensemble.sum()),
            'ensemble_accuracy':float(ensemble.mean()*100),'ensemble_D_correct':int(ensemble[hard].sum()),
            'parameters':rs[0]['parameters'],'train_bce_all':statistics.mean(r['final_metrics']['train']['bce_all'] for r in rs),
            'train_bce_D':statistics.mean(r['final_metrics']['train']['bce_disagreement'] for r in rs),
            'validation_bce_all':statistics.mean(r['final_metrics']['validation']['bce_all'] for r in rs),
            'validation_bce_D':statistics.mean(r['final_metrics']['validation']['bce_disagreement'] for r in rs),
            'train_D_accuracy':statistics.mean(r['final_metrics']['train']['accuracy_disagreement'] for r in rs)})
        for kind,values in [('single_mean',singles),('ensemble',ensemble[None])]:
            for group,mask in masks.items():
                subgroups.append({'variant':variant,'kind':kind,'subgroup':group,'questions':int(mask.sum()),
                    'accuracy':float(values[:,mask].mean()*100),'correct_mean':float(values[:,mask].sum(axis=1).mean())})
    write(OUT/'summary.json',summary);csv_write(OUT/'summary.csv',[{**r,'seed_accuracies':str(r['seed_accuracies'])} for r in summary])
    write(OUT/'subgroups.json',subgroups);csv_write(OUT/'subgroups.csv',subgroups)
    rng=np.random.default_rng(20261017);contrasts=[];repairs=[]
    for kind in ('single_mean','ensemble'):
        for name,coefs,primary in definitions():
            diff=sum(v*matrices[(k,kind)] for k,v in coefs.items())
            for group,mask in masks.items():
                by_seed=diff[:,mask].mean(axis=1)*100;point=diff[:,mask].mean(axis=0)
                contrasts.append({'contrast':name,'kind':kind,'subgroup':group,'questions':int(mask.sum()),
                    'difference_pp':float(point.mean()*100),'ci95_pp':interval(point,rng),'by_seed_pp':by_seed.tolist(),
                    'positive_seeds':int((by_seed>1e-10).sum()),'negative_seeds':int((by_seed<-1e-10).sum()),
                    'primary':primary and kind=='single_mean' and group in ('all','disagreement')})
                if len(coefs)==2:
                    left=next(k for k,v in coefs.items() if v==1);right=next(k for k,v in coefs.items() if v==-1)
                    a=matrices[(left,kind)][:,mask];b=matrices[(right,kind)][:,mask]
                    repairs.append({'contrast':name,'kind':kind,'subgroup':group,
                        'repairs_mean':float((a>b).sum(axis=1).mean()),'new_errors_mean':float((a<b).sum(axis=1).mean()),
                        'net_repairs_mean':float((a-b).sum(axis=1).mean())})
    write(OUT/'paired_comparisons.json',contrasts)
    csv_write(OUT/'paired_comparisons.csv',[{**r,'ci95_pp':str(r['ci95_pp']),'by_seed_pp':str(r['by_seed_pp'])} for r in contrasts])
    write(OUT/'primary_results.json',[r for r in contrasts if r['primary']]);write(OUT/'repairs.json',repairs);csv_write(OUT/'repairs.csv',repairs)
    changes=[]
    for scope in SCOPES:
        for feature in FEATURES:
            for seed in SEEDS:
                a=choice[f'{scope}_{feature}_Late_seed{seed}'];b=choice[f'{scope}_{feature}_Early_seed{seed}']
                different=a!=b;different_answer=np.array([answers[q][a[q]]!=answers[q][b[q]] for q in range(n)])
                changes.append({'scope':scope,'feature':feature,'seed':seed,'candidate_changes':int(different.sum()),
                    'answer_changes':int(different_answer.sum()),'same_answer_candidate_changes':int((different&~different_answer).sum())})
    write(OUT/'choice_changes.json',changes)
    np.savez(OUT/'per_question_validation.npz',labels=y,valid=valid,disagreement=hard,**choice);write(OUT/'question_ids.json',[r['id'] for r in bank])
    history=[]
    for run in runs:
        for h in read(HERE/run['run_dir']/'history.json'):
            history.append({'variant':run['variant'],'seed':run['seed'],
                **{k:h[k] for k in ('budget_epoch','optimizer_steps','draws','eligible_pool_passes','D_question_draws','stochastic_train_bce')},
                **{f'{split}_{key}':v for split in ('train','validation') for key,v in h[split].items()}})
    csv_write(OUT/'learning_curves.csv',history)
    costs={'training_runs':40,'updates_each':380,'total_updates':15200,'draws_each':95100,
        'total_question_draws':3804000,'optimization_seconds':sum(r['fit_seconds'] for r in runs),
        'diagnostic_seconds':sum(r['diagnostic_seconds'] for r in runs),
        'training_plus_diagnostics_seconds':sum(r['total_training_and_diagnostics_seconds'] for r in runs),
        'official_evaluations':48,'fresh_official_evaluation_seconds':sum(r['fresh_total_seconds'] for r in records),
        'additional_searches':0,'test_used':False}
    write(OUT/'cost.json',costs)
    exposures=[]
    for scope in SCOPES:
        r=next(r for r in runs if r['scope']==scope)
        exposures.append({'scope':scope,'eligible_questions':r['eligible_questions'],'updates':380,'draws':95100,
            'D_question_draws':r['D_question_draws'],'eligible_pool_passes':r['eligible_pool_passes']})
    write(OUT/'exposure_comparison.json',exposures)
    environment=read(OUT/'environment.json')
    primary_full=[r for r in contrasts if r['primary'] and r['subgroup']=='all']
    effect_names={'scope_main: D-All':'集中训练分歧题的平均效果',
        'early_at_C_main':'共识特征早期相对晚期注入',
        'early_feature_interaction':'早期注入是否特别帮助共识而非解析有效性',
        'early_scope_interaction':'早期注入效果是否因集中训练分歧题而改变'}
    findings=[]
    for r in primary_full:
        if r['ci95_pp'][0]>0:meaning='描述性区间位于零以上，仍需独立评价确认'
        elif r['ci95_pp'][1]<0:meaning='描述性区间位于零以下，仍需独立评价确认'
        else:meaning='描述性区间包含零，不能确认稳定差异'
        findings.append(f'- {effect_names[r["contrast"]]}：{ci(r)}pp；{meaning}。')
    lines=['# 训练分歧题与早期共识条件化：2×2×2实验','',
        '日期：2026-10-17（北京时间）。八组×五种子共40次训练，每次固定380次更新、95100次题目抽样；无早停、轮次选择、学习率搜索或优胜者选择。','',
        '## 主要发现','',*findings,'',
        '上面四项均为预设开发比较，未做多重比较校正。正向点估计不等于确认收益，负结果也不能推广为任何训练范围或注入方式都无效；具体单元与困难子组如下。','',
        '## 固定设计','',
        '- All使用4755道训练题，D仅使用2114道至少包含两个不同有效数值答案的题；分组只依赖官方解析器与输入候选，不依赖正确标签。所有组共用仅由全部训练输入拟合的标准化统计。','- P仅解析有效性，C使用原六维解析与一致性特征；P屏蔽五个共识维度。','- Early在编码器后、FFN与注意力前加上Wf；Late在交互及归一化后、评分头前加上Wf。均为6→128无偏置投影、零初始化，参数量均为272769，评分头不变。','- 固定S2四头主体，BCE、AdamW lr0.0003、weight_decay0.001、dropout0.1、clip1，batch256，末batch147，20个预算轮各19次更新。','- 每个种子的八组初始权重与函数逐项一致；同一scope内P/C及Early/Late共享精确抽样顺序。D在池内循环打乱，跨预算轮保留剩余索引。','',
        '## 统一推理','',
        '官方解析后只有一个有效答案组时直接选第一个有效候选，不加载或调用神经模型；至少两个有效组时读取四个候选embedding、计算分数，再排除解析无效候选。全部无效时使用神经打分回退，保证返回合法索引。','',
        '验证798题中443题确定性旁路、355题由模型选择。完整官方分数=(443+分歧题正确数)/798；两种准确率并非独立证据。全部48个预测器（40单模型、8五成员集成）都使用相同规则。','',
        '## 八组主结果','',
        '| scope / 特征 / 位置 | 完整验证五种子均值±SD % | 分歧355题均值±SD % | 集成正确数/798 | 集成分歧正确数/355 |',
        '|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["variant"]} | {r["mean_accuracy"]:.3f} ± {r["sample_sd"]:.3f} | {r["D_mean_accuracy"]:.3f} ± {r["D_sample_sd"]:.3f} | {r["ensemble_correct"]} | {r["ensemble_D_correct"]} |')
    lines+=['','均值为五个单模型的统计，集成是另一个固定预测器，不能混为同一种准确率；SD是样本标准差。','',
        '## 预设主要比较','',
        'D-All主效应等权平均四个特征/位置单元；共识早期效应等权平均两种训练范围；early_feature_interaction为(C早−晚)−(P早−晚)，再平均训练范围；early_scope_interaction为(D早−晚)−(All早−晚)，再平均特征。','',
        '| 比较 | 完整验证差值pp [95%区间] | 分歧题差值pp [95%区间] | 完整验证正向种子/5 |','|---|---:|---:|---:|']
    for name,_,primary in definitions():
        if primary:
            a=next(r for r in contrasts if r['contrast']==name and r['kind']=='single_mean' and r['subgroup']=='all')
            b=next(r for r in contrasts if r['contrast']==name and r['kind']=='single_mean' and r['subgroup']=='disagreement')
            lines.append(f'| {name} | {ci(a)} | {ci(b)} | {a["positive_seeds"]}/5 |')
    lines+=['','区间为10000次按题目配对bootstrap，先对同种子正确性平均。它们条件于已拟合模型、当前协议及反复验证开发背景；不包含调参不确定性，未校正多重比较。','',
        '## 单元配对与额外交互','', '| 比较 | 完整验证差值pp [95%区间] |','|---|---:|']
    for r in contrasts:
        if r['kind']=='single_mean' and r['subgroup']=='all' and not r['primary']:lines.append(f'| {r["contrast"]} | {ci(r)} |')
    lines+=['','## 困难子组：五种子均值','',
        '| 配置 | 单正确117 % | 错误共识60 % | 正确共识混合271 % | 并列88 % |','|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        values=[next(r['accuracy'] for r in subgroups if r['variant']==variant and r['kind']=='single_mean' and r['subgroup']==g) for g in ['one_correct','wrong_unique_plurality','correct_plurality_mixed','plurality_tie_mixed']]
        lines.append('| '+variant+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','单正确与错误共识子组重叠；不能累加。错误共识等结果是官方标签规则下的诊断，非独立benchmark。','',
        '## 学习曲线与共同曝光预算','',
        f'![Factorial results]({OUT.as_posix()}/factorial_results.png)','',
        f'![Learning curves]({OUT.as_posix()}/learning_curves.png)','',
        '曲线完整保留21个测量点（初始及20个预算轮）；训练与验证均为eval模式，诊断不消耗Torch训练随机状态。最后预算轮为唯一检查点，曲线不用于改选轮次。','',
        '| scope | 有效训练池 | 总抽样 | 更新 | 分歧题抽样 | 池遍历数 |','|---|---:|---:|---:|---:|---:|']
    for r in exposures:lines.append(f'| {r["scope"]} | {r["eligible_questions"]} | {r["draws"]} | {r["updates"]} | {r["D_question_draws"]} | {r["eligible_pool_passes"]:.3f} |')
    lines+=['','D删除了2641道确定性可决策题，同时增加了困难题的重复曝光并改变候选正例比例。因此检验的是等计算预算下的训练分配方案，不能把差值全部归因为消除了无用梯度。','',
        '## 训练拟合与资源','',
        '| 配置 | 训练全题BCE | 验证全题BCE | 训练分歧BCE | 验证分歧BCE | 训练分歧准确率 % |','|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["variant"]} | {r["train_bce_all"]:.4f} | {r["validation_bce_all"]:.4f} | {r["train_bce_D"]:.4f} | {r["validation_bce_D"]:.4f} | {r["train_D_accuracy"]:.3f} |')
    lines+=['','BCE均使用同一最后检查点、相同未加权题集测量，避免直接比较不同采样目标的随机训练损失。训练准确率只作为官方标签规则下的拟合诊断。','',
        f'40次训练加诊断{costs["training_plus_diagnostics_seconds"]:.1f}秒，其中优化{costs["optimization_seconds"]:.1f}秒、逐轮诊断{costs["diagnostic_seconds"]:.1f}秒；48次fresh预测加独立评分{costs["fresh_official_evaluation_seconds"]:.1f}秒。初始化、预处理、统计、绘图、部署包复核与原文件哈希另计。','',
        '| 集成 | fresh完整预测+评分秒 | 解包后资产MiB |','|---|---:|---:|']
    for r in records:
        if r['members']==5:lines.append(f'| {r["variant"]} | {r["fresh_total_seconds"]:.3f} | {r["asset_bytes"]/1024**2:.3f} |')
    lines+=['',f'本机{environment["cpu"]}；Python {environment["python"].split()[0]}、NumPy {environment["numpy"]}、Torch {environment["torch"]}；CPU串行，Torch4/interop1，OpenBLAS1。计时包括进程启动、导入、输入/资产加载、解析与全部成员及独立评分；没有官方Xeon或完整测试集认证。','',
        '## 验证及解释边界','',
        '- 同容量、初始函数、P屏蔽、直接注意力条件化、循环采样、候选排列等变、单题/batch和有限梯度检查均通过。每个训练模型的最终检查点另做等变和batch检查。','- 新进程输出与保存最后权重的全bank诊断推理逐项核对；443道旁路题全部正确。全部48个预测器的新ID、重排及单题、8个独立集成包复核见verification.json。','- Early相对Late同时改变特征影响FFN、注意力与归一化的路径；不能把差异唯一归因于注意力权重。注意力前单独注入、仅Q/K或仅V未包含于本轮。','- 两个注入位置均为新增匹配的加法投影；不能直接把此前拼接头的分数当作本轮Late基线。','- 早期共识可能改善比较，也可能更早强化错误共识；整体、分歧题及错误共识必须共同解释。训练BCE下降不等于选择准确率提高。','- 只有一个有效答案组的直接选择依赖官方至少有一个正确候选保证及原解析规则；不声称所有外部银行都具有相同正确性保证。','- 原文件只做按字节哈希保护；训练与预测不使用测试输入、参考答案或ID查表。无外部数据、额外预训练权重、API或新候选生成。','- 新设计实现与报告由Codex AI辅助，用户需审阅理解。官方数据、parser和scorer原样保留。','',
        '## 产物与复现','',
        'summary.csv/JSON为八组；primary_results.json为预设主要比较；paired_comparisons.json为全部单元与交互子组；repairs.csv为修正/改错；learning_curves.csv与各run的history为完整曲线；draw-counts.npy逐题记录训练曝光；cost.json和official_results.json记录成本和48份官方结果。README.md提供运行命令；portable中八个集成包独立可加载。未按本轮验证结果挑选新提交模型。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PRIMARY',json.dumps([r for r in contrasts if r['primary'] and r['subgroup']=='all'],ensure_ascii=False),flush=True)
    for r in summary:print('SUMMARY',json.dumps(r,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
