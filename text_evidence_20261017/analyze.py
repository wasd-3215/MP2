"""Paired development-set analysis; no further fitting or parameter search."""
import runtime
from utils import HERE,OUT,CODE,OLD,read,write
import csv
import time
import numpy as np
from common import load_inputs,read_jsonl,DATA,parse_answer
from evidence import extract

def interval(d):
    rng=np.random.default_rng(20261017);samples=np.empty(10000)
    for i in range(0,10000,500):samples[i:i+500]=100*d[rng.integers(0,len(d),(500,len(d)))].mean(1)
    return np.percentile(samples,[2.5,97.5]).tolist()

def main():
    records=read(OUT/'official_results.json')
    bank,_=load_inputs('validation');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    assert [r['id'] for r in bank]==[r['id'] for r in refs]
    answers=[[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    gold=[parse_answer(r['answer']) for r in refs]
    y=np.asarray([[a is not None and a==g for a in row] for row,g in zip(answers,gold)],bool)
    tick=time.perf_counter();features,details=extract(bank,detailed=True);extract_seconds=time.perf_counter()-tick
    np.save(OUT/'validation-evidence.npy',features,allow_pickle=False)
    choices={r['name']:np.asarray(read(OUT/'pipelines'/r['name']/'predictions.json')['selected']) for r in records}
    assert choices['baseline'].tolist()==read(OLD/'results/pipelines/S2_C_ensemble5_filtered/predictions.json')['selected']
    correct={n:y[np.arange(len(bank)),c] for n,c in choices.items()}
    base=correct['baseline']
    previous=[json.loads(line) for line in (CODE/'common_failure_diagnosis_20261017/evidence_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    common_ids={c['id'] for c in previous};defect_ids={c['id'] for c in previous if c['case'] in [2,13,17,20,24,27]}
    common=np.asarray([r['id'] in common_ids for r in bank]);defect=np.asarray([r['id'] in defect_ids for r in bank])
    active=np.asarray([len(set(a for a in row if a is not None))>=2 for row in answers])
    wrong_plurality=[]
    from collections import Counter
    for a,g in zip(answers,gold):
        counts=Counter(v for v in a if v is not None);top=max(counts.values(),default=0)
        leaders=[v for v,n in counts.items() if n==top]
        wrong_plurality.append(len(leaders)==1 and leaders[0]!=g)
    masks={'all':np.ones(len(bank),bool),'active_multi_answer':active,'common_failures42':common,
      'one_correct_candidate':y.sum(1)==1,'wrong_unique_plurality':np.asarray(wrong_plurality),
      'six_flawed_correct_trace_cases':defect,'any_format_evidence':(features[...,:3]!=0).any((1,2)),
      'any_arithmetic_violation':(features[...,4]>0).any(1)}
    results=[]
    for r in records:
        c=correct[r['name']];delta=c.astype(int)-base.astype(int)
        changed=np.asarray([a[int(i)]!=a[int(j)] for a,i,j in zip(answers,choices[r['name']],choices['baseline'])])
        results.append({**r,'repairs':int((c&~base).sum()),'harms':int((~c&base).sum()),'net':int(delta.sum()),
          'delta_pp':float(delta.mean()*100),'paired_bootstrap95_pp':interval(delta),
          'numerical_answer_changes':int(changed.sum()),'subgroups':{name:{'n':int(m.sum()),'correct':int(c[m].sum()),
           'repairs':int((c&~base&m).sum()),'harms':int((~c&base&m).sum())} for name,m in masks.items()}})
    comparison={}
    for other in ['format','arithmetic']:
        d=correct['combined'].astype(int)-correct[other].astype(int)
        comparison['combined_minus_'+other]={'net':int(d.sum()),'delta_pp':float(d.mean()*100),'ci95_pp':interval(d)}
    common_audit=[]
    for c in previous:
        q=c['row'];wrong=c['method_choices']['S2_C'];right=c['correct_indices']
        common_audit.append({'case':c['case'],'row':q,'id':c['id'],'baseline_wrong_candidate':wrong,
           'wrong_features':features[q,wrong].tolist(),'right_features':{str(i):features[q,i].tolist() for i in right},
           'any_detected_violation':bool((features[q,:,4]>0).any()),
           'wrong_candidate_violation':bool(features[q,wrong,4]>0),
           'predictor_correct':{n:bool(v[q]) for n,v in correct.items()}})
    write(OUT/'common42_coverage.json',common_audit)
    summary={'results':results,'combined_comparisons':comparison,'test_used':False,'validation_extraction_seconds':extract_seconds,
      'shared42_violation_questions':sum(c['any_detected_violation'] for c in common_audit),
      'shared42_baseline_wrong_candidate_violation':sum(c['wrong_candidate_violation'] for c in common_audit),
      'intervals_descriptive_development_set':True}
    write(OUT/'summary.json',summary)
    detail_lookup={(d['row'],d['candidate']):d for d in details}
    with (OUT/'answer_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['method','row','id','gold','baseline_answer','new_answer','repair','harm','old_features','new_features']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,selection in choices.items():
            if name=='baseline':continue
            for q,(old,new) in enumerate(zip(choices['baseline'],selection)):
                if answers[q][int(old)]==answers[q][int(new)]:continue
                writer.writerow({'method':name,'row':q,'id':bank[q]['id'],'gold':gold[q],'baseline_answer':answers[q][int(old)],
                  'new_answer':answers[q][int(new)],'repair':int(correct[name][q] and not base[q]),'harm':int(not correct[name][q] and base[q]),
                  'old_features':features[q,int(old)].tolist(),'new_features':features[q,int(new)].tolist()})
    lines=['# 输出结构与算术证据校正实验','',
      '冻结S2-C五种子主模型，训练集折外预测上拟合候选级线性分数残差。三次拟合分别只用输出结构、只用算术、合并证据；原S2-C为基线。所有规则在本轮验证评价前冻结。','',
      '| 方法 | 正确/798 | 准确率 | 修复/损害 | 净增 | 配对95%区间（百分点） |','|---|---:|---:|---:|---:|---|']
    for r in results:
        lo,hi=r['paired_bootstrap95_pp'];lines.append(f'| {r["name"]} | {r["correct"]}/798 | {r["accuracy"]:.3f}% | {r["repairs"]}/{r["harms"]} | {r["net"]:+d} | [{lo:.3f}, {hi:.3f}] |')
    lines+=['','## 固定设计','',
      '输出结构3项：最后显式数值框与官方实际解析不一致、数字前缀的代数框、文本确实结束于数学运算符。算术4项：可检查的不同等式数log1p、不成立等式数log1p、不成立比例、最后检查等式是否不成立。','',
      '只检查明确数值表达式，支持小数、分数、加减乘除和受限整数次幂。使用受限AST遍历及Fraction，绝不执行候选代码；变量、单位、含百分号的等式及可疑单位外置数学段不确定时跳过。绝对容差1e-6、相对容差0.001，允许常见小数近似；超过表达式/节点/数值/等式数量限制时不作错误结论。','',
      '校正为logit(五种子平均概率)+文本特征·β，原模型系数固定1、无截距、不缩放原始特征。仅使用训练集最终数值正确性标签，最小化候选BCE之和+0.5*||β||²，固定L2=1，确定性Newton收敛，无超参或阈值搜索。所有证据均为软分数，最终仍按原官方解析过滤None候选，全无效时保留原分数最大值。候选偏移完全相同的题保持原概率排序，避免数值裁剪改变基线。','',
      '## 训练审计与成本','']
    audit=read(OUT/'training_audit.json');fit=read(OUT/'fit_results.json');manual=read(OUT/'manual_training_audit.json')
    lines+=[f'提取训练集19,020个候选耗时{audit["extraction_seconds"]:.2f}秒；最终人工复核{manual["samples_reviewed"]}张训练卡。最初v0曾误读百分比教学写法、外置美分单位和Markdown粗体结尾；已在训练阶段修正并保留audit_history/v0，未在验证数据上修补提取定义。训练审计中的“候选最终正确但等式不成立”不等于提取器误报；两者分别记录。','',
      '| 特征 | 非零候选数 | 其中最终正确 | 其中最终错误 |','|---|---:|---:|---:|']
    for r in audit['feature_statistics']:lines.append(f'| {r["feature"]} | {r["candidate_nonzero"]} | {r["nonzero_officially_correct_candidates"]} | {r["nonzero_officially_wrong_candidates"]} |')
    lines+=['',f'新增主模型训练0次，线性校正拟合3次；准备及拟合阶段合计{fit["total_seconds"]:.2f}秒（不含前述提取审计）。已有基础模型OOF逐题留出，但校正器拟合全体OOF，以下训练表现为拟合内诊断，不是校正器无偏交叉验证成绩。','',
      '| 方法 | 拟合内净增题 | β |','|---|---:|---|']
    for r in fit['runs']:lines.append(f'| {r["name"]} | {r["repairs_vs_base"]-r["harms_vs_base"]:+d} | {r["beta"]} |')
    lines+=['','## 分组与42题覆盖','',
      '| 方法 | 多答案355题 | 共同失败42题 | 单正确候选117题 | 错误唯一多数60题 | 过程有错但数值正确6题 |','|---|---:|---:|---:|---:|---:|']
    for r in results:
        g=r['subgroups'];lines.append(f'| {r["name"]} | {g["active_multi_answer"]["correct"]}/355 | {g["common_failures42"]["correct"]}/42 | {g["one_correct_candidate"]["correct"]}/117 | {g["wrong_unique_plurality"]["correct"]}/60 | {g["six_flawed_correct_trace_cases"]["correct"]}/6 |')
    lines+=['',f'42道历史共同失败题中，本提取器在{summary["shared42_violation_questions"]}题至少发现一条候选等式不成立，在{summary["shared42_baseline_wrong_candidate_violation"]}题的S2-C原选候选中发现不成立等式。限制来自只处理纯数值，单位或变量表达式不作自动判断。逐题特征及各组修复见common42_coverage.json；这项覆盖审计没有反馈到特征定义或参数。','',
      '## 推理资源与复现','', '| 方法 | 本机完整预测+独立评分秒数 | 解压推理资产MiB |','|---|---:|---:|']
    for r in results:lines.append(f'| {r["name"]} | {r["fresh_total_seconds"]:.3f} | {r["asset_bytes_uncompressed"]/1024**2:.5f} |')
    lines+=['','每组均做全新进程官方预测与独立评分，另做共享便携资产重放。推理资产包括五个S2-C成员、标准化统计、配置和β，NPZ按内部展开计数；没有预测缓存、OOF数组或参考答案。Torch4/interop1、OpenBLAS1，数值工作串行。本机计时不能替代官方参考CPU的30秒认证。','',
      '特征设计受前一轮验证诊断启发，本轮参数和提取修复仅用训练数据。验证集已多次用于开发，配对区间仅作描述，不是独立新数据上的泛化确认。未用测试集。原数据、解析器、评分器、权重与旧实验均只读，保护结果及新输入检查见verification.json。','',
      '复现顺序：check_extractor.py、audit_train.py、固定人工训练审计输入、fit.py、evaluate.py、analyze.py、verify.py；Python3.12，PYTHONDONTWRITEBYTECODE=1且全部带-B。manual_training_audit.json是明确的人工检查输入，程序不自动重做人工判断。协议与验证前哈希分别见protocol.json、frozen_before_validation.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print([(r['name'],r['correct'],r['repairs'],r['harms']) for r in results],flush=True)

if __name__=='__main__':
    import json
    main()
