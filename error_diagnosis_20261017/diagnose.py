"""Local prediction diagnostics; no inference, training, or test-bank reads.

Long traces stay local. Only eight short, unreviewed evidence cards are exported.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,csv,hashlib
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent;CODE=HERE.parent;ROOT=CODE.parent
sys.path.insert(0,str(CODE))
from common import read_jsonl,parse_answer,check_data,DATA

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def excerpt(text,limit=650):
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if len(text)<=limit:return text,False
    prefix='\n'.join(lines[:2])[:130]
    tail='\n'.join(lines)[-480:]
    return prefix+'\n[... middle omitted; expand source if needed ...]\n'+tail,True
def main():
    check_data(DATA,['validation'])
    bank=read_jsonl(DATA/'candidates/gsm8k_validation.jsonl');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    assert len(bank)==len(refs)
    n=len(bank);ix=np.arange(n);answers=[];gold=[]
    for row,ref in zip(bank,refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        answers.append([parse_answer(c['text']) for c in row['candidates']]);gold.append(parse_answer(ref['answer']))
    y=np.array([[v is not None and v==g for v in a] for a,g in zip(answers,gold)],bool)
    assert y.any(axis=1).all()
    paths={
      'Pool_P':CODE/'interaction_consensus_20261017/results/pipelines/Pool_P_ensemble5_filtered/predictions.json',
      'S2_C':CODE/'interaction_consensus_20261017/results/pipelines/S2_C_ensemble5_filtered/predictions.json',
      'S2_route':CODE/'validity_filter_20261017/results/pipelines/s2_ensemble5_route_filtered/predictions.json',
      'S2_P_control':CODE/'interaction_consensus_20261017/results/pipelines/S2_P_ensemble5_filtered/predictions.json'}
    choices={};oks={}
    for name,path in paths.items():
        p=read(path);assert p['ids']==[row['id'] for row in bank]
        choices[name]=np.array(p['selected'],int);oks[name]=y[ix,choices[name]]
    primary=['Pool_P','S2_C','S2_route'];matrix=np.stack([oks[k] for k in primary])
    group_count=np.array([len(set(v for v in a if v is not None)) for a in answers]);hard=group_count>=2
    assert int(hard.sum())==355
    union=~matrix.all(axis=0);common=~matrix.any(axis=0);split=matrix.any(axis=0)&~matrix.all(axis=0)
    assert int(union.sum())==int(common.sum()+split.sum())
    rows=[];tags={};group_patterns=Counter()
    for q,row in enumerate(bank):
        counts=Counter(v for v in answers[q] if v is not None);top=max(counts.values());winners=[v for v,c in counts.items() if c==top]
        wrong_plurality=len(winners)==1 and winners[0]!=gold[q]
        singleton=all(c==1 for c in counts.values())
        t=[]
        if common[q]:t.append('all_three_wrong')
        if split[q]:t.append('methods_complementary')
        if wrong_plurality:t.append('wrong_unique_plurality')
        if singleton:t.append('all_valid_groups_singleton')
        if None in answers[q]:t.append('some_candidate_unparseable')
        if not oks['S2_C'][q] and oks['S2_P_control'][q]:t.append('C_damage_vs_same_architecture_P')
        if oks['S2_C'][q] and not oks['S2_P_control'][q]:t.append('C_gain_vs_same_architecture_P')
        tags[q]=t
        if union[q]:group_patterns[','.join(map(str,sorted(counts.values(),reverse=True)))]+=1
        rows.append({'row':q,'id':row['id'],'needs_model':bool(hard[q]),'correct_candidates':int(y[q].sum()),
          'valid_answer_groups':int(group_count[q]),'answer_group_sizes':str(sorted(counts.values(),reverse=True)),
          'primary_any_wrong':bool(union[q]),'primary_all_wrong':bool(common[q]),'tags':'|'.join(t),
          **{f'{name}_choice':int(selected[q]) for name,selected in choices.items()},
          **{f'{name}_correct':bool(v[q]) for name,v in oks.items()}})
    with (HERE/'question_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rng=np.random.default_rng(20261017);selected=[]
    strata=[('common_wrong_consensus',common&np.array(['wrong_unique_plurality' in tags[q] for q in ix])),
      ('common_singleton_groups',common&np.array(['all_valid_groups_singleton' in tags[q] for q in ix])),
      ('S2C_wrong_PoolP_right',~oks['S2_C']&oks['Pool_P']),
      ('S2C_right_PoolP_wrong',oks['S2_C']&~oks['Pool_P']),
      ('S2C_wrong_route_right',~oks['S2_C']&oks['S2_route']),
      ('S2C_right_route_wrong',oks['S2_C']&~oks['S2_route']),
      ('C_damage_same_architecture',~oks['S2_C']&oks['S2_P_control']),
      ('C_gain_same_architecture',oks['S2_C']&~oks['S2_P_control'])]
    for stratum,mask in strata:
        pool=[int(q) for q in np.flatnonzero(mask&union) if int(q) not in [v[1] for v in selected]]
        if pool:selected.append((stratum,int(rng.choice(pool))))
    cards=[]
    for stratum,q in selected:
        wrong=next((int(choices[k][q]) for k in ['S2_C','Pool_P','S2_route'] if not oks[k][q]),None)
        right=int(np.flatnonzero(y[q])[0]);candidate_cards=[]
        for index in [wrong,right]:
            if index is None:continue
            snippet,truncated=excerpt(bank[q]['candidates'][index]['text'])
            candidate_cards.append({'index':index,'parsed_answer':answers[q][index],'officially_correct':bool(y[q,index]),
                'text_excerpt':snippet,'truncated':truncated})
        cards.append({'card':len(cards)+1,'sample_stratum':stratum,'row':q,'id':bank[q]['id'],
          'question':bank[q]['question'],'official_answer':gold[q],'all_parsed_answers':answers[q],
          'method_choices':{k:int(v[q]) for k,v in choices.items()},'observable_tags':tags[q],
          'candidate_evidence':candidate_cards,'human_reason':'UNREVIEWED',
          'source_file':str(DATA/'candidates/gsm8k_validation.jsonl'),'source_line':q+1,
          'note':'Prediction-derived tags are not causal labels. Expand omitted text before assigning a reasoning error.'})
    with (HERE/'review_cards.jsonl').open('w',encoding='utf-8') as f:
        for card in cards:f.write(json.dumps(card,ensure_ascii=False)+'\n')
    full_chars=sum(len(row['question'])+sum(len(c['text']) for c in row['candidates']) for row in bank)
    card_chars=sum(len(c['question'])+sum(len(v['text_excerpt']) for v in c['candidate_evidence']) for c in cards)
    stats={'validation_questions':n,'deterministic_easy':int((group_count==1).sum()),'disagreement':int(hard.sum()),
       'primary_methods':primary,'review_error_union':int(union.sum()),'all_primary_wrong':int(common.sum()),
       'partly_correct_partly_wrong':int(split.sum()),'review_cards':len(cards),
       'selection_seed':20261017,'manual_causal_annotations_completed':0,
       'primary_correct':{k:int(oks[k].sum()) for k in primary},
       'error_group_size_patterns':dict(group_patterns),
       'full_bank_question_candidate_characters':full_chars,'eight_card_question_excerpt_characters':card_chars,
       'character_reduction_percent':100*(1-card_chars/full_chars),'character_counts_are_not_token_counts':True,
       'official_scores_recomputed':False,'test_used':False}
    write(HERE/'summary.json',stats)
    write(HERE/'sources.json',{'predictions':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},
       'script_sha256':sha(Path(__file__)),'data_split':'validation'})
    lines=['# 低上下文成本错题诊断索引','',
      f'自动分析全部{n}道验证题，模型只需判断的题有{stats["disagreement"]}道；三个代表预测器的错题去重后为{stats["review_error_union"]}道，其中{stats["all_primary_wrong"]}道共同错、{stats["partly_correct_partly_wrong"]}道体现互补。','',
      f'第一批只导出{len(cards)}张分层抽取的短证据卡：题目、官方答案、四个解析答案、各方法选择，以及一个错误候选和一个正确候选的截断文本。因果原因均标为UNREVIEWED。','',
      '## 阅读流程','',
      '1. question_index.csv用于本地筛选与统计，不需要整份送入模型。','2. 每次只读取review_cards.jsonl中的2–4张卡，优先识别可观察错误；无法判断时，按source_file/source_line回看相应候选全文。','3. 保留reason、evidence、confidence和needs_more_context四项标注；没有证据时留待复核，不能把统计标签当作推理错误。','4. 每批只保留结构化标注与新增模式摘要，避免在对话里重复粘贴历史推理。','5. 根据模式是否重复，再扩大抽样；频率统计仍以完整索引为准，分层8张卡不代表总体发生率。','',
      '## 自动标签与人工原因分开','',
      '共同失败、方法互补、错误最高票、单例答案组、解析失败与同结构C/P纠错方向可由预测及官方标签直接得到。条件遗漏、单位错误、算术错误、参考答案歧义等必须人工审阅文本，不能由程序直接认定。','',
      f'全部题目加候选文本共{full_chars:,}字符；第一批题目与证据片段共{card_chars:,}字符，减少约{stats["character_reduction_percent"]:.1f}%。这是字符量比较，不是精确token测量，实际还包括少量JSON元数据。','',
      '本轮只做自动索引和待审证据整理，没有重新训练、预测、评分或完成全量人工错因诊断。训练规则或学习参数不能由验证人工标签直接拟合；后续方法应在训练数据上建立并冻结后评价。','',
      '源数据、预测与评分原样保留；所有输出只用于研究诊断，不可作为部署查表。未读取测试集语义内容，原文件保护扫描仅按字节核对。']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    before=read(HERE/'protected_before.json');changed=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write(HERE/'protected_verification.json',{'passed':not changed,'files_checked':len(before),'changed':changed})
    assert not changed,changed
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
