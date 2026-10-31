"""Post-evaluation descriptive slices. Never fits, routes or changes predictions."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import runtime
from support import CODE,OUT,VARIANTS,SEEDS,read,write,sha
from grouping import layout
from common import load_inputs,read_jsonl,parse_answer,DATA
import numpy as np

def main():
    bank,_=load_inputs('validation');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    assert all(r['id']==q['id'] and r['question']==q['question'] for r,q in zip(bank,refs))
    y=np.asarray([[parse_answer(c['text'])==parse_answer(ref['answer']) for c in r['candidates']] for r,ref in zip(bank,refs)])
    _,membership,valid,counts=layout(bank);sizes=membership.sum(2);gold=(membership&y[:,None,:]).any(2).argmax(1)
    largest=sizes.max(1);unique=(sizes==largest[:,None]).sum(1)==1
    plurality_correct=unique&(sizes[np.arange(len(bank)),gold]==largest)
    masks={'unique_plurality_correct':plurality_correct,'unique_plurality_wrong':unique&~plurality_correct,'plurality_tied':~unique}
    patterns=['+'.join(map(str,sorted(s[s>0].tolist(),reverse=True)))+f';invalid={4-int(v.sum())}' for s,v in zip(sizes,valid)]
    masks.update({p:np.asarray([s==p for s in patterns]) for p in sorted(set(patterns))})
    frozen=np.load(OUT/'question_level_comparison.npz',allow_pickle=False)
    histpath=CODE/'interaction_consensus_20261017'/'results'/'pipelines'/'S2_C_ensemble5_filtered'/'predictions.json'
    historic=read(histpath);assert historic['ids']==[r['id'] for r in bank]
    old=y[np.arange(len(bank)),historic['selected']];assert old.sum()==740
    common=read_jsonl(CODE/'common_failure_diagnosis_20261017'/'results'/'reviewed_cases.jsonl')
    result={}
    for variant in VARIANTS:
        corr=frozen[f'ensemble_{variant}'];singles=frozen[f'singles_{variant}']
        result[variant]={'buckets':{k:{'n':int(m.sum()),'ensemble_correct':int(corr[m].sum()),'mean_single_correct':float(singles[:,m].sum(1).mean()),'historical_S2C_correct':int(old[m].sum())} for k,m in masks.items()},
           'versus_historical_S2C_context_only':{'repairs':int((corr&~old).sum()),'harms':int((~corr&old).sum()),'oracle_union':int((corr|old).sum())},
           'common42_rescued':[{'case':r['case'],'row':r['row'],'category':r['review']['category'],
                    'correct_seed_count':int(singles[:,r['row']].sum()),'review_evidence':r['review']['evidence']} for r in common if corr[r['row']]]}
    write(OUT/'posthoc_diagnostics.json',{'posthoc':True,'predictions_changed':False,'rules_fitted':False,'historical_prediction_sha256':sha(histpath),
         'not_matched_historical_comparison':True,'results':result,'script_sha256':sha(Path(__file__))})
    for v,r in result.items():print(v,r['versus_historical_S2C_context_only'],flush=True)
    print('patterns',{p:patterns.count(p) for p in sorted(set(patterns))},flush=True)

if __name__=='__main__':main()
