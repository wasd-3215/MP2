"""Frozen paired comparisons, explicit score contributions and interface checks."""
import runtime
from utils import *
from collections import Counter
import numpy as np
from common import load_inputs,read_jsonl,DATA,parse_answer
from evidence import extract,NAMES
from decisions import base_logits,choose,best

def interval(d):
    if not len(d):return None
    rng=np.random.default_rng(20261018);values=[]
    for _ in range(20):values.extend(100*d[rng.integers(0,len(d),(500,len(d)))].mean(1))
    return np.percentile(values,[2.5,97.5]).tolist()

def main():
    from method import probabilities,predict
    frozen=read(OUT/'frozen_before_validation.json')
    assert frozen['sources']==sources() and all(sha(p)==h for p,h in frozen['inputs'].items())
    bank,arrays=load_inputs('validation');refs=read_jsonl(DATA/'gsm8k_validation.jsonl')
    answers=[[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    gold=[parse_answer(r['answer']) for r in refs]
    y=np.asarray([[a is not None and a==g for a in row] for row,g in zip(answers,gold)],bool)
    tick=time.perf_counter();f,details=extract(bank,detailed=True);extraction=time.perf_counter()-tick
    np.save(OUT/'validation-evidence.npy',f,allow_pickle=False)
    p=probabilities(bank,arrays,members());z=base_logits(p)
    cards=[json.loads(s) for s in (CODE/'common_failure_diagnosis_20261017/evidence_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    common_ids={c['id'] for c in cards};flawed={c['id'] for c in cards if c['case'] in [2,13,17,20,24,27]}
    def wrong_plurality(a,g):
        counts=Counter(v for v in a if v is not None);m=max(counts.values(),default=0)
        leaders=[v for v,n in counts.items() if n==m]
        return len(leaders)==1 and leaders[0]!=g
    masks={'all':np.ones(len(bank),bool),'multi_answer':np.asarray([len({a for a in row if a is not None})>=2 for row in answers]),
      'wrong_plurality':np.asarray([wrong_plurality(a,g) for a,g in zip(answers,gold)]),'one_correct':y.sum(1)==1,
      'common42':np.asarray([r['id'] in common_ids for r in bank]),'flawed_correct6':np.asarray([r['id'] in flawed for r in bank]),
      'any_violation':(f[...,4]>0).any(1)}
    choices={n:np.asarray(read(OUT/'pipelines'/n/'predictions.json')['selected']) for n in frozen['pipelines']}
    correct={n:y[np.arange(len(bank)),v] for n,v in choices.items()};base=correct['B']
    results=[];changes=[]
    for record in read(OUT/'official_results.json'):
        name=record['name'];c=correct[name];d=c.astype(int)-base.astype(int)
        assert int(c.sum())==record['correct']
        config=read(OUT/'pipelines'/name/'pipeline.json')
        assert choose(bank,p,f,config).tolist()==choices[name].tolist()
        groups={s:{'n':int(m.sum()),'correct':int(c[m].sum()),'repairs':int((c&~base&m).sum()),'harms':int((~c&base&m).sum())} for s,m in masks.items()}
        changed=0
        for q,(old,new) in enumerate(zip(choices['B'],choices[name])):
            if answers[q][old]==answers[q][new]:continue
            changed+=1; cols=config['columns'];beta=np.asarray(config['beta'])
            contribution=(f[q,new,cols]-f[q,old,cols])*beta
            changes.append({'method':name,'row':q,'id':bank[q]['id'],'question':bank[q]['question'],'gold':gold[q],
              'old_index':int(old),'new_index':int(new),'old_answer':answers[q][old],'new_answer':answers[q][new],
              'repair':bool(c[q] and not base[q]),'harm':bool(not c[q] and base[q]),
              'baseline_old_minus_new_logit':float(z[q,old]-z[q,new]),
              'new_minus_old_contributions':dict(zip([NAMES[k] for k in cols],contribution.tolist())),
              'old_features':f[q,old].tolist(),'new_features':f[q,new].tolist(),
              'old_checked_pairs':details[4*q+old]['arithmetic']['pairs'],
              'new_checked_pairs':details[4*q+new]['arithmetic']['pairs'],
              'old_text':bank[q]['candidates'][old]['text'],'new_text':bank[q]['candidates'][new]['text']})
        results.append({**record,'repairs':int((c&~base).sum()),'harms':int((~c&base).sum()),'net':int(d.sum()),
          'delta_pp':100*float(d.mean()),'ci95_pp':interval(d),'numerical_changes':changed,'subgroups':groups})
    comparisons={}
    for name,weights in {'V-B':{'V':1,'B':-1},'N-B':{'N':1,'B':-1},'NV-V':{'NV':1,'V':-1},
                         'NV-N':{'NV':1,'N':-1},'interaction':{'NV':1,'N':-1,'V':-1,'B':1}}.items():
        d=sum(w*correct[n].astype(int) for n,w in weights.items())
        comparisons[name]={'net':int(d.sum()),'delta_pp':100*float(d.mean()),'ci95_pp':interval(d)}
    write(OUT/'analysis.json',{'results':results,'comparisons':comparisons,'validation_extraction_seconds':extraction,
      'intervals_descriptive':True,'test_used':False})
    with (OUT/'answer_changes.jsonl').open('w',encoding='utf-8') as out:
        for row in changes:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    # Useful boundary checks; no retraining or source mutation.
    ix=np.asarray([97,3,400,11,777,25,0,48,200,17,700,56,98]);perm=[2,0,3,1]
    subset=[{**bank[i],'id':f'new-audit:{j}'} for j,i in enumerate(ix)]
    small={'embeddings':arrays['embeddings'][ix]}
    pb=[{**r,'candidates':[{**r['candidates'][k],'index':j} for j,k in enumerate(perm)]} for r in subset]
    pf=extract(pb);pp=probabilities(pb,{'embeddings':small['embeddings'][:,perm]},members())
    assert np.array_equal(pf,f[ix][:,perm])
    permutation_error=float(np.max(np.abs(pp-p[ix][:,perm])));assert permutation_error<3e-5
    for n in frozen['pipelines']:
        path=OUT/'pipelines'/n;cfg=read(path/'pipeline.json')
        actual=predict(subset,small,path);assert actual==choices[n][ix].tolist()
        assert predict(subset[:1],{'embeddings':small['embeddings'][:1]},path)==actual[:1]
        assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},path)==[]
        assert all(type(i) is int and 0<=i<4 for i in actual)
        pc=choose(pb,pp,pf,cfg)
        assert all(parse_answer(pb[j]['candidates'][int(k)]['text'])==parse_answer(subset[j]['candidates'][actual[j]]['text']) for j,k in enumerate(pc))
        assert choose(subset,p[ix],np.zeros((len(ix),4,7)),cfg).tolist()==choices['B'][ix].tolist()
        for valid in [False,True]:
            b=[{'id':'boundary','question':'synthetic','candidates':[{'index':i,'text':'#### 7' if valid and i==2 else 'unknown'} for i in range(4)]}]
            selected=choose(b,p[:1],extract(b),cfg)
            assert selected.shape==(1,) and 0<=selected[0]<4
            if valid:assert selected[0]==2
    assert sources()==frozen['sources']
    write(OUT/'verification.json',{'passed':True,'fresh_official_evaluations':5,'portable_replays':5,
      'baseline_and_historical_choices_exact':True,'candidate_permutation_max_probability_error':permutation_error,
      'new_ids_reordered_single_empty_and_parser_fallback_passed':True,'no_evidence_preserves_baseline':True,
      'frozen_hashes_match':True,'test_used':False,'reference_machine_certification':False})
    print([(r['name'],r['correct'],r['repairs'],r['harms']) for r in results],flush=True)
    print(comparisons,flush=True)

if __name__=='__main__':main()
