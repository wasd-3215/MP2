"""Train-only extraction, coverage and seeded samples; does not load validation."""
import runtime
from utils import HERE,OUT,FUSION,read,write,sha
import time
import numpy as np
from common import load_inputs,parse_answer,read_jsonl,DATA,check_data
from evidence import extract,NAMES

def main():
    assert read(OUT/'extractor_checks.json')['passed']
    assert read(OUT/'extractor_checks.json')['source_sha256']==sha(HERE/'evidence.py')
    bank,_=load_inputs('train')
    check_data(DATA,['train'])
    refs=read_jsonl(DATA/'gsm8k_train.jsonl')
    assert [r['id'] for r in bank]==[r['id'] for r in refs]
    y=np.asarray([[parse_answer(c['text'])==parse_answer(g['answer']) for c in r['candidates']] for r,g in zip(bank,refs)],bool)
    assert np.array_equal(y,np.load(FUSION/'results/train-labels.npy'))
    start=time.perf_counter()
    features,details=extract(bank,detailed=True)
    seconds=time.perf_counter()-start
    np.save(OUT/'train-evidence.npy',features,allow_pickle=False)
    np.save(OUT/'train-labels.npy',y,allow_pickle=False)
    counts=[]
    for i,name in enumerate(NAMES):
        active=features[...,i]!=0
        counts.append({'feature':name,'candidate_nonzero':int(active.sum()),'question_nonzero':int(active.any(1).sum()),
          'nonzero_officially_correct_candidates':int((active&y).sum()),
          'nonzero_officially_wrong_candidates':int((active&~y).sum()),'mean':float(features[...,i].mean()),
          'max':float(features[...,i].max())})
    summary={'train_questions':len(bank),'train_candidates':4*len(bank),'feature_names':NAMES,
      'feature_statistics':counts,'extraction_seconds':seconds,'extractor_sha256':sha(HERE/'evidence.py'),
      'feature_sha256':sha(OUT/'train-evidence.npy'),'label_sha256':sha(OUT/'train-labels.npy'),
      'validation_loaded':False,'test_used':False,
      'association_is_not_extractor_error_rate':True}
    write(OUT/'training_audit.json',summary)
    rng=np.random.default_rng(20261017)
    strata=[('contradiction_official_correct',lambda d:d['arithmetic']['violations']>0 and y[d['row'],d['candidate']],8),
      ('contradiction_official_wrong',lambda d:d['arithmetic']['violations']>0 and not y[d['row'],d['candidate']],8),
      ('format_conflict',lambda d:features[d['row'],d['candidate'],0]>0,4),
      ('all_checked_consistent',lambda d:d['arithmetic']['checked_pairs']>0 and d['arithmetic']['violations']==0,4),
      ('only_unknown_arithmetic',lambda d:d['arithmetic']['unknown_pair_observations']>0 and d['arithmetic']['checked_pairs']==0,4)]
    samples=[];used=set()
    for name,condition,n in strata:
        pool=[i for i,d in enumerate(details) if condition(d) and i not in used]
        for ix in rng.choice(pool,size=min(n,len(pool)),replace=False):
            used.add(int(ix)); d=details[int(ix)];q,i=d['row'],d['candidate'];text=bank[q]['candidates'][i]['text']
            pairs=d['arithmetic']['pairs']
            shown=[p for p in pairs if p['violates']][:2] or pairs[:2]
            offsets=[p['source_offset'] for p in shown]
            snippet=text[max(0,min(offsets)-80):min(len(text),max(offsets)+450)] if offsets else text[-420:]
            samples.append({'sample':len(samples)+1,'stratum':name,'row':q,'candidate':i,'id':bank[q]['id'],
              'officially_correct':bool(y[q,i]),'features':features[q,i].tolist(),'format':d['format'],
              'checked_pairs':d['arithmetic']['checked_pairs'],'violations':d['arithmetic']['violations'],
              'evidence_pairs':shown,'snippet':snippet,'review':'UNREVIEWED'})
    with (OUT/'training_audit_samples.jsonl').open('w',encoding='utf-8') as f:
        for sample in samples:f.write(__import__('json').dumps(sample,ensure_ascii=False)+'\n')
    print(__import__('json').dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
