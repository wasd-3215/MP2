"""Train-input-only split and prospective audit sampling, before relation development."""
from pathlib import Path
import json
import hashlib
import re
import sys
import time
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
sys.path.insert(0,str(CODE))
from common import read_jsonl,parse_answer,DATA
FAMILIES=['scope_unit','entity_binding','condition_use','target_alignment']

def screen(question):
    q=question.lower()
    # Broad retrieval only. A hit is never a supported relationship or contradiction.
    return {
      'scope_unit':bool(re.search(r'\b(?:per|each|every)\b',q) and re.search(r'\b(?:total|altogether|in all|hours?|minutes?|days?|weeks?|miles?|feet|dollars?|cost)\b',q)),
      'entity_binding':bool(re.search(r'\b(?:boys?|males?|men|husbands?)\b',q) and re.search(r'\b(?:girls?|females?|women|wives?)\b',q)),
      'condition_use':bool(re.search(r'\b(?:second|twice|double|half|except|only|before|after|remaining)\b',q)),
      'target_alignment':bool(re.search(r'\b(?:how (?:many|much) (?:more|less)|additional|left|remaining|remain)\b',q))}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=HERE/'results';out.mkdir(parents=True,exist_ok=True)
    if (out/'split_and_sampling.json').exists():raise RuntimeError('Do not redraw frozen samples.')
    bank=read_jsonl(DATA/'candidates/gsm8k_train.jsonl')
    assert len(bank)==4755
    active=np.asarray([len({parse_answer(c['text']) for c in r['candidates'] if parse_answer(c['text']) is not None})>=2 for r in bank])
    rng=np.random.default_rng(20261018);perm=rng.permutation(len(bank));dev=perm[:2853];confirm=perm[2853:]
    screens=[screen(r['question']) for r in bank]
    selections={};counts={}
    for scope,indices,k in [('development',dev,12),('confirmation',confirm,25)]:
        selections[scope]={};counts[scope]={}
        for family in FAMILIES:
            pool=np.asarray([int(i) for i in indices if screens[int(i)][family] and active[int(i)]],int)
            chosen=rng.choice(pool,size=min(k,len(pool)),replace=False).tolist()
            selections[scope][family]=chosen
            counts[scope][family]={'all_screened':sum(screens[int(i)][family] for i in indices),'active_screened':len(pool),'sampled':len(chosen)}
    pool=np.asarray([int(i) for i in confirm if not any(screens[int(i)].values())],int)
    selections['negative_controls']=rng.choice(pool,size=min(20,len(pool)),replace=False).tolist()
    record={'seed':20261018,'train_ids':[r['id'] for r in bank],'development_rows':dev.tolist(),'confirmation_rows':confirm.tolist(),
      'samples':selections,'counts':counts,'active_train_questions':int(active.sum()),
      'positive_sample_rule':'Uniform without replacement within screened multi-valid-answer questions; input-only eligibility, independent per family; deduplicate reading, retain family memberships',
      'negative_sample_rule':'Uniform20 among confirmation questions with no family screen hit; not restricted to active',
      'labels_or_scores_loaded':False,'train_candidate_sha256':sha(DATA/'candidates/gsm8k_train.jsonl'),
      'protocol_sha256':sha(HERE.parent/'protocol.json'),'sampling_source_sha256':sha(Path(__file__)),'created':time.strftime('%Y-%m-%d %H:%M:%S')}
    write(out/'split_and_sampling.json',record)
    rows=sorted({i for v in selections['development'].values() for i in v})
    with (out/'development_cards.jsonl').open('w',encoding='utf-8') as f:
        for i in rows:
            f.write(json.dumps({'row':i,'id':bank[i]['id'],'families':[name for name,ix in selections['development'].items() if i in ix],
              'question':bank[i]['question'],'candidates':bank[i]['candidates']},ensure_ascii=False)+'\n')
    print(json.dumps({'counts':counts,'development_unique':len(rows),'negative_controls':len(selections['negative_controls'])}),flush=True)

if __name__=='__main__':main()
