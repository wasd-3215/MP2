"""Freeze probes, then train-only confirmation extraction; no reference IO."""
from pathlib import Path
import json
import hashlib
import time
import sys
from extractor import extract,check_sources
from prepare import FAMILIES,screen
from common import read_jsonl,DATA
HERE=Path(__file__).resolve().parent
OUT=HERE/'results'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def family_claims(evidence,family):
    if family=='condition_use':return evidence[family]['stage_comparisons']
    if family=='target_alignment':return [evidence[family]] if evidence[family] else []
    return evidence[family]

def unique_claims(row,family,evidence):
    claims=[]
    for candidate,e in enumerate(evidence):
        seen=set()
        for claim in family_claims(e,family):
            q,c=claim['question'],claim['candidate']
            if family=='scope_unit':key=(q['value'],q['numerator'],q['denominator'],c['value'])
            elif family=='entity_binding':key=(q['value'],q['entity'],c['value'])
            elif family=='condition_use':key=(q['stage'],q['unit'],q['multiplier'],c['multiplier'])
            else:key=(q['type'],c['type'])
            if key in seen:continue
            seen.add(key)
            claims.append({'claim_id':f'{row}:{family}:{candidate}:{len(seen)-1}','candidate_index':candidate,**claim})
    return claims

def freeze():
    assert (OUT/'development_review.json').exists()
    assert read(OUT/'development_review.json')['completed']
    if (OUT/'frozen_before_confirmation.json').exists():raise RuntimeError('Already frozen.')
    sources=[*HERE.glob('*.py'),HERE.parent/'protocol.json',HERE.parent/'arithmetic/evidence.py',HERE.parent/'arithmetic/decisions.py']
    write(OUT/'frozen_before_confirmation.json',{'sources':{str(p):sha(p) for p in sources},
      'sampling_sha256':sha(OUT/'split_and_sampling.json'),'development_review_sha256':sha(OUT/'development_review.json'),
      'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'reference_answers_loaded':False,'confirmation_review_started':False,
      'scope':'Bounded numeric rate/entity comparisons, explicit stage modifiers, target-operation compatibility; no semantic verifier claimed.'})

def check_frozen():
    frozen=read(OUT/'frozen_before_confirmation.json')
    assert all(sha(Path(p))==h for p,h in frozen['sources'].items())
    assert sha(OUT/'split_and_sampling.json')==frozen['sampling_sha256']
    assert sha(OUT/'development_review.json')==frozen['development_review_sha256']

def confirm():
    check_frozen()
    if (OUT/'confirmation_cards.jsonl').exists():raise RuntimeError('Confirmation already generated.')
    bank=read_jsonl(DATA/'candidates/gsm8k_train.jsonl');sampling=read(OUT/'split_and_sampling.json')
    selected={i for v in sampling['samples']['confirmation'].values() for i in v}|set(sampling['samples']['negative_controls'])
    population={f:{'development':{'questions_with_probes':0,'primitive_claims':0},'confirmation':{'questions_with_probes':0,'primitive_claims':0}} for f in FAMILIES}
    dev=set(sampling['development_rows']);cards=[];tick=time.perf_counter()
    for row,r in enumerate(bank):
        evidence=[extract(r['question'],c['text']) for c in r['candidates']]
        for c,e in zip(r['candidates'],evidence):check_sources(r['question'],c['text'],e)
        scope='development' if row in dev else 'confirmation'
        hit=screen(r['question'])
        for family in FAMILIES:
            claims=unique_claims(row,family,evidence)
            population[family][scope]['questions_with_probes']+=bool(claims)
            population[family][scope]['primitive_claims']+=len(claims)
        if row in selected:
            families=[f for f,ix in sampling['samples']['confirmation'].items() if row in ix]
            negative=row in sampling['samples']['negative_controls']
            # Negative controls show all probes, to reveal screening omissions.
            audited=FAMILIES if negative else families
            cards.append({'row':row,'id':r['id'],'question':r['question'],'families':families,'negative_control':negative,
              'screen':hit,'candidates':r['candidates'],'evidence':evidence,
              'claims':{f:unique_claims(row,f,evidence) for f in audited}})
    with (OUT/'confirmation_cards.jsonl').open('w',encoding='utf-8') as f:
        for r in cards:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(OUT/'population_coverage.json',{'counts':population,'train_extraction_seconds':time.perf_counter()-tick,
      'confirmation_unique_questions':len(cards),'confirmation_candidate_texts':4*len(cards),
      'reference_answers_loaded':False,'screened_counts_are_not_confirmed_support':True})
    check_frozen()
    print({'confirmation_questions':len(cards),'candidate_texts':4*len(cards),'coverage':population},flush=True)

def display(scope,start=0,end=None):
    filename='development_extraction.jsonl' if scope=='development' else 'confirmation_cards.jsonl'
    cards=[json.loads(s) for s in (OUT/filename).read_text(encoding='utf-8').splitlines()]
    for card in cards[start:end]:
        row=card['row'];families=card['families']
        if card.get('negative_control'):families=FAMILIES
        print(json.dumps({'row':row,'families':families,'negative':card.get('negative_control',False),'question':card['question']},ensure_ascii=False))
        for family in families:
            claims=unique_claims(row,family,card['evidence'])
            if not claims:print(f'{family}: no supported probe; UNKNOWN')
            for c in claims:
                q,v=c['question'],c['candidate']
                desc=lambda x:{k:x[k] for k in ['value','numerator','denominator','entity','stage','unit','multiplier','type'] if k in x}
                source=v['source']['text']
                if family=='target_alignment':source=c['candidate']['pair']['left']+' = '+c['candidate']['pair']['right']
                print(json.dumps({'id':c['claim_id'],'q':desc(q),'c':desc(v),'status':c['status'],'source':source[:420]},ensure_ascii=False))
        # All four final contexts shown, including unparseable/unknown candidates.
        print(json.dumps({'ends':[c['text'][-170:] for c in card['candidates']]},ensure_ascii=False))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='confirm':confirm()
    elif sys.argv[1]=='display':display(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
