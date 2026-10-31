"""Run probes on the previously selected development cards only."""
from pathlib import Path
import json
import hashlib
import time
from extractor import extract,check_sources,rates,counts,target,stage_multipliers
HERE=Path(__file__).resolve().parent

def main():
    start=time.perf_counter();checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    check('rate_denominator_value',rates('4 chairs every 6 hours')[0]['value']=='2/3')
    check('currency_rate',rates('Hiring costs $10 per hour.')[0]['numerator']=='dollar')
    check('count_not_fraction',counts('1/3 boys')==[])
    check('question_target_only',target('He has 5 left. How many in total?')['type']=='total')
    probe=extract('He earns $10 per hour. How much in total?','He earns $12 per hour. #### 24')
    check('value_difference_not_proven_semantic_conflict',probe['scope_unit'][0]['status']=='different_value' and not probe['scope_unit'][0]['binding_proven'])
    check('repeated_condition_not_use',extract('He needs a second coat.','He needs a second coat. #### 2')['condition_use']['actual_use']=='unknown')
    check('embedded_what_not_question',target("He loses half of what's left. How much remains?")['source']['text']=='How much remains?')
    check('stage_multiplier',stage_multipliers('In the fourth year it grows twice as much as the third year.')[0]['multiplier']=='2')
    check('stage_boundary',stage_multipliers('Year 3\nYear 4: twice as much.')[0]['stage']==4)
    cards=[json.loads(s) for s in (HERE/'results/development_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    result=[]
    for r in cards:
        evidence=[extract(r['question'],c['text']) for c in r['candidates']]
        for c,e in zip(r['candidates'],evidence):check_sources(r['question'],c['text'],e)
        result.append({**r,'evidence':evidence})
    with (HERE/'results/development_extraction.jsonl').open('w',encoding='utf-8') as f:
        for r in result:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (HERE/'results/development_checks.json').write_text(json.dumps({'passed':True,'checks':checks,'questions':len(cards),
      'source_sha256':hashlib.sha256((HERE/'extractor.py').read_bytes()).hexdigest(),'seconds':time.perf_counter()-start},indent=2)+'\n',encoding='utf-8')
    print({'questions':len(result),'checks':len(checks)},flush=True)

if __name__=='__main__':main()
