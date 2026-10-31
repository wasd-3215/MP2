"""Compact, label-blind source-span display; no inference or fitting."""
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'relation'))
from audit import unique_claims

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    scope=sys.argv[1];start,end=int(sys.argv[2]),int(sys.argv[3])
    filename='confirmation_cards.jsonl' if scope=='confirmation' else 'development_extraction.jsonl'
    rows=[json.loads(s) for s in (HERE/'relation/results'/filename).read_text(encoding='utf-8').splitlines()]
    for index,r in enumerate(rows[start:end],start):
        print(f'CARD{index} ROW{r["row"]} families={r["families"]} negative={r.get("negative_control",False)}')
        print(r['question'])
        families=list(r['claims']) if 'claims' in r else r['families']
        for f in families:
            claims=unique_claims(r['row'],f,r['evidence'])
            print(f+' claims='+str(len(claims)))
            for c in claims:
                q,v=c['question'],c['candidate']
                keys=['value','numerator','denominator','entity','stage','unit','multiplier','type']
                desc=lambda x:'/'.join(str(x[k]) for k in keys if k in x)
                src=v['source']['text']
                if f=='target_alignment':src=v['pair']['left']+' = '+v['pair']['right']
                print(c['claim_id']+' '+desc(q)+' -> '+desc(v)+' '+c['status']+' SRC '+src[:240].replace('\n',' '))
        for i,c in enumerate(r['candidates']):print(f'C{i} END '+c['text'][-130:].replace('\n',' '))

if __name__=='__main__':main()
