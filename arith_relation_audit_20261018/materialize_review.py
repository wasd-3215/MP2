"""Expand explicitly authored review dispositions to per-claim records; no labels."""
from pathlib import Path
import json
import fnmatch
import hashlib
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'relation'))
from audit import unique_claims,read,write,check_frozen,OUT

def main():
    check_frozen()
    source=OUT/'blind_review_input.json';review=read(source)
    assert not review['reference_answers_loaded'] and not review['model_scores_loaded']
    cards=[json.loads(s) for s in (OUT/'confirmation_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {int(q) for q in review['row_notes']}=={r['row'] for r in cards}
    def matched(key,patterns):return any(fnmatch.fnmatchcase(key,p) for p in patterns)
    annotations=[]
    for r in cards:
        for family in r['claims']:
            for c in unique_claims(r['row'],family,r['evidence']):
                key=c['claim_id'];good=matched(key,review['accepted_patterns']);bad=matched(key,review['rejected_semantic_patterns'])
                assert not (good and bad),key
                assert good or bad,('Unreviewed claim',key)
                annotations.append({'claim_id':key,'row':r['row'],'family':family,'candidate_index':c['candidate_index'],
                  'literal_fidelity':not matched(key,review['literal_error_patterns']),
                  'semantic_binding_supported':True if good else False,
                  'note':review['row_notes'][str(r['row'])],
                  'self_correction':'No retraction apparent in reviewed source; full-trace check not exhaustive'})
    value={**review,'completed':True,'reviewed_rows':[r['row'] for r in cards],'annotations':annotations,
      'review_input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'all_candidate_final_contexts_and_all_extracted_spans_read':True,'no_claim_labels_inferred_from_final_correctness':True}
    write(OUT/'confirmation_review_blind.json',value)
    print({'questions':len(cards),'claims':len(annotations),'literal_errors':sum(not a['literal_fidelity'] for a in annotations),
      'semantic_supported':sum(a['semantic_binding_supported'] for a in annotations)},flush=True)

if __name__=='__main__':main()
