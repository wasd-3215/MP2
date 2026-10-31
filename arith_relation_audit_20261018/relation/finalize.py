"""Reveal training labels only after the explicit first-pass review is frozen."""
from pathlib import Path
import json
import math
import time
from audit import OUT,HERE,read,write,sha,check_frozen,unique_claims
from prepare import FAMILIES
from common import read_jsonl,DATA,parse_answer

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-radius),min(1,center+radius)]

def main():
    check_frozen()
    review_path=OUT/'confirmation_review_blind.json';review=read(review_path)
    assert review['completed'] and not review['reference_answers_loaded']
    cards=[json.loads(s) for s in (OUT/'confirmation_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    assert set(review['reviewed_rows'])=={r['row'] for r in cards}
    annotations=review['annotations'];lookup={a['claim_id']:a for a in annotations}
    expected={c['claim_id'] for r in cards for f in r['claims'] for c in unique_claims(r['row'],f,r['evidence'])}
    assert set(lookup)==expected and len(lookup)==len(annotations)
    write(OUT/'blind_review_frozen_before_labels.json',{'review_sha256':sha(review_path),'cards_sha256':sha(OUT/'confirmation_cards.jsonl'),
      'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'label_reveal_follows_this_file':True})
    # First semantic reference read in the relation experiment happens here.
    refs=read_jsonl(DATA/'gsm8k_train.jsonl')
    summary={};contrasts=[]
    for family in FAMILIES:
        selected=[r for r in cards if family in r['families']]
        assertions=[];supported_questions=[];discriminating=[];useful=[]
        for r in selected:
            claims=unique_claims(r['row'],family,r['evidence'])
            assertions.extend(lookup[c['claim_id']] for c in claims)
            accepted=[c for c in claims if lookup[c['claim_id']]['semantic_binding_supported'] is True]
            if accepted:supported_questions.append(r['row'])
            # Only confirmed bound evidence participates; unknown is not a negative.
            scores={}
            for c in accepted:
                score=int(c['status'] in ['different_value','different_multiplier','different_operation'])
                scores[c['candidate_index']]=max(scores.get(c['candidate_index'],0),score)
            disc=len(set(scores.values()))>=2
            if disc:discriminating.append(r['row'])
            gold=parse_answer(refs[r['row']]['answer'])
            assert refs[r['row']]['id']==r['id']
            y=[parse_answer(c['text'])==gold and parse_answer(c['text']) is not None for c in r['candidates']]
            pairs=[{'candidate_a':a,'candidate_b':b,'a_evidence_conflict':scores[a],'b_evidence_conflict':scores[b],
              'a_final_correct':bool(y[a]),'b_final_correct':bool(y[b]),
              'direction_supports_correct':scores[a]<scores[b] if y[a] else scores[b]<scores[a]}
              for a in scores for b in scores if a<b and scores[a]!=scores[b] and y[a]!=y[b]]
            if pairs:useful.append(r['row'])
            if pairs:contrasts.append({'row':r['row'],'family':family,'pairs':pairs})
        n=len(assertions);literal=sum(a['literal_fidelity'] is True for a in assertions)
        bound=sum(a['semantic_binding_supported'] is True for a in assertions)
        # Unknown binding is not asserted wrong, but cannot count as proven reliable evidence.
        gates={'coverage20':len(supported_questions)>=20,'assertions40':n>=40,'questions20':sum(bool(unique_claims(r['row'],family,r['evidence'])) for r in selected)>=20,
          'semantic_fidelity95':n>0 and bound/n>=.95,'discriminating10':len(discriminating)>=10,'useful_correct_wrong_contrasts':bool(useful)}
        summary[family]={'sampled_questions':len(selected),'questions_with_probes':sum(bool(unique_claims(r['row'],family,r['evidence'])) for r in selected),
          'reviewed_assertions':n,'literal_faithful':literal,'literal_fidelity':literal/n if n else None,'literal_wilson95':wilson(literal,n),
          'semantic_binding_supported':bound,'semantic_support_fraction':bound/n if n else None,'semantic_wilson95':wilson(bound,n),
          'confirmed_supported_multianswer_questions':len(supported_questions),'supported_rows':supported_questions,
          'confirmed_discriminating_questions':len(discriminating),'discriminating_rows':discriminating,
          'correct_wrong_contrast_questions':len(useful),'contrast_rows':useful,
          'gates':gates,'decision':'GO_SOFT_FEATURE_EXPERIMENT' if all(gates.values()) else 'NO_GO_CURRENT_PROBE'}
    write(OUT/'relation_summary.json',{'families':summary,'label_reveal_after_review':True,'blind_review_sha256':sha(review_path),
      'contrasts':contrasts,'test_used':False,'validation_used':False,'new_relation_model_fits':0,
      'scope':'Current bounded probes only; unknown binding is insufficient evidence, not proof the relationship is wrong.',
      'precision_caveat':'Wilson intervals are descriptive at assertion level; assertions within questions correlated and samples stratified; not population reliability or recall.',
      'reviewer':'Codex single reviewer; not independent human review; no model generated labels used in training.'})
    check_frozen()
    print(json.dumps({f:{k:s[k] for k in ['sampled_questions','reviewed_assertions','literal_fidelity','semantic_support_fraction','confirmed_supported_multianswer_questions','confirmed_discriminating_questions','decision']} for f,s in summary.items()}),flush=True)

if __name__=='__main__':main()
