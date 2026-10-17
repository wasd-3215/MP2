"""Post hoc attribution only: distinguish invalid parsing from genuine answer ambiguity."""
import runtime
from utilities import HERE,read,write
import numpy as np
from common import load_bank,parse_answer


def main():
    out=HERE/'results';bank=load_bank('validation')
    answers=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    y=np.load(out/'validation-labels.npy')
    records=read(out/'pipelines.json');diagnostics=[]
    with np.load(out/'all-validation-choices.npz') as archive:
        for r in records:
            if r['group']!='routing':continue
            before=archive[r['base']];after=archive[r['name']]
            changed=np.asarray([answers[q][int(a)]!=answers[q][int(b)] for q,(a,b) in enumerate(zip(before,after))])
            invalid=np.asarray([answers[q][int(i)] is None for q,i in enumerate(before)])
            old_correct=y[np.arange(len(y)),before]>0;new_correct=y[np.arange(len(y)),after]>0
            diagnostics.append({'name':r['name'],'threshold':r['routing']['threshold'],
              'answer_changes':int(changed.sum()),'from_invalid_parse':int((changed&invalid).sum()),
              'from_valid_answer':int((changed&~invalid).sum()),
              'invalid_parse_repairs':int((changed&invalid&new_correct).sum()),
              'valid_answer_repairs':int((changed&~invalid&~old_correct&new_correct).sum()),
              'valid_answer_harms':int((changed&old_correct&~new_correct).sum()),
              'diagnostic_only':True})
    write(out/'routing_behavior.json',diagnostics)
    print('Routing behavior diagnosed; no predictions or selection changed')


if __name__=='__main__':main()
