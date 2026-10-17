"""Post hoc paired subgroup analysis of the observed consensus/minority tradeoff."""
import runtime
from utilities import HERE,read,write
import numpy as np
from common import load_bank
from explicit_features import answer_structure


def main():
    out=HERE/'results';y=np.load(out/'validation-labels.npy');bank=load_bank('validation')
    _,support,unique=answer_structure(bank)
    vote_correct=y[np.arange(len(y)),support.argmax(1)]>0
    masks={'one_correct':y.sum(1)==1,'multiple_valid_answer_groups':None,
           'unique_plurality_wrong':unique&~vote_correct}
    answers,_,_=answer_structure(bank)
    masks['multiple_valid_answer_groups']=np.asarray([len({v for v in a if v is not None})>=2 for a in answers])
    correct=lambda choice:y[np.arange(len(y)),choice]
    results=[];seeds=[17,29,43,71,101]
    with np.load(out/'all-validation-choices.npz') as p:
        comparisons=[('consistency_single_five_seed_mean','s2_single_five_seed_mean',
            np.mean([correct(p[f's2_consistency_seed{s}']) for s in seeds],axis=0),
            np.mean([correct(p[f's2_seed{s}']) for s in seeds],axis=0)),
          ('consistency_5','s2_5',correct(p['consistency_5']),correct(p['s2_5']))]
        for a,b,ac,bc in comparisons:
            for name,mask in masks.items():
                delta=(ac.astype(float)-bc)[mask]
                rng=np.random.default_rng(20261017);means=[]
                for start in range(0,10000,250):
                    ix=rng.integers(0,len(delta),(250,len(delta)))
                    means.extend(delta[ix].mean(1))
                results.append({'a':a,'b':b,'subgroup':name,'n':int(mask.sum()),
                   'a_accuracy_diagnostic':float(100*ac[mask].mean()),
                   'b_accuracy_diagnostic':float(100*bc[mask].mean()),
                   'delta_pp':float(100*delta.mean()),'ci95_pp':[float(v) for v in 100*np.quantile(means,[.025,.975])],
                   'post_hoc':True,'selection_changed':False,'multiplicity_corrected':False})
    write(out/'consensus_tradeoff.json',results)
    print('Consensus tradeoff diagnosed; no model selection changed')


if __name__=='__main__':main()
