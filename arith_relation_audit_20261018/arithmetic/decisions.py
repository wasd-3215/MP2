"""Frozen candidate-level residual, no reference labels or split identifiers."""
import numpy as np
from common import parse_answer

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(np.asarray(z,dtype=np.float64),-40,40)))

def base_logits(probabilities):
    p=np.clip(probabilities,1e-12,1-1e-12)
    return np.log(p)-np.log1p(-p)

def best(scores,valid):
    choice=scores.argmax(1)
    change=valid.any(1)&~valid[np.arange(len(valid)),choice]
    choice[change]=np.where(valid[change],scores[change],-np.inf).argmax(1)
    return choice

def choose(bank,probabilities,features,config):
    valid=np.asarray([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank],bool).reshape(-1,4)
    baseline=best(probabilities,valid)
    if not config['columns']:
        return baseline
    residual=features[...,config['columns']]@np.asarray(config['beta'],np.float64)
    score=base_logits(probabilities)+residual
    choice=best(score,valid)
    # A shared additive shift cannot alter rankings; preserve exact probability ties/clipping behavior.
    common_shift=np.ptp(residual,axis=1)==0
    choice[common_shift]=baseline[common_shift]
    return choice
