"""Official numeric groups, no reference data and no alternate answer parser."""
import numpy as np
from common import parse_answer

def layout(bank):
    membership=np.zeros((len(bank),4,4),bool);valid=np.zeros((len(bank),4),bool)
    counts=np.zeros(len(bank),np.int64);answers=[]
    for q,row in enumerate(bank):
        a=[parse_answer(c['text']) for c in row['candidates']];answers.append(a);groups={}
        for i,value in enumerate(a):
            if value is None:continue
            valid[q,i]=True
            if value not in groups:groups[value]=len(groups)
            membership[q,groups[value],i]=True
        counts[q]=len(groups)
    return answers,membership,valid,counts

def group_targets(membership,candidate_labels):
    target=(membership&np.asarray(candidate_labels,bool)[:,None,:]).any(2)
    assert (target.sum(1)==1).all()
    return target.argmax(1).astype(np.int64)

def choices_from_probabilities(membership,probabilities):
    result=np.zeros(len(membership),np.int64)
    valid_groups=membership.any(2)
    exists=valid_groups.any(1)
    chosen=np.where(valid_groups,probabilities,-np.inf).argmax(1)
    rows=np.flatnonzero(exists)
    result[rows]=membership[rows,chosen[rows]].argmax(1)
    return result

def softmax(z,membership):
    mask=membership.any(2);z=np.where(mask,z,-np.inf)
    top=np.max(z,axis=1,keepdims=True)
    safe=np.where(mask,z-top,-np.inf)
    value=np.exp(safe);return value/value.sum(1,keepdims=True)
