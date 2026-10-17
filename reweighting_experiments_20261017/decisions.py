"""Answer-aware aggregation and routing, independent of labels and IDs."""
import numpy as np
from explicit_features import answer_structure


def sigmoid(x):
    # Float64 avoids rounding moderately large float32 logits to tied probabilities.
    return 1/(1+np.exp(-np.clip(np.asarray(x,dtype=np.float64),-40,40)))


def aggregate(member_probabilities,bank,rule='mean_probability'):
    scores=np.mean(member_probabilities,axis=0)
    choices=scores.argmax(1)
    if rule=='mean_probability':return scores,choices
    if rule!='model_answer_vote':raise ValueError(rule)
    answers,_,_=answer_structure(bank)
    picked=member_probabilities.argmax(2)
    for q,a in enumerate(answers):
        counts={v:sum(a[int(i)]==v for i in picked[:,q]) for v in a if v is not None}
        if not counts:continue
        best=max(counts.values())
        eligible=[i for i in range(4) if a[i] is not None and counts[a[i]]==best]
        choices[q]=max(eligible,key=lambda i:(scores[q,i],-i))
    return scores,choices


def confidence(scores,choices,member_probabilities,bank,signal):
    if signal=='candidate_gap':
        sorted_scores=np.sort(scores,axis=1)
        return sorted_scores[:,-1]-sorted_scores[:,-2]
    answers,_,_=answer_structure(bank)
    out=np.zeros(len(bank),dtype=np.float64)
    picked=member_probabilities.argmax(2)
    for q,a in enumerate(answers):
        chosen=a[int(choices[q])]
        if chosen is None:continue
        if signal=='model_agreement':
            out[q]=np.mean([a[int(i)]==chosen for i in picked[:,q]])
        elif signal=='answer_gap':
            own=max(scores[q,i] for i in range(4) if a[i]==chosen)
            others=[scores[q,i] for i in range(4) if a[i] is not None and a[i]!=chosen]
            out[q]=max(0.,own-max(others)) if others else 1.
        else:raise ValueError(signal)
    return out


def route(scores,choices,conf,bank,threshold,min_support):
    result=choices.copy()
    if threshold is None:return result,np.zeros(len(bank),dtype=bool)
    answers,support,unique=answer_structure(bank)
    triggered=(conf<=threshold)&unique&(support.max(1)>=min_support)
    for q in np.flatnonzero(triggered):
        maximum=support[q].max()
        eligible=np.flatnonzero(support[q]==maximum)
        result[q]=max(eligible,key=lambda i:(scores[q,i],-int(i)))
    changed=np.asarray([answers[q][int(result[q])]!=answers[q][int(choices[q])] for q in range(len(bank))])
    return result,changed


def decide(member_probabilities,bank,rule='mean_probability',routing=None):
    scores,choices=aggregate(member_probabilities,bank,rule)
    if routing is not None:
        c=confidence(scores,choices,member_probabilities,bank,routing['signal'])
        choices,_=route(scores,choices,c,bank,routing['threshold'],routing['min_support'])
    return choices
