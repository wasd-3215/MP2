"""Input-only deterministic routing, neural disagreement scoring and validity filter."""
import runtime
runtime.configure_torch()
from support import read
from pathlib import Path
import numpy as np
import torch
from common import parse_answer
from models import Scorer
from features import extract

def layout(bank):
    answers=[[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    valid=np.asarray([[a is not None for a in row] for row in answers],bool).reshape(-1,4)
    groups=np.asarray([len({a for a in row if a is not None}) for row in answers],int)
    return answers,valid,groups
def sigmoid(z):return 1/(1+np.exp(-np.clip(np.asarray(z,dtype=np.float64),-40,40)))
def decide(bank,scores):
    _,valid,groups=layout(bank)
    choice=scores.argmax(axis=1)
    replace=valid.any(axis=1)&~valid[np.arange(len(bank)),choice]
    if replace.any():choice[replace]=np.where(valid[replace],scores[replace],-np.inf).argmax(axis=1)
    easy=groups==1
    if easy.any():choice[easy]=valid[easy].argmax(axis=1)
    return choice
def load_member(path):
    path=Path(path);model=Scorer(**read(path/'config.json')['model'])
    with np.load(path/'weights.npz',allow_pickle=False) as z:
        model.load_state_dict({k:torch.from_numpy(z[k].copy()) for k in z.files},strict=True)
    with np.load(path/'preprocessing.npz',allow_pickle=False) as z:prep={k:z[k].copy() for k in z.files}
    return model.eval(),prep
def infer(model,x,f):
    model.eval()
    with torch.inference_mode():return torch.cat([model(x[i:i+256],f[i:i+256]) for i in range(0,len(x),256)]).numpy()
def evaluate(bank,arrays,checkpoint):
    raw=np.asarray(arrays['embeddings'],dtype=np.float32)
    if raw.shape!=(len(bank),4,1024) or not np.isfinite(raw).all():raise ValueError('Expected finite [N,4,1024]')
    _,valid,groups=layout(bank)
    # When one valid group exists, first valid candidate is sufficient. All-invalid
    # rows use the neural fallback; they do not occur in the guaranteed course bank.
    needed=groups!=1
    result=np.zeros(len(bank),dtype=np.int64)
    if (~needed).any():result[~needed]=valid[~needed].argmax(axis=1)
    if needed.any():
        path=Path(checkpoint);manifest=read(path/'pipeline.json')
        sub=[row for row,keep in zip(bank,needed) if keep];feats=extract(sub);values=[]
        for relative in manifest['members']:
            model,prep=load_member((path/relative).resolve())
            x=torch.from_numpy((raw[needed]-prep['mean'])/prep['std'])
            f=torch.from_numpy((feats-prep['feature_mean'])/prep['feature_std'])
            values.append(sigmoid(infer(model,x,f)))
        scores=np.stack(values).mean(axis=0)
        result[needed]=decide(sub,scores)
    return [int(i) for i in result]
def predict(bank,arrays,checkpoint):
    if checkpoint is None:raise ValueError('Supply a frozen pipeline checkpoint')
    return evaluate(bank,arrays,checkpoint)
