"""Fresh inference on new banks, equal group-softmax ensemble, no references/cache."""
import runtime
runtime.configure_torch()
from pathlib import Path
import json
import numpy as np
import torch
from common import parse_answer
from grouping import layout,softmax,choices_from_probabilities
from features import extract
from models import Scorer

def load_member(path):
    path=Path(path);config=json.loads((path/'config.json').read_text(encoding='utf-8'))
    model=Scorer(**config['model'])
    with np.load(path/'weights.npz',allow_pickle=False) as a:
        model.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files},strict=True)
    with np.load(path/'preprocessing.npz',allow_pickle=False) as a:prep={k:a[k].copy() for k in a.files}
    return model.eval(),prep

def infer(model,x,f,m):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x[i:i+256],f[i:i+256],m[i:i+256])[1].numpy() for i in range(0,len(x),256)])

def predict(bank,arrays,checkpoint):
    raw=np.asarray(arrays['embeddings'],np.float32)
    if raw.shape!=(len(bank),4,1024) or not np.isfinite(raw).all():raise ValueError('Expected finite aligned [N,4,1024] embeddings')
    if not bank:return []
    _,membership,valid,counts=layout(bank)
    needed=counts>=2;result=np.zeros(len(bank),np.int64)
    easy=counts==1;result[easy]=valid[easy].argmax(1)
    if needed.any():
        if checkpoint is None:raise ValueError('Supply frozen pipeline for multiple-answer bank')
        path=Path(checkpoint);config=json.loads((path/'pipeline.json').read_text(encoding='utf-8'))
        sub=[r for r,k in zip(bank,needed) if k];raw_f=extract(sub)
        m=torch.from_numpy(membership[needed]);probabilities=[]
        for relative in config['members']:
            model,prep=load_member((path/relative).resolve())
            x=torch.from_numpy((raw[needed]-prep['mean'])/prep['std'])
            f=torch.from_numpy((raw_f-prep['feature_mean'])/prep['feature_std'])
            z=infer(model,x,f,m);probabilities.append(softmax(z,membership[needed]))
        result[needed]=choices_from_probabilities(membership[needed],np.stack(probabilities).mean(0))
    return [int(i) for i in result]
