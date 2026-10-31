"""Fresh frozen-S2 inference plus input-only text evidence; no cache/reference IO."""
import runtime
runtime.configure_torch()
from pathlib import Path
import json
import numpy as np
import torch
from models import Scorer
from features import extract as original_features
from evidence import extract as text_features
from decisions import sigmoid,choose

def load_member(path):
    path=Path(path)
    config=json.loads((path/'config.json').read_text(encoding='utf-8'))
    model=Scorer(**config['model'])
    with np.load(path/'weights.npz',allow_pickle=False) as a:
        model.load_state_dict({k:torch.from_numpy(a[k].copy()) for k in a.files},strict=True)
    with np.load(path/'preprocessing.npz',allow_pickle=False) as a:
        prep={k:a[k].copy() for k in a.files}
    return model.eval(),prep

def probabilities(bank,arrays,members):
    raw=np.asarray(arrays['embeddings'],np.float32)
    if raw.shape!=(len(bank),4,1024) or not np.isfinite(raw).all():
        raise ValueError('Expected finite aligned [N,4,1024] embeddings')
    rf=original_features(bank)
    result=[]
    for path in members:
        model,prep=load_member(path)
        x=torch.from_numpy((raw-prep['mean'])/prep['std'])
        f=torch.from_numpy((rf-prep['feature_mean'])/prep['feature_std'])
        with torch.inference_mode():
            z=np.concatenate([model(x[i:i+256],f[i:i+256]).numpy() for i in range(0,len(x),256)])
        result.append(sigmoid(z))
    return np.stack(result).mean(0)

def predict(bank,arrays,checkpoint):
    if checkpoint is None:raise ValueError('Supply frozen pipeline')
    if not bank:return []
    path=Path(checkpoint)
    config=json.loads((path/'pipeline.json').read_text(encoding='utf-8'))
    p=probabilities(bank,arrays,[(path/m).resolve() for m in config['members']])
    f=text_features(bank) if config['columns'] else None
    return [int(i) for i in choose(bank,p,f,config)]
