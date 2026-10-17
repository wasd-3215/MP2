"""Offline inference: only input bank and portable member bundles are accessed."""
import runtime
runtime.configure_torch()
from pathlib import Path
import json,sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from explicit_features import FEATURES,extract
from scorers import Scorer
from decisions import sigmoid,decide


def load_member(path):
    config=json.loads((path/'config.json').read_text(encoding='utf-8'))
    model=Scorer(**config['model'])
    with np.load(path/'weights.npz',allow_pickle=False) as archive:
        model.load_state_dict({k:torch.from_numpy(archive[k].copy()) for k in archive.files},strict=True)
    with np.load(path/'preprocessing.npz',allow_pickle=False) as archive:
        prep={k:archive[k].copy() for k in archive.files}
    return model.eval(),prep


def member_probabilities(bank,arrays,members):
    raw=np.asarray(arrays['embeddings'],dtype=np.float32)
    if raw.shape!=(len(bank),4,1024) or not np.isfinite(raw).all():
        raise ValueError('Expected aligned finite [N,4,1024] embeddings')
    feats=extract(bank)
    results=[]
    for path in members:
        model,prep=load_member(path)
        x=torch.from_numpy((raw-prep['mean'])/prep['std'])
        f=torch.from_numpy(((feats-prep['feature_mean'])/prep['feature_std'])[...,FEATURES[model.feature_mode]])
        with torch.inference_mode():
            logits=np.concatenate([model(x[i:i+256],f[i:i+256]).numpy() for i in range(0,len(x),256)])
        results.append(sigmoid(logits))
    return np.stack(results)


def predict(bank,arrays,checkpoint):
    if checkpoint is None:raise ValueError('Supply a bundle or pipeline checkpoint')
    path=Path(checkpoint)
    if (path/'pipeline.json').exists():
        manifest=json.loads((path/'pipeline.json').read_text(encoding='utf-8'))
        members=[(path/p).resolve() for p in manifest['members']]
        probabilities=member_probabilities(bank,arrays,members)
        choices=decide(probabilities,bank,manifest['rule'],manifest.get('routing'))
    else:
        probabilities=member_probabilities(bank,arrays,[path])
        choices=probabilities[0].argmax(1)
    return [int(i) for i in choices]
