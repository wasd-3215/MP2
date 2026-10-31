"""Fresh inference from portable weights and observable inputs only."""
import runtime
runtime.configure_torch()
from pathlib import Path
import json
import numpy as np
import torch
from models import Scorer
from features import extract
from decisions import sigmoid, decide

def load_member(path):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    model = Scorer(**config['model'])
    with np.load(path / 'weights.npz', allow_pickle=False) as a:
        model.load_state_dict({k: torch.from_numpy(a[k].copy()) for k in a.files}, strict=True)
    with np.load(path / 'preprocessing.npz', allow_pickle=False) as a:
        prep = {k: a[k].copy() for k in a.files}
    return model.eval(), prep

def infer(model, x, f):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x[i:i+256], f[i:i+256]).numpy() for i in range(0, len(x), 256)])

def member_probabilities(bank, arrays, paths):
    raw = np.asarray(arrays['embeddings'], dtype=np.float32)
    if raw.shape != (len(bank), 4, 1024) or not np.isfinite(raw).all():
        raise ValueError('Expected finite aligned [N,4,1024] embeddings')
    raw_f = extract(bank)
    result = []
    for path in paths:
        model, prep = load_member(path)
        x = torch.from_numpy((raw - prep['mean']) / prep['std'])
        f = torch.from_numpy((raw_f - prep['feature_mean']) / prep['feature_std'])
        result.append(sigmoid(infer(model, x, f)))
    return np.stack(result)

def evaluate(bank, arrays, checkpoint):
    path = Path(checkpoint)
    config = json.loads((path / 'pipeline.json').read_text(encoding='utf-8'))
    if not bank:
        return [], None
    probabilities = []
    for e in ('Pool_P', 'S2_C'):
        paths = [(path / p).resolve() for p in config['members'][e]]
        if not paths:
            probabilities.append(None)
        else:
            probabilities.append(member_probabilities(bank, arrays, paths))
    return [int(i) for i in decide(bank, probabilities, config)], probabilities

def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Supply frozen pipeline checkpoint')
    return evaluate(bank, arrays, checkpoint)[0]
