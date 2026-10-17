"""Fresh offline inference; no references, IDs or saved prediction lookup."""
import runtime
runtime.configure_torch()
from support import read
from pathlib import Path
import numpy as np
import torch
from common import parse_answer
from models import Scorer
from features import extract


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -40, 40)))


def filter_choices(bank, scores, choices):
    valid = np.asarray([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank], bool).reshape(-1, 4)
    result = np.asarray(choices, dtype=np.int64).copy()
    change = valid.any(axis=1) & ~valid[np.arange(len(bank)), result]
    if change.any():
        result[change] = np.where(valid[change], scores[change], -np.inf).argmax(axis=1)
    return result


def load_member(path):
    path = Path(path)
    model = Scorer(**read(path / 'config.json')['model'])
    with np.load(path / 'weights.npz', allow_pickle=False) as a:
        model.load_state_dict({k: torch.from_numpy(a[k].copy()) for k in a.files}, strict=True)
    with np.load(path / 'preprocessing.npz', allow_pickle=False) as a:
        prep = {k: a[k].copy() for k in a.files}
    return model.eval(), prep


def infer(model, x, f):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x[i:i+256], f[i:i+256]) for i in range(0, len(x), 256)]).numpy()


def evaluate(bank, arrays, checkpoint):
    path = Path(checkpoint)
    manifest = read(path / 'pipeline.json')
    raw = np.asarray(arrays['embeddings'], dtype=np.float32)
    if raw.shape != (len(bank), 4, 1024) or not np.isfinite(raw).all():
        raise ValueError('Expected finite aligned [N,4,1024] embeddings')
    raw_f = extract(bank)
    members = []
    for relative in manifest['members']:
        model, prep = load_member((path / relative).resolve())
        x = torch.from_numpy((raw - prep['mean']) / prep['std'])
        f = torch.from_numpy((raw_f - prep['feature_mean']) / prep['feature_std'])
        members.append(sigmoid(infer(model, x, f)))
    scores = np.stack(members).mean(axis=0)
    choices = scores.argmax(axis=1)
    if manifest['filter_invalid']:
        choices = filter_choices(bank, scores, choices)
    return [int(i) for i in choices], scores


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Supply a frozen pipeline')
    return evaluate(bank, arrays, checkpoint)[0]
