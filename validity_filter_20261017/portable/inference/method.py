"""Frozen scoring/routing followed by an optional official-parser validity filter.

Inference reads inputs and member assets only. It never reads references or saved
predictions. Invalid candidates remain present throughout neural scoring.
"""
import reference_method
from pathlib import Path
import json
import time
import numpy as np
from common import parse_answer
from decisions import aggregate, decide


def validity_filter(bank, scores, original):
    start = time.perf_counter()
    valid = np.asarray([[parse_answer(c['text']) is not None
                         for c in row['candidates']] for row in bank], dtype=bool).reshape(-1, 4)
    result = np.asarray(original, dtype=np.int64).copy()
    eligible = valid.any(axis=1)
    # For valid original choices preserve the choice exactly, including routing.
    change = eligible & ~valid[np.arange(len(bank)), result]
    if change.any():
        result[change] = np.where(valid[change], scores[change], -np.inf).argmax(axis=1)
    return result, {'seconds': time.perf_counter() - start,
                    'original_invalid': int((~valid[np.arange(len(bank)), original]).sum()),
                    'replaceable': int(change.sum()),
                    'all_invalid': int((~eligible).sum())}


def evaluate(bank, arrays, checkpoint):
    path = Path(checkpoint)
    manifest = json.loads((path / 'pipeline.json').read_text(encoding='utf-8'))
    members = [(path / p).resolve() for p in manifest['members']]
    probabilities = reference_method.member_probabilities(bank, arrays, members)
    scores, _ = aggregate(probabilities, bank, manifest['rule'])
    original = decide(probabilities, bank, manifest['rule'], manifest.get('routing'))
    if manifest['filter_invalid']:
        choices, diagnostics = validity_filter(bank, scores, original)
    else:
        choices, diagnostics = original, {'seconds': 0.0}
    return [int(i) for i in choices], scores, diagnostics


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Supply a frozen pipeline checkpoint')
    return evaluate(bank, arrays, checkpoint)[0]
