"""Pure observable features and frozen decisions; never opens reference labels."""
from collections import Counter
import numpy as np
from common import parse_answer

NAMES = ['Pool_answer_gap', 'S2_answer_gap', 'Pool_seed_answer_agreement',
         'S2_seed_answer_agreement', 'Pool_selected_answer_support',
         'S2_selected_answer_support', 'distinct_valid_answers', 'invalid_fraction']

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -40, 40)))

def layout(bank):
    answers = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    valid = np.asarray([[a is not None for a in row] for row in answers], bool).reshape(-1, 4)
    return answers, valid

def best(scores, valid):
    result = np.asarray(scores).argmax(axis=-1)
    change = valid.any(axis=1) & ~valid[np.arange(len(valid)), result]
    result[change] = np.where(valid[change], scores[change], -np.inf).argmax(axis=1)
    return result

def expert_choices(probabilities, valid):
    return best(probabilities.mean(axis=0), valid)

def gate_features(bank, probabilities):
    answers, valid = layout(bank)
    choices = [expert_choices(p, valid) for p in probabilities]
    selected_answers = [[a[int(i)] for a, i in zip(answers, choice)] for choice in choices]
    disagreement = np.asarray([a != b for a, b in zip(*selected_answers)], bool)
    features = np.zeros((len(bank), 8), np.float64)
    for e, probs in enumerate(probabilities):
        mean = probs.mean(axis=0)
        seed_choices = np.stack([best(s, valid) for s in probs])
        for q, a in enumerate(answers):
            counts = Counter(v for v in a if v is not None)
            groups = [max(mean[q, i] for i, v in enumerate(a) if v == answer) for answer in counts]
            groups.sort(reverse=True)
            features[q, e] = groups[0] - groups[1] if len(groups) >= 2 else 0.
            chosen = selected_answers[e][q]
            features[q, 2+e] = np.mean([a[int(i)] == chosen for i in seed_choices[:, q]])
            features[q, 4+e] = counts.get(chosen, 0) / 4
            features[q, 6] = len(counts) / 4
            features[q, 7] = 1 - valid[q].mean()
    assert np.isfinite(features).all()
    return features, choices, disagreement

def decide(bank, probabilities, config):
    answers, valid = layout(bank)
    mode = config['mode']
    if mode == 'S2_C':
        return expert_choices(probabilities[1], valid)
    if mode == 'Pool_P':
        return expert_choices(probabilities[0], valid)
    if mode in ('equal_average', 'fixed_weight'):
        w = config['pool_weight']
        return best(w * probabilities[0].mean(0) + (1-w) * probabilities[1].mean(0), valid)
    features, choices, disagreement = gate_features(bank, probabilities)
    gate = config['gate']
    cols = gate['columns']
    z = (features[:, cols] - np.asarray(gate['mean'])) / np.asarray(gate['std'])
    utility = gate['intercept'] + z @ np.asarray(gate['coefficients'])
    result = choices[1].copy()
    switch = disagreement & (utility > 0)
    result[switch] = choices[0][switch]
    return result
