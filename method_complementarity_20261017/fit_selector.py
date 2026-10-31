"""Fit tiny selectors on training OOF only; freeze all pipelines before validation."""
import runtime
from utils import HERE, OUT, FUSION, read, write, sha, sources, members
import os
import time
import numpy as np
from common import load_inputs
from decisions import sigmoid, gate_features, decide, NAMES

def main():
    assert read(OUT / 'training_verification.json')['passed']
    assert sources() == read(OUT / 'training_sources.json')
    tick = time.perf_counter()
    bank, _ = load_inputs('train')
    y = np.load(OUT / 'train-labels.npy')
    p = [sigmoid(np.load(OUT / 'oof-Pool_P.npy')),
         sigmoid(np.load(FUSION / 'results/oof-s2_consistency.npy'))]
    features, choices, disagreement = gate_features(bank, p)
    row = np.arange(len(bank))
    correct = [y[row, choice] for choice in choices]
    utility = correct[0] - correct[1]
    grid = []
    for w in read(HERE / 'protocol.json')['pool_weight_grid']:
        choice = decide(bank, p, {'mode': 'fixed_weight', 'pool_weight': w})
        grid.append({'pool_weight': w, 'correct': int(y[row, choice].sum())})
    weight = max(grid, key=lambda r: (r['correct'], -r['pool_weight']))['pool_weight']
    configs = [{'mode': 'S2_C'}, {'mode': 'Pool_P'}, {'mode': 'equal_average', 'pool_weight': .5},
               {'mode': 'fixed_weight', 'pool_weight': weight}]
    for mode, cols in [('gate8', list(range(8))), ('gate4', list(range(4)))]:
        x = features[disagreement][:, cols]
        mean, std = x.mean(0), np.maximum(x.std(0), 1e-4)
        z = np.column_stack([np.ones(len(x)), (x - mean) / std])
        penalty = np.eye(z.shape[1]) * 10.
        penalty[0, 0] = 0.
        beta = np.linalg.solve(z.T @ z + penalty, z.T @ utility[disagreement])
        config = {'mode': mode, 'gate': {'columns': cols, 'feature_names': [NAMES[i] for i in cols],
                  'mean': mean.tolist(), 'std': std.tolist(), 'intercept': float(beta[0]),
                  'coefficients': beta[1:].tolist(), 'ridge': 10., 'threshold': 0.}}
        configs.append(config)
    summaries = []
    for config in configs:
        dest = OUT / 'pipelines' / config['mode']
        used = ['S2_C'] if config['mode'] == 'S2_C' else (['Pool_P'] if config['mode'] == 'Pool_P' else ['Pool_P', 'S2_C'])
        manifest = {**config, 'members': {e: [os.path.relpath(p, dest) for p in members(e)] if e in used else [] for e in ('Pool_P', 'S2_C')}}
        write(dest / 'pipeline.json', manifest)
        choice = decide(bank, p, config)
        c = y[row, choice].astype(bool)
        base = correct[1].astype(bool)
        summaries.append({'name': config['mode'], 'correct': int(c.sum()),
           'repairs_vs_C': int((c & ~base).sum()), 'harms_vs_C': int((~c & base).sum()),
           'in_sample_selector_fit_diagnostic_only': config['mode'].startswith('gate'),
           'frozen_pipeline_sha256': sha(dest / 'pipeline.json')})
    info = {'train_questions': len(bank), 'answer_disagreements': int(disagreement.sum()),
        'Pool_only_correct': int((utility > 0).sum()), 'C_only_correct': int((utility < 0).sum()),
        'both_wrong_disagreements': int((disagreement & (correct[0] == 0) & (correct[1] == 0)).sum()),
        'base_correct': {'Pool_P': int(correct[0].sum()), 'S2_C': int(correct[1].sum())},
        'fixed_weight_grid': grid, 'selected_pool_weight': weight, 'selector_fit_seconds': time.perf_counter() - tick,
        'fit_diagnostics_not_meta_OOF_estimates': True, 'summaries': summaries,
        'validation_used': False, 'test_used': False}
    write(OUT / 'selector_fit.json', info)
    input_paths = [OUT / 'oof-Pool_P.npy', FUSION / 'results/oof-s2_consistency.npy', OUT / 'train-labels.npy']
    input_paths += [p for e in ('Pool_P', 'S2_C') for m in members(e) for p in m.iterdir()]
    write(OUT / 'frozen_before_validation.json', {'sources': sources(), 'inputs': {str(p): sha(p) for p in input_paths},
          'pipelines': {c['mode']: sha(OUT / 'pipelines' / c['mode'] / 'pipeline.json') for c in configs},
          'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'validation_used_for_fitting': False})
    print({k: info[k] for k in ['answer_disagreements', 'Pool_only_correct', 'C_only_correct', 'selected_pool_weight']}, flush=True)

if __name__ == '__main__':
    main()
