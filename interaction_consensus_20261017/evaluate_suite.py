"""Seventy-two fresh official evaluations; no training or model selection."""
import runtime
from support import HERE, CODE, OUT, SEEDS, VARIANTS, read, write, sha, command, asset_bytes, source_hashes
from pathlib import Path
import os
import sys


def main():
    assert read(OUT / 'training_verification.json')['passed']
    assert read(OUT / 'training_sources.json') == source_hashes()
    training = read(OUT / 'training_runs.json')
    write(OUT / 'evaluation_sources.json', {'evaluate_suite.py': sha(HERE / 'evaluate_suite.py')})
    records = []
    for variant in VARIANTS:
        for seeds in [[s] for s in SEEDS] + [SEEDS]:
            kind = f'seed{seeds[0]}' if len(seeds) == 1 else 'ensemble5'
            members = [OUT / 'runs' / variant / f'seed_{seed}/bundle' for seed in seeds]
            for seed, member in zip(seeds, members):
                run = next(r for r in training if r['variant'] == variant and r['seed'] == seed)
                assert {p.name: sha(p) for p in member.iterdir()} == run['bundle_hashes']
            for filtered in (False, True):
                mode = 'filtered' if filtered else 'raw'
                name = f'{variant}_{kind}_{mode}'
                folder = OUT / 'pipelines' / name
                folder.mkdir(parents=True, exist_ok=True)
                write(folder / 'pipeline.json', {'members': [os.path.relpath(p, folder) for p in members],
                      'filter_invalid': filtered, 'rule': 'mean_sigmoid_probability'})
                records.append({'name': name, 'variant': variant, 'architecture': variant.split('_')[0],
                    'feature_mode': variant.split('_')[1], 'seeds': seeds, 'members': len(seeds),
                    'mode': mode, 'checkpoint': str(folder),
                    'asset_bytes': asset_bytes(members, folder / 'pipeline.json')})
    assert len(records) == 72
    write(OUT / 'predictors.json', records)
    results = []
    for record in records:
        folder = Path(record['checkpoint'])
        if (folder / 'timing.json').exists():
            timing = read(folder / 'timing.json')
        else:
            prediction = command([sys.executable, '-B', str(CODE / 'predict.py'), '--method', str(HERE / 'method.py'),
                '--checkpoint', str(folder), '--split', 'validation', '--output', str(folder / 'predictions.json')], folder / 'predict.log')
            scoring = command([sys.executable, '-B', str(CODE / 'score_predictions.py'), '--split', 'validation',
                '--predictions', str(folder / 'predictions.json'), '--output', str(folder / 'score.json')], folder / 'score.log')
            timing = {'prediction_wall_seconds': prediction, 'scoring_wall_seconds': scoring,
                      'fresh_total_seconds': prediction + scoring}
            write(folder / 'timing.json', timing)
        result = {**record, **read(folder / 'score.json'), **timing}
        results.append(result)
        write(OUT / 'official_results.json', results)
        print(f'{len(results)}/72 {record["name"]}: {result["correct"]}/{result["n"]}, {result["fresh_total_seconds"]:.3f}s', flush=True)
    assert read(OUT / 'training_sources.json') == source_hashes()
    print('ALL 72 FRESH OFFICIAL EVALUATIONS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
