"""Create the fixed comparison list and run fresh official prediction/scoring.

No training, searches, test inputs, or reference reads occur in this script.
"""
import runtime
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
SOURCE = CODE / 'fusion_confidence_20261017'
OUT = HERE / 'results'
SEEDS = [17, 29, 43, 71, 101]
VARIANTS = ['s2', 's2_parse', 's2_consistency']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {p.name: sha(p) for p in HERE.glob('*.py')}


def asset_bytes(members, pipeline):
    total = pipeline.stat().st_size
    for member in members:
        for path in member.iterdir():
            if path.suffix == '.npz':
                with zipfile.ZipFile(path) as archive:
                    total += sum(v.file_size for v in archive.infolist())
            else:
                total += path.stat().st_size
    return total


def command(args, log):
    start = time.perf_counter()
    result = subprocess.run(args, cwd=CODE, env=os.environ.copy(), capture_output=True, text=True)
    Path(log).write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{log}: {result.stderr}')
    return time.perf_counter() - start


def main():
    OUT.mkdir(exist_ok=True)
    frozen = source_hashes()
    if (OUT / 'source_hashes.json').exists():
        assert read(OUT / 'source_hashes.json') == frozen
    else:
        write(OUT / 'source_hashes.json', frozen)
    route = read(SOURCE / 'results/pipelines/s2_5_route_model_agreement_support3/pipeline.json')['routing']
    assert route == {'signal': 'model_agreement', 'threshold': 0.8, 'min_support': 3}
    training = read(SOURCE / 'results/training_runs.json')
    records = []
    pairs = []
    def add(variant, seeds, filtered, routing=None):
        kind = 'ensemble5' if len(seeds) == 5 else f'seed{seeds[0]}'
        base = f'{variant}_{kind}' + ('_route' if routing else '')
        name = base + ('_filtered' if filtered else '_original')
        folder = OUT / 'pipelines' / name
        folder.mkdir(parents=True, exist_ok=True)
        members = [SOURCE / 'results/runs' / variant / f'seed_{s}/full/bundle' for s in seeds]
        for seed, member in zip(seeds, members):
            origin = next(r for r in training if r['variant'] == variant and r['seed'] == seed and r['role'] == 'full')
            assert origin['epochs'] == 20
            assert {p.name: sha(p) for p in member.iterdir()} == origin['bundle_hashes']
        manifest = {'members': [os.path.relpath(p, folder) for p in members],
                    'rule': 'mean_probability', 'routing': routing, 'filter_invalid': filtered}
        write(folder / 'pipeline.json', manifest)
        records.append({'name': name, 'variant': variant, 'seeds': seeds, 'members': len(seeds),
                        'filtered': filtered, 'routing': routing, 'checkpoint': str(folder),
                        'asset_bytes': asset_bytes(members, folder / 'pipeline.json')})
        return name
    for variant in VARIANTS:
        for seeds in [[s] for s in SEEDS] + [SEEDS]:
            a = add(variant, seeds, False)
            b = add(variant, seeds, True)
            pairs.append({'original': a, 'filtered': b})
    a = add('s2', SEEDS, False, route)
    b = add('s2', SEEDS, True, route)
    pairs.append({'original': a, 'filtered': b})
    write(HERE / 'protocol.json', {
        'date': '2026-10-17 Asia/Shanghai', 'split': 'validation', 'training_runs': 0,
        'source': 'fusion_confidence_20261017 fixed20 full-data models',
        'variants': VARIANTS, 'seeds': SEEDS, 'predictors': len(records), 'paired_filters': len(pairs),
        'filter': 'After complete scoring and frozen routing; replace invalid choice with highest original aggregate score among valid candidates; preserve valid/all-invalid choices',
        'tie_rule': 'earliest original candidate', 'routing': route,
        'no_recalibration_or_threshold_search': True, 'test_used': False,
        'bootstrap': {'unit': 'question', 'replicates': 10000, 'seed': 20261017,
                      'scope': 'descriptive conditional on frozen predictors and repeated validation development'},
        'selection': 'No model winner selected; all predeclared comparisons retained'})
    write(OUT / 'pairs.json', pairs)
    write(OUT / 'predictors.json', records)
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    except (ImportError, OSError):
        pass
    write(OUT / 'environment.json', {'python': sys.version, 'executable': sys.executable,
          'cpu': cpu, 'platform': platform.platform(), 'numerical_threads': 4,
          'openblas_threads': 1, 'serial_processes': True})
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
        print(f'{len(results)}/{len(records)} {record["name"]}: {result["correct"]}/{result["n"]} {result["fresh_total_seconds"]:.3f}s', flush=True)
    assert source_hashes() == frozen
    print('ALL FRESH PREDICTION AND OFFICIAL SCORES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
