"""Six frozen fresh official evaluations and portable inference replays."""
import runtime
from utils import HERE, CODE, OUT, read, write, sha, sources, command, members, asset_bytes
from pathlib import Path
import shutil
import sys

INFERENCE = ['method.py', 'decisions.py', 'runtime.py', 'models.py', 'reference_models.py', 'features.py', 'explicit_features.py']

def run(checkpoint, method, folder):
    folder.mkdir(parents=True, exist_ok=True)
    ps = command([sys.executable, '-B', str(CODE / 'predict.py'), '--method', str(method), '--checkpoint', str(checkpoint),
                  '--split', 'validation', '--output', str(folder / 'predictions.json')], folder / 'predict.log')
    ss = command([sys.executable, '-B', str(CODE / 'score_predictions.py'), '--split', 'validation',
                  '--predictions', str(folder / 'predictions.json'), '--output', str(folder / 'score.json')], folder / 'score.log')
    return {**read(folder / 'score.json'), 'prediction_wall_seconds': ps, 'scoring_wall_seconds': ss,
            'fresh_total_seconds': ps + ss}

def main():
    frozen = read(OUT / 'frozen_before_validation.json')
    assert sources() == frozen['sources']
    assert all(sha(p) == h for p, h in frozen['inputs'].items())
    portable = HERE / 'portable'
    infer = portable / 'inference'
    infer.mkdir(parents=True, exist_ok=True)
    for name in INFERENCE:
        shutil.copyfile(HERE / name, infer / name)
    for expert in ('Pool_P', 'S2_C'):
        for si, source in enumerate(members(expert)):
            target = portable / 'members' / expert / f'{si:02d}'
            shutil.copytree(source, target, dirs_exist_ok=True)
            assert {p.name: sha(p) for p in target.iterdir()} == {p.name: sha(p) for p in source.iterdir()}
    records = []
    for name, digest in frozen['pipelines'].items():
        folder = OUT / 'pipelines' / name
        assert sha(folder / 'pipeline.json') == digest
        manifest = read(folder / 'pipeline.json')
        timing = run(folder, HERE / 'method.py', folder)
        used_assets = [p for expert in ('Pool_P', 'S2_C') if manifest['members'][expert] for member in members(expert) for p in member.iterdir()]
        size = asset_bytes(used_assets + [folder / 'pipeline.json'])
        assert size <= 16 * 1024**2
        dest = portable / 'pipelines' / name
        local_manifest = {**manifest, 'members': {e: [f'../../members/{e}/{si:02d}' for si in range(5)] if manifest['members'][e] else [] for e in ('Pool_P', 'S2_C')}}
        write(dest / 'pipeline.json', local_manifest)
        replay_folder = OUT / 'portable_replays' / name
        replay = run(dest, infer / 'method.py', replay_folder)
        assert read(replay_folder / 'predictions.json')['selected'] == read(folder / 'predictions.json')['selected']
        assert replay['correct'] == timing['correct']
        result = {'name': name, **timing, 'asset_bytes_uncompressed': size, 'portable_replay': replay}
        records.append(result)
        write(OUT / 'official_results.json', records)
        print(f'{len(records)}/6 {name}: {timing["correct"]}/{timing["n"]}, {timing["fresh_total_seconds"]:.3f}s, {size/1024**2:.3f} MiB', flush=True)
    total_package = asset_bytes([p for p in portable.rglob('*') if p.is_file() and p.suffix != '.py'])
    assert total_package < 16 * 1024**2
    write(OUT / 'portable_package.json', {'all_six_predictors_shared_asset_bytes': total_package,
           'inference_source_bytes': sum((infer / name).stat().st_size for name in INFERENCE),
           'hashes': {str(p.relative_to(portable)): sha(p) for p in portable.rglob('*') if p.is_file()}})
    assert sources() == frozen['sources']

if __name__ == '__main__':
    main()
