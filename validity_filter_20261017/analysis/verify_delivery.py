"""Export self-contained bundles, replay them officially, verify original hashes."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_suite import HERE, CODE, OUT, read, write, sha, command, asset_bytes
import shutil


def main():
    assert read(OUT / 'new_bank_checks.json')['passed']
    records = read(OUT / 'official_results.json')
    inference = HERE / 'portable/inference'
    inference.mkdir(parents=True, exist_ok=True)
    names = ['method.py', 'reference_method.py', 'runtime.py', 'decisions.py',
             'explicit_features.py', 'scorers.py', 'reference_models.py']
    for name in names:
        target = inference / name
        if target.exists():
            assert sha(target) == sha(HERE / name)
        else:
            shutil.copyfile(HERE / name, target)
    deliveries = []
    for name in ['s2_ensemble5_filtered', 's2_parse_ensemble5_filtered',
                 's2_consistency_ensemble5_filtered', 's2_ensemble5_route_filtered']:
        record = next(r for r in records if r['name'] == name)
        source = Path(record['checkpoint'])
        manifest = read(source / 'pipeline.json')
        destination = HERE / 'portable' / name
        destination.mkdir(parents=True, exist_ok=True)
        member_paths = []
        for index, relative in enumerate(manifest['members']):
            member = (source / relative).resolve()
            target = destination / 'members' / f'{index:02d}'
            if not target.exists():
                shutil.copytree(member, target)
            assert {p.name: sha(p) for p in target.iterdir()} == {p.name: sha(p) for p in member.iterdir()}
            member_paths.append(target)
        manifest['members'] = [f'members/{i:02d}' for i in range(len(member_paths))]
        write(destination / 'pipeline.json', manifest)
        folder = OUT / 'delivery_check' / name
        folder.mkdir(parents=True, exist_ok=True)
        prediction_seconds = command([sys.executable, '-B', str(CODE / 'predict.py'), '--method', str(inference / 'method.py'),
            '--checkpoint', str(destination), '--split', 'validation', '--output', str(folder / 'predictions.json')], folder / 'predict.log')
        scoring_seconds = command([sys.executable, '-B', str(CODE / 'score_predictions.py'), '--split', 'validation',
            '--predictions', str(folder / 'predictions.json'), '--output', str(folder / 'score.json')], folder / 'score.log')
        assert read(folder / 'predictions.json')['selected'] == read(source / 'predictions.json')['selected']
        assert read(folder / 'score.json')['correct'] == record['correct']
        size = asset_bytes(member_paths, destination / 'pipeline.json')
        assert size <= 16 * 1024**2
        deliveries.append({'name': name, 'portable_bundle': str(destination), 'correct': record['correct'],
           'asset_bytes': size, 'fresh_prediction_scoring_seconds': prediction_seconds + scoring_seconds,
           'reference_cpu_certification': False,
           'asset_hashes': {str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file()}})
        print('PORTABLE REPLAY', name, record['correct'], flush=True)
    before = read(HERE / 'protected_before.json')
    root = CODE.parent
    changed = [name for name, digest in before.items() if not (root / name).is_file() or sha(root / name) != digest]
    write(OUT / 'protected_verification.json', {'files_checked': len(before), 'changed': changed, 'passed': not changed})
    assert not changed, changed
    frozen = read(OUT / 'source_hashes.json')
    assert all(sha(HERE / name) == digest for name, digest in frozen.items())
    research = read(OUT / 'analysis_sources_before.json')
    assert all(sha(HERE / name) == digest for name, digest in research.items())
    write(OUT / 'verification.json', {'passed': True, 'protected_files': len(before), 'protected_changes': [],
      'fresh_study_predictions_and_scores': 38, 'extra_portable_replays': 4,
      'new_bank_checks_predictors': 38, 'filter_pairs': 19,
      'baseline_replay_predictors': 19, 'all_pair_new_errors': 0, 'training_runs': 0,
      'test_used': False, 'inference_sources_unchanged': True, 'analysis_sources_unchanged': True,
      'portable_deliveries': deliveries, 'reference_cpu_certification': False})
    write(OUT / 'delivery_source_hashes.json', {str(p.relative_to(HERE)): sha(p)
          for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py', '.md')})
    print('VERIFICATION PASSED; PROTECTED FILES', len(before), flush=True)


if __name__ == '__main__':
    main()
