"""Read-only evidence inventory and integrity check for the experiment review."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot():
    return {p.relative_to(ROOT).as_posix(): digest(p)
            for p in sorted(ROOT.rglob('*'))
            if p.is_file() and HERE not in p.parents}


if sys.argv[1] == 'before':
    before = snapshot()
    (HERE / 'protected_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    reports = sorted(ROOT.glob('code/**/REPORT.md'))
    reports = [p for p in reports if HERE not in p.parents]
    sources = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)} for p in reports]
    (HERE / 'sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    print(json.dumps({'protected_files': len(before), 'source_reports': len(sources)}))
elif sys.argv[1] == 'after':
    before = json.loads((HERE / 'protected_before.json').read_text(encoding='utf-8'))
    after = snapshot()
    changed = [p for p, h in before.items() if p in after and after[p] != h]
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    scores = {}
    for name in ['V', 'H', 'NV', 'B', 'S2_route', 'Pool_P']:
        path = ROOT / 'code/final_test_20261028/results' / name / 'score.json'
        score = json.loads(path.read_text(encoding='utf-8'))
        assert score['split'] == 'test' and score['n'] == 467
        assert abs(score['accuracy'] - 100 * score['correct'] / score['n']) < 1e-10
        scores[name] = {k: score[k] for k in ['correct', 'n', 'accuracy']}
    result = {'protected_files': len(before), 'changed': changed, 'missing': missing,
              'added_outside_review': added, 'passed': not changed and not missing,
              'official_test_scores_read_only': scores,
              'no_training_prediction_or_scoring_executed': True}
    (HERE / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
