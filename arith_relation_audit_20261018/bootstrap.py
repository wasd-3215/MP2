"""Create a new experiment; snapshot every pre-existing workspace file first."""
from pathlib import Path
import hashlib
import json
import shutil
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    snapshot = HERE / 'protected_before.json'
    if snapshot.exists():
        raise RuntimeError('Snapshot already exists; do not replace it.')
    start = time.perf_counter()
    protected = {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*')
                 if p.is_file() and HERE not in p.parents}
    snapshot.write_text(json.dumps(protected, indent=2)+'\n', encoding='utf-8')
    arith = HERE / 'arithmetic'
    arith.mkdir()
    legacy = CODE / 'text_evidence_20261017'
    names = ['method.py','decisions.py','evidence.py','runtime.py','models.py',
             'reference_models.py','features.py','explicit_features.py','check_extractor.py']
    copied = {}
    for name in names:
        shutil.copyfile(legacy/name, arith/name)
        copied[name] = sha(arith/name)
    (HERE/'bootstrap_record.json').write_text(json.dumps({
        'existing_files': len(protected), 'copied_unchanged':copied,
        'snapshot_seconds':time.perf_counter()-start, 'test_semantic_content_used':False,
        'test_files_only_hashed_as_bytes':True},indent=2)+'\n',encoding='utf-8')
    print({'protected_files':len(protected),'copied':len(copied)},flush=True)

if __name__ == '__main__': main()
