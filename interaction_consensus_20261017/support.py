"""Paths, frozen definitions, safe resource accounting and serial subprocesses."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
OUT = HERE / 'results'
sys.path.insert(0, str(CODE))
SEEDS = [17, 29, 43, 71, 101]
ARCHITECTURES = ['I', 'Pool', 'S2']
VARIANTS = [f'{a}_{f}' for a in ARCHITECTURES for f in ('P', 'C')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    names = ['models.py', 'features.py', 'runtime.py', 'reference_models.py',
             'explicit_features.py', 'method.py', 'support.py', 'train.py', 'protocol.json']
    return {name: sha(HERE / name) for name in names}


def command(args, log):
    start = time.perf_counter()
    r = subprocess.run(args, cwd=CODE, env=os.environ.copy(), capture_output=True, text=True)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log).write_text(r.stdout + r.stderr, encoding='utf-8')
    if r.returncode:
        raise RuntimeError(f'{log}: {r.stderr}')
    return time.perf_counter() - start


def asset_bytes(members, manifest):
    total = Path(manifest).stat().st_size
    for member in members:
        for path in Path(member).iterdir():
            if path.suffix == '.npz':
                with zipfile.ZipFile(path) as archive:
                    total += sum(f.file_size for f in archive.infolist())
            else:
                total += path.stat().st_size
    return total


def protected():
    before = read(HERE / 'protected_before.json')
    changed = [name for name, h in before.items() if not (ROOT / name).is_file() or sha(ROOT / name) != h]
    write(OUT / 'protected_verification.json', {'passed': not changed, 'files_checked': len(before), 'changed': changed})
    assert not changed, changed
