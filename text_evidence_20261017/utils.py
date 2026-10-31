"""Read-only external assets, serial subprocesses, source and resource accounting."""
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
OLD = CODE / 'interaction_consensus_20261017'
FUSION = CODE / 'fusion_confidence_20261017'
sys.path.insert(0, str(CODE))
SEEDS = [17,29,43,71,101]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources():
    return {p.name:sha(p) for p in HERE.iterdir() if p.suffix=='.py' or p.name=='protocol.json'}

def members():
    return [OLD/f'results/runs/S2_C/seed_{s}/bundle' for s in SEEDS]

def command(args, log):
    start=time.perf_counter()
    r=subprocess.run(args,cwd=CODE,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    Path(log).write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode: raise RuntimeError(f'{log}: {r.stderr}')
    return time.perf_counter()-start

def asset_bytes(paths):
    size=0
    for p in set(Path(p).resolve() for p in paths):
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as a: size+=sum(v.file_size for v in a.infolist())
        else: size+=p.stat().st_size
    return size

def protected():
    original=read(HERE/'protected_before.json')
    changed=[p for p,h in original.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    result={'passed':not changed,'files_checked':len(original),'changed':changed}
    write(OUT/'protected_verification.json',result)
    return result
