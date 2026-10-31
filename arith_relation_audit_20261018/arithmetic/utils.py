from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
CODE = EXPERIMENT.parent
ROOT = CODE.parent
OUT = HERE/'results'
OLD = CODE/'interaction_consensus_20261017'
FUSION = CODE/'fusion_confidence_20261017'
LEGACY = CODE/'text_evidence_20261017'
sys.path.insert(0,str(CODE))
SEEDS = [17,29,43,71,101]

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sources():
    return {str(p):sha(p) for p in [*HERE.glob('*.py'),EXPERIMENT/'protocol.json']}
def members(): return [OLD/f'results/runs/S2_C/seed_{s}/bundle' for s in SEEDS]
def command(args,log):
    tick=time.perf_counter()
    env=os.environ.copy(); env['PYTHONDONTWRITEBYTECODE']='1'; env['PYTHONIOENCODING']='utf-8'
    result=subprocess.run(args,cwd=CODE,env=env,capture_output=True,text=True,encoding='utf-8')
    log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode: raise RuntimeError(str(log)+'\n'+result.stderr)
    return time.perf_counter()-tick
def asset_bytes(paths):
    size=0
    for p in set(Path(p).resolve() for p in paths):
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as a: size+=sum(i.file_size for i in a.infolist())
        else: size+=p.stat().st_size
    return size
def protected():
    original=read(EXPERIMENT/'protected_before.json')
    changed=[name for name,h in original.items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    result={'passed':not changed,'files_checked':len(original),'changed':changed}
    write(EXPERIMENT/'results/protected_verification.json',result)
    return result
