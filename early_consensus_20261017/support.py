"""Experiment paths, frozen definitions, serial evaluation and file protection."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
HERE=Path(__file__).resolve().parent
CODE=HERE.parent
ROOT=CODE.parent
OUT=HERE/'results'
sys.path.insert(0,str(CODE))
SEEDS=[17,29,43,71,101]
SCOPES=['All','D']
FEATURES=['P','C']
POSITIONS=['Late','Early']
VARIANTS=[f'{s}_{f}_{p}' for s in SCOPES for f in FEATURES for p in POSITIONS]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_hashes():
    names=['models.py','features.py','runtime.py','reference_models.py','explicit_features.py',
           'method.py','support.py','sampling.py','train.py','protocol.json']
    return {name:sha(HERE/name) for name in names}
def command(args,log):
    start=time.perf_counter();r=subprocess.run(args,cwd=CODE,env=os.environ.copy(),capture_output=True,text=True)
    Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(f'{log}: {r.stderr}')
    return time.perf_counter()-start
def asset_bytes(members,manifest):
    size=Path(manifest).stat().st_size
    for member in members:
        for p in Path(member).iterdir():
            if p.suffix=='.npz':
                with zipfile.ZipFile(p) as z:size+=sum(a.file_size for a in z.infolist())
            else:size+=p.stat().st_size
    return size
def protected():
    before=read(HERE/'protected_before.json')
    changed=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write(OUT/'protected_verification.json',{'passed':not changed,'files_checked':len(before),'changed':changed})
    assert not changed,changed
