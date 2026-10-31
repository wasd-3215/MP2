"""Frozen experiment definitions and read-only external data accounting."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
HERE=Path(__file__).resolve().parent;CODE=HERE.parent;ROOT=CODE.parent;OUT=HERE/'results'
sys.path.insert(0,str(CODE))
SEEDS=[17,29,43,71,101]
VARIANTS=[f'{r}_{s}' for r in ['CandBCE','CandMaxCE','GroupCE'] for s in ['NoSize','Size']]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources():return {p.name:sha(p) for p in HERE.iterdir() if p.suffix=='.py' or p.name=='protocol.json'}
def command(args,log):
    tick=time.perf_counter();r=subprocess.run(args,cwd=CODE,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
    Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(f'{log}: {r.stderr}')
    return time.perf_counter()-tick
def asset_bytes(paths):
    n=0
    for p in set(Path(p).resolve() for p in paths):
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as a:n+=sum(v.file_size for v in a.infolist())
        else:n+=p.stat().st_size
    return n
def protected():
    before=read(HERE/'protected_before.json')
    changed=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    result={'passed':not changed,'files_checked':len(before),'changed':changed};write(OUT/'protected_verification.json',result)
    return result
