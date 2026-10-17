"""Training utilities, official evaluation and original-file protection."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
HERE=Path(__file__).resolve().parent
CODE=HERE.parent
ROOT=CODE.parent
sys.path.insert(0,str(CODE))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected(output):
    before=read(HERE/'protected_before.json')
    changed=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    write(output/'protected_verification.json',{'files_checked':len(before),'changed':changed,'passed':not changed})
    assert not changed,changed


def command(args,log):
    tick=time.perf_counter()
    r=subprocess.run(args,cwd=CODE,env=os.environ.copy(),capture_output=True,text=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(r.stdout+r.stderr,encoding='utf-8')
    if r.returncode:raise RuntimeError(f'{log}: {r.stderr}')
    return time.perf_counter()-tick


def official(folder,bank,choices):
    from common import PROTOCOL
    write(folder/'validation-predictions.json',{'protocol':PROTOCOL,'split':'validation',
          'ids':[r['id'] for r in bank],'selected':[int(i) for i in choices]})
    seconds=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
         '--predictions',str(folder/'validation-predictions.json'),'--output',str(folder/'validation-score.json')],folder/'score.log')
    return {**read(folder/'validation-score.json'),'scoring_wall_seconds':seconds}


def asset_bytes(path):
    total=0
    for p in Path(path).rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.npz':
            with zipfile.ZipFile(p) as z:total+=sum(v.file_size for v in z.infolist())
        else:total+=p.stat().st_size
    return total


def source_hashes():
    names=['runtime.py','reference_models.py','scorers.py','explicit_features.py','utilities.py',
           'reference_training.py','weights.py','train.py','protocol.json']
    return {name:sha(HERE/name) for name in names}
