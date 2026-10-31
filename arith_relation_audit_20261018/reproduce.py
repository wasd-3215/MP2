"""Reproduce into a NEW sibling experiment directory, reusing fixed reading inputs."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    dest=args.output.resolve()
    if dest.parent!=HERE.parent or dest.exists():raise ValueError('Use a new immediate child directory of code/.')
    dest.mkdir()
    for name in ['bootstrap.py','protocol.json','materialize_review.py','verify_delivery.py','make_report.py']:
        shutil.copyfile(HERE/name,dest/name)
    for folder in ['arithmetic','relation']:
        (dest/folder).mkdir()
        for source in (HERE/folder).glob('*.py'):shutil.copyfile(source,dest/folder/source.name)
    # bootstrap normally copies the immutable inference sources; these are already
    # copied above, so create the same protected snapshot without overwriting them.
    import hashlib,json
    protected={f.relative_to(dest.parent.parent).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
      for f in dest.parent.parent.rglob('*') if f.is_file() and dest not in f.parents}
    (dest/'protected_before.json').write_text(json.dumps(protected,indent=2)+'\n',encoding='utf-8')
    for key,value in {'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','BLIS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'4'}.items():os.environ[key]=value
    def run(name,*extra):subprocess.run([sys.executable,'-B',str(dest/name),*extra],cwd=dest.parent,env=os.environ.copy(),check=True)
    run('arithmetic/run.py');run('arithmetic/analyze.py')
    run('relation/prepare.py');run('relation/develop.py')
    target=dest/'relation/results'
    for name in ['development_review.json','blind_review_input.json']:shutil.copyfile(HERE/'relation/results'/name,target/name)
    run('relation/audit.py','freeze');run('relation/audit.py','confirm')
    run('materialize_review.py');run('relation/finalize.py');run('verify_delivery.py');run('make_report.py')
    print('Reproduction complete. Manual reading inputs were replayed, not independently re-annotated.',flush=True)

if __name__=='__main__':main()
