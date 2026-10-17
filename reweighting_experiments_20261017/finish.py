"""Freeze analysis definitions before scoring the six secondary ensembles."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time


def main():
    here=Path(__file__).resolve().parent
    assert len(json.loads((here/'results/training_runs.json').read_text()))==30
    sources=json.loads((here/'analysis_sources_before.json').read_text())
    assert all(hashlib.sha256((here/n).read_bytes()).hexdigest()==h for n,h in sources.items())
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
    for key in ['OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
    stages=[('analyze.py',sys.executable),('verify.py',sys.executable),('report.py',sys.executable),
            ('plot_results.py','D:/anaconda/python.exe')]
    for name,executable in stages:
        print('START',name,flush=True);tick=time.perf_counter()
        with (here/'results'/f'{Path(name).stem}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([executable,'-B',str(here/name)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        if result.returncode:raise RuntimeError(f'{name} failed; inspect results log')
        print('DONE',name,f'{time.perf_counter()-tick:.2f}s',flush=True)
    print('ALL EVALUATION AND DELIVERY STEPS COMPLETE',flush=True)


if __name__=='__main__':main()
