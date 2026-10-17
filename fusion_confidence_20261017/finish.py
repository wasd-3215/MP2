"""After training, run all dependent evaluation/delivery steps sequentially."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time


def main():
    here=Path(__file__).resolve().parent
    assert len(json.loads((here/'results/training_runs.json').read_text()))==140
    frozen=json.loads((here/'analysis_sources_final_before.json').read_text())['hashes']
    assert all(hashlib.sha256((here/name).read_bytes()).hexdigest()==digest for name,digest in frozen.items())
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
    for key in ['OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
    plotting_python=Path('D:/anaconda/python.exe')
    stages=[('check_implementation.py',sys.executable),('analyze.py',sys.executable),
            ('verify.py',sys.executable),('routing_diagnostics.py',sys.executable),
            ('tradeoff_diagnostics.py',sys.executable),
            ('report.py',sys.executable),('plot_results.py',str(plotting_python))]
    for name,executable in stages:
        print('START',name,flush=True);tick=time.perf_counter()
        with (here/'results'/f'{Path(name).stem}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([executable,'-B',str(here/name)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        if result.returncode:raise RuntimeError(f'{name} failed; see its results log')
        print('DONE',name,f'{time.perf_counter()-tick:.2f}s',flush=True)
    print('ALL EVALUATION AND DELIVERY STEPS COMPLETE',flush=True)


if __name__=='__main__':main()
