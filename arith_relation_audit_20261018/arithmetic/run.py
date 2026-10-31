"""Freeze, fit three residuals, then five official evaluations and portable replays."""
import runtime
from utils import *
import shutil
import numpy as np
from decisions import sigmoid,base_logits,choose
from evidence import NAMES

def newton(x,z,y):
    beta=np.zeros(x.shape[1],np.float64)
    def loss(b):
        t=z+x@b
        return float(np.sum(np.logaddexp(0,t)-y*t)+.5*(b@b))
    history=[]
    for iteration in range(80):
        t=z+x@beta; p=1/(1+np.exp(-np.clip(t,-700,700)))
        g=x.T@(p-y)+beta
        h=(x.T*(p*(1-p)))@x+np.eye(x.shape[1])
        history.append({'iteration':iteration,'objective':loss(beta),'gradient':float(np.max(np.abs(g)))})
        if np.max(np.abs(g))<1e-7: break
        d=np.linalg.solve(h,g); scale=1.; before=loss(beta)
        for _ in range(40):
            proposal=beta-scale*d
            if loss(proposal)<=before-1e-4*scale*float(g@d):break
            scale*=.5
        beta=proposal
        if np.max(np.abs(scale*d))<1e-10:break
    p=1/(1+np.exp(-np.clip(z+x@beta,-700,700)))
    gradient=float(np.max(np.abs(x.T@(p-y)+beta)))
    assert gradient<1e-5 and np.isfinite(beta).all()
    return beta,{'iterations':len(history),'gradient_max_abs':gradient,'history':history}

def evaluate(path,method,folder):
    folder.mkdir(parents=True,exist_ok=True)
    a=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(method),'--checkpoint',str(path),
      '--split','validation','--output',str(folder/'predictions.json')],folder/'predict.log')
    b=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
      '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')],folder/'score.log')
    return {**read(folder/'score.json'),'prediction_wall_seconds':a,'scoring_wall_seconds':b,'fresh_total_seconds':a+b}

def main():
    from common import load_inputs,read_jsonl,DATA,parse_answer
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'frozen_before_validation.json').exists():raise RuntimeError('Already frozen; do not refit.')
    protocol=read(EXPERIMENT/'protocol.json')
    write(OUT/'protocol_before_fit.json',{'protocol_sha256':sha(EXPERIMENT/'protocol.json'),'sources':sources()})
    command([sys.executable,'-B',str(HERE/'check_extractor.py')],OUT/'extractor_checks.log')
    assert sha(HERE/'evidence.py')==sha(LEGACY/'evidence.py')
    tick=time.perf_counter()
    bank,_=load_inputs('train'); refs=read_jsonl(DATA/'gsm8k_train.jsonl')
    y=np.asarray([[parse_answer(c['text']) is not None and parse_answer(c['text'])==parse_answer(g['answer'])
      for c in r['candidates']] for r,g in zip(bank,refs)],bool)
    assert [r['id'] for r in bank]==[r['id'] for r in refs]==read(FUSION/'results/folds.json')['ids']
    assert np.array_equal(y,np.load(LEGACY/'results/train-labels.npy',allow_pickle=False))
    f=np.load(LEGACY/'results/train-evidence.npy',allow_pickle=False)
    assert sha(LEGACY/'results/train-evidence.npy')==read(LEGACY/'results/training_audit.json')['feature_sha256']
    logits=np.load(FUSION/'results/oof-s2_consistency.npy',allow_pickle=False)
    p=sigmoid(logits).mean(0);z=base_logits(p);base=choose(bank,p,None,{'columns':[]})
    base_correct=y[np.arange(len(bank)),base]
    fit=[]
    for name,columns in protocol['arithmetic']['variants'].items():
        t=time.perf_counter()
        if columns:beta,info=newton(f[...,columns].reshape(-1,len(columns)),z.ravel(),y.ravel().astype(float))
        else:beta=np.empty(0);info={'iterations':0}
        folder=OUT/'pipelines'/name
        config={'name':name,'columns':columns,'feature_names':[NAMES[c] for c in columns],'beta':beta.tolist(),
          'baseline_logit_clip':1e-12,'members':[os.path.relpath(m,folder) for m in members()]}
        write(folder/'pipeline.json',config)
        c=y[np.arange(len(bank)),choose(bank,p,f,config)]
        fit.append({'name':name,'parameters':len(beta),'beta':beta.tolist(),'optimizer':info,
          'fit_seconds':time.perf_counter()-t,'training_fit_diagnostic_only':True,'train_correct':int(c.sum()),
          'train_repairs':int((c&~base_correct).sum()),'train_harms':int((~c&base_correct).sum())})
    old=read(LEGACY/'results/pipelines/arithmetic/pipeline.json')
    folder=OUT/'pipelines/H'
    write(folder/'pipeline.json',{**old,'name':'H','members':[os.path.relpath(m,folder) for m in members()]})
    inputs=[LEGACY/'results/train-evidence.npy',LEGACY/'results/train-labels.npy',FUSION/'results/oof-s2_consistency.npy',
      FUSION/'results/folds.json',FUSION/'results/question-folds.npy',LEGACY/'results/pipelines/arithmetic/pipeline.json']
    inputs += [p for m in members() for p in m.iterdir()]
    frozen={'sources':sources(),'inputs':{str(p):sha(p) for p in inputs},
      'pipelines':{n:sha(OUT/'pipelines'/n/'pipeline.json') for n in ['B','N','V','NV','H']},
      'validation_loaded_before_freeze':False,'test_used':False,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')}
    write(OUT/'fit_results.json',{'runs':fit,'new_residual_fits':3,'new_base_training_runs':0,
      'fit_preparation_seconds':time.perf_counter()-tick,'feature_extraction_reused_not_charged_as_new':True})
    write(OUT/'frozen_before_validation.json',frozen)
    infer=EXPERIMENT/'portable/inference';infer.mkdir(parents=True,exist_ok=True)
    for n in ['method.py','decisions.py','evidence.py','runtime.py','models.py','reference_models.py','features.py','explicit_features.py']:
        shutil.copyfile(HERE/n,infer/n)
    for i,m in enumerate(members()):shutil.copytree(m,EXPERIMENT/f'portable/members/{i:02d}')
    records=[]
    for name in frozen['pipelines']:
        folder=OUT/'pipelines'/name;cfg=read(folder/'pipeline.json')
        result=evaluate(folder,HERE/'method.py',folder)
        dest=EXPERIMENT/'portable/pipelines'/name
        write(dest/'pipeline.json',{**cfg,'members':[f'../../members/{i:02d}' for i in range(5)]})
        replay=evaluate(dest,infer/'method.py',OUT/'portable_replays'/name)
        assert read(folder/'predictions.json')['selected']==read(OUT/'portable_replays'/name/'predictions.json')['selected']
        size=asset_bytes([p for i in range(5) for p in (EXPERIMENT/f'portable/members/{i:02d}').iterdir()]+[dest/'pipeline.json'])
        assert size<=16*1024**2
        records.append({'name':name,**result,'asset_bytes_uncompressed':size,'portable_replay':replay})
        write(OUT/'official_results.json',records)
        print(name,result['correct'],result['fresh_total_seconds'],flush=True)
    assert sources()==frozen['sources']
    assert all(sha(p)==h for p,h in frozen['inputs'].items())
    assert read(OUT/'pipelines/B/predictions.json')['selected']==read(OLD/'results/pipelines/S2_C_ensemble5_filtered/predictions.json')['selected']
    assert read(OUT/'pipelines/H/predictions.json')['selected']==read(LEGACY/'results/pipelines/arithmetic/predictions.json')['selected']
    print('Five evaluations and five portable replays complete.',flush=True)

if __name__=='__main__':main()
