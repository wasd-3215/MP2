"""Fit three bias-free residuals on base-model training OOF, never validation."""
import runtime
from utils import HERE,OUT,FUSION,read,write,sha,sources,members
import os
import platform
import sys
import time
import winreg
import numpy as np
from common import read_jsonl,DATA
from decisions import sigmoid,base_logits,choose
from evidence import NAMES

def newton(x,z,y,penalty):
    beta=np.zeros(x.shape[1],np.float64)
    history=[]
    def loss(b):
        value=z+x@b
        return float(np.sum(np.logaddexp(0,value)-y*value)+.5*penalty*(b@b))
    before=loss(beta)
    for iteration in range(80):
        value=z+x@beta
        p=1/(1+np.exp(-np.clip(value,-700,700)))
        gradient=x.T@(p-y)+penalty*beta
        hessian=(x.T*(p*(1-p)))@x+penalty*np.eye(x.shape[1])
        grad=float(np.max(np.abs(gradient)))
        history.append({'iteration':iteration,'objective':loss(beta),'gradient_max_abs':grad})
        if grad<1e-7:break
        direction=np.linalg.solve(hessian,gradient)
        old=loss(beta);scale=1.
        for backtrack in range(40):
            proposal=beta-scale*direction
            if loss(proposal)<=old-1e-4*scale*float(gradient@direction):break
            scale*=.5
        beta=proposal
        if float(np.max(np.abs(scale*direction)))<1e-10:break
    value=z+x@beta
    final_gradient=x.T@(1/(1+np.exp(-np.clip(value,-700,700)))-y)+penalty*beta
    assert np.isfinite(beta).all() and loss(beta)<=before+1e-8
    assert np.max(np.abs(final_gradient))<1e-5
    return beta,{'objective_initial':before,'objective_final':loss(beta),'gradient_max_abs':float(np.max(np.abs(final_gradient))),
       'iterations':len(history),'history':history,'converged':True}

def main():
    assert read(OUT/'extractor_checks.json')['source_sha256']==sha(HERE/'evidence.py')
    audit=read(OUT/'training_audit.json')
    assert audit['extractor_sha256']==sha(HERE/'evidence.py')
    assert sha(OUT/'train-evidence.npy')==audit['feature_sha256']
    assert read(OUT/'manual_training_audit.json')['passed']
    frozen=sources();start=time.perf_counter()
    bank=read_jsonl(DATA/'candidates/gsm8k_train.jsonl')
    fold_document=read(FUSION/'results/folds.json')
    assert fold_document['ids']==[r['id'] for r in bank]
    assert np.array_equal(np.asarray(fold_document['folds']),np.load(FUSION/'results/question-folds.npy'))
    logits=np.load(FUSION/'results/oof-s2_consistency.npy')
    y=np.load(OUT/'train-labels.npy')
    assert np.array_equal(y,np.load(FUSION/'results/train-labels.npy'))
    f=np.load(OUT/'train-evidence.npy')
    assert logits.shape==(5,len(bank),4) and f.shape==(len(bank),4,len(NAMES))
    p=sigmoid(logits).mean(0);z=base_logits(p)
    protocol=read(HERE/'protocol.json')
    configs=[];records=[]
    base_choice=choose(bank,p,None,{'columns':[]})
    base_correct=y[np.arange(len(bank)),base_choice]
    for name,columns in protocol['variants'].items():
        tick=time.perf_counter()
        if columns:
            beta,diagnostic=newton(f[...,columns].reshape(-1,len(columns)),z.ravel(),y.ravel().astype(float),protocol['L2'])
        else:beta=np.empty(0);diagnostic={'converged':True,'iterations':0}
        folder=OUT/'pipelines'/name
        config={'name':name,'columns':columns,'feature_names':[NAMES[c] for c in columns],'beta':beta.tolist(),
                'baseline_logit_clip':1e-12,'members':[os.path.relpath(m,folder) for m in members()]}
        write(folder/'pipeline.json',config);configs.append(config)
        choice=choose(bank,p,f,config);c=y[np.arange(len(bank)),choice]
        record={'name':name,'parameters':len(beta),'fit_seconds':time.perf_counter()-tick,'optimizer':diagnostic,
           'training_fit_diagnostic_only':True,'correct':int(c.sum()),'repairs_vs_base':int((c&~base_correct).sum()),
           'harms_vs_base':int((~c&base_correct).sum()),'beta':beta.tolist()}
        records.append(record)
        print(name,'beta',beta.tolist(),'in-sample net',int(c.sum())-int(base_correct.sum()),flush=True)
    input_paths=[FUSION/'results/oof-s2_consistency.npy',FUSION/'results/folds.json',FUSION/'results/question-folds.npy',
        OUT/'train-evidence.npy',OUT/'train-labels.npy',OUT/'manual_training_audit.json']
    input_paths += [p for m in members() for p in m.iterdir()]
    write(OUT/'frozen_before_validation.json',{'sources':frozen,'inputs':{str(p):sha(p) for p in input_paths},
       'pipelines':{c['name']:sha(OUT/'pipelines'/c['name']/'pipeline.json') for c in configs},
       'validation_used_for_fitting':False,'feature_design_from_prior_diagnosis':True,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S')})
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    write(OUT/'environment.json',{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
      'cpu':cpu,'platform':platform.platform(),'fit_device':'cpu','fit_blas_threads':1,'inference_torch_threads':4,
      'inference_interop_threads':1,'serial_numerical_jobs':True})
    write(OUT/'fit_results.json',{'runs':records,'new_base_model_training_runs':0,'new_residual_fits':3,
      'L2':protocol['L2'],'total_seconds':time.perf_counter()-start,'feature_names':NAMES,'validation_loaded':False,
      'test_used':False,'train_diagnostics_not_meta_OOF_accuracy':True})
    assert frozen==sources()

if __name__=='__main__':main()
