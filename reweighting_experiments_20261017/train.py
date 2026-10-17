"""Thirty paired fixed-budget fits; no validation-dependent training choices."""
from reference_training import labels,infer,checks
from utilities import HERE,CODE,write,read,sha,protected,official,source_hashes
import math,platform,random,sys,time
import numpy as np
import torch
from common import load_inputs
from explicit_features import FEATURES,extract,fit_standardization
from scorers import Scorer,configuration
from weights import make_weights,groups


def weighted_bce(logits,target,weights):
    per_question=torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1)
    return (per_question*weights).mean()


def train_one(model_name,variant,scheme,seed,protocol,data,prep,weight,bank,output,sources):
    group=f'{model_name}_{scheme}';folder=output/'runs'/group/f'seed_{seed}'
    identity={'group':group,'variant':variant,'seed':seed,'sources':sources}
    if (folder/'result.json').exists():
        r=read(folder/'result.json');assert r['identity']==identity
        for name,digest in r['bundle_hashes'].items():assert sha(folder/'bundle'/name)==digest
        print('RESUME',group,seed,flush=True);return r
    folder.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    tx,tf,ty,vx,vf,vy=data
    cols=FEATURES[configuration(variant)['feature_mode']]
    f=tf[...,cols];ef=vf[...,cols]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=Scorer(**configuration(variant))
    torch.manual_seed(seed+20000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    generator=torch.Generator().manual_seed(seed+10000)
    w=torch.from_numpy(weight.astype(np.float32));history=[]
    for epoch in range(1,protocol['epochs']+1):
        model.train();order=torch.randperm(len(tx),generator=generator)
        total=0.;clipped=0;grad_norm_sum=0.
        for begin in range(0,len(order),protocol['batch_size']):
            ix=order[begin:begin+protocol['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=weighted_bce(model(tx[ix],f[ix]),ty[ix],w[ix])
            assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'])
            clipped+=int(norm>protocol['gradient_clip']);grad_norm_sum+=norm.item()
            optimizer.step();total+=loss.item()*len(ix)
        model.eval()
        with torch.inference_mode():
            train_logits=infer(model,tx,f);val_logits=infer(model,vx,ef)
        def bce(z,t):return float(np.mean(np.logaddexp(0,z)-t*z))
        steps=math.ceil(len(tx)/protocol['batch_size'])
        history.append({'epoch':epoch,'steps':epoch*steps,'draws':epoch*len(tx),
          'stochastic_weighted_train_bce':total/len(tx),'clean_unweighted_train_bce':bce(train_logits,ty.numpy()),
          'clean_unweighted_validation_bce':bce(val_logits,vy.numpy()),
          'gradient_clipping_fraction':clipped/steps,'mean_preclip_gradient_norm':grad_norm_sum/steps})
    semantic=checks(model,vx,ef)
    bundle=folder/'bundle';bundle.mkdir(exist_ok=True)
    write(bundle/'config.json',{'model':configuration(variant)})
    np.savez(bundle/'weights.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    np.savez(bundle/'preprocessing.npz',**prep)
    np.save(folder/'validation-logits.npy',val_logits,allow_pickle=False)
    np.save(folder/'train-logits.npy',train_logits,allow_pickle=False)
    write(folder/'history.json',history)
    training_seconds=time.perf_counter()-tick
    score=official(folder,bank,val_logits.argmax(1))
    result={'identity':identity,'group':group,'model':model_name,'variant':variant,'scheme':scheme,'seed':seed,
       'epochs':protocol['epochs'],'optimizer_steps':history[-1]['steps'],'training_draws':history[-1]['draws'],
       'training_wall_seconds':training_seconds,'official':score,'checks':semantic,
       'parameters':sum(p.numel() for p in model.parameters()),'history_final':history[-1],
       'run_dir':str(folder.relative_to(HERE)),
       'bundle_hashes':{p.name:sha(p) for p in bundle.iterdir()}}
    write(folder/'result.json',result)
    print(f'DONE {group} seed={seed}: {score["correct"]}/798, {training_seconds:.1f}s',flush=True)
    return result


def main():
    output=HERE/'results';output.mkdir(exist_ok=True)
    p=read(HERE/'protocol.json');sources=source_hashes()
    if (output/'training_sources.json').exists():assert read(output/'training_sources.json')==sources
    write(output/'training_sources.json',sources);protected(output)
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    except (ImportError,OSError):pass
    write(output/'environment.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
       'cpu':cpu,'platform':platform.platform(),'executable':sys.executable,'device':'cpu',
       'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
       'started_local':time.strftime('%Y-%m-%d %H:%M:%S')})
    train,ta=load_inputs('train');val,va=load_inputs('validation')
    ty=labels('train',train);vy=labels('validation',val)
    weights,train_groups,weight_summary=make_weights(train,ty,p['hard_to_other_mixed_weight_ratio'])
    val_groups=groups(val,vy)
    write(output/'weight_summary.json',weight_summary)
    np.savez(output/'training_weights.npz',**weights)
    np.savez(output/'train-groups.npz',**train_groups);np.savez(output/'validation-groups.npz',**val_groups)
    np.save(output/'train-labels.npy',ty,allow_pickle=False);np.save(output/'validation-labels.npy',vy,allow_pickle=False)
    write(output/'data_summary.json',{'train':len(train),'validation':len(val),'validation_subgroups':{k:int(v.sum()) for k,v in val_groups.items()},
       'test_used':False,'no_validation_fitted_weights':True})
    mean,std=fit_standardization(ta['embeddings']);raw_f=extract(train);vraw_f=extract(val)
    fm,fs=fit_standardization(raw_f)
    prep={'mean':mean,'std':std,'feature_mean':fm,'feature_std':fs}
    data=(torch.from_numpy((ta['embeddings']-mean)/std),torch.from_numpy((raw_f-fm)/fs),torch.from_numpy(ty),
          torch.from_numpy((va['embeddings']-mean)/std),torch.from_numpy((vraw_f-fm)/fs),torch.from_numpy(vy))
    all_runs=[]
    for model_name,variant in p['models'].items():
        for scheme in p['weight_schemes']:
            for seed in p['seeds']:
                all_runs.append(train_one(model_name,variant,scheme,seed,p,data,prep,weights[scheme],val,output,sources))
                write(output/'training_runs.json',all_runs)
    protected(output)
    print('COMPLETE 30 fixed-budget fits',flush=True)


if __name__=='__main__':main()
