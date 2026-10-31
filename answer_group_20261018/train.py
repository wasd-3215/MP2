"""Thirty final-checkpoint fits; no validation or test inputs opened here."""
import runtime
from support import HERE,OUT,SEEDS,VARIANTS,read,write,sources,sha
from method import infer
from models import Scorer,objective
from grouping import layout,group_targets,choices_from_probabilities
from sampling import CyclicSampler
from features import extract,fit_standardization
from common import load_inputs,read_jsonl,parse_answer,DATA,check_data
import hashlib,platform,random,sys,time
import numpy as np
import torch

def labels(split,bank):
    assert split in ('train','validation')
    check_data(DATA,[split]);refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert len(refs)==len(bank)
    result=[]
    for row,ref in zip(bank,refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        a=parse_answer(ref['answer']);assert a is not None
        result.append([parse_answer(c['text'])==a for c in row['candidates']])
    y=np.asarray(result,np.float32);assert (y.sum(1)>0).all()
    return y

def digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def main():
    assert read(OUT/'implementation_checks.json')['passed']
    frozen=sources()
    if (OUT/'training_sources.json').exists():assert read(OUT/'training_sources.json')==frozen
    write(OUT/'training_sources.json',frozen)
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:cpu=winreg.QueryValueEx(k,'ProcessorNameString')[0]
    except (ImportError,OSError):pass
    write(OUT/'environment.json',{'cpu':cpu,'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
          'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
          'device':'cpu','serial_jobs':True,'validation_read_during_training':False,'test_used':False})
    bank,arrays=load_inputs('train');y=labels('train',bank)
    _,membership,valid,counts=layout(bank);pool=np.flatnonzero(counts>=2)
    assert len(bank)==4755 and len(pool)==2114
    target=group_targets(membership,y)
    raw=extract(bank);mean,std=fit_standardization(arrays['embeddings']);fm,fs=fit_standardization(raw)
    prep={'mean':mean,'std':std,'feature_mean':fm,'feature_std':fs}
    x=torch.from_numpy((arrays['embeddings']-mean)/std);f=torch.from_numpy((raw-fm)/fs)
    m=torch.from_numpy(membership);t=torch.from_numpy(target);yy=torch.from_numpy(y)
    np.savez(OUT/'training_layout.npz',counts=counts,pool=pool,labels=y,group_targets=target)
    write(OUT/'training_inputs.json',{'n':len(bank),'active':len(pool),'one_valid_group':int((counts==1).sum()),
          'zero_valid_groups':int((counts==0).sum()),'scope_selection_references_used':False,'standardization_inputs':'all training'})
    results=[]
    for variant in VARIANTS:
        representation,size_mode=variant.split('_')
        for seed in SEEDS:
            folder=OUT/'runs'/variant/f'seed_{seed}';identity={'variant':variant,'seed':seed,'sources':frozen}
            if (folder/'result.json').exists():
                result=read(folder/'result.json');assert result['identity']==identity
                assert all(sha(folder/'bundle'/k)==h for k,h in result['bundle_hashes'].items())
                results.append(result);print('RESUME',variant,seed,flush=True);continue
            folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=Scorer(representation,size_mode);initial=digest(model)
            torch.manual_seed(seed+20000)
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
            sampler=CyclicSampler(pool,seed+10000);order_digest=hashlib.sha256();exposures=np.zeros(len(bank),np.int64)
            history=[]
            for epoch in range(1,21):
                model.train();order=sampler.draw(4755);order_digest.update(order.numpy().tobytes())
                np.add.at(exposures,order.numpy(),1);loss_sum=0.
                for begin in range(0,4755,256):
                    ix=order[begin:begin+256];opt.zero_grad(set_to_none=True)
                    loss=objective(model,model(x[ix],f[ix],m[ix]),yy[ix],m[ix],t[ix])
                    assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    opt.step();loss_sum+=loss.item()*len(ix)
                history.append({'budget_epoch':epoch,'optimizer_steps':epoch*19,'draws':epoch*4755,'stochastic_loss':loss_sum/4755})
            assert exposures.sum()==95100 and exposures[counts<2].sum()==0
            fit_seconds=time.perf_counter()-start
            z=infer(model,x[pool],f[pool],m[pool]);choice=choices_from_probabilities(membership[pool],z)
            train_accuracy=float(100*y[pool,choice].mean())
            permutation=torch.tensor([2,0,3,1]);sample=torch.from_numpy(pool[:17])
            with torch.inference_mode():
                direct=model(x[sample],f[sample],m[sample])[1]
                permuted=model(x[sample][:,permutation],f[sample][:,permutation],m[sample][:,:,permutation])[1]
            error=float((direct-permuted).abs().max());assert error<1e-4
            inactive=5 if size_mode=='NoSize' else 4
            assert not model.base.head[0].weight[:,-inactive:].count_nonzero().item()
            bundle=folder/'bundle';bundle.mkdir(exist_ok=True)
            write(bundle/'config.json',{'model':{'representation':representation,'size_mode':size_mode},'seed':seed,'budget_epoch':20})
            np.savez(bundle/'weights.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
            np.savez(bundle/'preprocessing.npz',**prep);np.save(folder/'train_group_logits.npy',z,allow_pickle=False)
            write(folder/'history.json',history)
            result={'identity':identity,'initial_state_sha256':initial,'draw_order_sha256':order_digest.hexdigest(),
                    'draws':int(exposures.sum()),'updates':380,'pool_exposure_min':int(exposures[pool].min()),
                    'pool_exposure_max':int(exposures[pool].max()),'train_active_accuracy':train_accuracy,
                    'fit_seconds':fit_seconds,'parameters':sum(p.numel() for p in model.parameters()),
                    'permutation_max_error':error,'bundle_hashes':{p.name:sha(p) for p in bundle.iterdir()}}
            write(folder/'result.json',result);results.append(result)
            print('FIT',variant,seed,'train_active',round(train_accuracy,3),'seconds',round(fit_seconds,2),flush=True)
    for seed in SEEDS:
        same=[r for r in results if r['identity']['seed']==seed]
        assert len({r['initial_state_sha256'] for r in same})==1
        assert len({r['draw_order_sha256'] for r in same})==1
    assert sources()==frozen
    write(OUT/'training_summary.json',{'complete':True,'fits':len(results),'total_fit_seconds':sum(r['fit_seconds'] for r in results),
          'matched_initial_states_and_draw_orders':True,'validation_used':False,'test_used':False,'results':results})

if __name__=='__main__':main()
