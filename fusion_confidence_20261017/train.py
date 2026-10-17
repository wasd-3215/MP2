"""Fixed-epoch full-data and genuinely held-out, question-level OOF training."""
import runtime
runtime.configure_torch()
from utilities import HERE,CODE,write,read,sha,protected,official,source_hashes
import math,platform,random,sys,time
import numpy as np
import torch
from common import load_inputs,read_jsonl,parse_answer,DATA,check_data
from explicit_features import FEATURES,extract,fit_standardization
from scorers import Scorer,configuration


def labels(split,bank):
    assert split in ('train','validation')
    check_data(DATA,[split])
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert len(refs)==len(bank)
    out=[]
    for r,g in zip(bank,refs):
        assert r['id']==g['id'] and r['question']==g['question']
        a=parse_answer(g['answer']);assert a is not None
        out.append([int(parse_answer(c['text'])==a) for c in r['candidates']])
    y=np.asarray(out,dtype=np.float32)
    assert (y.sum(1)>0).all()
    return y


def infer(model,x,f):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x[i:i+256],f[i:i+256]) for i in range(0,len(x),256)]).numpy()


def checks(model,x,f):
    order=[2,0,3,1]
    a=infer(model,x[:13],f[:13])
    b=infer(model,x[:13,order],f[:13,order])
    error=float(np.max(np.abs(a[:,order]-b)))
    batch=float(np.max(np.abs(a[:1]-infer(model,x[:1],f[:1]))))
    assert error<3e-5 and batch<3e-5,(error,batch)
    return {'candidate_permutation_max_error':error,'single_batch_max_error':batch}


def train_one(variant,seed,fold,indices,held,raw_x,raw_f,y,val_x,val_f,bank,protocol,output,sources):
    role='full' if fold is None else f'fold_{fold}'
    folder=output/'runs'/variant/f'seed_{seed}'/role
    identity={'variant':variant,'seed':seed,'role':role,'sources':sources}
    if (folder/'result.json').exists():
        result=read(folder/'result.json');assert result['identity']==identity
        for name,digest in result['bundle_hashes'].items():assert sha(folder/'bundle'/name)==digest
        print('RESUME',variant,seed,role,flush=True)
        return result
    start=time.perf_counter();folder.mkdir(parents=True,exist_ok=True)
    mode=configuration(variant)['feature_mode'];cols=FEATURES[mode]
    mean,std=fit_standardization(raw_x[indices])
    fm,fs=fit_standardization(raw_f[indices])
    prep={'mean':mean,'std':std,'feature_mean':fm,'feature_std':fs}
    def tensors(x,f):
        return torch.from_numpy((x-mean)/std),torch.from_numpy(((f-fm)/fs)[...,cols])
    x,f=tensors(raw_x[indices],raw_f[indices]);target=torch.from_numpy(y[indices])
    if fold is None:ex,ef=tensors(val_x,val_f)
    else:ex,ef=tensors(raw_x[held],raw_f[held])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=Scorer(**configuration(variant))
    # Reset dropout RNG after model creation so feature-head construction cannot alter masks.
    torch.manual_seed(seed+20000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    generator=torch.Generator().manual_seed(seed+10000)
    history=[]
    for epoch in range(1,protocol['epochs']+1):
        model.train();order=torch.randperm(len(x),generator=generator);acc=0.
        for begin in range(0,len(order),protocol['batch_size']):
            ix=order[begin:begin+protocol['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ix],f[ix]),target[ix])
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip']);optimizer.step()
            acc+=loss.item()*len(ix)
        history.append({'epoch':epoch,'stochastic_train_bce':acc/len(x),'steps':epoch*math.ceil(len(x)/protocol['batch_size'])})
    logits=infer(model,ex,ef)
    assert np.isfinite(logits).all()
    np.save(folder/'evaluation-logits.npy',logits,allow_pickle=False)
    np.save(folder/'held_indices.npy',held,allow_pickle=False)
    bundle=folder/'bundle';bundle.mkdir(exist_ok=True)
    write(bundle/'config.json',{'model':configuration(variant)})
    np.savez(bundle/'weights.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    np.savez(bundle/'preprocessing.npz',**prep)
    write(folder/'history.json',history)
    semantic=checks(model,ex,ef)
    result={'identity':identity,'variant':variant,'seed':seed,'fold':fold,'role':role,
            'epochs':protocol['epochs'],'optimizer_steps':history[-1]['steps'],'train_questions':len(indices),
            'held_questions':len(held),'parameters':sum(p.numel() for p in model.parameters()),
            'training_wall_seconds':time.perf_counter()-start,'checks':semantic,
            'run_dir':str(folder.relative_to(HERE)),
            'bundle_hashes':{p.name:sha(p) for p in bundle.iterdir()}}
    if fold is None:
        score=official(folder,bank,logits.argmax(1));result['official']=score
        tr=infer(model,x,f)
        def bce(z,t):return float(np.mean(np.logaddexp(0,z)-t*z))
        result['train_bce']=bce(tr,y[indices])
        result['validation_bce']=bce(logits,labels('validation',bank))
        result['train_selection_accuracy_diagnostic']=float(100*y[indices][np.arange(len(indices)),tr.argmax(1)].mean())
        print(f'DONE {variant} seed={seed} full {score["correct"]}/{score["n"]} {result["training_wall_seconds"]:.1f}s',flush=True)
    else:
        print(f'DONE {variant} seed={seed} {role} {result["training_wall_seconds"]:.1f}s',flush=True)
    write(folder/'result.json',result)
    return result


def main():
    output=HERE/'results';output.mkdir(exist_ok=True)
    p=read(HERE/'protocol.json');sources=source_hashes()
    if (output/'training_sources.json').exists():assert read(output/'training_sources.json')==sources
    write(output/'training_sources.json',sources);protected(output)
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:cpu=winreg.QueryValueEx(k,'ProcessorNameString')[0]
    except (ImportError,OSError):pass
    write(output/'environment.json',{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
       'torch':torch.__version__,'cpu':cpu,'platform':platform.platform(),'torch_threads':torch.get_num_threads(),
       'interop_threads':torch.get_num_interop_threads(),'device':'cpu','started_local':time.strftime('%Y-%m-%d %H:%M:%S')})
    train,ta=load_inputs('train');val,va=load_inputs('validation')
    x=ta['embeddings'];vx=va['embeddings'];f=extract(train);vf=extract(val)
    y=labels('train',train);vy=labels('validation',val)
    np.save(output/'train-labels.npy',y,allow_pickle=False);np.save(output/'validation-labels.npy',vy,allow_pickle=False)
    rng=np.random.default_rng(p['fold_seed']);order=rng.permutation(len(train))
    folds=np.empty(len(train),dtype=np.int64)
    for i,ix in enumerate(np.array_split(order,p['folds'])):folds[ix]=i
    np.save(output/'question-folds.npy',folds,allow_pickle=False)
    write(output/'folds.json',{'ids':[r['id'] for r in train],'folds':folds.tolist()})
    write(output/'data_summary.json',{'train':len(train),'validation':len(val),
      'train_correct_histogram':np.bincount(y.sum(1).astype(int),minlength=5).tolist(),
      'validation_correct_histogram':np.bincount(vy.sum(1).astype(int),minlength=5).tolist(),
      'test_read':False,'feature_names':__import__('explicit_features').NAMES})
    all_runs=[]
    for variant in p['full_variants']:
        for seed in p['seeds']:
            all_runs.append(train_one(variant,seed,None,np.arange(len(train)),np.arange(len(val)),x,f,y,vx,vf,val,p,output,sources))
            write(output/'training_runs.json',all_runs)
    for variant in p['oof_variants']:
        arrays=[]
        for seed in p['seeds']:
            oof=np.full((len(train),4),np.nan,dtype=np.float32)
            for fold in range(p['folds']):
                held=np.flatnonzero(folds==fold);indices=np.flatnonzero(folds!=fold)
                assert not np.intersect1d(indices,held).size
                r=train_one(variant,seed,fold,indices,held,x,f,y,vx,vf,val,p,output,sources)
                all_runs.append(r);write(output/'training_runs.json',all_runs)
                run=HERE/r['run_dir'];assert np.array_equal(np.load(run/'held_indices.npy'),held)
                oof[held]=np.load(run/'evaluation-logits.npy')
            assert np.isfinite(oof).all();arrays.append(oof)
        np.save(output/f'oof-{variant}.npy',np.stack(arrays),allow_pickle=False)
    protected(output)
    print('COMPLETE',len(all_runs),'training runs',flush=True)


if __name__=='__main__':main()
