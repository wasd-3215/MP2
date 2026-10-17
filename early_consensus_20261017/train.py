"""Forty fixed-budget fits; input-only pool selection and final checkpoints."""
import runtime
from support import HERE,CODE,OUT,SEEDS,VARIANTS,read,write,sha,source_hashes,protected
from method import infer,sigmoid,layout,decide
from models import Scorer
from sampling import CyclicSampler
from features import extract,fit_standardization
from common import load_inputs,read_jsonl,parse_answer,DATA,check_data
import hashlib,platform,random,sys,time
import numpy as np
import torch

def labels(split,bank):
    assert split in ('train','validation');check_data(DATA,[split])
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl');assert len(refs)==len(bank)
    result=[]
    for row,ref in zip(bank,refs):
        assert row['id']==ref['id'] and row['question']==ref['question']
        answer=parse_answer(ref['answer']);assert answer is not None
        result.append([parse_answer(c['text'])==answer for c in row['candidates']])
    y=np.asarray(result,dtype=np.float32);assert (y.sum(axis=1)>0).all()
    return y
def state_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
def metrics(z,y,bank,hard):
    scores=sigmoid(z);raw=scores.argmax(axis=1);selected=decide(bank,scores)
    losses=np.mean(np.logaddexp(0,z.astype(np.float64))-y*z,axis=1)
    return {'bce_all':float(losses.mean()),'bce_disagreement':float(losses[hard].mean()),
        'accuracy':float(100*y[np.arange(len(y)),selected].mean()),
        'accuracy_disagreement':float(100*y[np.arange(len(y)),selected][hard].mean()),
        'neural_raw_accuracy':float(100*y[np.arange(len(y)),raw].mean())}
def main():
    OUT.mkdir(exist_ok=True);assert read(OUT/'implementation_checks.json')['passed']
    protocol=read(HERE/'protocol.json');sources=source_hashes()
    if (OUT/'training_sources.json').exists():assert read(OUT/'training_sources.json')==sources
    write(OUT/'training_sources.json',sources);protected()
    cpu=platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    except (ImportError,OSError):pass
    write(OUT/'environment.json',{'cpu':cpu,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'numpy':np.__version__,'torch':torch.__version__,'torch_threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),'openblas_threads':1,'device':'cpu','serial_jobs':True})
    train,ta=load_inputs('train');validation,va=load_inputs('validation')
    y,vy=labels('train',train),labels('validation',validation)
    groups=layout(train)[2];vgroups=layout(validation)[2]
    hard=groups>=2;vhard=vgroups>=2
    assert int(hard.sum())==2114 and int(vhard.sum())==355
    rf,vf=extract(train),extract(validation)
    mean,std=fit_standardization(ta['embeddings']);fm,fs=fit_standardization(rf)
    prep={'mean':mean,'std':std,'feature_mean':fm,'feature_std':fs}
    x=torch.from_numpy((ta['embeddings']-mean)/std);f=torch.from_numpy((rf-fm)/fs)
    vx=torch.from_numpy((va['embeddings']-mean)/std);vfeatures=torch.from_numpy((vf-fm)/fs)
    target=torch.from_numpy(y)
    np.save(OUT/'train-labels.npy',y,allow_pickle=False);np.save(OUT/'validation-labels.npy',vy,allow_pickle=False)
    np.save(OUT/'train-disagreement.npy',hard,allow_pickle=False)
    write(OUT/'data_summary.json',{'train':len(train),'validation':len(validation),'train_D':int(hard.sum()),
        'validation_D':int(vhard.sum()),'train_easy':int((groups==1).sum()),'validation_easy':int((vgroups==1).sum()),
        'pool_selection_references_used':False,'standardization_fit_split':'all training inputs, shared across scopes','test_used':False})
    results=[]
    for variant in VARIANTS:
        scope,feature,position=variant.split('_');pool=np.flatnonzero(hard) if scope=='D' else np.arange(len(train))
        for seed in SEEDS:
            folder=OUT/'runs'/variant/f'seed_{seed}';identity={'variant':variant,'seed':seed,'sources':sources}
            if (folder/'result.json').exists():
                result=read(folder/'result.json');assert result['identity']==identity
                assert all(sha(folder/'bundle'/k)==v for k,v in result['bundle_hashes'].items())
                results.append(result);print('RESUME',variant,seed,flush=True);continue
            folder.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=Scorer(feature,position);initial_digest=state_digest(model)
            torch.manual_seed(seed+20000)
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
            sampler=CyclicSampler(pool,seed+10000);order_hash=hashlib.sha256();exposures=np.zeros(len(train),dtype=np.int64)
            history=[];fit_seconds=diagnostic_seconds=0.;initial=None
            for epoch in range(protocol['budget_epochs']+1):
                stochastic=None
                if epoch:
                    tick=time.perf_counter();model.train();order=sampler.draw(4755)
                    order_hash.update(order.numpy().tobytes());np.add.at(exposures,order.numpy(),1)
                    acc=0.
                    for begin in range(0,4755,256):
                        ix=order[begin:begin+256];optimizer.zero_grad(set_to_none=True)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ix],f[ix]),target[ix])
                        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        optimizer.step();acc+=loss.item()*len(ix)
                    stochastic=acc/4755;fit_seconds+=time.perf_counter()-tick
                tick=time.perf_counter();rng=torch.get_rng_state().clone()
                tz,vz=infer(model,x,f),infer(model,vx,vfeatures);assert torch.equal(rng,torch.get_rng_state())
                row={'budget_epoch':epoch,'optimizer_steps':epoch*19,'draws':epoch*4755,
                    'eligible_pool_passes':epoch*4755/len(pool),'D_question_draws':int(exposures[hard].sum()),
                    'stochastic_train_bce':stochastic,'train':metrics(tz,y,train,hard),
                    'validation':metrics(vz,vy,validation,vhard)}
                history.append(row)
                if epoch==0:initial=vz.copy()
                diagnostic_seconds+=time.perf_counter()-tick
            assert int(exposures.sum())==95100 and np.isfinite(vz).all()
            if scope=='D':assert not exposures[~hard].any()
            if feature=='P':assert torch.count_nonzero(model.feature_projection.weight[:,1:]).item()==0
            bundle=folder/'bundle';bundle.mkdir(exist_ok=True)
            write(bundle/'config.json',{'model':{'feature_mode':feature,'position':position}})
            np.savez(bundle/'weights.npz',**{k:v.detach().numpy() for k,v in model.state_dict().items()})
            np.savez(bundle/'preprocessing.npz',**prep)
            for name,value in [('validation-logits',vz),('train-logits',tz),('initial-validation-logits',initial),('draw-counts',exposures)]:np.save(folder/f'{name}.npy',value,allow_pickle=False)
            write(folder/'history.json',history)
            perm=[2,0,3,1];a=infer(model,vx[:13],vfeatures[:13]);b=infer(model,vx[:13,perm],vfeatures[:13,perm])
            error=float(np.max(np.abs(a[:,perm]-b)));batch=float(np.max(np.abs(a[:1]-infer(model,vx[:1],vfeatures[:1]))))
            assert error<3e-5 and batch<3e-5
            result={'identity':identity,'variant':variant,'scope':scope,'feature':feature,'position':position,'seed':seed,
                'budget_epochs':20,'optimizer_steps':380,'draws':95100,'eligible_questions':len(pool),
                'D_question_draws':int(exposures[hard].sum()),'eligible_pool_passes':95100/len(pool),
                'parameters':sum(p.numel() for p in model.parameters()),'initial_state_sha256':initial_digest,
                'sample_order_sha256':order_hash.hexdigest(),'fit_seconds':fit_seconds,'diagnostic_seconds':diagnostic_seconds,
                'total_training_and_diagnostics_seconds':time.perf_counter()-start,'final_metrics':history[-1],
                'run_dir':str(folder.relative_to(HERE)),'permutation_max_error':error,'single_batch_max_error':batch,
                'bundle_hashes':{p.name:sha(p) for p in bundle.iterdir()}}
            write(folder/'result.json',result);results.append(result);write(OUT/'training_runs.json',results)
            print(f'{len(results)}/40 {variant} seed{seed}: all={row["validation"]["accuracy"]:.3f} D={row["validation"]["accuracy_disagreement"]:.3f} {result["total_training_and_diagnostics_seconds"]:.1f}s',flush=True)
    assert len(results)==40
    for seed in SEEDS:
        rs=[r for r in results if r['seed']==seed]
        assert len({r['initial_state_sha256'] for r in rs})==1
        arrays=[np.load(HERE/r['run_dir']/'initial-validation-logits.npy') for r in rs]
        assert all(np.array_equal(arrays[0],v) for v in arrays)
        for scope in ('All','D'):assert len({r['sample_order_sha256'] for r in rs if r['scope']==scope})==1
    assert source_hashes()==sources
    write(OUT/'training_runs.json',results)
    write(OUT/'training_verification.json',{'passed':True,'runs':40,'all_eight_cells_initial_state_and_function_identical_per_seed':True,
        'within_scope_exact_common_draw_order':True,'last_budget_only':True,'updates_each':380,'draws_each':95100,
        'D_draws_from_easy_questions':0,'eval_diagnostics_preserve_rng':True,'test_used':False})
    print('ALL 40 FIXED-BUDGET RUNS COMPLETE',flush=True)
if __name__=='__main__':main()
