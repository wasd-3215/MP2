"""Frozen ablations, train-OOF threshold selection, official validation and reporting."""
import runtime
from utilities import HERE,CODE,read,write,sha,protected,official,asset_bytes,source_hashes
from pathlib import Path
import csv,os,shutil,sys,time
import numpy as np
from common import load_bank
from explicit_features import answer_structure,FEATURES
from scorers import configuration
from decisions import sigmoid,aggregate,confidence,route,decide

OUT=HERE/'results'
P=read(HERE/'protocol.json')


def dump_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def correctness(y,choice):return y[np.arange(len(y)),choice].astype(bool)


def interval(delta):
    rng=np.random.default_rng(P['bootstrap']['seed']);means=[]
    for start in range(0,P['bootstrap']['samples'],250):
        indices=rng.integers(0,len(delta),(min(250,P['bootstrap']['samples']-start),len(delta)))
        means.extend(np.mean(delta[indices],axis=1))
    return [float(v) for v in np.quantile(means,[.025,.975])*100]


def calibration_fit(c,target):
    mean=float(np.mean(c));std=max(float(np.std(c)),1e-4)
    x=np.stack([np.ones(len(c)),(c-mean)/std],1)
    w=np.array([np.log((target.mean()+1e-4)/(1-target.mean()+1e-4)),0.])
    penalty=np.diag([1e-8,.01])
    for _ in range(100):
        prob=sigmoid(x@w);gradient=x.T@(prob-target)+penalty@w
        hessian=x.T@((prob*(1-prob))[:,None]*x)+penalty
        step=np.linalg.solve(hessian,gradient);w-=step
        if np.max(np.abs(step))<1e-8:break
    return {'mean':mean,'std':std,'intercept':float(w[0]),'slope':float(w[1]),'fit':'train OOF selected-candidate correctness; no validation fitting'}


def calibration_apply(c,fit):return sigmoid(fit['intercept']+fit['slope']*(c-fit['mean'])/fit['std'])


def calibration_metrics(prob,target):
    bins=[];ece=0.
    for i in range(10):
        mask=(prob>=i/10)&((prob<(i+1)/10) if i<9 else (prob<=1))
        if not mask.any():continue
        m=float(prob[mask].mean());a=float(target[mask].mean())
        bins.append({'low':i/10,'high':(i+1)/10,'n':int(mask.sum()),'mean_probability':m,'accuracy_diagnostic':a})
        ece+=float(mask.mean())*abs(m-a)
    return {'brier':float(np.mean((prob-target)**2)),'ece_10_equal_width':ece,'bins':bins}


def main():
    assert read(OUT/'training_sources.json')==source_hashes()
    runs=read(OUT/'training_runs.json');assert len(runs)==140
    protected(OUT)
    train=load_bank('train');val=load_bank('validation')
    y=np.load(OUT/'validation-labels.npy');ty=np.load(OUT/'train-labels.npy')
    answers,support,unique=answer_structure(val)
    full={};members={};records=[];preds={};probabilities={};eligible={}
    full_runs=[r for r in runs if r['fold'] is None]
    for variant in P['full_variants']:
        selected=[r for r in full_runs if r['variant']==variant]
        assert [r['seed'] for r in selected]==P['seeds']
        full[variant]=np.stack([np.load(HERE/r['run_dir']/'evaluation-logits.npy') for r in selected])
        members[variant]=[HERE/r['run_dir']/'bundle' for r in selected]
        for i,r in enumerate(selected):
            name=f'{variant}_seed{r["seed"]}'
            choice=full[variant][i].argmax(1);preds[name]=choice
            probabilities[name]=sigmoid(full[variant][i:i+1])
            eligible[name]=r['seed']==17
            records.append({'name':name,'group':'single','variant':variant,'members':1,
              'correct':r['official']['correct'],'accuracy':r['official']['accuracy'],
              'eligible':eligible[name],'feature_dimensions':len(FEATURES[configuration(variant)['feature_mode']]),
              'asset_bytes':asset_bytes(members[variant][i]),'base':None,'rule':'mean_probability','routing':None,
              'member_paths':[str(members[variant][i])], 'official_score_path':str(HERE/r['run_dir']/'validation-score.json')})

    def add_pipeline(name,group,mpaths,probs,rule='mean_probability',routing=None,base=None,can_select=True):
        folder=OUT/'pipelines'/name;folder.mkdir(parents=True,exist_ok=True)
        manifest={'members':[os.path.relpath(p,folder) for p in mpaths],'rule':rule,'routing':routing}
        write(folder/'pipeline.json',manifest)
        choice=decide(probs,val,rule,routing)
        score=official(folder,val,choice)
        size=sum(asset_bytes(p) for p in mpaths)+(folder/'pipeline.json').stat().st_size
        rec={'name':name,'group':group,'variant':None,'members':len(mpaths),
             'correct':score['correct'],'accuracy':score['accuracy'],'eligible':can_select,
             'feature_dimensions':sum(len(FEATURES[read(p/'config.json')['model']['feature_mode']]) for p in mpaths),
             'asset_bytes':size,'base':base,'rule':rule,'routing':routing,
             'member_paths':[str(p) for p in mpaths],'official_score_path':str(folder/'validation-score.json')}
        if base:
            before=correctness(y,preds[base]);after=correctness(y,choice)
            rec['repairs']=int((~before&after).sum());rec['harms']=int((before&~after).sum())
            rec['answer_changes']=sum(answers[q][int(choice[q])]!=answers[q][int(preds[base][q])] for q in range(len(val)))
        records.append(rec);preds[name]=choice;probabilities[name]=probs;eligible[name]=can_select
        print(f'SCORED {name}: {score["correct"]}/798',flush=True)
        return rec

    ensemble_data={};oof_data={};paths={}
    for name,variants in P['ensemble_definitions'].items():
        mpaths=[path for v in variants for path in members[v]]
        probs=sigmoid(np.concatenate([full[v] for v in variants]))
        oof=sigmoid(np.concatenate([np.load(OUT/f'oof-{v}.npy') for v in variants]))
        ensemble_data[name]=probs;oof_data[name]=oof;paths[name]=mpaths
        for rule in P['ensemble_rules']:
            suffix='' if rule=='mean_probability' else '_answer_vote'
            add_pipeline(name+suffix,'ensemble',mpaths,probs,rule,base='s2_seed17')
    oof_data['s2_seed17']=sigmoid(np.load(OUT/'oof-s2.npy')[:1])
    ensemble_data['s2_seed17']=probabilities['s2_seed17'];paths['s2_seed17']=members['s2'][:1]

    # Official, deterministic generated-answer vote baseline; ties choose first candidate.
    majority=support.argmax(1)
    vote_score=official(OUT/'answer_majority',val,majority)
    vote_correct=correctness(y,majority)
    complementarity=[]
    for name in ['s2_seed17','s2_5','heterogeneous_15','consistency_5']:
        c=correctness(y,preds[name])
        complementarity.append({'model':name,'model_correct':int(c.sum()),'vote_correct':int(vote_correct.sum()),
          'vote_only_correct':int((~c&vote_correct).sum()),'model_only_correct':int((c&~vote_correct).sum()),
          'both_wrong':int((~c&~vote_correct).sum()),'oracle_either_correct_diagnostic':int((c|vote_correct).sum())})
    for a,b in [('s2_seed17','s2_5'),('s2_5','heterogeneous_15'),('s2_5','consistency_5')]:
        ca,cb=correctness(y,preds[a]),correctness(y,preds[b])
        complementarity.append({'model':a+' versus '+b,'model_correct':int(ca.sum()),'vote_correct':int(cb.sum()),
          'vote_only_correct':int((~ca&cb).sum()),'model_only_correct':int((ca&~cb).sum()),
          'both_wrong':int((~ca&~cb).sum()),'oracle_either_correct_diagnostic':int((ca|cb).sum())})
    write(OUT/'complementarity.json',complementarity)

    route_fits=[];calibrations=[];confidence_bins=[];candidate_search=[]
    for base in P['routing_bases']:
        tprob=oof_data[base];vprob=ensemble_data[base]
        ts,tc=aggregate(tprob,train);vs,vc=aggregate(vprob,val)
        target=correctness(ty,tc);vtarget=correctness(y,vc)
        for signal in P['confidence_signals']:
            tconf=confidence(ts,tc,tprob,train,signal);vconf=confidence(vs,vc,vprob,val,signal)
            fit=calibration_fit(tconf,target.astype(float));calibrated=calibration_apply(vconf,fit)
            mixed=y.sum(1)<4
            calibrations.append({'base':base,'signal':signal,'fit':fit,
              'validation':calibration_metrics(calibrated,vtarget),
              'validation_mixed':calibration_metrics(calibrated[mixed],vtarget[mixed])})
            # Bin boundaries determined on OOF train, never validation labels.
            edges=np.unique(np.quantile(tconf,[0,.1,.25,.5,.75,.9,1]))
            cuts=edges[1:-1]
            bins=np.searchsorted(cuts,vconf,side='right')
            for bi in np.unique(bins):
                m=bins==bi
                confidence_bins.append({'base':base,'signal':signal,'bin':int(bi),'n':int(m.sum()),
                  'mean_signal':float(vconf[m].mean()),'model_correct':int(vtarget[m].sum()),
                  'vote_correct':int(vote_correct[m].sum()),
                  'vote_repairs':int((~vtarget&vote_correct&m).sum()),
                  'vote_harms':int((vtarget&~vote_correct&m).sum())})
            for minimum in P['vote_min_support']:
                thresholds=[None]+[float(t) for t in np.unique(np.quantile(tconf,P['threshold_quantiles']))]
                candidates=[]
                for threshold in thresholds:
                    choice,changed=route(ts,tc,tconf,train,threshold,minimum)
                    corr=correctness(ty,choice)
                    candidates.append({'threshold':threshold,'correct_diagnostic':int(corr.sum()),
                      'answer_changes':int(changed.sum()),'repairs':int((~target&corr).sum()),
                      'harms':int((target&~corr).sum())})
                best=max(candidates,key=lambda r:(r['correct_diagnostic'],-r['answer_changes'],
                    -float('inf') if r['threshold'] is None else -r['threshold']))
                # Prefer disabled if it has the same correctness and same number of changes.
                disabled=candidates[0]
                if (best['correct_diagnostic'],best['answer_changes'])==(disabled['correct_diagnostic'],0):best=disabled
                routing={'signal':signal,'threshold':best['threshold'],'min_support':minimum}
                name=f'{base}_route_{signal}_support{minimum}'
                rec=add_pipeline(name,'routing',paths[base],vprob,routing=routing,base=base)
                _,changed=route(vs,vc,vconf,val,best['threshold'],minimum)
                rec['fallback_fraction']=float(changed.mean())
                route_fits.append({'name':name,'base':base,'routing':routing,'oof_selected':best,
                  'oof_base_correct_diagnostic':int(target.sum()),'n_oof':len(train),
                  'selection_labels':'train OOF only','validation_fallback_fraction':float(changed.mean())})
                candidate_search.append({'name':name,'candidates':candidates})
        # Unconditional clear-vote controls separate vote conditions from confidence filtering.
        for minimum in P['vote_min_support']:
            add_pipeline(f'{base}_always_support{minimum}','unconditional_vote',paths[base],vprob,
                routing={'signal':'answer_gap','threshold':2.,'min_support':minimum},base=base,can_select=False)
    write(OUT/'routing_fits.json',route_fits);write(OUT/'routing_search.json',candidate_search)
    write(OUT/'calibration.json',calibrations);write(OUT/'confidence_bins.json',confidence_bins)
    dump_csv(OUT/'confidence_bins.csv',confidence_bins)

    # Five-seed feature summaries and paired question bootstrap (seeds not independent questions).
    feature_rows=[];paired=[]
    for variant in P['full_variants']:
        these=[r for r in full_runs if r['variant']==variant]
        acc=np.asarray([r['official']['accuracy'] for r in these])
        row={'variant':variant,'mean_accuracy':float(acc.mean()),'sd_accuracy':float(acc.std(ddof=1)),
          'mean_train_bce':float(np.mean([r['train_bce'] for r in these])),
          'mean_validation_bce':float(np.mean([r['validation_bce'] for r in these])),
          'parameters':these[0]['parameters'],'feature_dimensions':len(FEATURES[configuration(variant)['feature_mode']]),
          'seed_accuracies':acc.tolist()}
        feature_rows.append(row)
    def compare_features(a,b):
        ac=np.stack([correctness(y,z.argmax(1)) for z in full[a]])
        bc=np.stack([correctness(y,z.argmax(1)) for z in full[b]])
        d=ac.astype(float)-bc
        paired.append({'a':a,'b':b,'kind':'five_seed_mean','delta_pp':float(d.mean()*100),
          'ci95_pp':interval(d.mean(0)),'seed_deltas_pp':(100*d.mean(1)).tolist(),
          'repairs_mean':float(np.mean((ac&~bc).sum(1))),'harms_mean':float(np.mean((~ac&bc).sum(1)))})
    for a,b in [('s0','s2'),('pool','s2'),('s2_parse','s2'),('s2_consistency','s2_parse'),
                ('s2_consistency','s2'),('s2_style','s2_parse'),('s2_all','s2_consistency'),('features_only','s2')]:compare_features(a,b)
    comparisons=[('s2_5','s2_seed17'),('heterogeneous_15','s2_5'),('consistency_5','s2_5')]
    comparisons += [(r['name'],r['base']) for r in records if r['group']=='routing']
    for a,b in comparisons:
        d=correctness(y,preds[a]).astype(float)-correctness(y,preds[b])
        paired.append({'a':a,'b':b,'kind':'one_frozen_pipeline','delta_pp':float(d.mean()*100),'ci95_pp':interval(d)})
    write(OUT/'feature_summary.json',feature_rows)
    dump_csv(OUT/'feature_summary.csv',[{**r,'seed_accuracies':str(r['seed_accuracies'])} for r in feature_rows])
    write(OUT/'paired_comparisons.json',paired)

    subgroup_rows=[]
    masks={'one_correct':y.sum(1)==1,'mixed':y.sum(1)<4,'all_four_correct':y.sum(1)==4,
      'multiple_valid_answer_groups':np.asarray([len({v for v in a if v is not None})>=2 for a in answers]),
      'unique_plurality_correct':unique&vote_correct,'unique_plurality_wrong':unique&~vote_correct,
      'plurality_tie':~unique,'any_invalid_parse':np.asarray([None in a for a in answers])}
    for r in records:
        c=correctness(y,preds[r['name']])
        for group,mask in masks.items():
            subgroup_rows.append({'name':r['name'],'subgroup':group,'n':int(mask.sum()),
              'correct_diagnostic':int(c[mask].sum()),'accuracy_diagnostic':float(100*c[mask].mean()) if mask.any() else None})
    dump_csv(OUT/'subgroups.csv',subgroup_rows)
    np.savez(OUT/'all-validation-choices.npz',**{k:v for k,v in preds.items()})
    write(OUT/'pipelines.json',records)
    dump_csv(OUT/'pipelines.csv',[{k:v for k,v in r.items() if k in ('name','group','members','correct','accuracy','eligible','asset_bytes')} for r in records])

    candidates=[r for r in records if r['eligible'] and r['asset_bytes']<=16*1024**2]
    winner=min(candidates,key=lambda r:(-r['correct'],r['members'],r['feature_dimensions'],r['name']))
    bundle=OUT/'selected_bundle';bundle.mkdir(exist_ok=True)
    for i,path in enumerate(winner['member_paths']):
        destination=bundle/'members'/f'{i:02d}';destination.mkdir(parents=True,exist_ok=True)
        for file in Path(path).iterdir():shutil.copyfile(file,destination/file.name)
    write(bundle/'pipeline.json',{'members':[f'members/{i:02d}' for i in range(winner['members'])],
       'rule':winner['rule'],'routing':winner['routing']})
    selection={**winner,'validation_accuracy':winner['accuracy'],'selection_protocol':P['final_selection'],
      'portable_asset_bytes':asset_bytes(bundle),'bundle_sha256':{str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}}
    assert selection['portable_asset_bytes']<=16*1024**2
    write(OUT/'selection.json',selection)
    write(OUT/'cost.json',{'full_runs':40,'oof_runs':100,'epochs_per_run':20,
      'optimizer_steps':sum(r['optimizer_steps'] for r in runs),
      'training_wall_seconds_sum':sum(r['training_wall_seconds'] for r in runs),
      'all_fits_accounted':len(runs),'official_validation_pipelines':len(records)+1,
      'threshold_candidates_total':sum(len(r['candidates']) for r in candidate_search),
      'calibration_models':len(calibrations),'test_used':False})
    protected(OUT)
    print('SELECTED',winner['name'],winner['accuracy'],flush=True)


if __name__=='__main__':main()
