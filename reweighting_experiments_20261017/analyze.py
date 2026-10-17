"""Predeclared paired comparisons and secondary five-member ensembles."""
import runtime
from utilities import HERE,CODE,read,write,sha,protected,official,asset_bytes,source_hashes
from pathlib import Path
import csv,os,shutil
import numpy as np
from common import load_bank
from decisions import sigmoid

OUT=HERE/'results'
P=read(HERE/'protocol.json')
SUBGROUPS=['all','wrong_unique_plurality','one_correct','correct_unique_plurality_mixed','plurality_tie']


def dump_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def interval(delta):
    rng=np.random.default_rng(P['bootstrap']['seed']);samples=[]
    for _ in range(P['bootstrap']['samples']//250):
        index=rng.integers(0,len(delta),(250,len(delta)))
        samples.extend(delta[index].mean(1))
    return [float(v) for v in 100*np.quantile(samples,[.025,.975])]


def correct(y,pred):return y[np.arange(len(y)),pred].astype(np.float64)


def main():
    runs=read(OUT/'training_runs.json');assert len(runs)==30
    assert read(OUT/'training_sources.json')==source_hashes()
    protected(OUT)
    bank=load_bank('validation');y=np.load(OUT/'validation-labels.npy');ty=np.load(OUT/'train-labels.npy')
    with np.load(OUT/'validation-groups.npz') as a:vg={k:a[k] for k in a.files}
    with np.load(OUT/'train-groups.npz') as a:tg={k:a[k] for k in a.files}
    logits={};preds={};members={};vc={};tc={};summary=[];subgroup_rows=[];records=[]
    for m in P['models']:
        for scheme in P['weight_schemes']:
            group=f'{m}_{scheme}';selected=[r for r in runs if r['group']==group]
            assert [r['seed'] for r in selected]==P['seeds']
            logits[group]=np.stack([np.load(HERE/r['run_dir']/'validation-logits.npy') for r in selected])
            train_logits=np.stack([np.load(HERE/r['run_dir']/'train-logits.npy') for r in selected])
            members[group]=[HERE/r['run_dir']/'bundle' for r in selected]
            vc[group]=np.stack([correct(y,z.argmax(1)) for z in logits[group]])
            tc[group]=np.stack([correct(ty,z.argmax(1)) for z in train_logits])
            a=np.asarray([r['official']['accuracy'] for r in selected])
            row={'group':group,'model':m,'scheme':scheme,'mean_accuracy':float(a.mean()),
              'sd_accuracy':float(a.std(ddof=1)),'seed_accuracies':a.tolist(),
              'parameters':selected[0]['parameters'],
              'train_bce':float(np.mean([r['history_final']['clean_unweighted_train_bce'] for r in selected])),
              'validation_bce':float(np.mean([r['history_final']['clean_unweighted_validation_bce'] for r in selected]))}
            for name,mask in vg.items():
                accuracies=100*vc[group][:,mask].mean(1)
                subgroup_rows.append({'group':group,'split':'validation','subgroup':name,'n':int(mask.sum()),
                  'mean_correct_diagnostic':float(vc[group][:,mask].sum(1).mean()),
                  'mean_accuracy_diagnostic':float(accuracies.mean()),'sd_accuracy':float(accuracies.std(ddof=1))})
                if name in SUBGROUPS:row[name+'_accuracy']=float(accuracies.mean())
            for name,mask in tg.items():
                accuracies=100*tc[group][:,mask].mean(1)
                subgroup_rows.append({'group':group,'split':'train','subgroup':name,'n':int(mask.sum()),
                  'mean_correct_diagnostic':float(tc[group][:,mask].sum(1).mean()),
                  'mean_accuracy_diagnostic':float(accuracies.mean()),'sd_accuracy':float(accuracies.std(ddof=1))})
            summary.append(row)
            for i,r in enumerate(selected):
                name=f'{group}_seed{r["seed"]}';pred=logits[group][i].argmax(1);preds[name]=pred
                records.append({'name':name,'group':group,'kind':'single','members':1,
                  'correct':r['official']['correct'],'accuracy':r['official']['accuracy'],
                  'member_paths':[str(members[group][i])],
                  'official_score_path':str(HERE/r['run_dir']/'validation-score.json'),
                  'asset_bytes':asset_bytes(members[group][i]),'eligible':r['seed']==17})
    write(OUT/'group_summary.json',summary)
    dump_csv(OUT/'group_summary.csv',[{**r,'seed_accuracies':str(r['seed_accuracies'])} for r in summary])
    dump_csv(OUT/'subgroup_summary.csv',subgroup_rows)
    write(OUT/'subgroup_summary.json',subgroup_rows)

    comparisons=[(P['primary_comparison']['a'],P['primary_comparison']['b'])]
    comparisons += [tuple(pair) for pair in P['secondary_comparisons']]
    paired=[];repairs=[]
    for a,b in comparisons:
        differences=vc[a]-vc[b]
        for subgroup in SUBGROUPS:
            mask=vg[subgroup];delta=differences[:,mask]
            r={'a':a,'b':b,'subgroup':subgroup,'n':int(mask.sum()),
              'primary':a=='C_H' and b=='C_K' and subgroup=='wrong_unique_plurality',
              'delta_pp':float(100*delta.mean()),'ci95_pp':interval(delta.mean(0)),
              'seed_deltas_pp':(100*delta.mean(1)).tolist(),
              'positive_seeds':int((delta.mean(1)>0).sum()),
              'repairs_mean':float(((vc[a][:,mask]>0)&(vc[b][:,mask]==0)).sum(1).mean()),
              'harms_mean':float(((vc[a][:,mask]==0)&(vc[b][:,mask]>0)).sum(1).mean())}
            paired.append(r)
            repairs.append({k:r[k] for k in ['a','b','subgroup','n','delta_pp','repairs_mean','harms_mean']})
    write(OUT/'paired_comparisons.json',paired);dump_csv(OUT/'repair_harm.csv',repairs)
    dump_csv(OUT/'paired_comparisons.csv',[{**r,'ci95_pp':str(r['ci95_pp']),'seed_deltas_pp':str(r['seed_deltas_pp'])} for r in paired])

    interactions=[]
    for reference in ['U','K']:
        delta=(vc['C_H']-vc['C_'+reference])-(vc['P_H']-vc['P_'+reference])
        for subgroup in SUBGROUPS:
            mask=vg[subgroup];d=delta[:,mask]
            interactions.append({'comparison':f'(C_H-C_{reference})-(P_H-P_{reference})',
              'subgroup':subgroup,'n':int(mask.sum()),'delta_pp':float(100*d.mean()),
              'ci95_pp':interval(d.mean(0)),'seed_deltas_pp':(100*d.mean(1)).tolist()})
    write(OUT/'interactions.json',interactions)
    practical={}
    for m in ['P','C']:
        d=vc[m+'_H']-vc[m+'_U'];hard_delta=float(100*d[:,vg['wrong_unique_plurality']].mean());overall_delta=float(100*d.mean())
        practical[m]={'reference':m+'_U','target':m+'_H','hard_delta_pp':hard_delta,'overall_delta_pp':overall_delta,
          'hard_gain_at_least_10pp':hard_delta>=10,'overall_loss_at_most_0p5pp':overall_delta>=-.5,
          'point_target_met':hard_delta>=10 and overall_delta>=-.5,
          'not_a_significance_test':True}
    write(OUT/'practical_target.json',practical)

    ensembles=[]
    for group,z in logits.items():
        name=group+'_ensemble5';pred=sigmoid(z).mean(0).argmax(1);preds[name]=pred
        folder=OUT/'pipelines'/name
        write(folder/'pipeline.json',{'members':[os.path.relpath(path,folder) for path in members[group]],
              'rule':'mean_probability','routing':None})
        score=official(folder,bank,pred);c=correct(y,pred)
        r={'name':name,'group':group,'kind':'ensemble','members':5,'correct':score['correct'],
           'accuracy':score['accuracy'],'member_paths':[str(p) for p in members[group]],
           'official_score_path':str(folder/'validation-score.json'),
           'asset_bytes':sum(asset_bytes(p) for p in members[group])+(folder/'pipeline.json').stat().st_size,
           'eligible':True}
        records.append(r)
        sub={name:float(100*c[mask].mean()) for name,mask in vg.items()}
        ensembles.append({**r,'subgroups_diagnostic':sub})
        print(f'ENSEMBLE {name}: {score["correct"]}/798 hard={sub["wrong_unique_plurality"]:.3f}',flush=True)
    write(OUT/'ensemble_summary.json',ensembles)
    secondary=[]
    for a,b in comparisons:
        ca=correct(y,preds[a+'_ensemble5']);cb=correct(y,preds[b+'_ensemble5'])
        for subgroup in ['all','wrong_unique_plurality','one_correct']:
            mask=vg[subgroup];delta=(ca-cb)[mask]
            secondary.append({'a':a+'_ensemble5','b':b+'_ensemble5','subgroup':subgroup,
              'delta_pp':float(100*delta.mean()),'ci95_pp':interval(delta),'secondary':True})
    write(OUT/'ensemble_comparisons.json',secondary)
    write(OUT/'pipelines.json',records)
    np.savez(OUT/'validation-choices.npz',**preds)

    # Historical uniform runs are read-only context, never substitutes for new matched controls.
    replay=[]
    for m,variant in P['models'].items():
        for i,seed in enumerate(P['seeds']):
            previous=CODE/'fusion_confidence_20261017/results/runs'/variant/f'seed_{seed}'/'full'
            old=np.load(previous/'evaluation-logits.npy');new=logits[m+'_U'][i]
            replay.append({'model':m,'seed':seed,'max_abs_logit_error':float(np.max(np.abs(new-old))),
               'identical_candidate_choices':bool(np.array_equal(old.argmax(1),new.argmax(1))),
               'changed_choices':int((old.argmax(1)!=new.argmax(1)).sum()),
               'old_correct_diagnostic':int(correct(y,old.argmax(1)).sum()),
               'new_official_correct':next(r['official']['correct'] for r in runs if r['group']==m+'_U' and r['seed']==seed),
               'note':'float32 reduction order and gradient clipping can differ; all primary comparisons use new runs'})
    write(OUT/'uniform_replay.json',replay)
    curves=[]
    for r in runs:
        for h in read(HERE/r['run_dir']/'history.json'):curves.append({'group':r['group'],'seed':r['seed'],**h})
    dump_csv(OUT/'learning_curves.csv',curves)

    candidates=[r for r in records if r['eligible'] and r['asset_bytes']<=16*1024**2]
    selected=min(candidates,key=lambda r:(-r['correct'],r['members'],r['name']))
    bundle=OUT/'selected_bundle';bundle.mkdir(exist_ok=True)
    for i,path in enumerate(selected['member_paths']):
        destination=bundle/'members'/f'{i:02d}';destination.mkdir(parents=True,exist_ok=True)
        for source in Path(path).iterdir():shutil.copyfile(source,destination/source.name)
    write(bundle/'pipeline.json',{'members':[f'members/{i:02d}' for i in range(selected['members'])],
       'rule':'mean_probability','routing':None})
    write(OUT/'selection.json',{**selected,'selection_protocol':P['selection'],
      'portable_asset_bytes':asset_bytes(bundle),
      'bundle_sha256':{str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}})
    write(OUT/'cost.json',{'training_runs':len(runs),'epochs_per_run':20,
      'training_wall_seconds_sum':sum(r['training_wall_seconds'] for r in runs),
      'optimizer_steps':sum(r['optimizer_steps'] for r in runs),'draws':sum(r['training_draws'] for r in runs),
      'official_validation_predictors':len(records),'checkpoint_or_weight_search':False,'test_used':False})
    protected(OUT)
    print('SELECTED',selected['name'],selected['accuracy'],flush=True)


if __name__=='__main__':main()
