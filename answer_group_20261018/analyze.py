"""Fixed contrasts and question-paired uncertainty; no fitted validation rules."""
from support import HERE,OUT,CODE,SEEDS,VARIANTS,read,write
from grouping import layout
from train import labels
from common import load_inputs,read_jsonl,parse_answer
import numpy as np

def bootstrap(delta,mask):
    values=np.asarray(delta)[mask].astype(np.float64);rng=np.random.default_rng(20261018)
    means=np.concatenate([values[rng.integers(0,len(values),(min(250,10000-i),len(values)))].mean(1) for i in range(0,10000,250)])
    return {'delta_percentage_points':float(100*values.mean()),'ci95_percentage_points':(100*np.quantile(means,[.025,.975])).tolist(),'n':len(values)}

def main():
    assert read(OUT/'evaluation_summary.json')['complete']
    bank,_=load_inputs('validation');y=labels('validation',bank)
    answers,membership,valid,counts=layout(bank);sizes=membership.sum(2)
    gold_group=(membership&y.astype(bool)[:,None,:]).any(2).argmax(1)
    largest=sizes.max(1);unique=(sizes==largest[:,None]).sum(1)==1
    wrongplurality=unique&(sizes[np.arange(len(bank)),gold_group]<largest)
    common=read_jsonl(CODE/'common_failure_diagnosis_20261017'/'results'/'reviewed_cases.jsonl')
    commonids={r['id'] for r in common};assert len(commonids)==42
    masks={'all':np.ones(len(bank),bool),'multi_answer':counts>=2,'one_correct':y.sum(1)==1,
           'wrong_unique_plurality':wrongplurality,'all_valid_groups_singleton':(sizes<=1).all(1),
           'historical_common42':np.asarray([r['id'] in commonids for r in bank])}
    assert len(bank)==798 and masks['multi_answer'].sum()==355 and masks['one_correct'].sum()==117 and wrongplurality.sum()==60
    correctness={};ensembles={};table=[]
    for variant in VARIANTS:
        choices=np.asarray([read(OUT/'evaluation'/variant/f'seed_{seed}'/'predictions.json')['selected'] for seed in SEEDS])
        correctness[variant]=y[np.arange(len(bank))[None,:],choices].astype(bool)
        selected=np.asarray(read(OUT/'evaluation'/variant/'ensemble'/'predictions.json')['selected'])
        ensembles[variant]=y[np.arange(len(bank)),selected].astype(bool)
        stats={}
        for name,mask in masks.items():
            single=correctness[variant][:,mask].sum(1);ens=int(ensembles[variant][mask].sum());n=int(mask.sum())
            stats[name]={'n':n,'single_correct':single.tolist(),'mean_single_accuracy':float(100*single.mean()/n),
                         'seed_accuracy_sd':float(100*single.std(ddof=1)/n),'ensemble_correct':ens,'ensemble_accuracy':100*ens/n}
        train=[read(OUT/'runs'/variant/f'seed_{seed}'/'result.json')['train_active_accuracy'] for seed in SEEDS]
        table.append({'variant':variant,'train_active_mean_accuracy':float(np.mean(train)),'metrics':stats})
    pairs=[('primary_group_representation_NoSize','GroupCE_NoSize','CandMaxCE_NoSize'),
           ('group_representation_Size','GroupCE_Size','CandMaxCE_Size'),
           ('loss_control_NoSize','CandMaxCE_NoSize','CandBCE_NoSize'),
           ('loss_control_Size','CandMaxCE_Size','CandBCE_Size')]
    pairs += [(f'explicit_size_{name}',f'{name}_Size',f'{name}_NoSize') for name in ('CandBCE','CandMaxCE','GroupCE')]
    contrasts={}
    for name,a,b in pairs:
        delta=correctness[a].mean(0)-correctness[b].mean(0)
        contrasts[name]={'a':a,'b':b,'paired_seed_delta_pp':(100*(correctness[a].mean(1)-correctness[b].mean(1))).tolist(),
                         'mean_seed':{k:bootstrap(delta,m) for k,m in masks.items()},
                         'ensemble':{k:bootstrap(ensembles[a].astype(float)-ensembles[b],m) for k,m in masks.items()},
                         'ensemble_repairs':int((ensembles[a]&~ensembles[b]).sum()),'ensemble_harms':int((~ensembles[a]&ensembles[b]).sum())}
    interaction=(correctness['GroupCE_Size'].mean(0)-correctness['CandMaxCE_Size'].mean(0))-(correctness['GroupCE_NoSize'].mean(0)-correctness['CandMaxCE_NoSize'].mean(0))
    contrasts['group_by_explicit_size_interaction']={'mean_seed':{k:bootstrap(interaction,m) for k,m in masks.items()}}
    joint=np.stack(list(ensembles.values()));commonmask=masks['historical_common42']
    changed=[]
    for i,row in enumerate(bank):
        if commonmask[i]:changed.append({'row':i,'id':row['id'],'case':next(r['case'] for r in common if r['id']==row['id']),
                             'rescued_by':[v for v in VARIANTS if ensembles[v][i]],'answer_groups':int(counts[i]),'wrong_unique_plurality':bool(wrongplurality[i])})
    arrays={**{f'singles_{k}':v for k,v in correctness.items()},**{f'ensemble_{k}':v for k,v in ensembles.items()},**{f'mask_{k}':v for k,v in masks.items()}}
    np.savez(OUT/'question_level_comparison.npz',**arrays)
    write(OUT/'analysis.json',{'table':table,'contrasts':contrasts,'subgroup_counts':{k:int(v.sum()) for k,v in masks.items()},
         'common42_cases':changed,'common42_union_rescued':int(joint[:,commonmask].any(0).sum()),
         'validation_union_correct':int(joint.any(0).sum()),'bootstrap_resamples':10000,'bootstrap_seed':20261018,
         'uncertainty_scope':'Questions resampled with all paired-seed correctness held together. Conditional on these fitted seeds; repeated validation use and multiple contrasts are not corrected.'})
    print([(r['variant'],r['metrics']['all']['single_correct'],r['metrics']['all']['ensemble_correct']) for r in table],flush=True)

if __name__=='__main__':main()
