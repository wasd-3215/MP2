"""48 serial fresh official full-validation predictions and scores."""
import runtime
from support import HERE,CODE,OUT,SEEDS,VARIANTS,read,write,sha,source_hashes,command,asset_bytes
from pathlib import Path
import os,sys
def main():
    assert read(OUT/'training_verification.json')['passed']
    assert read(OUT/'training_sources.json')==source_hashes()
    write(OUT/'evaluation_sources.json',{'evaluate_suite.py':sha(HERE/'evaluate_suite.py')})
    runs=read(OUT/'training_runs.json');records=[]
    for variant in VARIANTS:
        scope,feature,position=variant.split('_')
        for seeds in [[s] for s in SEEDS]+[SEEDS]:
            kind=f'seed{seeds[0]}' if len(seeds)==1 else 'ensemble5';name=f'{variant}_{kind}'
            folder=OUT/'pipelines'/name;folder.mkdir(parents=True,exist_ok=True)
            members=[OUT/'runs'/variant/f'seed_{seed}/bundle' for seed in seeds]
            for seed,member in zip(seeds,members):
                run=next(r for r in runs if r['variant']==variant and r['seed']==seed)
                assert {p.name:sha(p) for p in member.iterdir()}==run['bundle_hashes']
            write(folder/'pipeline.json',{'members':[os.path.relpath(m,folder) for m in members],
                'rule':'mean_sigmoid_probability','one_valid_group_bypass':True,'final_validity_filter':True})
            records.append({'name':name,'variant':variant,'scope':scope,'feature':feature,'position':position,
                'seeds':seeds,'members':len(seeds),'checkpoint':str(folder),'asset_bytes':asset_bytes(members,folder/'pipeline.json')})
    write(OUT/'predictors.json',records);results=[]
    for r in records:
        folder=Path(r['checkpoint'])
        if (folder/'timing.json').exists():timing=read(folder/'timing.json')
        else:
            p=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(HERE/'method.py'),
                '--checkpoint',str(folder),'--split','validation','--output',str(folder/'predictions.json')],folder/'predict.log')
            s=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
                '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')],folder/'score.log')
            timing={'prediction_wall_seconds':p,'scoring_wall_seconds':s,'fresh_total_seconds':p+s};write(folder/'timing.json',timing)
        result={**r,**read(folder/'score.json'),**timing};results.append(result);write(OUT/'official_results.json',results)
        print(f'{len(results)}/48 {r["name"]}: {result["correct"]}/{result["n"]}, {result["fresh_total_seconds"]:.3f}s',flush=True)
    assert read(OUT/'training_sources.json')==source_hashes()
    print('ALL 48 FRESH OFFICIAL EVALUATIONS COMPLETE',flush=True)
if __name__=='__main__':main()
