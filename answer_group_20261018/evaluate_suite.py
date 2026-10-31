"""Fresh original runner/scorer subprocesses, with separately deployable cells."""
from support import HERE,CODE,OUT,SEEDS,VARIANTS,read,write,sources,sha,command,asset_bytes
from pathlib import Path
import os,shutil,sys

INFERENCE=['method.py','runtime.py','models.py','reference_models.py','grouping.py','features.py','explicit_features.py']

def evaluate(method,checkpoint,folder):
    folder.mkdir(parents=True,exist_ok=True)
    prediction=folder/'predictions.json';score=folder/'score.json'
    predict_wall=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(method),'--checkpoint',str(checkpoint),
                          '--split','validation','--output',str(prediction)],folder/'predict.log')
    score_wall=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--predictions',str(prediction),
                        '--split','validation','--output',str(score)],folder/'score.log')
    result={'prediction_wall_seconds':predict_wall,'scoring_wall_seconds':score_wall,'complete_wall_seconds':predict_wall+score_wall,
            'runner_prediction_seconds':read(prediction)['prediction_seconds'],**read(score)}
    write(folder/'timing.json',result);return result

def main():
    assert read(OUT/'training_summary.json')['complete']
    assert sources()==read(OUT/'training_sources.json')
    records=[]
    for variant in VARIANTS:
        bundles=[OUT/'runs'/variant/f'seed_{seed}'/'bundle' for seed in SEEDS]
        for tag,members in [(f'seed_{s}',[b]) for s,b in zip(SEEDS,bundles)]+[('ensemble',bundles)]:
            config=OUT/'pipelines'/variant/tag;config.mkdir(parents=True,exist_ok=True)
            write(config/'pipeline.json',{'members':[os.path.relpath(b,config).replace('\\','/') for b in members],
                  'aggregation':'equal mean of group softmax probabilities','variant':variant})
            assets=[HERE/p for p in INFERENCE]+[config/'pipeline.json']+[p for b in members for p in b.iterdir()]
            size=asset_bytes(assets);assert size<=16*1024**2
            result=evaluate(HERE/'method.py',config,OUT/'evaluation'/variant/tag)
            row={'variant':variant,'tag':tag,'assets_uncompressed_bytes':size,**result};records.append(row)
            write(OUT/'evaluation_progress.json',records)
            print('EVAL',variant,tag,result['correct'],'seconds',round(result['complete_wall_seconds'],3),flush=True)
    replays=[]
    for variant in VARIANTS:
        portable=OUT/'portable'/variant;portable.mkdir(parents=True,exist_ok=True)
        for filename in INFERENCE:shutil.copy2(HERE/filename,portable/filename)
        for seed in SEEDS:
            shutil.copytree(OUT/'runs'/variant/f'seed_{seed}'/'bundle',portable/'members'/f'seed_{seed}',dirs_exist_ok=True)
        write(portable/'pipeline.json',{'members':[f'members/seed_{seed}' for seed in SEEDS],
              'aggregation':'equal mean of group softmax probabilities','variant':variant})
        assets=list(portable.glob('*.py'))+[portable/'pipeline.json']+list((portable/'members').rglob('*.*'))
        size=asset_bytes(assets);assert size<=16*1024**2
        manifest={str(p.relative_to(portable)).replace('\\','/'):sha(p) for p in assets}
        write(portable/'manifest.json',manifest)
        replay=evaluate(portable/'method.py',portable,OUT/'portable_replay'/variant)
        original=read(OUT/'evaluation'/variant/'ensemble'/'predictions.json')
        repeat=read(OUT/'portable_replay'/variant/'predictions.json')
        assert original['ids']==repeat['ids'] and original['selected']==repeat['selected']
        replays.append({'variant':variant,'identical_choices':True,'assets_uncompressed_bytes':size,**replay})
        print('PORTABLE',variant,replay['correct'],flush=True)
    assert sources()==read(OUT/'training_sources.json')
    write(OUT/'evaluation_summary.json',{'complete':True,'official_evaluations':36,'portable_replays':6,'results':records,'portable':replays})

if __name__=='__main__':main()
