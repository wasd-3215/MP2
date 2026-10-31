"""Four frozen, fresh official evaluations and portable replays."""
import runtime
from utils import HERE,CODE,OUT,read,write,sha,sources,command,members,asset_bytes
import shutil
import sys

INFERENCE=['method.py','decisions.py','evidence.py','runtime.py','models.py','reference_models.py','features.py','explicit_features.py']

def run(checkpoint,method,folder):
    folder.mkdir(parents=True,exist_ok=True)
    prediction=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(method),'--checkpoint',str(checkpoint),
        '--split','validation','--output',str(folder/'predictions.json')],folder/'predict.log')
    scoring=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
        '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')],folder/'score.log')
    return {**read(folder/'score.json'),'prediction_wall_seconds':prediction,'scoring_wall_seconds':scoring,'fresh_total_seconds':prediction+scoring}

def main():
    frozen=read(OUT/'frozen_before_validation.json')
    assert sources()==frozen['sources']
    assert all(sha(p)==h for p,h in frozen['inputs'].items())
    portable=HERE/'portable';infer=portable/'inference';infer.mkdir(parents=True,exist_ok=True)
    for name in INFERENCE:shutil.copyfile(HERE/name,infer/name)
    for i,member in enumerate(members()):
        dest=portable/'members'/f'{i:02d}';shutil.copytree(member,dest,dirs_exist_ok=True)
        assert {p.name:sha(p) for p in member.iterdir()}=={p.name:sha(p) for p in dest.iterdir()}
    results=[]
    for name,digest in frozen['pipelines'].items():
        path=OUT/'pipelines'/name
        assert sha(path/'pipeline.json')==digest
        config=read(path/'pipeline.json')
        timing=run(path,HERE/'method.py',path)
        size=asset_bytes([p for m in members() for p in m.iterdir()]+[path/'pipeline.json'])
        assert size<16*1024**2
        destination=portable/'pipelines'/name
        write(destination/'pipeline.json',{**config,'members':[f'../../members/{i:02d}' for i in range(5)]})
        replay_folder=OUT/'portable_replays'/name
        replay=run(destination,infer/'method.py',replay_folder)
        assert read(replay_folder/'predictions.json')['selected']==read(path/'predictions.json')['selected']
        assert timing['correct']==replay['correct']
        record={'name':name,**timing,'asset_bytes_uncompressed':size,'portable_replay':replay}
        results.append(record);write(OUT/'official_results.json',results)
        print(f'{len(results)}/4 {name}: {timing["correct"]}/{timing["n"]}, {timing["fresh_total_seconds"]:.3f}s, {size/1024**2:.4f} MiB',flush=True)
    assert sources()==frozen['sources']
    write(OUT/'portable_package.json',{'all_variants_shared_assets_bytes':asset_bytes([p for p in portable.rglob('*') if p.is_file() and p.suffix!='.py']),
      'inference_source_bytes':sum((infer/n).stat().st_size for n in INFERENCE),
      'hashes':{str(p.relative_to(portable)):sha(p) for p in portable.rglob('*') if p.is_file()}})

if __name__=='__main__':main()
