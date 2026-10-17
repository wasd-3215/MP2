"""Fresh-process replay, resource accounting, new-bank behavior and protected hashes."""
import runtime
from utilities import HERE,CODE,read,write,sha,protected,command,asset_bytes,source_hashes
from pathlib import Path
import os,sys,time
import numpy as np
from common import load_inputs


def main():
    out=HERE/'results';selection=read(out/'selection.json');records=read(out/'pipelines.json')
    assert read(out/'implementation_checks.json')['passed']
    assert read(out/'training_sources.json')==source_hashes()
    bundle=out/'selected_bundle'
    assert {str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}==selection['bundle_sha256']
    names=['s2_seed17','s2_5','heterogeneous_15','consistency_5','s2_consistency_seed17','features_only_seed17',selection['name']]
    names=list(dict.fromkeys(names));resources=[]
    for name in names:
        r=next(r for r in records if r['name']==name)
        checkpoint=bundle if name==selection['name'] else (Path(r['member_paths'][0]) if r['group']=='single' else out/'pipelines'/name)
        folder=out/'benchmarks'/name;folder.mkdir(parents=True,exist_ok=True)
        predict_seconds=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(HERE/'method.py'),
            '--checkpoint',str(checkpoint),'--split','validation','--output',str(folder/'predictions.json')],folder/'predict.log')
        score_seconds=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
            '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')],folder/'score.log')
        score=read(folder/'score.json');assert score['correct']==r['correct']
        source=read(Path(r['official_score_path']).parent/'validation-predictions.json')['selected']
        assert read(folder/'predictions.json')['selected']==source
        size=asset_bytes(bundle) if name==selection['name'] else r['asset_bytes']
        resources.append({'name':name,'members':r['members'],'accuracy':score['accuracy'],
          'prediction_wall_seconds':predict_seconds,'scoring_wall_seconds':score_seconds,
          'total_fresh_wall_seconds':predict_seconds+score_seconds,'uncompressed_asset_bytes':size,
          'local_under_30_seconds':predict_seconds+score_seconds<=30,'under_16_mib':size<=16*1024**2})
        print('BENCHMARK',name,f'{predict_seconds+score_seconds:.3f}s',flush=True)
    write(out/'resources.json',resources)
    # Import torch only after isolated process timing; parent does no simultaneous numerical work.
    from method import predict
    bank,arrays=load_inputs('validation')
    index=np.array([97,3,400,11,777,25,0,48,200,17,700,56,98])
    subset=[{**bank[i],'id':f'new-bank:{j}'} for j,i in enumerate(index)]
    subarrays={'embeddings':arrays['embeddings'][index]}
    checks=[]
    for r in records:
        if r['group']=='single':checkpoint=Path(r['member_paths'][0])
        else:checkpoint=out/'pipelines'/r['name']
        expected=read(Path(r['official_score_path']).parent/'validation-predictions.json')['selected']
        actual=predict(subset,subarrays,checkpoint)
        assert actual==[expected[i] for i in index],r['name']
        assert predict(subset[:1],{'embeddings':subarrays['embeddings'][:1]},checkpoint)==actual[:1]
        assert all(type(i) is int and 0<=i<4 for i in actual)
        checks.append(r['name'])
        if len(checks)%10==0:print('NEW BANK CHECKS',len(checks),flush=True)
    selected_expected=read(Path(selection['official_score_path']).parent/'validation-predictions.json')['selected']
    assert predict(subset,subarrays,bundle)==[selected_expected[i] for i in index]
    protected(out)
    result={'passed':True,'all_pipelines_new_ids_reordered_and_single_question':checks,
      'native_integer_outputs':True,'selected_portable_bundle_replayed':True,
      'protected_files_checked':len(read(HERE/'protected_before.json')),'protected_changes':[],
      'training_sources_unchanged':True,'test_used':False,
      'selected_local_resource_check':next(r for r in resources if r['name']==selection['name']),
      'reference_cpu_certification':False}
    write(out/'verification.json',result)
    write(out/'delivery_source_hashes.json',{p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md') or p.name=='protocol.json'})
    print('VERIFICATION PASSED',flush=True)


if __name__=='__main__':main()
