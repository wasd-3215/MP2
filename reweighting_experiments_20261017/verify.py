"""Fresh official replay and inference checks for every new predictor."""
import runtime
from utilities import HERE,CODE,read,write,sha,protected,command,asset_bytes,source_hashes
from pathlib import Path
import sys
import numpy as np
from common import load_inputs


def main():
    out=HERE/'results';records=read(out/'pipelines.json');selection=read(out/'selection.json')
    assert read(out/'implementation_checks.json')['passed']
    assert read(out/'training_sources.json')==source_hashes()
    frozen=read(HERE/'analysis_sources_before.json')
    assert all(sha(HERE/n)==h for n,h in frozen.items())
    bundle=out/'selected_bundle'
    assert {str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}==selection['bundle_sha256']
    names=[r['name'] for r in records if r['kind']=='ensemble']+['P_U_seed17','C_U_seed17','C_H_seed17',selection['name']]
    names=list(dict.fromkeys(names));resources=[]
    for name in names:
        r=next(r for r in records if r['name']==name)
        path=bundle if name==selection['name'] else (Path(r['member_paths'][0]) if r['kind']=='single' else out/'pipelines'/name)
        folder=out/'benchmarks'/name;folder.mkdir(parents=True,exist_ok=True)
        a=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(HERE/'method.py'),
          '--checkpoint',str(path),'--split','validation','--output',str(folder/'predictions.json')],folder/'predict.log')
        b=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
          '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')],folder/'score.log')
        assert read(folder/'score.json')['correct']==r['correct']
        expected=read(Path(r['official_score_path']).parent/'validation-predictions.json')['selected']
        assert read(folder/'predictions.json')['selected']==expected
        size=asset_bytes(bundle) if name==selection['name'] else r['asset_bytes']
        resources.append({'name':name,'members':r['members'],'accuracy':r['accuracy'],
          'prediction_wall_seconds':a,'scoring_wall_seconds':b,'total_fresh_wall_seconds':a+b,
          'uncompressed_asset_bytes':size,'under_16_mib':size<=16*1024**2,'local_under_30_seconds':a+b<=30})
        print('BENCHMARK',name,f'{a+b:.3f}s',flush=True)
    write(out/'resources.json',resources)
    from method import predict
    bank,arrays=load_inputs('validation');indices=np.array([97,3,400,11,777,25,0,48,200,17,700,56,98])
    subbank=[{**bank[i],'id':f'audit-new:{j}'} for j,i in enumerate(indices)]
    subarrays={'embeddings':arrays['embeddings'][indices]};passed=[]
    for r in records:
        path=Path(r['member_paths'][0]) if r['kind']=='single' else out/'pipelines'/r['name']
        expected=read(Path(r['official_score_path']).parent/'validation-predictions.json')['selected']
        actual=predict(subbank,subarrays,path)
        assert actual==[expected[i] for i in indices],r['name']
        assert predict(subbank[:1],{'embeddings':subarrays['embeddings'][:1]},path)==actual[:1]
        assert all(type(v) is int and 0<=v<4 for v in actual)
        passed.append(r['name'])
        if len(passed)%10==0:print('NEW BANK CHECKS',len(passed),flush=True)
    expected=read(Path(selection['official_score_path']).parent/'validation-predictions.json')['selected']
    assert predict(subbank,subarrays,bundle)==[expected[i] for i in indices]
    protected(out)
    write(out/'verification.json',{'passed':True,'all_predictors_new_ids_reordered_13_and_single_question':passed,
      'selected_portable_bundle_replayed':True,'native_integer_output':True,'weights_not_required_at_inference':True,
      'training_sources_unchanged':True,'analysis_and_inference_sources_unchanged':True,
      'protected_files_checked':len(read(HERE/'protected_before.json')),'protected_changes':[],
      'test_used':False,'reference_cpu_certification':False})
    print('VERIFICATION PASSED',flush=True)


if __name__=='__main__':main()
