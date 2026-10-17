"""Portable bundles, fresh official replays, arbitrary-bank behavior and hashes."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
from support import HERE,CODE,OUT,VARIANTS,read,write,sha,source_hashes,protected,command,asset_bytes
import shutil
def main():
    assert read(OUT/'training_sources.json')==source_hashes()
    assert read(OUT/'training_verification.json')['passed']
    assert len(read(OUT/'fresh_replay.json'))==48
    records=read(OUT/'official_results.json');inference=HERE/'portable/inference';inference.mkdir(parents=True,exist_ok=True)
    for name in ['method.py','runtime.py','support.py','models.py','reference_models.py','features.py','explicit_features.py']:
        target=inference/name
        if target.exists():assert sha(target)==sha(HERE/name)
        else:shutil.copyfile(HERE/name,target)
    deliveries=[]
    for variant in VARIANTS:
        name=f'{variant}_ensemble5';source=OUT/'pipelines'/name;manifest=read(source/'pipeline.json')
        dest=HERE/'portable'/name;dest.mkdir(parents=True,exist_ok=True);members=[]
        for i,relative in enumerate(manifest['members']):
            member=(source/relative).resolve();target=dest/'members'/f'{i:02d}'
            if not target.exists():shutil.copytree(member,target)
            assert {p.name:sha(p) for p in target.iterdir()}=={p.name:sha(p) for p in member.iterdir()}
            members.append(target)
        manifest['members']=[f'members/{i:02d}' for i in range(5)];write(dest/'pipeline.json',manifest)
        folder=OUT/'delivery_check'/name;folder.mkdir(parents=True,exist_ok=True)
        ps=command([sys.executable,'-B',str(CODE/'predict.py'),'--method',str(inference/'method.py'),'--checkpoint',str(dest),
            '--split','validation','--output',str(folder/'predictions.json')],folder/'predict.log')
        ss=command([sys.executable,'-B',str(CODE/'score_predictions.py'),'--split','validation',
            '--predictions',str(folder/'predictions.json'),'--output',str(folder/'score.json')],folder/'score.log')
        assert read(folder/'predictions.json')['selected']==read(source/'predictions.json')['selected']
        record=next(r for r in records if r['name']==name);assert read(folder/'score.json')['correct']==record['correct']
        size=asset_bytes(members,dest/'pipeline.json');assert size<=16*1024**2
        deliveries.append({'name':name,'correct':record['correct'],'asset_bytes':size,'fresh_prediction_scoring_seconds':ps+ss,
            'asset_hashes':{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}})
        print('PORTABLE REPLAY',name,record['correct'],flush=True)
    import numpy as np
    from common import load_inputs
    from method import predict,layout
    bank,arrays=load_inputs('validation');index=np.array([97,3,400,11,777,25,0,48,200,17,700,56,98])
    subset=[{**bank[i],'id':f'new-audit:{j}'} for j,i in enumerate(index)];inputs={'embeddings':arrays['embeddings'][index]}
    _,valid,groups=layout(bank);hard=int(np.flatnonzero(groups>=2)[0]);easy=int(np.flatnonzero(groups==1)[0])
    hardbank=[{**bank[hard],'id':'single-new-disagreement'}];easybank=[{**bank[easy],'id':'single-new-easy'}]
    checked=[]
    for r in records:
        expected=read(Path(r['checkpoint'])/'predictions.json')['selected'];actual=predict(subset,inputs,Path(r['checkpoint']))
        assert actual==[expected[i] for i in index],r['name']
        assert predict(hardbank,{'embeddings':arrays['embeddings'][hard:hard+1]},Path(r['checkpoint']))==[expected[hard]]
        assert predict(easybank,{'embeddings':arrays['embeddings'][easy:easy+1]},'not-a-real-checkpoint')==[int(valid[easy].argmax())]
        assert all(type(i) is int and 0<=i<4 for i in actual);checked.append(r['name'])
    protected()
    assert read(OUT/'training_sources.json')==source_hashes()
    assert read(OUT/'evaluation_sources.json')['evaluate_suite.py']==sha(HERE/'evaluate_suite.py')
    frozen=read(OUT/'analysis_sources_before.json');assert all(sha(HERE/p)==h for p,h in frozen.items())
    write(OUT/'verification.json',{'passed':True,'training_runs':40,'official_evaluations':48,'extra_portable_official_replays':8,
        'fresh_replay_exact':48,'new_id_reordered_and_single_disagreement_checks':checked,
        'single_easy_bank_without_loading_checkpoint':True,'native_int_outputs':True,'initial_weights_and_function_matched_all_eight_cells':True,
        'training_draws_exactly_matched_within_scope':True,'easy_validation_correct':443,'neural_validation_questions':355,
        'protected_files':len(read(HERE/'protected_before.json')),'protected_changes':[],'portable_deliveries':deliveries,
        'test_used':False,'inference_training_evaluation_analysis_sources_unchanged':True,'reference_cpu_certification':False})
    write(OUT/'delivery_source_hashes.json',{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md')})
    print('ALL VERIFICATION PASSED',flush=True)
if __name__=='__main__':main()
