"""Finalize after the strict verifier detected a Git FETCH_HEAD change.

Preserve the failed all-file gate and its baseline. Recheck experiment artifacts
and original non-Git files; never restore or edit Git metadata or original files.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime
from support import HERE,CODE,ROOT,OUT,VARIANTS,read,write,sha,source_hashes,asset_bytes
import numpy as np
from common import load_inputs
from method import predict,layout

def main():
    write(OUT/'verification_finalize_sources.json',{'analysis/finalize_verification.py':sha(Path(__file__)),
       'reason':'Strict original verifier completed portable and input checks but stopped on changed code/.git/FETCH_HEAD; no scientific or inference code changed'})
    assert read(OUT/'training_sources.json')==source_hashes()
    assert read(OUT/'training_verification.json')['passed']
    assert read(OUT/'evaluation_sources.json')['evaluate_suite.py']==sha(HERE/'evaluate_suite.py')
    frozen=read(OUT/'analysis_sources_before.json');assert all(sha(HERE/p)==h for p,h in frozen.items())
    records=read(OUT/'official_results.json');assert len(read(OUT/'fresh_replay.json'))==48
    inference=HERE/'portable/inference'
    for name in ['method.py','runtime.py','support.py','models.py','reference_models.py','features.py','explicit_features.py']:
        assert sha(inference/name)==sha(HERE/name)
    deliveries=[]
    for variant in VARIANTS:
        name=f'{variant}_ensemble5';source=OUT/'pipelines'/name;dest=HERE/'portable'/name
        folder=OUT/'delivery_check'/name;manifest=read(dest/'pipeline.json')
        assert read(folder/'predictions.json')['selected']==read(source/'predictions.json')['selected']
        score=read(folder/'score.json');record=next(r for r in records if r['name']==name)
        assert score['correct']==record['correct']
        members=[(dest/p).resolve() for p in manifest['members']]
        original=[(source/p).resolve() for p in read(source/'pipeline.json')['members']]
        for a,b in zip(members,original):assert {p.name:sha(p) for p in a.iterdir()}=={p.name:sha(p) for p in b.iterdir()}
        size=asset_bytes(members,dest/'pipeline.json');assert size<=16*1024**2
        deliveries.append({'name':name,'correct':score['correct'],'asset_bytes':size,
            'fresh_official_replay_completed_before_strict_hash_failure':True,
            'fresh_wall_seconds_not_retained_because_verifier_aborted':True,
            'official_result_path':str(folder/'score.json'),
            'asset_hashes':{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}})
    bank,arrays=load_inputs('validation');index=np.array([97,3,400,11,777,25,0,48,200,17,700,56,98])
    subset=[{**bank[i],'id':f'new-audit:{j}'} for j,i in enumerate(index)];inputs={'embeddings':arrays['embeddings'][index]}
    _,valid,groups=layout(bank);hard=int(np.flatnonzero(groups>=2)[0]);easy=int(np.flatnonzero(groups==1)[0]);checked=[]
    for r in records:
        expected=read(Path(r['checkpoint'])/'predictions.json')['selected'];actual=predict(subset,inputs,Path(r['checkpoint']))
        assert actual==[expected[i] for i in index]
        assert predict([{**bank[hard],'id':'single-new-hard'}],{'embeddings':arrays['embeddings'][hard:hard+1]},Path(r['checkpoint']))==[expected[hard]]
        assert predict([{**bank[easy],'id':'single-new-easy'}],{'embeddings':arrays['embeddings'][easy:easy+1]},'not-a-real-checkpoint')==[int(valid[easy].argmax())]
        assert all(type(i) is int and 0<=i<4 for i in actual);checked.append(r['name'])
    before=read(HERE/'protected_before.json');changed=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    project_changed=[p for p in changed if '.git' not in Path(p).parts]
    metadata=[p for p in changed if '.git' in Path(p).parts]
    assert not project_changed,project_changed
    assert all(Path(p).name=='FETCH_HEAD' for p in metadata),metadata
    write(OUT/'protected_verification.json',{'passed':not changed,'files_checked':len(before),'changed':changed,
       'non_git_original_changes':project_changed,'git_metadata_changes':metadata,'baseline_updated':False,'original_files_restored_or_overwritten':False})
    write(OUT/'verification.json',{'passed':not changed,'experiment_checks_passed':True,
       'all_original_files_hashes_match':not changed,'non_git_original_files_unchanged':True,
       'strict_gate_note':'Only Git FETCH_HEAD hash differs; all-file protection remains failed, not suppressed. No Git command was executed by this experiment; origin of metadata update not established.',
       'training_runs':40,'official_evaluations':48,'extra_portable_official_replays':8,'fresh_replay_exact':48,
       'new_id_reordered_and_single_disagreement_checks':checked,'single_easy_bank_without_loading_checkpoint':True,
       'native_int_outputs':True,'initial_weights_and_function_matched_all_eight_cells':True,
       'training_draws_exactly_matched_within_scope':True,'easy_validation_correct':443,'neural_validation_questions':355,
       'protected_files':len(before),'protected_changes':changed,'git_metadata_changes':metadata,
       'portable_deliveries':deliveries,'test_used':False,'inference_training_evaluation_analysis_sources_unchanged':True,
       'additional_finalization_script_audited_separately':True,'reference_cpu_certification':False})
    report=OUT/'REPORT.md';text=report.read_text(encoding='utf-8');marker='## 原文件核验补充'
    if marker not in text:
        text+='\n'+marker+'\n\n'
        text+=f'严格全文件哈希门未通过：{len(before)}个原文件中，只有`code/.git/FETCH_HEAD`发生变化，其余{len(before)-len(changed)}个哈希一致。该差异是Git元数据；数据、源码、历史实验与模型资产未发现变化。本实验未执行Git操作，元数据更新来源未确定。没有覆盖该文件或更新基准哈希。\n\n'
        text+='模型实验检查、48个新输入检查、8个独立包官方复现均通过。verification.json分别记录experiment_checks_passed=true与严格passed=false；原严格核验失败记录保留。额外finalize_verification.py仅完成分项核对与披露，没有更改训练、推理、统计定义或任何原文件。\n'
        report.write_text(text,encoding='utf-8')
    write(OUT/'delivery_source_hashes.json',{str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py','.md')})
    print('EXPERIMENT CHECKS PASSED; STRICT ORIGINAL-FILE HASH EXCEPTION:',changed,flush=True)
if __name__=='__main__':main()
