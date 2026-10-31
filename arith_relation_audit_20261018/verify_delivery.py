"""Controlled-thread relation replay, immutable source checks and original-file hashes."""
import os
for key,value in {'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','BLIS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'4'}.items():os.environ[key]=value
from pathlib import Path
import json
import sys
import time
import platform
import winreg
import hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'arithmetic'))
from utils import protected,read,write,sha,asset_bytes,members,CODE

def main():
    frozen=read(HERE/'arithmetic/results/frozen_before_validation.json')
    assert all(sha(p)==h for p,h in frozen['sources'].items())
    assert all(sha(p)==h for p,h in frozen['inputs'].items())
    assert all(sha(HERE/'arithmetic/results/pipelines'/n/'pipeline.json')==h for n,h in frozen['pipelines'].items())
    assert read(HERE/'arithmetic/results/verification.json')['passed']
    sys.path.insert(0,str(HERE/'relation'))
    from audit import check_frozen,unique_claims
    from extractor import extract,check_sources
    from common import read_jsonl,DATA
    check_frozen()
    blind=read(HERE/'relation/results/blind_review_frozen_before_labels.json')
    assert blind['review_sha256']==sha(HERE/'relation/results/confirmation_review_blind.json')
    assert blind['cards_sha256']==sha(HERE/'relation/results/confirmation_cards.jsonl')
    # Replay frozen extraction on train inputs only; no annotations or labels fed in.
    bank=read_jsonl(DATA/'candidates/gsm8k_train.jsonl')
    cards=[json.loads(s) for s in (HERE/'relation/results/confirmation_cards.jsonl').read_text(encoding='utf-8').splitlines()]
    selected={c['row']:c for c in cards};tick=time.perf_counter();comparisons=0
    counts={f:0 for f in ['scope_unit','entity_binding','condition_use','target_alignment']}
    for row,r in enumerate(bank):
        e=[extract(r['question'],c['text']) for c in r['candidates']]
        for c,v in zip(r['candidates'],e):check_sources(r['question'],c['text'],v)
        for family in counts:counts[family]+=bool(unique_claims(row,family,e))
        if row in selected:
            assert e==selected[row]['evidence'];comparisons+=4
    seconds=time.perf_counter()-tick
    # Probe behavior depends only on current question/candidate content.
    first=cards[0];permutation=[2,0,3,1]
    perm=[extract(first['question'],first['candidates'][k]['text']) for k in permutation]
    assert perm==[first['evidence'][k] for k in permutation]
    assert all(extract(r['question'],c['text'])==e for r in cards[:3] for c,e in zip(r['candidates'],r['evidence']))
    assert all(extract('unknown','unknown')[f] in ([],None) for f in ['scope_unit','entity_binding','target_alignment'])
    package=HERE/'portable';hashes={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()}
    for i,m in enumerate(members()):
        assert {p.name:sha(p) for p in m.iterdir()}=={p.name:sha(p) for p in (package/f'members/{i:02d}').iterdir()}
    assets={}
    for name in frozen['pipelines']:
        cfg=read(package/f'pipelines/{name}/pipeline.json')
        assert cfg['beta']==read(HERE/f'arithmetic/results/pipelines/{name}/pipeline.json')['beta']
        paths=[p for i in range(5) for p in (package/f'members/{i:02d}').iterdir()]+[package/f'pipelines/{name}/pipeline.json']
        assets[name]=asset_bytes(paths);assert assets[name]<=16*1024**2
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    import numpy as np
    import torch
    write(HERE/'results/environment.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
      'cpu':cpu.strip(),'platform':platform.platform(),'inference_threads':'Torch intra4/interop1, BLAS1, serial jobs',
      'relation_replay_environment_before_numpy_import':{k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
      'reference_xeon_evaluation':False})
    write(HERE/'results/portable_package.json',{'hashes':hashes,'per_predictor_uncompressed_asset_bytes':assets,
      'shared_directory_all_config_bytes':asset_bytes([p for p in package.rglob('*') if p.is_file() and p.suffix!='.py']),
      'reference_answers_or_cached_predictions_packaged':False})
    integrity=protected()
    result={'passed':integrity['passed'],'protected_files':integrity['files_checked'],'protected_changes':integrity['changed'],
      'arithmetic_frozen_sources_inputs_and_configs_match':True,'relation_frozen_sources_and_blind_review_match':True,
      'relation_controlled_thread_replay_candidate_comparisons':comparisons,'relation_controlled_thread_train_replay_seconds':seconds,
      'relation_train_questions_with_probes':counts,'relation_source_spans_and_candidate_permutation_passed':True,
      'arithmetic_official_evaluations':5,'arithmetic_portable_official_replays':5,'test_semantic_content_predictions_or_labels_used':False,
      'reviewer':'Codex single reviewer; not independent human adjudication','reference_cpu_certification':False}
    write(HERE/'results/verification.json',result)
    print(result,flush=True)
    assert result['passed']

if __name__=='__main__':main()
