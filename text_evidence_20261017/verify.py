"""Exact expert replay, evidence equivariance, arbitrary-input and integrity checks."""
import runtime
from utils import HERE,OUT,OLD,read,write,sha,sources,protected,members
import numpy as np
from common import load_inputs,parse_answer
from method import probabilities,predict
from decisions import choose,sigmoid,best
from evidence import extract,candidate_evidence

def main():
    frozen=read(OUT/'frozen_before_validation.json')
    assert sources()==frozen['sources'] and all(sha(p)==h for p,h in frozen['inputs'].items())
    bank,arrays=load_inputs('validation')
    p=probabilities(bank,arrays,members())
    old=np.stack([sigmoid(np.load(OLD/f'results/runs/S2_C/seed_{s}/validation-logits.npy')) for s in [17,29,43,71,101]]).mean(0)
    error=float(np.max(np.abs(p-old)));assert error<1e-6
    f=np.load(OUT/'validation-evidence.npy')
    ix=np.asarray([97,3,400,11,777,25,0,48,200,17,700,56,98])
    subset=[{**bank[i],'id':f'new-audit:{j}'} for j,i in enumerate(ix)]
    small={'embeddings':arrays['embeddings'][ix]}
    perm=[2,0,3,1]
    perm_bank=[{**r,'candidates':[{**r['candidates'][k],'index':j} for j,k in enumerate(perm)]} for r in subset]
    perm_arrays={'embeddings':small['embeddings'][:,perm]}
    perm_p=probabilities(perm_bank,perm_arrays,members())
    perm_f=extract(perm_bank)
    assert np.array_equal(perm_f,f[ix][:,perm])
    perm_error=float(np.max(np.abs(perm_p-p[ix][:,perm])));assert perm_error<3e-5
    checked=[]
    for name in frozen['pipelines']:
        path=OUT/'pipelines'/name;config=read(path/'pipeline.json')
        expected=read(path/'predictions.json')['selected']
        assert choose(bank,p,f,config).tolist()==expected
        actual=predict(subset,small,path);assert actual==[expected[i] for i in ix]
        assert predict(subset[:1],{'embeddings':small['embeddings'][:1]},path)==actual[:1]
        assert all(type(i) is int and 0<=i<4 for i in actual)
        new=choose(perm_bank,perm_p,perm_f,config)
        for q,k in enumerate(new):
            assert parse_answer(perm_bank[q]['candidates'][int(k)]['text'])==parse_answer(subset[q]['candidates'][actual[q]]['text'])
        invalid=[{'id':'all-invalid','question':'synthetic','candidates':[{'index':i,'text':'no numeric final marker'} for i in range(4)]}]
        valid=[{**invalid[0],'id':'one-valid','candidates':[{'index':i,'text':'#### 7' if i==2 else 'no marker'} for i in range(4)]}]
        for b in [invalid,valid]:
            values=choose(b,p[:1],extract(b),config)
            assert len(values)==1 and 0<=int(values[0])<4
            if b is valid:assert values.tolist()==[2]
        no_text=np.zeros((len(subset),4,7))
        valid_mask=np.asarray([[parse_answer(c['text']) is not None for c in r['candidates']] for r in subset])
        assert np.array_equal(choose(subset,p[ix],no_text,config),best(p[ix],valid_mask))
        checked.append(name)
    integrity=protected()
    verification={'passed':integrity['passed'],'experiment_checks_passed':True,'fresh_official_evaluations':4,'portable_official_replays':4,
      'baseline_probability_replay_max_error':error,'candidate_permutation_probability_max_error':perm_error,
      'text_evidence_candidate_permutation_exact':True,'new_ids_reordered_single_question_and_parser_fallback_checks':checked,
      'no_evidence_preserves_original_choices':True,'source_and_frozen_asset_hashes_unchanged':True,
      'protected_files':integrity['files_checked'],'protected_changes':integrity['changed'],'test_used':False,
      'official_reference_cpu_certification':False}
    write(OUT/'verification.json',verification)
    write(OUT/'delivery_source_hashes.json',sources())
    print(verification,flush=True)
    assert integrity['passed'],integrity['changed']

if __name__=='__main__':main()
