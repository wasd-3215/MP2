"""Integrity and portable unseen-input contracts, without any test split access."""
from support import HERE,OUT,VARIANTS,SEEDS,read,write,sources,sha,protected,asset_bytes
from method import predict
from grouping import layout
from common import load_inputs,parse_answer
import copy
import numpy as np

def main():
    assert sources()==read(OUT/'training_sources.json')
    training=read(OUT/'training_summary.json');evaluation=read(OUT/'evaluation_summary.json')
    assert training['complete'] and len(training['results'])==30
    assert evaluation['official_evaluations']==36 and len(evaluation['portable'])==6
    for r in training['results']:
        bundle=OUT/'runs'/r['identity']['variant']/f"seed_{r['identity']['seed']}"/'bundle'
        assert all(sha(bundle/name)==h for name,h in r['bundle_hashes'].items())
    bank,arrays=load_inputs('validation');answers,_,_,counts=layout(bank)
    active=np.flatnonzero(counts>=2);ix=active[:13][::-1]
    newbank=copy.deepcopy([bank[i] for i in ix])
    for i,row in enumerate(newbank):row['id']=f'unseen-group-contract-{i}'
    newarrays={'embeddings':arrays['embeddings'][ix].copy()}
    permutation=[2,0,3,1];permbank=copy.deepcopy(newbank)
    for row in permbank:
        row['candidates']=[row['candidates'][p] for p in permutation]
        for i,c in enumerate(row['candidates']):c['index']=i
    permarrays={'embeddings':newarrays['embeddings'][:,permutation].copy()}
    contracts=[]
    for variant in VARIANTS:
        bundle=OUT/'portable'/variant;manifest=read(bundle/'manifest.json')
        assert all(sha(bundle/name)==h for name,h in manifest.items())
        size=asset_bytes([bundle/name for name in manifest]);assert size<=16*1024**2
        direct=predict(newbank,newarrays,bundle);permuted=predict(permbank,permarrays,bundle)
        expected=read(OUT/'evaluation'/variant/'ensemble'/'predictions.json')['selected']
        # Ties among equivalent candidates need not preserve candidate index.
        for k,i in enumerate(ix):
            a=parse_answer(newbank[k]['candidates'][direct[k]]['text'])
            b=parse_answer(bank[i]['candidates'][expected[i]]['text'])
            c=parse_answer(permbank[k]['candidates'][permuted[k]]['text'])
            assert a==b==c
        single=predict([newbank[0]],{'embeddings':newarrays['embeddings'][:1]},bundle)
        assert single==direct[:1] and all(type(c) is int and 0<=c<4 for c in direct)
        contracts.append({'variant':variant,'new_ids_reordered13':True,'candidate_permutation_numeric_answer':True,'single_question_batch':True,
                          'portable_manifest':True,'assets_uncompressed_bytes':size})
    easy=copy.deepcopy(bank[int(np.flatnonzero(counts==1)[0])]);fake=HERE/'does-not-exist'
    assert 0<=predict([easy],{'embeddings':np.zeros((1,4,1024),np.float32)},fake)[0]<4
    invalid=copy.deepcopy(easy)
    for c in invalid['candidates']:c['text']='No final answer'
    assert predict([invalid],{'embeddings':np.zeros((1,4,1024),np.float32)},fake)==[0]
    assert predict([],{'embeddings':np.empty((0,4,1024),np.float32)},fake)==[]
    protection=protected();assert protection['passed']
    write(OUT/'verification.json',{'passed':True,'training_sources_frozen':True,'protected':protection,'portable_contracts':contracts,
          'one_group_bypass':True,'zero_group_legal_fallback':True,'empty_bank':True,'test_semantics_opened':False,
          'local_complete_predict_score_seconds_range':[min(r['complete_wall_seconds'] for r in evaluation['results']),max(r['complete_wall_seconds'] for r in evaluation['results'])],
          'official_reference_cpu_30s_certified':False})
    print('VERIFICATION PASSED',protection['files_checked'],flush=True)

if __name__=='__main__':main()
