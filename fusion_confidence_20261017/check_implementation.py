"""Meaningful invariance, grouping, routing and OOF preprocessing checks."""
import runtime
runtime.configure_torch()
from utilities import HERE,CODE,read,write,sha
import importlib.util
import numpy as np
import torch
from common import load_bank,load_inputs
from scorers import Scorer,configuration
from explicit_features import extract,FEATURES,fit_standardization,answer_structure
from decisions import confidence,route,aggregate


def main():
    spec=importlib.util.spec_from_file_location('previous_stage_models',CODE/'interaction_stage_20261017/models.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.manual_seed(2026);x=torch.randn(13,4,1024)
    f=extract(load_bank('train')[:13]);checks=[]
    for variant,layout in [('s2','s2_before_keep_norm_after'),('s0','s0_after')]:
        torch.manual_seed(17);new=Scorer(**configuration(variant)).eval()
        torch.manual_seed(17);prior=old.CandidateScorer(layout=layout).eval()
        assert all(torch.equal(v,prior.state_dict()[k]) for k,v in new.base.state_dict().items())
        assert torch.equal(new(x,torch.zeros(13,4,0)),prior(x))
        checks.append(variant+' identical to prior architecture/initialization')
    for variant in read(HERE/'protocol.json')['full_variants']:
        torch.manual_seed(17);model=Scorer(**configuration(variant)).eval()
        feats=torch.from_numpy(f[...,FEATURES[model.feature_mode]])
        a=model(x,feats);order=[2,0,3,1]
        assert torch.allclose(a[:,order],model(x[:,order],feats[:,order]),atol=3e-5,rtol=1e-5)
        if variant.startswith('s2_'):
            torch.manual_seed(17);base=Scorer(**configuration('s2')).eval()
            assert torch.equal(a,base(x,torch.zeros(13,4,0)))
        checks.append(variant+' candidate equivariance and zero feature initialization')
    bank=[{'id':'synthetic','question':'unused','candidates':[{'index':i,'text':t} for i,t in enumerate(
        ['#### 42','#### 42.0','#### 17','#### 8'])]}]
    probabilities=np.array([[[.90,.89,.30,.20]]])
    scores,choice=aggregate(probabilities,bank)
    cg=confidence(scores,choice,probabilities,bank,'candidate_gap')
    ag=confidence(scores,choice,probabilities,bank,'answer_gap')
    assert np.allclose(cg,.01) and np.allclose(ag,.60)
    assert answer_structure(bank)[1].tolist()==[[2,2,1,1]]
    corrected,changed=route(scores,np.array([2]),np.array([0.]),bank,.1,2)
    assert corrected.tolist()==[0] and changed.tolist()==[True]
    no_three,_=route(scores,np.array([2]),np.array([0.]),bank,.1,3)
    assert no_three.tolist()==[2]
    tied=[{**bank[0],'candidates':[{'index':i,'text':f'#### {v}'} for i,v in enumerate([42,42,17,17])]}]
    no_tie,_=route(scores,np.array([2]),np.array([0.]),tied,.1,2)
    assert no_tie.tolist()==[2]
    checks+=['numeric equivalent answer grouping','same-answer candidate ambiguity versus answer gap',
      'unique-plurality fallback and support condition','tied plurality never triggers']
    # Check actual fitted preprocessing for every OOF fold of one seed, all four variants.
    folds=np.load(HERE/'results/question-folds.npy');bank,arrays=load_inputs('train');raw=arrays['embeddings'];rawf=extract(bank)
    normalization=[]
    for fold in range(5):
        index=np.flatnonzero(folds!=fold);held=np.flatnonzero(folds==fold)
        mean,std=fit_standardization(raw[index]);fm,fs=fit_standardization(rawf[index])
        for variant in read(HERE/'protocol.json')['oof_variants']:
            folder=HERE/'results/runs'/variant/'seed_17'/f'fold_{fold}'
            assert np.array_equal(np.load(folder/'held_indices.npy'),held)
            with np.load(folder/'bundle/preprocessing.npz') as p:
                assert np.array_equal(p['mean'],mean) and np.array_equal(p['std'],std)
                assert np.array_equal(p['feature_mean'],fm) and np.array_equal(p['feature_std'],fs)
            normalization.append(f'{variant}/fold{fold}')
    write(HERE/'results/implementation_checks.json',{'passed':True,'checks':checks,
      'actual_oof_preprocessing_checked':normalization,
      'all_four_candidates_same_fold':True,'test_used':False})
    print('IMPLEMENTATION CHECKS PASSED',len(checks),len(normalization))


if __name__=='__main__':main()
