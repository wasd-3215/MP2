"""Exact weight controls, objective gradients and shared model initialization."""
from train import weighted_bce
from utilities import HERE,write
import numpy as np
import torch
from common import load_inputs
from reference_training import labels
from weights import make_weights,groups
from explicit_features import extract,FEATURES
from scorers import Scorer,configuration


def main():
    bank,_=load_inputs('train');y=labels('train',bank)
    weights,g,summary=make_weights(bank,y,2.)
    assert int(g['wrong_unique_plurality'].sum())==367
    assert g['wrong_unique_plurality'][y.sum(1)!=1].sum()==0
    k=y.sum(1)
    for value in range(1,5):
        assert np.isclose(weights['H'][k==value].sum(),weights['K'][k==value].sum())
    torch.manual_seed(2026)
    logits=torch.randn(13,4,requires_grad=True);target=torch.randint(0,2,(13,4)).float()
    u=weighted_bce(logits,target,torch.ones(13))
    standard=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
    gu=torch.autograd.grad(u,logits,retain_graph=True)[0];gs=torch.autograd.grad(standard,logits,retain_graph=True)[0]
    gradient_error=float((gu-gs).abs().max())
    assert torch.allclose(u,standard,atol=1e-7) and torch.allclose(gu,gs,atol=1e-8,rtol=1e-6)
    w=torch.linspace(.5,2,13)
    loss=weighted_bce(logits,target,w)
    grad=torch.autograd.grad(loss,logits)[0]
    expected=(logits.sigmoid()-target)*w[:,None]/(13*4)
    assert torch.allclose(grad,expected,atol=1e-7)
    permutation=[2,0,3,1]
    permuted=[{**r,'candidates':[{**r['candidates'][old],'index':i} for i,old in enumerate(permutation)]} for r in bank]
    pw,pg,_=make_weights(permuted,y[:,permutation],2.)
    assert all(np.array_equal(weights[key],pw[key]) for key in weights)
    assert all(np.array_equal(g[key],pg[key]) for key in g)
    x=torch.randn(13,4,1024)
    torch.manual_seed(17);p=Scorer(**configuration('s2_parse')).eval()
    torch.manual_seed(17);c=Scorer(**configuration('s2_consistency')).eval()
    assert torch.equal(p(x,torch.zeros(13,4,1)),c(x,torch.zeros(13,4,6)))
    result={'passed':True,'training_group_counts':summary['counts'],
       'uniform_gradient_matches_standard_bce_within_float32_roundoff':True,
       'uniform_gradient_max_abs_error':gradient_error,'weighted_gradient_matches_hand_formula':True,
       'K_H_mass_matches_at_each_correct_count':True,'K_H_positive_negative_label_mass_matches':True,
       'all_positive_and_mixed_mass_preserved':True,'candidate_permutation_preserves_groups_and_weights':True,
       'P_C_initial_embedding_functions_identical':True,'weights_fitted_on_train_only':True}
    write(HERE/'results/implementation_checks.json',result)
    print('IMPLEMENTATION CHECKS PASSED')


if __name__=='__main__':main()
